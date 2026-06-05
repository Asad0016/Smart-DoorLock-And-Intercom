"""
webrtc_stream.py

WebRTC streaming engine for DoorLock intercom.
Lives in: Hardware/webrtc_stream.py

Wiring:
  - Video  → CameraManager.get_next_frame()        (returns RGB ndarray from picamera2)
  - Audio  → INMP441MicRecorder.frames[]            (list of int16 bytes, filled by _record_loop)
  - Signal → Supabase REST (webrtc_sessions table)  (SDP offer/answer exchange)

Flow:
  1. AlertManager.start_intercom_stream(user_id) is called from webhook.py
  2. camera.start_preview_stream() + mic.start_recording() are started
  3. WebRTCSession.start() launches async loop in background thread
  4. Pi creates SDP offer → writes to Supabase webrtc_sessions table
  5. App reads offer → creates answer → patches row with status='answer'
  6. Pi polls for answer → sets remote description → stream goes live P2P
  7. AlertManager.stop_intercom_stream() tears everything down cleanly
"""

import asyncio
import threading
import time
import fractions

import cv2
import numpy as np
import aiohttp

from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, AudioStreamTrack
from av import VideoFrame, AudioFrame

from logs.logger import getLogger

logger = getLogger("WebRTCStream")


# ══════════════════════════════════════════════════════════════
#  VIDEO TRACK
#  Calls camera.get_next_frame() every tick.
#  picamera2 returns RGB ndarray → convert to BGR for av.
# ══════════════════════════════════════════════════════════════

class CameraVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, camera):
        super().__init__()
        self._camera        = camera   # CameraManager instance
        self._pts           = 0
        self._fps           = 15
        self._clock_rate    = 90000
        self._frame_dur     = self._clock_rate // self._fps
        self._blank         = np.zeros((480, 640, 3), dtype=np.uint8)

    async def recv(self) -> VideoFrame:
        # Pace ourselves to target FPS
        await asyncio.sleep(1.0 / self._fps)

        # get_next_frame() returns (H,W,3) RGB or None if paused/sleeping
        raw = self._camera.get_next_frame()

        if raw is None:
            raw = self._blank

        # Handle RGBA from picamera2 if streaming config returns 4 channels
        if raw.ndim == 3 and raw.shape[2] == 4:
            raw = cv2.cvtColor(raw, cv2.COLOR_RGBA2RGB)

        # picamera2 → RGB, av VideoFrame.from_ndarray expects bgr24
        bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

        av_frame                = VideoFrame.from_ndarray(bgr, format="bgr24")
        av_frame.pts            = self._pts
        av_frame.time_base      = fractions.Fraction(1, self._clock_rate)
        self._pts              += self._frame_dur
        return av_frame


# ══════════════════════════════════════════════════════════════
#  AUDIO TRACK
#  Reads directly from mic.frames[] — the same list that
#  INMP441MicRecorder._record_loop() appends int16 bytes into.
#  No monkey-patching needed — we just drain new entries each tick.
# ══════════════════════════════════════════════════════════════

class MicAudioTrack(AudioStreamTrack):
    kind = "audio"

    # 20ms frame at 48kHz = 960 samples — standard WebRTC packet size
    SAMPLES_PER_FRAME = 960

    def __init__(self, mic):
        super().__init__()
        self._mic           = mic      # INMP441MicRecorder instance
        self._pts           = 0
        self._sample_rate   = 48000    # must match mic.rate = 48000
        self._clock_rate    = 48000
        self._buffer        = np.array([], dtype=np.int16)
        self._read_index    = 0        # how many mic.frames entries we've consumed

    def _pull_new_frames(self):
        """
        mic._record_loop runs in its own daemon thread and appends
        volume-adjusted int16 bytes to mic.frames[].
        We read only entries added since our last call.
        """
        current_frames = self._mic.frames          # direct list reference
        new_entries    = current_frames[self._read_index:]
        self._read_index = len(current_frames)

        for chunk in new_entries:
            samples      = np.frombuffer(chunk, dtype=np.int16)
            self._buffer = np.concatenate([self._buffer, samples])

    async def recv(self) -> AudioFrame:
        # Pace to 20ms per packet
        await asyncio.sleep(self.SAMPLES_PER_FRAME / self._sample_rate)

        self._pull_new_frames()

        # Slice one 20ms frame; pad with silence if buffer not full yet
        if len(self._buffer) >= self.SAMPLES_PER_FRAME:
            samples      = self._buffer[:self.SAMPLES_PER_FRAME]
            self._buffer = self._buffer[self.SAMPLES_PER_FRAME:]
        else:
            samples = np.zeros(self.SAMPLES_PER_FRAME, dtype=np.int16)

        # aiortc needs: float32 planar, shape (1, N), range [-1.0, 1.0]
        audio_float = (samples.astype(np.float32) / 32768.0).reshape(1, -1)

        av_frame             = AudioFrame.from_ndarray(audio_float, format="fltp", layout="mono")
        av_frame.pts         = self._pts
        av_frame.sample_rate = self._sample_rate
        av_frame.time_base   = fractions.Fraction(1, self._clock_rate)
        self._pts           += self.SAMPLES_PER_FRAME
        return av_frame


# ══════════════════════════════════════════════════════════════
#  WEBRTC SESSION
#  Manages one full peer connection lifecycle.
#  Signalling travels via Supabase REST (webrtc_sessions table).
#  Actual media travels P2P directly to the phone — nothing
#  goes through Supabase after the handshake.
# ══════════════════════════════════════════════════════════════

class WebRTCSession:

    def __init__(
        self,
        camera,
        mic,
        supabase_url: str,
        supabase_key: str,
        user_id: str,
    ):
        self._camera        = camera
        self._mic           = mic
        self._supabase_url  = supabase_url.rstrip("/")
        self._supabase_key  = supabase_key
        self._user_id       = user_id

        self.pc: RTCPeerConnection | None = None
        self._running       = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    # ── Public API — called from AlertManager (sync context) ──

    def start(self):
        """Launch async session in a dedicated background thread."""
        if self._running:
            logger.warning("WebRTCSession.start() called but session already running.")
            return

        self._running = True
        self._thread  = threading.Thread(
            target   = self._run_event_loop,
            daemon   = True,
            name     = "WebRTCSession",
        )
        self._thread.start()
        logger.info("WebRTC session thread launched.")

    def stop(self):
        """Signal the session to tear down and close the peer connection."""
        logger.info("WebRTCSession.stop() called.")
        self._running = False

        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._teardown(), self._loop)

    # ── Thread entry point ────────────────────────────────────

    def _run_event_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._session())
        except Exception as e:
            logger.error(f"WebRTC event loop error: {e}")
        finally:
            self._loop.close()
            logger.info("WebRTC event loop closed.")

    # ── Main async session coroutine ──────────────────────────

    async def _session(self):
        self.pc = RTCPeerConnection()

        # Attach tracks built from your existing hardware objects
        self.pc.addTrack(CameraVideoTrack(self._camera))
        self.pc.addTrack(MicAudioTrack(self._mic))

        # ── Step 1: Create and publish SDP offer ──────────────
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        published = await self._publish_offer(self.pc.localDescription)
        if not published:
            logger.error("Failed to publish SDP offer — aborting session.")
            await self._teardown()
            return

        logger.info("SDP offer published to Supabase. Waiting for app answer (30s timeout)…")

        # ── Step 2: Poll Supabase for app's SDP answer ────────
        answer_data = await self._poll_for_answer(timeout_s=30)
        if not answer_data:
            logger.error("No SDP answer received from app within 30s — aborting.")
            await self._teardown()
            return

        # ── Step 3: Set remote description → stream goes live ─
        answer = RTCSessionDescription(
            sdp  = answer_data["sdp"],
            type = answer_data["type"],
        )
        await self.pc.setRemoteDescription(answer)
        logger.info("SDP answer applied — WebRTC stream is LIVE.")

        # ── Step 4: Hold session open until stop() is called ──
        while self._running:
            await asyncio.sleep(1)

        await self._teardown()

    # ── Supabase signalling helpers ───────────────────────────

    async def _publish_offer(self, local_desc) -> bool:
        """Write or overwrite Pi's SDP offer into webrtc_sessions table via UPSERT."""
        url     = f"{self._supabase_url}/rest/v1/webrtc_sessions"
        
        # 🔥 OPTIMIZED: Added resolution headers to handle rapid sequential clicks gracefully
        headers = self._headers()
        headers["Prefer"] = "resolution=merge-duplicates" 

        payload = {
            "user_id": self._user_id,
            "sdp":     local_desc.sdp,
            "type":    local_desc.type,
            "status":  "offer",
        }
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    url, json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as r:
                    if r.status in (200, 201):
                        logger.info(f"Offer published/updated successfully. HTTP {r.status}")
                        return True
                    logger.error(f"Offer publish failed. HTTP {r.status}: {await r.text()}")
                    return False
        except Exception as e:
            logger.error(f"Offer publish exception: {e}")
            return False

    async def _poll_for_answer(self, timeout_s: int) -> dict | None:
        """
        Poll webrtc_sessions table for a row with status='answer'
        belonging to this user_id. Retries every 1s up to timeout_s.
        """
        # 🔥 OPTIMIZED: Ordered by an auto-incrementing id or timestamp if available, 
        # or strictly grabbing the single live updated answer context.
        url     = (
            f"{self._supabase_url}/rest/v1/webrtc_sessions"
            f"?user_id=eq.{self._user_id}"
            f"&status=eq.answer"
            f"&select=sdp,type"
            f"&limit=1"
        )
        headers  = self._headers()
        deadline = time.time() + timeout_s

        async with aiohttp.ClientSession() as s:
            while time.time() < deadline and self._running:
                try:
                    async with s.get(
                        url, headers=headers,
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as r:
                        data = await r.json()
                        if data and len(data) > 0:
                            logger.info("Valid SDP answer found in Supabase! Completing handshake...")
                            return data[0]
                except Exception as e:
                    logger.debug(f"Polling for answer: {e}")

                await asyncio.sleep(1)

        logger.warning("WebRTC Handshake Timeout: The phone application did not write back its SDP answer.")
        return None

    async def _teardown(self):
        """Close peer connection and delete signalling row from Supabase."""
        if self.pc:
            await self.pc.close()
            self.pc = None
            logger.info("RTCPeerConnection closed.")

        await self._delete_signalling_row()

    async def _delete_signalling_row(self):
        """Clean up the offer/answer row so stale entries don't accumulate."""
        url     = f"{self._supabase_url}/rest/v1/webrtc_sessions?user_id=eq.{self._user_id}"
        headers = self._headers()
        try:
            async with aiohttp.ClientSession() as s:
                async with s.delete(
                    url, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as r:
                    logger.info(f"Signalling row deleted from Supabase. HTTP {r.status}")
        except Exception as e:
            logger.error(f"Failed to delete signalling row: {e}")

    def _headers(self) -> dict:
        return {
            "apikey":        self._supabase_key,
            "Authorization": f"Bearer {self._supabase_key}",
            "Content-Type":  "application/json",
            "Prefer":        "return=minimal",
        }