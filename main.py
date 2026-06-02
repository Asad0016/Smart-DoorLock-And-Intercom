"""
main.py - Face Recognition Door Lock System (Updated & Refactored)

Key behaviour:
  • Smart-train: only retrains when encodings.pickle is missing/empty
    OR the dataset folder contents actually changed (hash mismatch).
    Startup with an unchanged dataset → NO retrain.

  • Motion detected → camera starts → recording begins IMMEDIATELY.
  • While recording:
      - Known face OR RFID authorised  → abort clip, delete temp file, sleep camera.
      - Full RECORDING_DURATION elapsed with NO authorised person → upload clip
        to Supabase, delete local copy, sleep camera, wait for next motion.
"""

import hashlib
import json
import os
import tempfile
import threading
import time
from Hardware.relay import MagneticLock
import cv2
from Hardware.relay import MagneticLock
# Environmental control variables parsing library
from dotenv import load_dotenv

from Cloud.AlertManager import AlertManager
from Cloud.supabase import SupabaseManager
from Hardware.camera import CameraManager
from Hardware.rfid import RFIDManager
from Hardware.MotionSensor import PIRSENSOR
from logs.logger import getLogger
from software.FaceDetection import ModelTraining
from Hardware.mic import INMP441MicRecorder
# Imported cleanly from your new decoupled module
from Cloud.webhook import start_webhook_server

# ── Load Configuration from Environment File ─────────────────
# Path layout context parsing configuration variables
env_path = "/home/doorlock/DoorLock/Cloud/credential.env"
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://qiyqwbkuknogegycahqm.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_jYBUpnRA1U2PWu_ssvOvMA_sumfxMy8")

# ─────────────────────────── CONFIG ───────────────────────────
DATASET_PATH        = "/home/doorlock/DoorLock/dataset"
ENCODINGS_PATH      = "/home/doorlock/DoorLock/dataset/encodings.pickle"
DATASET_HASH_PATH   = "/home/doorlock/DoorLock/dataset/.dataset_hash.json"

DETECTION_MODEL     = "hog"
ENCODING_MODEL      = "large"
TOLERANCE           = 0.5

CAMERA_RESOLUTION         = (1920, 1080)
CAMERA_PREVIEW_RESOLUTION = (640, 480)
CAMERA_FPS                = 30
RECORDING_DURATION        = 10   # seconds of footage before uploading
RECORDING_FPS             = 20

CAMERA_IDLE_TIMEOUT       = 8    # seconds of no motion before sleeping camera

WEBHOOK_SECRET      = "DoorLock123"
WEBHOOK_HOST        = "0.0.0.0"
WEBHOOK_PORT        = 5050

logger = getLogger("Main")


# ══════════════════════════════════════════════════════════════
#  DATASET HASH  — content-based, immune to mtime/copy changes
# ══════════════════════════════════════════════════════════════

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def compute_dataset_hash(dataset_path: str) -> str:
    """
    SHA-256 built from:
      • The relative file path  (catches renames / new files)
      • The SHA-256 of the file's actual bytes  (catches content changes)
    """
    outer = hashlib.sha256()
    for root, dirs, files in sorted(os.walk(dataset_path)):
        dirs.sort()
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _IMAGE_EXTENSIONS:
                continue                      # skip pickle, json, hidden files, etc.
            full = os.path.join(root, fname)
            rel  = os.path.relpath(full, dataset_path)

            # Hash the actual file contents
            inner = hashlib.sha256()
            with open(full, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    inner.update(chunk)

            # Contribute  "relative/path → content_hash"  to the outer digest
            outer.update(f"{rel}={inner.hexdigest()}\n".encode())

    return outer.hexdigest()


def load_saved_hash() -> str | None:
    if os.path.exists(DATASET_HASH_PATH):
        try:
            with open(DATASET_HASH_PATH, "r") as f:
                return json.load(f).get("hash")
        except Exception:
            pass
    return None


def save_dataset_hash(digest: str) -> None:
    os.makedirs(os.path.dirname(DATASET_HASH_PATH), exist_ok=True)
    with open(DATASET_HASH_PATH, "w") as f:
        json.dump({"hash": digest, "timestamp": time.time()}, f)


# ══════════════════════════════════════════════════════════════
#  SMART TRAINING
# ══════════════════════════════════════════════════════════════

def smart_train(model: ModelTraining, force: bool = False) -> bool:
    current_hash = compute_dataset_hash(DATASET_PATH)
    saved_hash   = load_saved_hash()
    encodings_ok = (
        os.path.exists(ENCODINGS_PATH)
        and os.path.getsize(ENCODINGS_PATH) > 0
    )

    if not force and encodings_ok and current_hash == saved_hash:
        logger.info("Smart-train: Hashes match perfectly. Loading encodings into RAM.")
        return model._load_encodings()

    reason = "Forced retrain" if force else ("Missing encodings" if not encodings_ok else "Dataset modification detected")
    logger.info(f"Smart-train: Starting training [{reason}] …")
    
    success = model.train_model(force=True, current_hash=current_hash)
    return success


# ══════════════════════════════════════════════════════════════
#  SHARED STATE  (thread-safe)
# ══════════════════════════════════════════════════════════════

class DoorLockState:
    def __init__(self):
        self._lock         = threading.Lock()
        self.authorized    = False
        self.motion_active = False
        self.motion_locked = False
        self.camera_ready  = False

    def set_authorized(self, value: bool) -> None:
        with self._lock:
            self.authorized = value

    def consume_authorized(self) -> bool:
        with self._lock:
            val = self.authorized
            self.authorized = False
            return val

    def set_motion(self, value: bool) -> None:
        with self._lock:
            self.motion_active = value

    def is_motion(self) -> bool:
        with self._lock:
            if self.motion_locked:
                return False
            return self.motion_active

    def lock_motion(self) -> None:
        with self._lock:
            self.motion_locked = True
        logger.info("Motion gate LOCKED — PIR ignored until record/upload cycle completes.")

    def unlock_motion(self) -> None:
        with self._lock:
            self.motion_locked = False
            self.camera_ready  = False
        logger.info("Motion gate UNLOCKED — PIR active, system ready for next event.")

    def mark_camera_ready(self) -> None:
        with self._lock:
            self.camera_ready = True

    def is_camera_ready(self) -> bool:
        with self._lock:
            return self.camera_ready

    def reset_camera_ready(self) -> None:
        with self._lock:
            self.camera_ready = False


# ══════════════════════════════════════════════════════════════
#  BACKGROUND SENSOR THREAD  (PIR + RFID)
# ══════════════════════════════════════════════════════════════

def sensor_polling_thread(pir: PIRSENSOR, rfid: RFIDManager, state: DoorLockState) -> None:
    logger.info("Sensor polling thread started (PIR + RFID, 100 ms interval).")
    rfid_available = rfid.reader is not None
    if not rfid_available:
        logger.warning("Sensor thread: RFID reader unavailable — PIR + face recognition active.")
        
    while True:
        state.set_motion(pir.is_motion_active())

        if rfid_available:
            try:
                if rfid.is_authorized_card():
                    logger.info("RFID: authorised card tapped → flagging state.")
                    state.set_authorized(True)
            except Exception as e:
                logger.debug(f"RFID poll error (non-fatal): {e}")

        time.sleep(0.1)


# ══════════════════════════════════════════════════════════════
#  CAMERA STARTUP THREAD
# ══════════════════════════════════════════════════════════════

def camera_startup_thread(camera: CameraManager, state: DoorLockState) -> None:
    logger.info("Camera startup thread: starting preview stream …")
    camera.start_preview_stream(fps=CAMERA_FPS)

    for _ in range(100):
        frame = camera.get_next_frame()
        if frame is not None:
            state.mark_camera_ready()
            logger.info("Camera startup thread: first frame received — camera is ready.")
            return
        time.sleep(0.05)

    state.mark_camera_ready()
    logger.warning("Camera startup thread: timed out waiting for first frame.")


# ══════════════════════════════════════════════════════════════
#  UPLOAD & CLEANUP
# ═══════════════════════════════════════════════════════════

def upload_and_cleanup(
    storage: SupabaseManager | None,
    video_path: str,
    state: DoorLockState,
) -> None:
    try:
        if storage and video_path and os.path.exists(video_path):
            logger.info("Uploading intruder clip to Supabase …")
            try:
                url = storage.upload_and_get_url(video_path)
                if url:
                    logger.info(f"Upload complete. Public link: {url}")
                else:
                    logger.error("Upload returned no URL.")
            except Exception as e:
                logger.error(f"Upload failed: {e}")
        else:
            if not storage:
                logger.warning("Supabase not connected — clip NOT uploaded.")
            else:
                logger.warning("Temp video path missing — nothing to upload.")
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
            logger.info("Local clip deleted from RPi storage.")
        state.unlock_motion()


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _sleep_camera(camera: CameraManager, state: DoorLockState) -> dict:
    camera.stop_preview_stream()
    state.reset_camera_ready()
    logger.info("Camera sleeping — waiting for next motion event.")
    return {
        "camera_active":     False,
        "camera_starting":   False,
        "first_frame_time":  None,
        "last_motion_time":  None,
    }

def _merge_audio_video(video_path: str, audio_path: str | None) -> str:
    """Merge mic WAV into the video using ffmpeg. Returns path to merged file."""
    if not audio_path or not os.path.exists(audio_path):
        logger.warning("No audio file to merge — uploading video only.")
        return video_path

    merged_path = video_path.replace(".mp4", "_merged.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",       # no re-encode, fast
        "-c:a", "aac",        # convert WAV → AAC for mp4 container
        "-shortest",          # trim to the shorter stream
        merged_path
    ]
    try:
        import subprocess
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0:
            logger.info(f"Audio merged → {merged_path}")
            os.remove(video_path)
            os.remove(audio_path)
            return merged_path
        else:
            logger.error(f"ffmpeg merge failed: {result.stderr.decode()}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return video_path   # fall back to video-only
    except Exception as e:
        logger.error(f"ffmpeg error: {e}")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return video_path
# ══════════════════════════════════════════════════════════════
#  MAIN EXECUTION CONTEXT
# ══════════════════════════════════════════════════════════════

def main():
    # ── 1. Supabase Initialization ────────────────────────────
    logger.info("Connecting to Supabase …")
    storage = None
    try:
        storage = SupabaseManager()
        logger.info("Supabase connected.")
    except Exception as e:
        logger.warning(f"Supabase unavailable: {e}  →  continuing offline.")

    # ── 2. Sync Cloud Assets ──────────────────────────────────
    if storage:
        logger.info("Importing dataset from cloud bucket …")
        try:
            storage.import_entire_dataset(DATASET_PATH)
            logger.info("Dataset import complete.")
        except Exception as e:
            logger.error(f"Dataset import failed: {e}")

    # ── 3. Face Model Setup ───────────────────────────────────
    logger.info("Initialising Face Recognition model …")
    model = ModelTraining(
        dataset_path    = DATASET_PATH,
        encodings_path  = ENCODINGS_PATH,
        detection_model = DETECTION_MODEL,
        encoding_model  = ENCODING_MODEL,
        tolerance       = TOLERANCE,
    )

    # ── 4. Structural Verification & Boot Training ───────────
    dataset_has_images = any(
        os.path.splitext(fname)[1].lower() in _IMAGE_EXTENSIONS
        for _, _, files in os.walk(DATASET_PATH)
        for fname in files
    )

    if dataset_has_images:
        current_hash = compute_dataset_hash(DATASET_PATH)
        encodings_exist = os.path.exists(ENCODINGS_PATH) and os.path.getsize(ENCODINGS_PATH) > 0
        saved_hash = load_saved_hash()
        if encodings_exist and saved_hash is None:
            logger.info("Found existing encodings but no hash file. Creating hash to prevent false retraining.")
            save_dataset_hash(current_hash)
        smart_train(model, force=False)  
    else:
        logger.warning(
            "Dataset directory contains no images — skipping training. "
            "Training will trigger automatically when the first image arrives via webhook."
        )

    # ── 5. Modularized Webhook Server Bootstrap ───────────────
    # 💡 UPDATED: Initializing alert_mgr with dynamic credentials read from .env
# initialise with your actual GPIO pin
    
    # Handing over runtime dependencies securely to webhook.py
    start_webhook_server(
        alert_mgr=alert_mgr,
        model=model,
        smart_train_func=smart_train,
        host=WEBHOOK_HOST,
        port=WEBHOOK_PORT,
        secret=WEBHOOK_SECRET
    )

    # ── 6. Hardware Context Deployment ────────────────────────
    logger.info("Initialising PIR sensor …")
    pir = PIRSENSOR(pir_pin=23)

    logger.info("Initialising RFID reader …")
    rfid = RFIDManager()

    state = DoorLockState()

    threading.Thread(
        target=sensor_polling_thread,
        args=(pir, rfid, state),
        daemon=True,
        name="SensorPolling",
    ).start()

    # ── 7. Camera Handlers ────────────────────────────────────
    logger.info("Initialising camera …")
    camera = CameraManager(
        resolution         = CAMERA_RESOLUTION,
        framerate          = CAMERA_FPS,
        preview_resolution = CAMERA_PREVIEW_RESOLUTION,
    )
    if not camera.initialize_camera():
        logger.error("Camera failed to initialise. Exiting.")
        pir.cleanup()
        rfid.stop()
        return
    logger.info("initializing microphone...")
    mic = INMP441MicRecorder()
    logger.info("initializing lock driver...")
    lock = MagneticLock(pin=18)


    alert_mgr = AlertManager(
        base_storage_path = DATASET_PATH,
        secret_key        = WEBHOOK_SECRET,
        supabase_url      = SUPABASE_URL,
        supabase_key      = SUPABASE_KEY,
        camera_manager    = camera,
        mic_recorder      = mic,
        lock              = lock,
    )
    logger.info("System ready — waiting for motion …  (press 'q' to quit)")
    # ── Main Loop State Engine ────────────────────────────────
    camera_active     = False
    camera_starting   = False
    recording         = False
    video_writer      = None
    temp_video_path   = None
    record_start_time = None
    last_motion_time  = None
    first_frame_time  = None

    try:
        while True:
            motion_now = state.is_motion()

            # ── Trigger camera startup on motion ──────────────
            if motion_now:
                last_motion_time = time.time()
                if not camera_active and not camera_starting:
                    logger.info("Motion detected → launching camera startup thread …")
                    
                    # 💡 UPSTREAM REALTIME WARNING TRIGGER
                    # Background thread executes instantly to warn the database & app
                    alert_mgr.trigger_cloud_alert("motion_detected", "Someone is at the door!")
                    
                    state.reset_camera_ready()
                    camera_starting = True
                    threading.Thread(
                        target=camera_startup_thread,
                        args=(camera, state),
                        daemon=True,
                        name="CameraStartup",
                    ).start()

            # ── Camera became ready → begin recording immediately ──
            if camera_starting and state.is_camera_ready():
                camera_active    = True
                camera_starting  = False
                first_frame_time = time.time()

                if not recording:
                    logger.info("Camera live → locking motion gate and starting recording immediately …")
                    state.lock_motion()

                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                        temp_video_path = tmp.name

                    fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        temp_video_path, fourcc, RECORDING_FPS, CAMERA_PREVIEW_RESOLUTION
                    )
                    recording         = True
                    record_start_time = time.time()
                    logger.warning(f"Recording started → {temp_video_path}")
                    mic.start_recording()
                    logger.info("Mic recording started alongside video.")
            # ── Idle timeout (only when NOT recording) ────────
            if (
                camera_active
                and not recording
                and not motion_now
                and first_frame_time is not None
            ):
                idle_baseline = max(first_frame_time, last_motion_time or 0)
                if (time.time() - idle_baseline) > CAMERA_IDLE_TIMEOUT:
                    logger.info(f"No motion for {CAMERA_IDLE_TIMEOUT}s → sleeping camera.")
                    rst = _sleep_camera(camera, state)
                    camera_active    = rst["camera_active"]
                    camera_starting  = rst["camera_starting"]
                    first_frame_time = rst["first_frame_time"]
                    last_motion_time = rst["last_motion_time"]

            if not camera_active:
                time.sleep(0.05)
                continue

            # ── Frame Processing Sequence ────────────────────
            frame = camera.get_next_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # ── Neural Network Processing ─────────────────────
            frame = model.process_frame(frame)
            frame = model.draw_results(frame)
            detected_names = model.get_detected_names()

            known_face_present = any(
                name.lower() not in ("unknown", "") for name in detected_names
            )

            rfid_authorized = state.consume_authorized()

            # ── Access Control Decision Matrix ────────────────
            if recording:
                elapsed = time.time() - record_start_time

                # Authorized → abort clip, sleep camera instantly
                if known_face_present or rfid_authorized:
                    reason = "Known face" if known_face_present else "RFID"
                    logger.info(f"{reason} detected → authorised access. Aborting clip and sleeping camera.")
                    
                    # Telemetry push on success operations
                    if known_face_present:
                        authorized_user = [n for n in detected_names if n.lower() not in ("unknown", "")][0]
                        alert_mgr.trigger_cloud_alert("face_success", f"Welcome back! {authorized_user} unlocked the door.")
                    else:
                        alert_mgr.trigger_cloud_alert("rfid_success", "Door unlocked via authorized RFID Tag.")
                    threading.Thread(
                        target=lock.unlock,  # direct hardware call to unlock the door
                        kwargs={"hold_time": 3.0},
                        daemon=True,
                        name="DoorUnlock",
                    ).start()
                    video_writer.release()
                    mic_path = mic.stop_recording(custom_filename=temp_video_path.replace(".mp4", "_audio.wav"))
                    if mic_path and os.path.exists(mic_path):
                        os.remove(mic_path)
                        logger.info("Associated mic recording deleted.")
                    if temp_video_path and os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
                        logger.info("Aborted clip deleted.")

                    recording         = False
                    video_writer      = None
                    temp_video_path   = None
                    record_start_time = None

                    rst = _sleep_camera(camera, state)
                    camera_active    = rst["camera_active"]
                    camera_starting  = rst["camera_starting"]
                    first_frame_time = rst["first_frame_time"]
                    last_motion_time = rst["last_motion_time"]
                    state.unlock_motion()
                    continue

                # Stream current matrix array onto disk
                video_writer.write(frame)

                # Recording duration reached without clearance → upload alert sequence
                if elapsed >= RECORDING_DURATION:
                    logger.warning(f"Recording complete ({RECORDING_DURATION}s) — breach verified → uploading clip …")
                    
                    # Log unauthorized breach event on the cloud telemetry table
                    alert_mgr.trigger_cloud_alert("unauthorized_breach", "Security Breach: Unrecognized entity verified at physical asset terminal.")
                    
                    video_writer.release()
                    mic_path = mic.stop_recording(custom_filename="temp_intruder_mic.wav")
                    if mic_path and os.path.exists(mic_path):
                        clip_path = _merge_audio_video(temp_video_path, mic_path)
                    else:
                        clip_path = temp_video_path
                    temp_video_path   = None
                    video_writer      = None
                    recording         = False
                    record_start_time = None

                    rst = _sleep_camera(camera, state)
                    camera_active    = rst["camera_active"]
                    camera_starting  = rst["camera_starting"]
                    first_frame_time = rst["first_frame_time"]
                    last_motion_time = rst["last_motion_time"]

                    threading.Thread(
                        target=upload_and_cleanup,
                        args=(storage, clip_path, state),
                        daemon=True,
                        name="UploadClip",
                    ).start()
                    continue

            # ── Display Interface Engine ──────────────────────
            if recording and record_start_time:
                remaining = max(0, RECORDING_DURATION - (time.time() - record_start_time))
                hud_text  = f"RECORDING {remaining:.1f}s"
                hud_color = (0, 0, 255)
            else:
                hud_text  = "Monitoring"
                hud_color = (0, 255, 0)

            cv2.putText(
                frame, hud_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2,
            )
            cv2.imshow("Face Recognition Door Lock", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("System interrupted by user.")
    finally:
        if video_writer:
            video_writer.release()
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        cv2.destroyAllWindows()
        camera.cleanup()
        pir.cleanup()
        mic.cleanup()
        rfid.stop()
        lock.cleanup()  
        logger.info("System shutdown complete.")


if __name__ == "__main__":
    main()