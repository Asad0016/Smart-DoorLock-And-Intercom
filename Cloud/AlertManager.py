import os
import time
import threading
import requests
from logs.logger import getLogger

from software.webrtc_stream import WebRTCSession
from Hardware.relay import MagneticLock


class AlertManager:
    def __init__(
        self,
        base_storage_path: str,
        secret_key: str,
        supabase_url: str,
        supabase_key: str,
        trainer_instance=None,
        camera_manager=None,
        mic_recorder=None,
        lock: MagneticLock = None,    # injected from main.py like camera and mic
    ):
        self.logger = getLogger("AlertManager")
        self.base_path = os.path.abspath(base_storage_path)
        self.secret_key = secret_key
        self.trainer = trainer_instance

        self.supabase_url = supabase_url.strip().rstrip("/")
        self.supabase_key = supabase_key.strip()
        self.cloud_alert_endpoint = f"{self.supabase_url}/rest/v1/door_alerts"

        # hardware refs
        self._camera = camera_manager
        self._mic = mic_recorder
        self._lock = lock             # MagneticLock instance

        # WebRTC session state
        self._webrtc_session: WebRTCSession | None = None
        self._webrtc_lock = threading.Lock()

        os.makedirs(self.base_path, exist_ok=True)
        self.logger.info(f"AlertManager initialised. Dataset: {self.base_path}")

    # ══════════════════════════════════════════════════════════
    #  EXISTING METHODS — unchanged
    # ══════════════════════════════════════════════════════════

    def verify_alert_auth(self, incoming_secret: str) -> bool:
        if not incoming_secret:
            self.logger.warning("Auth failure: missing security headers.")
            return False
        is_valid = incoming_secret == self.secret_key
        if not is_valid:
            self.logger.warning("Auth failure: invalid token.")
        return is_valid

    def handle_new_face_sync(self, data: dict) -> tuple[dict, int]:
        user_name  = data.get("user_name")
        public_url = data.get("public_url")
        event_type = data.get("event_type", "new_file")

        self.logger.info(f"Face sync received: user={user_name} event={event_type}")

        if not user_name or not public_url:
            self.logger.error("Missing user_name or public_url.")
            return {"status": "error", "message": "Missing user_name or public_url"}, 400

        target_dir = os.path.join(self.base_path, user_name)
        os.makedirs(target_dir, exist_ok=True)

        if event_type == "new_folder":
            self.logger.info(f"Profile node initialised for {user_name}")
            return {"status": "success", "message": f"Storage node created for '{user_name}'."}, 200

        file_name  = f"{user_name}_{int(time.time())}.jpg"
        final_path = os.path.join(target_dir, file_name)

        try:
            self.logger.info(f"Downloading asset: {file_name}")
            response = requests.get(public_url, stream=True, timeout=30)
            if response.status_code == 200:
                with open(final_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                self.logger.info(f"Asset saved: {final_path}")
                return {"status": "success", "local_path": final_path}, 200

            self.logger.error(f"Download failed. HTTP {response.status_code}")
            return {"status": "error", "message": f"HTTP {response.status_code}"}, 500

        except Exception as e:
            self.logger.error(f"Download exception: {e}")
            return {"status": "error", "message": str(e)}, 500

    def trigger_cloud_alert(self, event_type: str, message: str):
        threading.Thread(
            target=self._send_cloud_alert_worker,
            args=(event_type, message),
            daemon=True,
        ).start()

    def _send_cloud_alert_worker(self, event_type: str, message: str):
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        payload = {"event_type": event_type, "message": message, "status": "unread"}
        try:
            r = requests.post(self.cloud_alert_endpoint, json=payload, headers=headers, timeout=5)
            if r.status_code in (200, 201):
                self.logger.info(f"Cloud alert sent: [{event_type}]")
            else:
                self.logger.error(f"Cloud alert failed [{event_type}]: HTTP {r.status_code}")
        except requests.exceptions.Timeout:
            self.logger.error(f"Cloud alert timeout: [{event_type}]")
        except Exception as e:
            self.logger.error(f"Cloud alert exception [{event_type}]: {e}")

    def handle_intruder_alert(self, data: dict) -> tuple[dict, int]:
        self.logger.warning("INTRUDER ALERT: Hardware lockdown tripped!")
        return {"status": "lockdown_sequence_fired"}, 200

    # ══════════════════════════════════════════════════════════
    #  INTERCOM START
    # ══════════════════════════════════════════════════════════

    def start_intercom_stream(self, user_id: str):
        with self._webrtc_lock:
            if self._webrtc_session is not None:
                self.logger.warning("Intercom already active — ignoring duplicate request.")
                return

            if self._camera is None or self._mic is None:
                self.logger.error("Cannot start intercom: camera or mic not injected.")
                return

            self.logger.info(f"Starting intercom stream for user_id={user_id}")

            if not self._camera.streaming:
                self._camera.start_preview_stream(fps=15)
                self.logger.info("Camera preview stream started for intercom.")

            self._mic.start_recording()
            self.logger.info("Mic recording started for intercom.")

            session = WebRTCSession(
                camera       = self._camera,
                mic          = self._mic,
                supabase_url = self.supabase_url,
                supabase_key = self.supabase_key,
                user_id      = user_id,
            )
            session.start()
            self._webrtc_session = session
            self.logger.info("WebRTC session launched.")

    # ══════════════════════════════════════════════════════════
    #  INTERCOM STOP
    # ══════════════════════════════════════════════════════════

    def stop_intercom_stream(self):
        with self._webrtc_lock:
            if self._webrtc_session is None:
                self.logger.warning("No active intercom session to stop.")
                return

            self.logger.info("Stopping intercom stream.")

            self._webrtc_session.stop()
            self._webrtc_session = None

            mic_path = self._mic.stop_recording(custom_filename="intercom_discard.wav")
            if mic_path and os.path.exists(mic_path):
                os.remove(mic_path)
                self.logger.info("Intercom mic audio discarded.")

            self._camera.stop_preview_stream()
            self.logger.info("Intercom stream fully torn down.")

    # ══════════════════════════════════════════════════════════
    #  DOOR LATCH — uses MagneticLock.unlock() directly
    # ══════════════════════════════════════════════════════════

    def trigger_door_latch(self, hold_time: float = 3.0):
        """
        Called by webhook.py on 'unlock' command.
        Delegates entirely to MagneticLock.unlock() which handles
        GPIO LOW → hold → GPIO HIGH and its own error/safety fallback.
        Runs in a background thread so the webhook response returns instantly.
        """
        if self._lock is None:
            self.logger.error("Cannot unlock: MagneticLock not injected into AlertManager.")
            return

        self.logger.info("Door unlock command received — firing latch thread.")

        def _unlock_worker():
            success = self._lock.unlock(hold_time=hold_time)
            if success:
                self.trigger_cloud_alert("door_unlocked", "Door unlocked remotely via app.")
            else:
                self.trigger_cloud_alert("door_unlock_failed", "Door unlock hardware error.")

        threading.Thread(target=_unlock_worker, daemon=True, name="DoorLatch").start()