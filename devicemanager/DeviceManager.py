"""
DeviceManager.py
Owns and manages the full lifecycle of every hardware component,
cloud service, and software module in the DoorLock system.

Main.py only creates a DeviceManager and calls .start() / .stop().
All initialization, wiring, and smart-train logic lives here.
"""

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
import cv2

from config import (
    SUPABASE_URL, SUPABASE_KEY,
    DATASET_PATH, ENCODINGS_PATH, DATASET_HASH_PATH,
    DETECTION_MODEL, ENCODING_MODEL, TOLERANCE, CV_SCALER,
    CAMERA_RESOLUTION, CAMERA_PREVIEW_RESOLUTION, CAMERA_FPS,
    RECORDING_DURATION, RECORDING_FPS, CAMERA_IDLE_TIMEOUT,
    LOCK_GPIO_PIN, PIR_GPIO_PIN,
    WEBHOOK_SECRET, WEBHOOK_HOST, WEBHOOK_PORT,
    MIC_DEVICE_STRING, MIC_RATE, MIC_CHANNELS, MIC_VOLUME_GAIN, MIC_OUTPUT_FOLDER,
    IMAGE_EXTENSIONS,
)

from Cloud.AlertManager        import AlertManager
from Cloud.supabase            import SupabaseManager
from Cloud.webhook             import start_webhook_server
from Hardware.camera           import CameraManager
from Hardware.mic              import INMP441MicRecorder
from Hardware.relay            import MagneticLock
from Hardware.rfid             import RFIDManager
from Hardware.MotionSensor     import PIRSENSOR
from software.FaceDetection    import ModelTraining
from logs.logger               import getLogger


# ══════════════════════════════════════════════════════════════
#  SHARED STATE  (thread-safe) — lives inside DeviceManager
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
            return False if self.motion_locked else self.motion_active

    def lock_motion(self) -> None:
        with self._lock:
            self.motion_locked = True

    def unlock_motion(self) -> None:
        with self._lock:
            self.motion_locked = False
            self.camera_ready  = False

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
#  DEVICE MANAGER
# ══════════════════════════════════════════════════════════════

class DeviceManager:
    """
    Single owner of every resource in the DoorLock system.
    
    Lifecycle:
        dm = DeviceManager()
        dm.boot()           ← initializes everything
        dm.run_loop()       ← blocking main loop
        dm.shutdown()       ← called automatically on exit
    """

    def __init__(self):
        self.logger = getLogger("DeviceManager")

        # ── Cloud ──────────────────────────────────────────────
        self.storage   : SupabaseManager | None = None
        self.alert_mgr : AlertManager    | None = None

        # ── Hardware ───────────────────────────────────────────
        self.camera : CameraManager      | None = None
        self.mic    : INMP441MicRecorder | None = None
        self.lock   : MagneticLock       | None = None
        self.pir    : PIRSENSOR          | None = None
        self.rfid   : RFIDManager        | None = None

        # ── Software ───────────────────────────────────────────
        self.model  : ModelTraining      | None = None

        # ── Runtime state ──────────────────────────────────────
        self.state  = DoorLockState()
        self._running = False

        # ── Main loop variables ────────────────────────────────
        self._camera_active     = False
        self._camera_starting   = False
        self._recording         = False
        self._video_writer      = None
        self._temp_video_path   = None
        self._record_start_time = None
        self._last_motion_time  = None
        self._first_frame_time  = None

    # ══════════════════════════════════════════════════════════
    #  BOOT SEQUENCE
    # ══════════════════════════════════════════════════════════

    def boot(self) -> bool:
        """
        Full system initialization in correct dependency order.
        Returns False if any critical component fails.
        """
        self.logger.info("═══════ DoorLock DeviceManager Boot Sequence ═══════")

        if not self._init_cloud():
            self.logger.warning("Cloud unavailable — continuing offline.")

        if not self._init_model():
            self.logger.error("Face model failed — cannot continue.")
            return False

        encodings_exist = (
        os.path.exists(ENCODINGS_PATH)
        and os.path.getsize(ENCODINGS_PATH) > 0
        )
        if not encodings_exist:
            self.logger.info("No encodings found — running initial training.")
            self._smart_train(force=True)
        else:
            self.logger.info("Encodings already loaded — skipping boot training.")

        if not self._init_hardware():
            self.logger.error("Hardware initialization failed — cannot continue.")
            return False

        self._init_alert_manager()
        self._init_webhook()
        self._start_sensor_thread()

        self.logger.info("═══════ Boot complete — system ready ═══════")
        return True

    # ── Cloud ──────────────────────────────────────────────────

    def _init_cloud(self) -> bool:
        self.logger.info("Connecting to Supabase…")
        try:
            self.storage = SupabaseManager()
            self.logger.info("Supabase connected.")
            self._sync_dataset()
            return True
        except Exception as e:
            self.logger.warning(f"Supabase unavailable: {e}")
            self.storage = None
            return False

    def _sync_dataset(self):
        if not self.storage:
            return
        self.logger.info("Syncing dataset from cloud bucket…")
        try:
            self.storage.import_entire_dataset(DATASET_PATH)
            self.logger.info("Dataset sync complete.")
        except Exception as e:
            self.logger.warning(f"Dataset sync failed (using local): {e}")

    # ── Face Model ─────────────────────────────────────────────

    def _init_model(self) -> bool:
        self.logger.info("Initialising Face Recognition model…")
        try:
            self.model = ModelTraining(
                dataset_path    = DATASET_PATH,
                encodings_path  = ENCODINGS_PATH,
                detection_model = DETECTION_MODEL,
                encoding_model  = ENCODING_MODEL,
                cv_scaler       = CV_SCALER,
                tolerance       = TOLERANCE,
            )
            self.logger.info("Face model initialized.")
            return True
        except Exception as e:
            self.logger.error(f"Model init failed: {e}")
            return False

    # ── Smart Train ────────────────────────────────────────────

    def _smart_train(self, force: bool = False) -> bool:
        """
        Content-based smart training — only retrains when dataset changed.
        Computes SHA-256 over all image files and compares to saved hash.
        """
        dataset_has_images = any(
            os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            for _, _, files in os.walk(DATASET_PATH)
            for f in files
        )

        if not dataset_has_images:
            self.logger.warning("No images in dataset — skipping training.")
            return False

        current_hash  = self._compute_dataset_hash()
        saved_hash    = self._load_saved_hash()
        encodings_ok  = (
            os.path.exists(ENCODINGS_PATH)
            and os.path.getsize(ENCODINGS_PATH) > 0
        )

        if not force and encodings_ok and current_hash == saved_hash:
            self.logger.info("Smart-train: dataset unchanged — loading encodings.")
            return self.model._load_encodings()

        reason = "forced" if force else ("no encodings" if not encodings_ok else "dataset changed")
        self.logger.info(f"Smart-train: retraining [{reason}]…")

        if encodings_ok and saved_hash is None:
            self._save_dataset_hash(current_hash)

        return self.model.train_model(force=force, current_hash=current_hash)

    def _compute_dataset_hash(self) -> str:
        outer = hashlib.sha256()
        for root, dirs, files in sorted(os.walk(DATASET_PATH)):
            dirs.sort()
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() not in IMAGE_EXTENSIONS:
                    continue
                full  = os.path.join(root, fname)
                rel   = os.path.relpath(full, DATASET_PATH)
                inner = hashlib.sha256()
                with open(full, "rb") as fh:
                    for chunk in iter(lambda: fh.read(65536), b""):
                        inner.update(chunk)
                outer.update(f"{rel}={inner.hexdigest()}\n".encode())
        return outer.hexdigest()

    def _load_saved_hash(self) -> str | None:
        if os.path.exists(DATASET_HASH_PATH):
            try:
                with open(DATASET_HASH_PATH) as f:
                    return json.load(f).get("hash")
            except Exception:
                pass
        return None

    def _save_dataset_hash(self, digest: str) -> None:
        os.makedirs(os.path.dirname(DATASET_HASH_PATH), exist_ok=True)
        with open(DATASET_HASH_PATH, "w") as f:
            json.dump({"hash": digest, "timestamp": time.time()}, f)

    # ── Hardware ───────────────────────────────────────────────

    def _init_hardware(self) -> bool:
        self.logger.info("Initialising PIR sensor…")
        try:
            self.pir = PIRSENSOR(pir_pin=PIR_GPIO_PIN)
        except Exception as e:
            self.logger.error(f"PIR failed: {e}")
            return False

        self.logger.info("Initialising RFID reader…")
        try:
            self.rfid = RFIDManager()
        except Exception as e:
            self.logger.warning(f"RFID unavailable: {e}")
            self.rfid = None

        self.logger.info("Initialising camera…")
        try:
            self.camera = CameraManager(
                resolution         = CAMERA_RESOLUTION,
                framerate          = CAMERA_FPS,
                preview_resolution = CAMERA_PREVIEW_RESOLUTION,
            )
            if not self.camera.initialize_camera():
                self.logger.error("Camera failed to initialize.")
                return False
        except Exception as e:
            self.logger.error(f"Camera init error: {e}")
            return False

        self.logger.info("Initialising microphone…")
        try:
            self.mic = INMP441MicRecorder(
                rate          = MIC_RATE,
                channels      = MIC_CHANNELS,
                device_string = MIC_DEVICE_STRING,
                output_folder = MIC_OUTPUT_FOLDER,
                volume_gain   = MIC_VOLUME_GAIN,
            )
        except Exception as e:
            self.logger.warning(f"Mic init error: {e}")
            self.mic = None

        self.logger.info("Initialising door lock relay…")
        try:
            self.lock = MagneticLock(pin=LOCK_GPIO_PIN)
        except Exception as e:
            self.logger.error(f"Lock relay init error: {e}")
            self.lock = None

        self.logger.info("All hardware initialized.")
        return True

    # ── AlertManager ───────────────────────────────────────────

    def _init_alert_manager(self):
        self.logger.info("Initialising AlertManager…")
        self.alert_mgr = AlertManager(
            base_storage_path = DATASET_PATH,
            secret_key        = WEBHOOK_SECRET,
            supabase_url      = self.storage.supabase_url if self.storage else SUPABASE_URL,
            supabase_key      = self.storage.supabase_key if self.storage else SUPABASE_KEY,
            camera_manager    = self.camera,
            mic_recorder      = self.mic,
            lock              = self.lock,
        )
        self.logger.info("AlertManager ready.")

    # ── Webhook ────────────────────────────────────────────────

    def _init_webhook(self):
        self.logger.info(f"Starting webhook server on port {WEBHOOK_PORT}…")
        start_webhook_server(
            alert_mgr        = self.alert_mgr,
            model            = self.model,
            smart_train_func = self._smart_train,
            host             = WEBHOOK_HOST,
            port             = WEBHOOK_PORT,
            secret           = WEBHOOK_SECRET,
        )
        self.logger.info("Webhook server running.")

    # ── Sensor polling thread ──────────────────────────────────

    def _start_sensor_thread(self):
        threading.Thread(
            target = self._sensor_polling_loop,
            daemon = True,
            name   = "SensorPolling",
        ).start()
        self.logger.info("Sensor polling thread started.")

    def _sensor_polling_loop(self):
        rfid_ok = self.rfid and self.rfid.reader is not None
        if not rfid_ok:
            self.logger.warning("RFID unavailable — PIR + face recognition only.")

        while True:
            self.state.set_motion(self.pir.is_motion_active())
            if rfid_ok:
                try:
                    if self.rfid.is_authorized_card():
                        self.logger.info("RFID: authorized card tapped.")
                        self.state.set_authorized(True)
                except Exception as e:
                    self.logger.debug(f"RFID poll error: {e}")
            time.sleep(0.1)

    # ══════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ══════════════════════════════════════════════════════════

    def run_loop(self):
        """Blocking main loop — call this from main.py after boot()."""
        self._running = True
        self.logger.info("Main loop started — waiting for motion… (press 'q' to quit)")

        try:
            while self._running:
                self._tick()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user.")
        finally:
            self.shutdown()

    def _tick(self):
        """One iteration of the main loop."""
        motion_now = self.state.is_motion()

        # ── Motion detected → start camera ────────────────────
        if motion_now:
            self._last_motion_time = time.time()
            if not self._camera_active and not self._camera_starting:
                self.logger.info("Motion detected → starting camera…")
                self.alert_mgr.trigger_cloud_alert("motion_detected", "Someone is at the door!")
                self.state.reset_camera_ready()
                self._camera_starting = True
                threading.Thread(
                    target = self._camera_startup,
                    daemon = True,
                    name   = "CameraStartup",
                ).start()

        # ── Camera ready → start recording ────────────────────
        if self._camera_starting and self.state.is_camera_ready():
            self._camera_active    = True
            self._camera_starting  = False
            self._first_frame_time = time.time()

            if not self._recording:
                self._start_recording()

        # ── Idle timeout when not recording ───────────────────
        if (
            self._camera_active
            and not self._recording
            and not motion_now
            and self._first_frame_time is not None
        ):
            baseline = max(self._first_frame_time, self._last_motion_time or 0)
            if (time.time() - baseline) > CAMERA_IDLE_TIMEOUT:
                self.logger.info("Idle timeout — sleeping camera.")
                self._sleep_camera()

        if not self._camera_active:
            time.sleep(0.05)
            return

        # ── Get frame ─────────────────────────────────────────
        frame = self.camera.get_next_frame()
        if frame is None:
            time.sleep(0.01)
            return

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # ── Face recognition ───────────────────────────────────
        frame          = self.model.process_frame(frame)
        frame          = self.model.draw_results(frame)
        detected_names = self.model.get_detected_names()

        authorized_name    = None
        known_face_present = False
        if detected_names:
            recognized = [n for n in detected_names if n.lower() not in ("unknown", "")]
            if recognized:
                authorized_name    = max(set(recognized), key=recognized.count)
                known_face_present = True

        rfid_authorized = self.state.consume_authorized()

        # ── Recording logic ────────────────────────────────────
        if self._recording:
            elapsed = time.time() - self._record_start_time

            if known_face_present or rfid_authorized:
                self._handle_authorized(known_face_present, authorized_name)
                return

            self._video_writer.write(frame)

            if elapsed >= RECORDING_DURATION:
                self._handle_breach()
                return

        # ── HUD display ────────────────────────────────────────
        if self._recording and self._record_start_time:
            remaining = max(0, RECORDING_DURATION - (time.time() - self._record_start_time))
            hud_text  = f"RECORDING {remaining:.1f}s"
            hud_color = (0, 0, 255)
        else:
            hud_text  = "Monitoring"
            hud_color = (0, 255, 0)

        cv2.putText(frame, hud_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2)
        cv2.imshow("Face Recognition Door Lock", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            self._running = False

    # ── Recording helpers ──────────────────────────────────────

    def _camera_startup(self):
        self.camera.start_preview_stream(fps=CAMERA_FPS)
        for _ in range(100):
            if self.camera.get_next_frame() is not None:
                self.state.mark_camera_ready()
                self.logger.info("Camera ready — first frame received.")
                return
            time.sleep(0.05)
        self.state.mark_camera_ready()
        self.logger.warning("Camera startup timed out.")

    def _start_recording(self):
        self.logger.info("Starting recording…")
        self.state.lock_motion()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            self._temp_video_path = tmp.name

        fourcc             = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(
            self._temp_video_path, fourcc, RECORDING_FPS, CAMERA_PREVIEW_RESOLUTION
        )
        self._recording         = True
        self._record_start_time = time.time()
        self.logger.warning(f"Recording → {self._temp_video_path}")

        if self.mic:
            self.mic.start_recording()
            self.logger.info("Mic recording started.")

    def _handle_authorized(self, known_face: bool, name: str | None):
        reason = "Known face" if known_face else "RFID"
        self.logger.info(f"{reason} → authorized. Aborting clip.")

        if known_face and name:
            self.alert_mgr.trigger_cloud_alert("face_success", f"Welcome back! {name} unlocked the door.")
        else:
            self.alert_mgr.trigger_cloud_alert("rfid_success", "Door unlocked via authorized RFID.")

        if self.lock:
            threading.Thread(
                target = self.lock.unlock,
                kwargs = {"hold_time": 3.0},
                daemon = True,
                name   = "DoorUnlock",
            ).start()

        self._video_writer.release()

        if self.mic:
            mic_path = self.mic.stop_recording(
                custom_filename=self._temp_video_path.replace(".mp4", "_audio.wav")
            )
            if mic_path and os.path.exists(mic_path):
                os.remove(mic_path)

        if self._temp_video_path and os.path.exists(self._temp_video_path):
            os.remove(self._temp_video_path)
            self.logger.info("Aborted clip deleted.")

        self._reset_recording_state()
        self._sleep_camera()
        self.state.unlock_motion()

    def _handle_breach(self):
        self.logger.warning(f"Recording complete ({RECORDING_DURATION}s) — breach → uploading…")
        self.alert_mgr.trigger_cloud_alert(
            "unauthorized_breach",
            "Security Breach: Unrecognized entity at terminal."
        )

        self._video_writer.release()

        mic_path  = self.mic.stop_recording(custom_filename="temp_intruder_mic.wav") if self.mic else None
        clip_path = self._merge_audio_video(self._temp_video_path, mic_path) \
                    if mic_path and os.path.exists(mic_path) \
                    else self._temp_video_path

        self._reset_recording_state()
        self._sleep_camera()

        threading.Thread(
            target = self._upload_and_cleanup,
            args   = (clip_path,),
            daemon = True,
            name   = "UploadClip",
        ).start()

    def _reset_recording_state(self):
        self._recording         = False
        self._video_writer      = None
        self._temp_video_path   = None
        self._record_start_time = None

    def _sleep_camera(self):
        self.camera.stop_preview_stream()
        self.state.reset_camera_ready()
        self._camera_active    = False
        self._camera_starting  = False
        self._first_frame_time = None
        self._last_motion_time = None
        self.logger.info("Camera sleeping.")

    def _merge_audio_video(self, video_path: str, audio_path: str | None) -> str:
        if not audio_path or not os.path.exists(audio_path):
            self.logger.warning("No audio to merge — video only.")
            return video_path

        merged = video_path.replace(".mp4", "_merged.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            merged,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode == 0:
                self.logger.info(f"Audio merged → {merged}")
                os.remove(video_path)
                os.remove(audio_path)
                return merged
            else:
                self.logger.error(f"ffmpeg failed: {result.stderr.decode()}")
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                return video_path
        except Exception as e:
            self.logger.error(f"ffmpeg error: {e}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return video_path

    def _upload_and_cleanup(self, video_path: str):
        try:
            if self.storage and video_path and os.path.exists(video_path):
                self.logger.info("Uploading intruder clip…")
                try:
                    url = self.storage.upload_and_get_url(video_path)
                    if url:
                        self.logger.info(f"Upload complete: {url}")
                    else:
                        self.logger.error("Upload returned no URL.")
                except Exception as e:
                    self.logger.error(f"Upload failed: {e}")
            else:
                self.logger.warning("Supabase not connected — clip not uploaded.")
        finally:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                self.logger.info("Local clip deleted.")
            self.state.unlock_motion()

    # ══════════════════════════════════════════════════════════
    #  SHUTDOWN
    # ══════════════════════════════════════════════════════════

    def shutdown(self):
        self.logger.info("Shutting down DeviceManager…")
        self._running = False

        if self._video_writer:
            self._video_writer.release()
        if self._temp_video_path and os.path.exists(self._temp_video_path):
            os.remove(self._temp_video_path)

        cv2.destroyAllWindows()

        if self.camera:
            self.camera.cleanup()
        if self.mic:
            self.mic.cleanup()
        if self.lock:
            self.lock.cleanup()
        if self.pir:
            self.pir.cleanup()
        if self.rfid:
            self.rfid.stop()

        self.logger.info("DeviceManager shutdown complete.")