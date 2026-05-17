# """
# main.py - Face Recognition Door Lock System
# Fixed Version - Press 's' to record, upload directly to Supabase
# """

# import time
# import cv2
# import os
# import tempfile

# from Hardware.camera import CameraManager
# from software.FaceDetection import ModelTraining
# from Cloud.supabase import SupabaseManager

# # ========================= CONFIG =========================
# DATASET_PATH    = "/home/doorlock/DoorLock/dataset/Naveed"
# ENCODINGS_PATH  = "/home/doorlock/DoorLock/dataset/encodings.pickle"
# DETECTION_MODEL = "hog"
# ENCODING_MODEL  = "large"
# TOLERANCE       = 0.5

# CAMERA_RESOLUTION         = (1920, 1080)
# CAMERA_PREVIEW_RESOLUTION = (640, 480)
# CAMERA_FPS                = 30
# RECORDING_DURATION        = 10      # seconds
# RECORDING_FPS             = 20

# # ========================= MAIN =========================
# def main():
#     # 1. Face Model
#     print("[INFO] Loading Face Recognition Model...")
#     model = ModelTraining(
#         dataset_path=DATASET_PATH,
#         encodings_path=ENCODINGS_PATH,
#         detection_model=DETECTION_MODEL,
#         encoding_model=ENCODING_MODEL,
#         tolerance=TOLERANCE,
#     )
#     model.train_model()

#     # 2. Camera
#     print("[INFO] Initializing Camera...")
#     camera = CameraManager(
#         resolution=CAMERA_RESOLUTION,
#         framerate=CAMERA_FPS,
#         preview_resolution=CAMERA_PREVIEW_RESOLUTION,
#     )

#     if not camera.initialize_camera():
#         print("[ERROR] Camera failed to initialize!")
#         return

#     camera.start_preview_stream(fps=CAMERA_FPS)

#     # 3. Supabase
#     storage = None
#     try:
#         storage = SupabaseManager()
#         print("[INFO] Supabase Connected Successfully")
#     except Exception as e:
#         print(f"[WARNING] Supabase Failed: {e}")

#     print("\n[INFO] System Ready!")
#     print("   Press 's' → Start 10s Recording (uploads to Supabase)")
#     print("   Press 'q' → Quit\n")

#     # Recording state
#     recording        = False
#     video_writer     = None
#     temp_video_path  = None
#     record_start_time = None

#     try:
#         while True:
#             # --- Get frame ---
#             frame = camera.get_next_frame()
#             if frame is None:
#                 time.sleep(0.01)
#                 continue

#             # Convert BGRA → BGR if needed
#             if frame.ndim == 3 and frame.shape[2] == 4:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

#             # --- Face processing ---
#             frame = model.process_frame(frame)
#             frame = model.draw_results(frame)
#             detected_names = model.get_detected_names()

#             # --- Status overlay ---
#             if recording:
#                 elapsed  = time.time() - record_start_time
#                 remaining = max(0, RECORDING_DURATION - elapsed)
#                 status_text  = f"RECORDING  {remaining:.1f}s left"
#                 status_color = (0, 0, 255)
#             else:
#                 status_text  = "Monitoring  |  press S to record"
#                 status_color = (0, 255, 0)

#             cv2.putText(frame, status_text, (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

#             # --- Show frame FIRST so waitKey works reliably ---
#             cv2.imshow("Face Recognition Door Lock", frame)
#             key = cv2.waitKey(1) & 0xFF

#             # --- Quit ---
#             if key == ord('q'):
#                 print("[INFO] Quitting...")
#                 break

#             # ================== START RECORDING ==================
#             # Press 's' → always start recording (no Unknown check)
#             if key in (ord('s'), ord('S')) and not recording:
#                 print("[INFO] 's' pressed → Starting 10s Recording...")

#                 with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
#                     temp_video_path = tmp.name

#                 fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
#                 video_writer = cv2.VideoWriter(
#                     temp_video_path, fourcc, RECORDING_FPS, CAMERA_PREVIEW_RESOLUTION
#                 )
#                 recording         = True
#                 record_start_time = time.time()

#             # ================== WRITE & STOP RECORDING ==================
#             if recording:
#                 video_writer.write(frame)
#                 elapsed = time.time() - record_start_time

#                 if elapsed >= RECORDING_DURATION:
#                     print("[INFO] 10s complete → Releasing video writer...")
#                     video_writer.release()
#                     video_writer = None
#                     recording    = False

#                     # Upload to Supabase
#                     if storage and temp_video_path and os.path.exists(temp_video_path):
#                         print("[INFO] Uploading to Supabase...")
#                         try:
#                             url = storage.upload_and_get_url(temp_video_path)
#                             if url:
#                                 print(f"[SUCCESS] Upload complete!")
#                                 print(f"Link: {url}")
#                             else:
#                                 print("[ERROR] Upload returned no URL.")
#                         except Exception as e:
#                             print(f"[ERROR] Upload failed: {e}")
#                     else:
#                         if not storage:
#                             print("[WARNING] Supabase not connected – video NOT uploaded.")
#                         else:
#                             print("[WARNING] Temp file missing – nothing to upload.")

#                     # Clean up temp file
#                     if temp_video_path and os.path.exists(temp_video_path):
#                         os.remove(temp_video_path)
#                     temp_video_path = None

#     except KeyboardInterrupt:
#         print("\n[INFO] Interrupted by user.")
#     finally:
#         if video_writer:
#             video_writer.release()
#         if temp_video_path and os.path.exists(temp_video_path):
#             os.remove(temp_video_path)
#         cv2.destroyAllWindows()
#         camera.cleanup()
#         print("[INFO] System Shutdown Complete.")


# if __name__ == "__main__":
#     main()
"""
main.py - Face Recognition Door Lock System
==========================================
Startup Flow:
  1. Connect to Supabase Cloud → import full dataset
  2. Smart-train: hash the dataset; skip if unchanged, retrain if new data found
  3. Launch Flask webhook listener (background thread) for Supabase Edge Function alerts
  4. Run live camera loop with face recognition + press-S recording

Webhook Secret (Supabase Dashboard → MY_RPI_SECRET): DoorLock123
"""

import hashlib
import json
import os
import pickle
import tempfile
import threading
import time

import cv2
from flask import Flask, jsonify, request

from Cloud.AlertManager import AlertManager
from Cloud.supabase import SupabaseManager
from Hardware.camera import CameraManager
from logs.logger import getLogger
from software.FaceDetection import ModelTraining

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
RECORDING_DURATION        = 10   # seconds
RECORDING_FPS             = 20

WEBHOOK_SECRET      = "DoorLock123"
WEBHOOK_HOST        = "0.0.0.0"
WEBHOOK_PORT        = 5050

logger = getLogger("Main")


# ══════════════════════════════════════════════════════════════
#  DATASET FINGERPRINTING  (detect new folders / images)
# ══════════════════════════════════════════════════════════════

def compute_dataset_hash(dataset_path: str) -> str:
    """
    Walk the dataset directory and build a stable SHA-256 fingerprint
    that covers every file path + its size + last-modified time.
    Returns a hex digest string.
    """
    hasher = hashlib.sha256()
    for root, dirs, files in sorted(os.walk(dataset_path)):
        dirs.sort()   # ensure deterministic traversal order
        for fname in sorted(files):
            if fname.startswith("."):   # skip hidden housekeeping files
                continue
            full = os.path.join(root, fname)
            rel  = os.path.relpath(full, dataset_path)
            stat = os.stat(full)
            # include relative path, size, and modification time in the hash
            hasher.update(f"{rel}|{stat.st_size}|{stat.st_mtime}".encode())
    return hasher.hexdigest()


def load_saved_hash() -> str | None:
    """Load the previously persisted dataset hash (None if first run)."""
    if os.path.exists(DATASET_HASH_PATH):
        try:
            with open(DATASET_HASH_PATH, "r") as f:
                return json.load(f).get("hash")
        except Exception:
            pass
    return None


def save_dataset_hash(digest: str):
    """Persist the current dataset hash to disk."""
    os.makedirs(os.path.dirname(DATASET_HASH_PATH), exist_ok=True)
    with open(DATASET_HASH_PATH, "w") as f:
        json.dump({"hash": digest, "timestamp": time.time()}, f)


# ══════════════════════════════════════════════════════════════
#  SMART TRAINING  (skip if dataset hasn't changed)
# ══════════════════════════════════════════════════════════════

def smart_train(model: ModelTraining, force: bool = False) -> bool:
    """
    Compute a fingerprint of the dataset directory.
    - If the fingerprint matches the last saved one AND encodings exist → skip.
    - Otherwise → train and save the new fingerprint.
    Returns True if training was performed, False if it was skipped.
    """
    current_hash = compute_dataset_hash(DATASET_PATH)
    saved_hash   = load_saved_hash()
    encodings_ok = os.path.exists(ENCODINGS_PATH) and os.path.getsize(ENCODINGS_PATH) > 0

    if not force and encodings_ok and current_hash == saved_hash:
        logger.info("✅ Dataset unchanged and encodings present → skipping training.")
        return False

    reason = "forced" if force else ("no encodings" if not encodings_ok else "dataset changed")
    logger.info(f"🔄 Training model [{reason}] ...")
    model.train_model()
    save_dataset_hash(current_hash)
    logger.info("✅ Model training complete. Hash saved.")
    return True


# ══════════════════════════════════════════════════════════════
#  WEBHOOK SERVER  (Flask in a background thread)
# ══════════════════════════════════════════════════════════════

def build_webhook_app(alert_mgr: AlertManager, model: ModelTraining) -> Flask:
    """
    Construct and return the Flask app that handles Supabase Edge Function
    webhook calls.  Two routes are registered:

      POST /webhook/face-sync     → new image / new folder uploaded to storage
      POST /webhook/intruder      → security / lockdown trigger
    """
    app = Flask(__name__)

    # ── Silence Flask's noisy access log in the console ──────────
    import logging as _logging
    _logging.getLogger("werkzeug").setLevel(_logging.ERROR)

    # ── /webhook/face-sync ────────────────────────────────────────
    @app.route("/webhook/face-sync", methods=["POST"])
    def face_sync():
        incoming_secret = request.headers.get("X-Webhook-Secret", "")

        if not alert_mgr.verify_alert_auth(incoming_secret):
            logger.warning("🚫 Rejected webhook: invalid or missing secret.")
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json(silent=True) or {}
        result, status_code = alert_mgr.handle_new_face_sync(data)

        # ── Retrain only when a new image (not just a folder) arrived ──
        if status_code == 200 and not (
            data.get("event_type") == "new_folder_only"
            or str(data.get("file_path", "")).endswith("/")
        ):
            logger.info("📸 New image synced → triggering smart retrain …")
            threading.Thread(
                target=smart_train,
                kwargs={"model": model, "force": False},
                daemon=True,
            ).start()

        return jsonify(result), status_code

    # ── /webhook/intruder ─────────────────────────────────────────
    @app.route("/webhook/intruder", methods=["POST"])
    def intruder_alert():
        incoming_secret = request.headers.get("X-Webhook-Secret", "")

        if not alert_mgr.verify_alert_auth(incoming_secret):
            logger.warning("🚫 Rejected intruder webhook: invalid or missing secret.")
            return jsonify({"error": "Unauthorized"}), 401

        data   = request.get_json(silent=True) or {}
        result, status_code = alert_mgr.handle_intruder_alert(data)
        return jsonify(result), status_code

    return app


def start_webhook_server(alert_mgr: AlertManager, model: ModelTraining):
    """Launch the Flask webhook listener in a daemon thread."""
    app = build_webhook_app(alert_mgr, model)
    thread = threading.Thread(
        target=lambda: app.run(host=WEBHOOK_HOST, port=WEBHOOK_PORT, use_reloader=False),
        daemon=True,
        name="WebhookServer",
    )
    thread.start()
    logger.info(f"🌐 Webhook server listening on {WEBHOOK_HOST}:{WEBHOOK_PORT}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():

    # ── 1. Cloud: connect to Supabase ────────────────────────────
    logger.info("☁️  Connecting to Supabase …")
    storage = None
    try:
        storage = SupabaseManager()
        logger.info("✅ Supabase connected.")
    except Exception as e:
        logger.warning(f"⚠️  Supabase unavailable: {e}. Continuing offline.")

    # ── 2. Cloud: pull full dataset to local disk ─────────────────
    if storage:
        logger.info("📦 Importing dataset from cloud bucket …")
        try:
            storage.import_entire_dataset(DATASET_PATH)
            logger.info("✅ Dataset import complete.")
        except Exception as e:
            logger.error(f"❌ Dataset import failed: {e}")

    # ── 3. Face model: smart-train (skip if unchanged) ────────────
    logger.info("🧠 Initialising Face Recognition model …")
    model = ModelTraining(
        dataset_path    = DATASET_PATH,
        encodings_path  = ENCODINGS_PATH,
        detection_model = DETECTION_MODEL,
        encoding_model  = ENCODING_MODEL,
        tolerance       = TOLERANCE,
    )

    # Check whether the dataset has any images at all before attempting training
    dataset_has_images = any(
        fname.lower().endswith((".jpg", ".jpeg", ".png"))
        for _, _, files in os.walk(DATASET_PATH)
        for fname in files
    )

    if dataset_has_images:
        smart_train(model, force=False)
    else:
        logger.warning("⚠️  Dataset appears empty — skipping training until images arrive.")

    # ── 4. AlertManager + Webhook server ─────────────────────────
    alert_mgr = AlertManager(
        base_storage_path = DATASET_PATH,
        secret_key        = WEBHOOK_SECRET,
    )
    start_webhook_server(alert_mgr, model)

    # ── 5. Camera ────────────────────────────────────────────────
    logger.info("📷 Initialising camera …")
    camera = CameraManager(
        resolution        = CAMERA_RESOLUTION,
        framerate         = CAMERA_FPS,
        preview_resolution= CAMERA_PREVIEW_RESOLUTION,
    )

    if not camera.initialize_camera():
        logger.error("❌ Camera failed to initialise. Exiting.")
        return

    camera.start_preview_stream(fps=CAMERA_FPS)

    logger.info("\n✅ System Ready!")
    logger.info("   Press 's' → Start 10-second recording (uploads to Supabase)")
    logger.info("   Press 'q' → Quit\n")

    # ── 6. Main camera loop ───────────────────────────────────────
    recording         = False
    video_writer      = None
    temp_video_path   = None
    record_start_time = None

    try:
        while True:
            frame = camera.get_next_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # BGRA → BGR if camera returns alpha channel
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Face detection + drawing
            frame = model.process_frame(frame)
            frame = model.draw_results(frame)

            # Status overlay
            if recording:
                elapsed   = time.time() - record_start_time
                remaining = max(0, RECORDING_DURATION - elapsed)
                status_text  = f"RECORDING  {remaining:.1f}s left"
                status_color = (0, 0, 255)
            else:
                status_text  = "Monitoring  |  press S to record"
                status_color = (0, 255, 0)

            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            cv2.imshow("Face Recognition Door Lock", frame)
            key = cv2.waitKey(1) & 0xFF

            # ── Quit ──────────────────────────────────────────────
            if key == ord('q'):
                logger.info("Quit key pressed.")
                break

            # ── Start recording ───────────────────────────────────
            if key in (ord('s'), ord('S')) and not recording:
                logger.info("'s' pressed → starting 10-second recording …")
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                    temp_video_path = tmp.name
                fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    temp_video_path, fourcc, RECORDING_FPS, CAMERA_PREVIEW_RESOLUTION
                )
                recording         = True
                record_start_time = time.time()

            # ── Write frames / stop recording ────────────────────
            if recording:
                video_writer.write(frame)
                elapsed = time.time() - record_start_time

                if elapsed >= RECORDING_DURATION:
                    logger.info("Recording complete → releasing writer …")
                    video_writer.release()
                    video_writer = None
                    recording    = False

                    # Upload clip to Supabase
                    if storage and temp_video_path and os.path.exists(temp_video_path):
                        logger.info("Uploading clip to Supabase …")
                        try:
                            url = storage.upload_and_get_url(temp_video_path)
                            if url:
                                logger.info(f"✅ Upload complete! Link: {url}")
                            else:
                                logger.error("Upload returned no URL.")
                        except Exception as e:
                            logger.error(f"Upload failed: {e}")
                    else:
                        logger.warning(
                            "Supabase not connected or temp file missing — clip NOT uploaded."
                        )

                    # Cleanup temp file
                    if temp_video_path and os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
                    temp_video_path = None

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        if video_writer:
            video_writer.release()
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        cv2.destroyAllWindows()
        camera.cleanup()
        logger.info("System shutdown complete.")


if __name__ == "__main__":
    main()