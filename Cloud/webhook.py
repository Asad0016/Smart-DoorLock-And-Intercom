"""
webhook.py - Isolated Flask Webhook Server for Face Synchronization
Handles Supabase routing requests, token verification, and sequential model retraining.
"""

import threading
from config import WEBHOOK_SECRET
from flask import Flask, jsonify, request
from Cloud.AlertManager import AlertManager
from software.FaceDetection import ModelTraining
from logs.logger import getLogger

logger = getLogger("WebhookServer")

# Global lock to prevent overlapping model training tasks
training_lock = threading.Lock()

def build_webhook_app(alert_mgr: AlertManager, model: ModelTraining, smart_train_func, webhook_secret: str) -> Flask:
    app = Flask(__name__)
    import logging as _logging
    _logging.getLogger("werkzeug").setLevel(_logging.ERROR)

    @app.route("/new-face", methods=["POST"])
    def face_sync():
        # 1. Verify custom gateway security handshake
        incoming_secret = request.headers.get("X-Gateway-Secret", "")
        if incoming_secret != webhook_secret:
            logger.warning("Auth failure: Unauthorized payload dropped by webhook server.")
            return jsonify({"error": "Unauthorized"}), 401

        # Force parse json even if headers are slightly malformed from the proxy/edge
        data = request.get_json(force=True, silent=True) or {}
        
        # 2. Execute the download/mirror logic via AlertManager
        result, status_code = alert_mgr.handle_new_face_sync(data)

        if status_code == 200:
            # Check if this is a dummy node or an actual image download sync
            event_type = data.get("event_type", "")
            
            if event_type == "new_folder":
                logger.info("Webhook: Local profile node synchronized. Training skipped.")
            else:
                # 3. Safe Sequential Background Training Thread
                def safe_training_worker():
                    # Thread execution check: if another training is active, it waits in queue safely
                    if not training_lock.acquire(blocking=False):
                        logger.warning("Webhook: Model retraining already in progress. Queueing or skipping thread execution.")
                        return
                    try:
                        logger.info("Webhook: Core lock acquired → Initiating smart retrain worker context...")
                        smart_train_func(force=False)
                        logger.info("Webhook: Smart retraining sequence finished successfully. Lock released.")
                    except Exception as e:
                        logger.error(f"Critical failure inside training worker thread execution: {e}")
                    finally:
                        training_lock.release()

                # Launching the wrapper task on a separate daemon thread
                threading.Thread(
                    target=safe_training_worker,
                    daemon=True,
                    name="WebhookRetrain",
                ).start()

        return jsonify(result), status_code

    @app.route("/webhook/intruder", methods=["POST"])
    def intruder_alert():
        incoming_secret = request.headers.get("X-Gateway-Secret", "")
        if incoming_secret != webhook_secret:
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json(force=True, silent=True) or {}
        result, status_code = alert_mgr.handle_intruder_alert(data)
        return jsonify(result), status_code

    # ── FIXED: Moved inside the factory scope BEFORE 'return app' ──
    @app.route("/door-command", methods=["POST"])
    def door_command_trigger():
        # Security handshake check
        incoming_secret = request.headers.get("X-Gateway-Secret", "")
        if incoming_secret != webhook_secret:
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json(force=True, silent=True) or {}
        command = data.get("command") # 'view_live_feed', 'unlock', ya 'stop_video_call'
        user_id = data.get("user_id")
        logger.info(f"Door command received via Edge Function: {command}")

        # Execute streaming, stopping, or unlocking context
        if command == "view_live_feed":
            threading.Thread(
                target=alert_mgr.start_intercom_stream,
                args=(user_id,),
                daemon=True,
                name="IntercomStream"
            ).start()
            return jsonify({"status": "success", "message": "Streaming initialization sequence kicked off."}), 200

        elif command == "unlock":
            alert_mgr.trigger_door_latch()
            return jsonify({"status": "success", "message": "Latching hardware triggered."}), 200

        elif command == "stop_video_call":
            # Direct cleanup execution without thread context locking overhead
            alert_mgr.stop_intercom_stream()
            return jsonify({"status": "success", "message": "Intercom stream teardown executed safely."}), 200

        return jsonify({"status": "error", "message": "Unknown command context"}), 400
    @app.route("/set-pin", methods=["POST"])
    def set_pin_trigger():
        incoming_secret = request.headers.get("X-Gateway-Secret", "")
        if incoming_secret != webhook_secret:
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json(force=True, silent=True) or {}
        
        # Pure object ko bhejhein taake handle_local_pin_schedule 'record' ko parse kar sakay
        success = alert_mgr.update_local_pin_schedule(data)
        
        if success:
            return jsonify({"status": "success", "message": "Keypad set_pin function triggered."}), 200
        else:
            return jsonify({"status": "error", "message": "Failed to update keypad pin."}), 500
    @app.route("/sync-rfid", methods=["POST"])
    def sync_rfid_trigger():
        # Security Key Check
        incoming_secret = request.headers.get("X-Gateway-Secret", "")
        if incoming_secret != webhook_secret:
            return jsonify({"error": "Unauthorized"}), 401

        # Edge function ka banaya hua payload extract karna
        payload = request.get_json(force=True, silent=True) or {}
        
        # AlertManager ke function ko poora payload pass karein
        success = alert_mgr.update_local_rfid_cache(payload)
        if success:
            return jsonify({
                "status": "success", 
                "message": "RFID manager cache updated successfully."
            }), 200
        else:
            return jsonify({
                "status": "error", 
                "message": "Failed to inject RFID card into hardware layer."
            }), 500
    return app
def start_webhook_server(alert_mgr: AlertManager, model: ModelTraining, smart_train_func, host: str, port: int, secret: str) -> None:
    """
    Spawns the Flask webhook server runner context inside a non-blocking background daemon thread.
    """
    app = build_webhook_app(alert_mgr, model, smart_train_func, secret)
    
    threading.Thread(
        target=lambda: app.run(host=host, port=port, use_reloader=False),
        daemon=True,
        name="WebhookServerLoop",
    ).start()
    
    logger.info(f"Webhook tracking operational gateway bound to http://{host}:{port}/new-face")