"""
webhook.py - Isolated Flask Webhook Server for Face Synchronization
Handles Supabase routing requests, token verification, and model retraining threads.
"""

import threading
from flask import Flask, jsonify, request
from Cloud.AlertManager import AlertManager
from software.FaceDetection import ModelTraining
from logs.logger import getLogger

logger = getLogger("WebhookServer")

def build_webhook_app(alert_mgr: AlertManager, model: ModelTraining, smart_train_func, webhook_secret: str) -> Flask:
    app = Flask(__name__)
    import logging as _logging
    _logging.getLogger("werkzeug").setLevel(_logging.ERROR)

    @app.route("/new-face", methods=["POST"])
    def face_sync():
        # Verify custom gateway security handshake
        incoming_secret = request.headers.get("X-Gateway-Secret", "")
        if incoming_secret != webhook_secret:
            logger.warning("Auth failure: Unauthorized payload dropped by webhook server.")
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json(silent=True) or {}
        
        # Execute the download/mirror logic via AlertManager
        result, status_code = alert_mgr.handle_new_face_sync(data)

        if status_code == 200:
            event_type = data.get("event_type", "")
            file_path  = str(data.get("file_path", ""))

            if event_type == "new_folder" or file_path.endswith("/"):
                logger.info("Webhook: New profile directory mirrored safely. No training needed.")
            else:
                logger.info("Webhook: Live image sync completed → dispatching smart retrain background worker...")
                
                # Launching the training sequence on a separate daemon thread
                threading.Thread(
                    target=smart_train_func,
                    kwargs={"model": model, "force": True},
                    daemon=True,
                    name="WebhookRetrain",
                ).start()

        return jsonify(result), status_code

    @app.route("/webhook/intruder", methods=["POST"])
    def intruder_alert():
        incoming_secret = request.headers.get("X-Gateway-Secret", "")
        if incoming_secret != webhook_secret:
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json(silent=True) or {}
        result, status_code = alert_mgr.handle_intruder_alert(data)
        return jsonify(result), status_code

    return app


def start_webhook_server(alert_mgr: AlertManager, model: ModelTraining, smart_train_func, host: str, port: int, secret: str) -> None:
    """
    Spawns the Flask webhook server runner context inside a non-blocking background daemon thread.
    """
    app = build_webhook_app(alert_mgr, model, smart_train_func, secret)
    
    threading.Thread(
        target=lambda: app.run(host=host, port=port, use_reloader=False),
        daemon=True,
        name="WebhookServer",
    ).start()
    
    logger.info(f"Webhook tracking operational gateway bound to http://{host}:{port}/new-face")