import os
import requests
from flask import Flask, request, jsonify
from logs.logger import getLogger

class AlertManager:
    """
    Alert Manager Class
    Handles live real-time webhooks, authorization, and automatic sync routing.
    """
    def __init__(self, base_storage_path : str, secret_key: str):
        """
        Constructor: Initializes tracking paths and gateway authentication secrets.
        :param base_storage_path: The root directory to save dataset additions (e.g., '/home/doorlock/DoorLock/dataset')
        :param secret_key: The string passphrase matching the Supabase Dashboard variable (MY_RPI_SECRET)
        """
        self.logger = getLogger("AlertManager")
        self.base_path = os.path.abspath(base_storage_path)
        self.secret_key = secret_key

        # Automatically check and verify base path layout
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
            self.logger.info(f"📁 Root dataset location instantiated: {self.base_path}")

    def verify_alert_auth(self, incoming_secret: str) -> bool:
        """
        Validates token credentials from incoming transmission packets.
        """
        if not incoming_secret:
            self.logger.warning("Auth failure: Received request missing security headers.")
            return False
            
        is_valid = incoming_secret == self.secret_key
        if not is_valid:
            self.logger.warning("Auth failure: Incoming request provided an invalid token.")
        return is_valid

    def handle_new_face_sync(self, data: dict) -> tuple[dict, int]:
        """
        Core Action Unit: Processes 'New Image' or 'New Folder' webhooks on the fly.
        """
        event_type = data.get("event_type")
        file_path = data.get("file_path")
        parent_folder = data.get("parent_folder")
        public_url = data.get("publicUrl")

        self.logger.info(f"🔔 Live Alert Received! Action Type: {event_type} | Resource Path: {file_path}")

        # -------------------------------------------------------------
        # Live Stream Event points to an Empty Folder creation
        # -------------------------------------------------------------
        if event_type == "new_folder_only" or file_path.endswith("/"):
            # Normalize target folder name extraction
            folder_name = parent_folder if parent_folder else file_path.replace("/", "")
            target_folder_path = os.path.join(self.base_path, folder_name)
            
            if not os.path.exists(target_folder_path):
                os.makedirs(target_folder_path)
                self.logger.info(f"📁 Live Webhook Processed: Mirror directory created at {target_folder_path}")
                return {"status": "success", "message": f"Folder '{folder_name}' initialized locally."}, 200
            
            return {"status": "success", "message": f"Folder '{folder_name}' already existed."}, 200

        # -------------------------------------------------------------
        # Live Stream Event points to a New File/Image Upload
        # -------------------------------------------------------------
        else:
            # Map out exact target directory tree path structure
            target_dir = os.path.join(self.base_path, parent_folder) if parent_folder else self.base_path
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                self.logger.info(f"📁 Dynamically generating missing parent link path: {target_dir}")

            file_name = file_path.split("/")[-1]
            final_file_path = os.path.join(target_dir, file_name)

            try:
                self.logger.info(f"📥 Pulling live asset from gateway: {file_name}")
                response = requests.get(public_url, stream=True, timeout=30)
                
                if response.status_code == 200:
                    with open(final_file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    self.logger.info(f"✅ Real-time pipeline sync complete. File saved to: {final_file_path}")
                    
                    # 💡 DEV NOTE: You can inject your Face Recognition model training/reload function here!
                    return {"status": "success", "local_path": final_file_path}, 200
                
                self.logger.error(f"❌ Target download pipeline failed. HTTP Code: {response.status_code}")
                return {"status": "error", "message": f"HTTP mirror download failed with code {response.status_code}"}, 500
                
            except Exception as e:
                self.logger.error(f"❌ Crash encountered over streaming download pipeline: {str(e)}")
                return {"status": "error", "message": str(e)}, 500

    def handle_intruder_alert(self, data: dict) -> tuple[dict, int]:
        """
        Extensible Slot Example: Easily process security locks or sirens in the future
        without modifying storage system structures.
        """
        self.logger.warning("🚨 ALERT BOUND: Hardware lockdown conditions tripped!")
        # Put your GPIO/Relay board triggers here
        return {"status": "lockdown_sequence_fired"}, 200