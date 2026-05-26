import os
import requests
import threading  # Asynchronous non-blocking cloud execution layers
from flask import Flask, request, jsonify
from pathlib import Path
from logs.logger import getLogger

class AlertManager:
    """
    Alert Manager Class
    Handles live real-time webhooks, authorization, automatic sync routing,
    and generalized upstream cloud notifications without locking edge cycles.
    """
    def __init__(self, base_storage_path: str, secret_key: str, supabase_url: str, supabase_key: str, trainer_instance=None):
        """
        Constructor: Initializes tracking paths, gateway secrets, and cloud telemetry credentials.
        :param base_storage_path: Root directory to save datasets (e.g., '/home/doorlock/DoorLock/dataset')
        :param secret_key: String passphrase matching incoming server validation headers (MY_RPI_SECRET)
        :param supabase_url: Your Supabase Project Endpoint URL (e.g., https://xyz.supabase.co)
        :param supabase_key: Your Supabase Anon/Service Key credentials
        :param trainer_instance: Embedded FaceDetection dynamic training handler context (Optional)
        """
        self.logger = getLogger("AlertManager")
        self.base_path = os.path.abspath(base_storage_path)
        self.secret_key = secret_key
        self.trainer = trainer_instance 

        # ☁️ DYNAMIC CLOUD TELEMETRY ROUTING VARIABLES
        self.supabase_url = supabase_url.strip().rstrip('/')
        self.supabase_key = supabase_key.strip()
        self.cloud_alert_endpoint = f"{self.supabase_url}/rest/v1/door_alerts"

        # Verify or generate the target local dataset system tree
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
            self.logger.info(f"Root dataset location instantiated: {self.base_path}")

    def verify_alert_auth(self, incoming_secret: str) -> bool:
        """
        Validates token credentials from incoming transaction packets.
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
        public_url = data.get("public_url")  

        self.logger.info(f"Live Alert Received! Action Type: {event_type} | Resource Path: {file_path}")

        if not file_path:
            return {"status": "error", "message": "Missing file_path context metadata"}, 400

        # -------------------------------------------------------------
        # Live Stream Event points to a Folder creation context
        # -------------------------------------------------------------
        if event_type == "new_folder" or file_path.endswith("/") or ".emptyFolderPlaceholder" in file_path:
            folder_name = parent_folder if parent_folder else file_path.split("/")[0]
            target_folder_path = os.path.join(self.base_path, folder_name)
            
            if not os.path.exists(target_folder_path):
                os.makedirs(target_folder_path)
                self.logger.info(f"Live Webhook Processed: Mirror directory created at {target_folder_path}")
                return {"status": "success", "message": f"Folder '{folder_name}' initialized locally."}, 200
            
            return {"status": "success", "message": f"Folder '{folder_name}' already existed."}, 200

        # -------------------------------------------------------------
        # Live Stream Event points to a New File/Image Upload context
        # -------------------------------------------------------------
        else:
            if not public_url:
                self.logger.error("Download rejected: Missing valid resource public_url path.")
                return {"status": "error", "message": "Missing asset URL target context"}, 400

            target_dir = os.path.join(self.base_path, parent_folder) if parent_folder else self.base_path
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                self.logger.info(f"Dynamically generating missing parent link path: {target_dir}")

            file_name = file_path.split("/")[-1]
            final_file_path = os.path.join(target_dir, file_name)

            try:
                self.logger.info(f"Pulling live asset from gateway: {file_name}")
                response = requests.get(public_url, stream=True, timeout=30)
                
                if response.status_code == 200:
                    with open(final_file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    self.logger.info(f"Real-time pipeline sync complete. File saved to: {final_file_path}")
                    
                    # AUTOMATIC INCREMENTAL TRAINING INTEGRATION
                    if self.trainer is not None:
                        self.logger.info("Triggering live downstream Incremental Training for newly arrived asset...")
                        training_success = self.trainer.train_model(force=False)
                        if training_success:
                            self.logger.info("Models refreshed and context caches active on the fly.")
                        else:
                            self.logger.warning("Model training execution complete with warnings.")

                    return {"status": "success", "local_path": final_file_path}, 200
                
                self.logger.error(f"Target download pipeline failed. HTTP Code: {response.status_code}")
                return {"status": "error", "message": f"HTTP mirror download failed with code {response.status_code}"}, 500
                
            except Exception as e:
                self.logger.error(f"Crash encountered over streaming download pipeline: {str(e)}")
                return {"status": "error", "message": str(e)}, 500

    # -------------------------------------------------------------
    # ⚡ GENERALIZED UPSTREAM CLOUD ALERT SYSTEM (NON-BLOCKING)
    # -------------------------------------------------------------
    def trigger_cloud_alert(self, event_type: str, message: str):
        """
        Public generalized method to push any system event notification or telemetry log to Supabase.
        Spawns a detached background thread dynamically to protect physical loop timing profiles.
        :param event_type: Type designation token (e.g., 'motion_detected', 'rfid_denied', 'face_success')
        :param message: Human-readable notification context to target real-time mobile app receivers
        """
        self.logger.info(f"Dispatching clean cloud telemetry thread worker for action: [{event_type}]")
        
        # Packing parameters securely into parallel execution loops
        worker = threading.Thread(
            target=self._send_cloud_alert_worker, 
            args=(event_type, message)
        )
        worker.daemon = True
        worker.start()

    def _send_cloud_alert_worker(self, event_type: str, message: str):
        """
        Internal worker container mapping safe HTTP pipeline sequences directly to Cloud tables.
        """
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        
        payload = {
            "event_type": event_type,
            "message": message,
            "status": "unread"
        }
        
        try:
            # 5-second connection ceiling guard
            response = requests.post(self.cloud_alert_endpoint, json=payload, headers=headers, timeout=5)
            
            if response.status_code in [200, 201]:
                self.logger.info(f"Cloud Alert Sync: Event [{event_type}] committed cleanly to Supabase.")
            else:
                self.logger.error(f"Cloud Alert Sync Failed for [{event_type}]: Remote HTTP {response.status_code}")
                
        except requests.exceptions.Timeout:
            self.logger.error(f"Cloud Alert Sync Timeout: Dropping network execution thread on frame state [{event_type}].")
        except Exception as e:
            self.logger.error(f"Cloud Alert Sync Exception on channel [{event_type}]: Connection layout missing -> {e}")

    def handle_intruder_alert(self, data: dict) -> tuple[dict, int]:
        """
        Extensible Slot Example: Process downstream physical latch operations or panic devices.
        """
        self.logger.warning("INTRUDER ALERT BOUND: Hardware lockdown conditions tripped!")
        return {"status": "lockdown_sequence_fired"}, 200