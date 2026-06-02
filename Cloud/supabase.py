import time
import os
import requests
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import Optional
from logs.logger import getLogger


class SupabaseManager:
    """
    Supabase Storage Manager
    Handles two separate buckets:
      - DATASET_BUCKET    → face recognition dataset (folders + images)
      - RECORDINGS_BUCKET → 10-second video clips recorded by the camera
    """

    def __init__(self, env_file: str = None):

        self.logger = getLogger("SupabaseManager")

        if env_file is None:
            env_file = "/home/doorlock/DoorLock/Cloud/credentials.env"

        self.logger.info(f"Loading env file from: {env_file}")
        if not load_dotenv(env_file):
            self.logger.warning(f"Could not load {env_file} file")

        self._supabase_url: str  = os.getenv("SUPABASE_URL")
        self._supabase_key: str  = os.getenv("SUPABASE_KEY")

        # ── Two separate buckets ──────────────────────────────────
        self._dataset_bucket:    str = os.getenv("DATASET_BUCKET",    "faces")
        self._recordings_bucket: str = os.getenv("RECORDINGS_BUCKET", "Recordings")

        self._client: Optional[Client] = None

        if not all([self._supabase_url, self._supabase_key]):
            self.logger.error("Missing SUPABASE_URL or SUPABASE_KEY in .env file!")
            raise ValueError("Supabase credentials are missing in environment file.")

        self._initialize_client()
        self.logger.info(
            f"SupabaseManager ready  |  dataset bucket: '{self._dataset_bucket}'"
            f"  |  recordings bucket: '{self._recordings_bucket}'"
        )

    # ─────────────────────────────────────────────────────────────
    #  Internal
    # ─────────────────────────────────────────────────────────────

    def _initialize_client(self):
        self._client = create_client(self._supabase_url, self._supabase_key)

    # ─────────────────────────────────────────────────────────────
    #  Getters / Setters (kept for backward compatibility)
    # ─────────────────────────────────────────────────────────────

    def get_dataset_bucket(self) -> str:
        return self._dataset_bucket

    def get_recordings_bucket(self) -> str:
        return self._recordings_bucket

    # ─────────────────────────────────────────────────────────────
    #  Generic helpers  (bucket-agnostic, used internally)
    # ─────────────────────────────────────────────────────────────

    def _upload_file_to_bucket(self, bucket: str, file_path: str, file_name: Optional[str] = None) -> bool:
        """Upload a local file into the specified bucket."""
        if not self._client:
            self.logger.error("Supabase client not initialised.")
            return False

        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return False

        try:
            if not file_name:
                file_name = os.path.basename(file_path)

            with open(file_path, "rb") as f:
                self._client.storage.from_(bucket).upload(file_name, f)

            self.logger.info(f"✅ Uploaded '{file_name}' → bucket '{bucket}'")
            return True

        except Exception as e:
            self.logger.error(f"❌ Upload to '{bucket}' failed: {e}")
            return False

    def _get_public_url(self, bucket: str, file_name: str) -> Optional[str]:
        """Return the public URL for a file in the specified bucket."""
        if not self._client:
            return None
        try:
            url = self._client.storage.from_(bucket).get_public_url(file_name)
            return url
        except Exception as e:
            self.logger.error(f"Failed to get public URL from '{bucket}': {e}")
            return None

    # ─────────────────────────────────────────────────────────────
    #  RECORDINGS bucket  (video clips)
    # ─────────────────────────────────────────────────────────────

    def upload_recording(self, file_path: str, file_name: Optional[str] = None) -> Optional[str]:
        """
        Upload a video clip to the Recordings bucket and return its public URL.
        This is the method called from main.py after the 10-second recording.
        """
        if self._upload_file_to_bucket(self._recordings_bucket, file_path, file_name):
            actual_name = file_name or os.path.basename(file_path)
            return self._get_public_url(self._recordings_bucket, actual_name)
        return None

    # ── Kept for backward compatibility (main.py used upload_and_get_url) ──
    def upload_and_get_url(self, file_path: str, file_name: Optional[str] = None) -> Optional[str]:
        """Alias → uploads to the Recordings bucket."""
        return self.upload_recording(file_path, file_name)

    # ─────────────────────────────────────────────────────────────
    #  DATASET bucket  (face images / folders)
    # ─────────────────────────────────────────────────────────────

    def get_public_url(self, file_name: str) -> Optional[str]:
        """Get public URL for a file inside the dataset bucket."""
        return self._get_public_url(self._dataset_bucket, file_name)

    def list_files(self):
        """List all files in the dataset bucket."""
        if not self._client:
            return None
        try:
            files = self._client.storage.from_(self._dataset_bucket).list()
            file_names = [f['name'] for f in files]
            self.logger.info(f"Total files in dataset bucket: {len(file_names)}")
            return file_names
        except Exception as e:
            self.logger.error(f"Failed to list dataset files: {e}")
            return None

    def import_entire_dataset(self, local_destination_path: str):
        """
        Scans the root bucket folders first, then enters each individual directory
        to check for new or missing images. Only downloads files that do not
        exist locally on the system.
        """
        if not self._client:
            self.logger.error("Supabase client not initialized for dataset import.")
            return

        base_path = os.path.abspath(local_destination_path)

        try:
            # Step 1: List items at the root level of the bucket
            root_items = self._client.storage.from_(self._dataset_bucket).list(path='')
            
            if not root_items:
                self.logger.info("Dataset bucket is completely empty.")
                return

            # Filter out root directories (folder entries lack file ids or end with a slash)
            folders = [item['name'] for item in root_items if item.get('id') is None or item['name'].endswith('/')]
            
            if not folders:
                folders = [''] # Target root directory if no explicit subfolders are found

            self.logger.info(f"Folders found on cloud storage: {folders}")

            # Step 2: Traverse inside each identified directory to discover actual files
            for folder in folders:
                folder_clean = folder.strip('/')
                self.logger.info(f"Scanning cloud folder: '{folder_clean}'...")
                
                bucket_files = self._client.storage.from_(self._dataset_bucket).list(path=folder_clean)
                
                if not bucket_files:
                    continue

                for item in bucket_files:
                    file_name = item.get('name', '')
                    
                    # Ignore directory metadata placeholders
                    if not file_name or '.emptyFolderPlaceholder' in file_name:
                        continue
                    
                    # Generate structural cloud resource path and target system file path
                    remote_file_path = f"{folder_clean}/{file_name}" if folder_clean else file_name
                    final_local_file = os.path.join(base_path, remote_file_path)

                    # Create parent folder structure locally if missing
                    os.makedirs(os.path.dirname(final_local_file), exist_ok=True)

                    # Verify if the real file exists locally and has non-zero size
                    if os.path.isfile(final_local_file) and os.path.getsize(final_local_file) > 0:
                        continue

                    # Fetch valid asset URI for downloading the target file
                    public_url = self.get_public_url(remote_file_path)
                    if not public_url:
                        self.logger.warning(f"Could not resolve URL for resource: {remote_file_path}")
                        continue

                    self.logger.info(f"New image resource detected. Downloading: {remote_file_path}")
                    res = requests.get(public_url, stream=True, timeout=30)

                    if res.status_code == 200:
                        with open(final_local_file, 'wb') as f:
                            for chunk in res.iter_content(chunk_size=8192):
                                f.write(chunk)
                        self.logger.info(f"Successfully saved locally: {remote_file_path}")
                    else:
                        self.logger.error(f"Download request failed (HTTP {res.status_code}): {remote_file_path}")

            self.logger.info("Dataset synchronization process complete.")

        except Exception as e:
            self.logger.error(f"Dataset import sequence encountered an exception: {e}")