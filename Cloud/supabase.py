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
            env_file = "/home/doorlock/DoorLock/supabase/credentials.env"

        self.logger.info(f"Loading env file from: {env_file}")
        if not load_dotenv(env_file):
            self.logger.warning(f"Could not load {env_file} file")

        self._supabase_url: str  = os.getenv("SUPABASE_URL")
        self._supabase_key: str  = os.getenv("SUPABASE_KEY")

        # ── Two separate buckets ──────────────────────────────────
        self._dataset_bucket:    str = os.getenv("DATASET_BUCKET",    "dataset")
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
        Recursively scan the DATASET bucket, mirror its folder structure,
        and download every image file to local_destination_path.
        Skips .emptyFolderPlaceholder entries.
        """
        if not self._client:
            self.logger.error("Supabase client not initialised for dataset import.")
            return

        base_path = os.path.abspath(local_destination_path)

        try:
            files = self._client.storage.from_(self._dataset_bucket).list(
                path='', options={"recursive": True}
            )

            if not files:
                self.logger.info("Dataset bucket is empty. Nothing to import.")
                return

            self.logger.info(f"Found {len(files)} items in dataset bucket.")

            for item in files:
                file_path = item.get('name', '')

                # Skip Supabase placeholder entries
                if '.emptyFolderPlaceholder' in file_path:
                    continue

                path_parts    = file_path.split("/")
                is_nested     = len(path_parts) > 1
                parent_folder = path_parts[0] if is_nested else None
                file_name     = path_parts[-1]

                target_dir = os.path.join(base_path, parent_folder) if parent_folder else base_path
                os.makedirs(target_dir, exist_ok=True)

                if file_path.endswith("/"):
                    # Pure folder entry — directory already created above
                    continue

                final_local_file = os.path.join(target_dir, file_name)

                # Skip files already downloaded (no re-download on every restart)
                if os.path.exists(final_local_file):
                    self.logger.info(f"⏭️  Already exists locally, skipping: {final_local_file}")
                    continue

                public_url = self.get_public_url(file_path)
                if not public_url:
                    self.logger.warning(f"Could not get URL for: {file_path}")
                    continue

                self.logger.info(f"📥 Downloading: {file_path}")
                res = requests.get(public_url, stream=True, timeout=30)

                if res.status_code == 200:
                    with open(final_local_file, 'wb') as f:
                        for chunk in res.iter_content(chunk_size=8192):
                            f.write(chunk)
                    self.logger.info(f"✅ Saved: {final_local_file}")
                else:
                    self.logger.error(f"❌ Download failed (HTTP {res.status_code}): {file_path}")

            self.logger.info("📦 Dataset import complete.")

        except Exception as e:
            self.logger.error(f"Dataset import crashed: {e}")