"""
model_training.py

OOP-structured Face Recognition Model Training & Processing
Integrates with CameraManager for Raspberry Pi / Arducam setups.

Class: ModelTraining
  - train_model()     : Encode only new/modified faces (appends and skips up-to-date items)
  - process_frame()   : Detect & recognize faces in a single frame
  - needs_training()  : Check if dataset has changed since last encoding
"""

import os
import pickle
import hashlib
import json
from time import time

import cv2
import numpy as np
import face_recognition
from imutils import paths
from pathlib import Path

try:
    from logs.logger import getLogger
    logger = getLogger("FaceDetection")
except ImportError:
    import logging
    logger = logging.getLogger("FaceDetection")
    logging.basicConfig(level=logging.INFO)


class ModelTraining:
    """
    Handles face recognition model training and real-time frame processing.

    Training Strategy (Incremental Upgrades):
      - Keeps your manifest file (.dataset_manifest.json) matching encodings.pickle.
      - On execution, isolates specific individual file path mismatches.
      - Extracts features ONLY from new paths, keeping existing encodings preserved.
    """

    MANIFEST_FILENAME = ".dataset_manifest.json"

    def __init__(
        self,
        dataset_path: str = "dataset",
        encodings_path: str = "encodings.pickle",
        detection_model: str = "hog",
        encoding_model: str = "large",
        cv_scaler: int = 4,
        tolerance: float = 0.5,
    ):
        self.dataset_path = Path(dataset_path)
        self.encodings_path = Path(encodings_path)
        self.manifest_path = self.encodings_path.parent / self.MANIFEST_FILENAME
        self.detection_model = detection_model
        self.encoding_model = encoding_model
        self.cv_scaler = cv_scaler
        self.tolerance = tolerance

        # Runtime state – populated by train_model() or _load_encodings()
        self.known_encodings: list = []
        self.known_names: list = []

        # Per-frame detection state updated by process_frame()
        self._face_locations: list = []
        self._face_names: list = []

        logger.info(
            f"FaceDetection initialized | dataset={dataset_path} "
            f"encodings={encodings_path} scaler=1/{cv_scaler}"
        )
        
        # Pre-load whatever is currently saved on disk right away
        self._load_encodings()

    # ──────────────────────────────────────────────
    # PUBLIC: TRAINING
    # ──────────────────────────────────────────────

    def needs_training(self) -> bool:
        """
        Return True if training or updating is required.
        """
        if not self.encodings_path.exists():
            logger.info("needs_training=True : encodings.pickle not found")
            return True

        if not self.manifest_path.exists():
            logger.info("needs_training=True : manifest not found")
            return True

        if self._build_manifest() != self._load_manifest():
            logger.info("needs_training=True : dataset modifications or additions detected")
            return True

        logger.info("needs_training=False : dataset unchanged, encodings are current")
        return False

    def train_model(self, force: bool = False, current_hash: str = None) -> bool:
        """
        Processes and extracts encodings for new or updated files only.
        Appends data to the cached profile list to protect runtime performance.
        """
        # Short-circuit if force flag is absent and everything checks out
        if not force and not self.needs_training():
            logger.info("Training skipped – encodings are up-to-date. Loading from disk...")
            return self._load_encodings()

        # Build current states and pull disk records
        live_manifest = self._build_manifest()
        saved_manifest = {} if force else self._load_manifest()

        paths_to_train = []
        for rel_path, fingerprint in live_manifest.items():
            # If the file is completely new or modified, mark it for extraction
            if saved_manifest.get(rel_path) != fingerprint:
                paths_to_train.append(os.path.join(str(self.dataset_path), rel_path))

        # Check for true deletions safely using set math on hashes
        live_fingerprints = set(live_manifest.values())
        saved_fingerprints = set(saved_manifest.values())
        
        # If any old fingerprint is missing from the new set, an absolute deletion occurred
        removed_items_detected = not saved_fingerprints.issubset(live_fingerprints)

        # Handle complete rebuild or incremental operations structurally
        if force or removed_items_detected:
            logger.info("Dataset items removed or forced update initialized. Clearing caches for full rebuild.")
            self.known_encodings = []
            self.known_names = []
            paths_to_train = [os.path.join(str(self.dataset_path), p) for p in live_manifest.keys()]
        else:
            # Safely retain operational vectors from disk cache before handling new files
            self._load_encodings()

        if not paths_to_train:
            logger.info("Incremental Train: No unindexed image objects to train.")
            return True

        logger.info(f"[INCREMENTAL TRAINING] Processing {len(paths_to_train)} target file(s)...")

        new_encodings = []
        new_names = []

        # Extract features exclusively from unparsed imagery
        for i, image_path in enumerate(paths_to_train):
            person_name = Path(image_path).parent.name
            logger.info(f"   [{i + 1}/{len(paths_to_train)}] Extracting: {person_name} ← {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Unable to read file: {image_path}")
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model=self.detection_model)
            encodings = face_recognition.face_encodings(rgb, boxes)

            for encoding in encodings:
                new_encodings.append(encoding)
                new_names.append(person_name)

        # Commit memory upgrades back to system cache files
        if new_encodings or removed_items_detected or force:
            if not removed_items_detected and not force:
                # Merge phase
                self.known_encodings.extend(new_encodings)
                self.known_names.extend(new_names)
            else:
                # Rebuild phase
                self.known_encodings = new_encodings
                self.known_names = new_names

            data = {"encodings": self.known_encodings, "names": self.known_names}
            with open(self.encodings_path, "wb") as f:
                f.write(pickle.dumps(data))

            self._save_manifest(live_manifest)
            logger.info(f"Feature processing complete. Total entries now active: {len(self.known_encodings)}")
        else:
            logger.warning("No face profiles located in newly added assets.")

        # Update execution session tracking hash
        if current_hash:
            hash_path = self.dataset_path / ".dataset_hash.json"
            try:
                with open(hash_path, "w") as f:
                    json.dump({"hash": current_hash, "timestamp": time()}, f)
                logger.info(f"Successfully saved sync hash: {current_hash[:8]}...")
            except Exception as e:
                logger.error(f"Failed to write .dataset_hash.json: {e}")

        return True

    # ──────────────────────────────────────────────
    # PUBLIC: REAL-TIME PROCESSING
    # ──────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect and recognize faces in a single BGR frame.
        """
        if not self.known_encodings:
            logger.warning("process_frame: no encodings in memory – call train_model() first.")
            return frame

        small = cv2.resize(frame, (0, 0), fx=1 / self.cv_scaler, fy=1 / self.cv_scaler)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(
            rgb_small, face_locations, model=self.encoding_model
        )

        self._face_locations = face_locations
        self._face_names = [self._match_face(enc) for enc in face_encodings]

        return frame

    def draw_results(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw bounding boxes and name labels on the frame.
        """
        for (top, right, bottom, left), name in zip(self._face_locations, self._face_names):
            top    *= self.cv_scaler
            right  *= self.cv_scaler
            bottom *= self.cv_scaler
            left   *= self.cv_scaler

            cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
            cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), (244, 42, 3), cv2.FILLED)
            cv2.putText(
                frame, name,
                (left + 6, top - 6),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1,
            )

        return frame

    def get_detected_names(self) -> list:
        """Return person names detected in the most recent frame."""
        return list(self._face_names)

    def get_detected_face_boxes(self) -> list:
        """Return face bounding boxes (scaled to original resolution) from the last frame."""
        return [
            (t * self.cv_scaler, r * self.cv_scaler, b * self.cv_scaler, l * self.cv_scaler)
            for (t, r, b, l) in self._face_locations
        ]

    # ──────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────

    def _match_face(self, face_encoding: np.ndarray) -> str:
        """Compare one encoding against all known faces; return closest match name."""
        if not self.known_encodings:
            return "Unknown"
        matches = face_recognition.compare_faces(
            self.known_encodings, face_encoding, tolerance=self.tolerance
        )
        distances = face_recognition.face_distance(self.known_encodings, face_encoding)
        best_idx = int(np.argmin(distances))
        return self.known_names[best_idx] if matches[best_idx] else "Unknown"

    def _load_encodings(self) -> bool:
        """Load serialized encodings from disk into memory."""
        try:
            if os.path.exists(self.encodings_path) and os.path.getsize(self.encodings_path) > 0:
                with open(self.encodings_path, "rb") as f:
                    data = pickle.loads(f.read())
                self.known_encodings = data["encodings"]
                self.known_names     = data["names"]
                logger.info(
                    f"Loaded {len(self.known_encodings)} encoding(s) for "
                    f" {len(set(self.known_names))} person(s) from {self.encodings_path}"
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load encodings: {e}")
            return False

    def _build_manifest(self) -> dict:
        """
        Walk the dataset folder and return a fingerprint dict.
        """
        manifest = {}
        if not self.dataset_path.exists():
            return manifest

        for img_path in sorted(paths.list_images(str(self.dataset_path))):
            rel = os.path.relpath(img_path, str(self.dataset_path))
            mtime = os.path.getmtime(img_path)
            size = os.path.getsize(img_path)
            fingerprint = hashlib.md5(f"{rel}:{mtime:.6f}:{size}".encode()).hexdigest()
            manifest[rel] = fingerprint

        return manifest

    def _save_manifest(self, manifest: dict) -> None:
        try:
            with open(self.manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.debug(f"Manifest saved -> {self.manifest_path}")
        except Exception as e:
            logger.error(f"Failed to write dataset manifest file: {e}")

    def _load_manifest(self) -> dict:
        try:
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load manifest file: {e}")
            return {}