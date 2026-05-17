"""
model_training.py

OOP-structured Face Recognition Model Training & Processing
Integrates with CameraManager for Raspberry Pi / Arducam setups.

Class: ModelTraining
  - train_model()     : Encode all faces in dataset folder (skips if up-to-date)
  - process_frame()   : Detect & recognize faces in a single frame
  - needs_training()  : Check if dataset has changed since last encoding
"""

import os
import pickle
import hashlib
import json

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

    Training Strategy (needs_training check):
      - A manifest file (.dataset_manifest.json) is stored alongside encodings.pickle.
      - The manifest records a fingerprint of every image path + mtime + size.
      - On startup (or on demand), compare the live dataset state against the manifest.
      - If anything changed (new image, deleted image, modified image) → retrain.

    Args:
        dataset_path   (str)  : Path to dataset root. Subfolders = person names.
        encodings_path (str)  : Path to save/load encodings.pickle.
        detection_model(str)  : "hog" (CPU-friendly) or "cnn" (GPU, more accurate).
        encoding_model (str)  : "small" or "large" for face_encodings quality.
        cv_scaler      (int)  : Downscale factor for faster real-time detection.
        tolerance      (float): Face match distance threshold (lower = stricter).
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
            f"ModelTraining initialized | dataset={dataset_path} "
            f"encodings={encodings_path} scaler=1/{cv_scaler}"
        )

    # ──────────────────────────────────────────────
    # PUBLIC: TRAINING
    # ──────────────────────────────────────────────

    def needs_training(self) -> bool:
        """
        Return True if (re)training is required.

        Checks (in order):
          1. encodings.pickle absent  → must train.
          2. manifest absent          → must train.
          3. live dataset hash ≠ saved manifest → dataset changed, retrain.
          4. All match               → no training needed.
        """
        if not self.encodings_path.exists():
            logger.info("needs_training=True : encodings.pickle not found")
            return True

        if not self.manifest_path.exists():
            logger.info("needs_training=True : manifest not found")
            return True

        if self._build_manifest() != self._load_manifest():
            logger.info("needs_training=True : dataset has changed since last training")
            return True

        logger.info("needs_training=False : dataset unchanged, encodings are current")
        return False

    def train_model(self) -> bool:
        """
        Encode all faces in the dataset folder and save to encodings.pickle.

        Automatically skips if needs_training() is False (loads existing encodings).

        Returns:
            True  – success (or skipped because already up-to-date).
            False – no images found or an unrecoverable error occurred.
        """
        if not self.needs_training():
            logger.info("Training skipped – encodings are up-to-date. Loading from disk...")
            return self._load_encodings()

        image_paths = list(paths.list_images(str(self.dataset_path)))
        if not image_paths:
            logger.error(f"No images found in dataset: {self.dataset_path}")
            return False

        logger.info(f"[TRAINING] Processing {len(image_paths)} image(s)...")
        known_encodings: list = []
        known_names: list = []

        for i, image_path in enumerate(image_paths):
            person_name = Path(image_path).parent.name
            logger.info(f"  [{i + 1}/{len(image_paths)}] {person_name} ← {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"  Could not read image, skipping: {image_path}")
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model=self.detection_model)
            encodings = face_recognition.face_encodings(rgb, boxes)

            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(person_name)

        if not known_encodings:
            logger.error("Training failed: no face encodings could be extracted.")
            return False

        # Persist encodings to disk
        data = {"encodings": known_encodings, "names": known_names}
        with open(self.encodings_path, "wb") as f:
            f.write(pickle.dumps(data))

        # Persist manifest so next run can detect changes
        self._save_manifest(self._build_manifest())

        # Cache in memory for immediate use
        self.known_encodings = known_encodings
        self.known_names = known_names

        logger.info(
            f"[TRAINING] Complete. {len(known_encodings)} encoding(s) for "
            f"{len(set(known_names))} person(s) → saved to {self.encodings_path}"
        )
        return True

    # ──────────────────────────────────────────────
    # PUBLIC: REAL-TIME PROCESSING
    # ──────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect and recognize faces in a single BGR frame.

        Stores results internally; retrieve via get_detected_names() or
        visualize via draw_results().

        Args:
            frame: BGR numpy array (from OpenCV or Picamera2.capture_array()).

        Returns:
            The original (unmodified) frame.
        """
        if not self.known_encodings:
            logger.warning("process_frame: no encodings in memory – call train_model() first.")
            return frame

        # Downscale for faster detection
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

        Must be called after process_frame() on the same frame.

        Args:
            frame: BGR numpy array (annotated in-place).

        Returns:
            Annotated frame.
        """
        for (top, right, bottom, left), name in zip(self._face_locations, self._face_names):
            # Scale coordinates back up to original resolution
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
        matches   = face_recognition.compare_faces(
            self.known_encodings, face_encoding, tolerance=self.tolerance
        )
        distances = face_recognition.face_distance(self.known_encodings, face_encoding)
        best_idx  = int(np.argmin(distances))
        return self.known_names[best_idx] if matches[best_idx] else "Unknown"

    def _load_encodings(self) -> bool:
        """Load serialized encodings from disk into memory."""
        try:
            with open(self.encodings_path, "rb") as f:
                data = pickle.loads(f.read())
            self.known_encodings = data["encodings"]
            self.known_names     = data["names"]
            logger.info(
                f"Loaded {len(self.known_encodings)} encoding(s) for "
                f"{len(set(self.known_names))} person(s) from {self.encodings_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load encodings: {e}")
            return False

    def _build_manifest(self) -> dict:
        """
        Walk the dataset folder and return a fingerprint dict.

        Key  : relative image path (stable across machines).
        Value: md5 of "<relpath>:<mtime>:<size>" — detects modifications too.
        """
        manifest = {}
        if not self.dataset_path.exists():
            return manifest

        for img_path in sorted(paths.list_images(str(self.dataset_path))):
            rel         = os.path.relpath(img_path, str(self.dataset_path))
            mtime       = os.path.getmtime(img_path)
            size        = os.path.getsize(img_path)
            fingerprint = hashlib.md5(f"{rel}:{mtime:.6f}:{size}".encode()).hexdigest()
            manifest[rel] = fingerprint

        return manifest

    def _save_manifest(self, manifest: dict) -> None:
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.debug(f"Manifest saved → {self.manifest_path}")

    def _load_manifest(self) -> dict:
        try:
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}