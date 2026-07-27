"""
Stage 6 of the ML pipeline: Prediction API. Loads the persisted TensorFlow
model + label list and exposes a simple `predict(text) -> (category, confidence)`
used by the document-upload pipeline to auto-categorize new PDFs.
"""
import json
import logging
import os
from typing import Optional, Tuple

from config.settings import settings
from src.ml.dataset_prep import clean_text

logger = logging.getLogger(__name__)


class DocumentClassifier:
    _instance: Optional["DocumentClassifier"] = None

    def __init__(self):
        self.model = None
        self.labels = None
        self._load()

    def _load(self) -> None:
        if not (os.path.exists(settings.model_path) and os.path.exists(settings.labels_path)):
            logger.warning(
                "No trained classifier found at %s. Run `python -m src.ml.train_classifier` first. "
                "Classification will be skipped until a model is trained.",
                settings.model_path,
            )
            return
        import tensorflow as tf  # local import: keeps API startup fast when model isn't needed yet
        self.model = tf.keras.models.load_model(settings.model_path)
        with open(settings.labels_path) as f:
            self.labels = json.load(f)
        logger.info("Loaded classifier with %d categories.", len(self.labels))

    @classmethod
    def get_instance(cls) -> "DocumentClassifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_ready(self) -> bool:
        return self.model is not None and self.labels is not None

    def predict(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        """Returns (category, confidence) or (None, None) if no model is trained yet."""
        if not self.is_ready():
            return None, None

        import tensorflow as tf
        cleaned = clean_text(text[:5000])  # cap input length for speed
        probs = self.model.predict(tf.constant([cleaned], dtype=tf.string), verbose=0)[0]
        idx = int(probs.argmax())
        return self.labels[idx], float(probs[idx])
