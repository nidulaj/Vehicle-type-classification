"""Vehicle classifier wrapper using fine-tuned MobileNetV2.

Responsible for loading the MobileNetV2 Keras model, preprocessing vehicle crops
using the unified Phase 2 preprocessing function, and outputting classification
predictions across the 5 target classes.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf

from src.data.preprocessor import preprocess_image
from src.utils.logger import get_logger
from src.utils.path_helpers import resolve_path

logger = get_logger("vehicle_classifier")


class VehicleClassifier:
    """MobileNetV2 vehicle classification wrapper."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/vehicle_classifier_best.keras",
        idx_to_class: Optional[Dict[int, str]] = None,
        class_indices_path: Union[str, Path] = "artifacts/class_indices.json",
    ):
        """Initialize the MobileNetV2 vehicle classifier.

        Args:
            model_path: Path to trained Keras model file (.keras or .h5).
            idx_to_class: Optional mapping from integer class index to class name.
                          If omitted, loaded from class_indices_path.
            class_indices_path: Path to class_indices.json if idx_to_class is not provided.
        """
        self.model_path = resolve_path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Classifier model file not found at: {self.model_path}")

        # Resolve class mapping
        if idx_to_class is not None:
            self.idx_to_class = idx_to_class
        else:
            indices_file = resolve_path(class_indices_path)
            if not indices_file.exists():
                raise FileNotFoundError(f"Class indices file not found at: {indices_file}")
            with open(indices_file, "r", encoding="utf-8") as f:
                raw_indices = json.load(f)
            # Invert {"Bus": 0, "Car": 1, ...} -> {0: "Bus", 1: "Car", ...}
            self.idx_to_class = {int(v): k for k, v in raw_indices.items()}

        logger.info(f"Loading MobileNetV2 classifier from {self.model_path}")
        self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
        logger.info(f"Classifier initialized with {len(self.idx_to_class)} classes: {list(self.idx_to_class.values())}")

    def predict(self, crop: np.ndarray) -> Tuple[str, float]:
        """Classify a single cropped vehicle patch.

        Uses the shared Phase 2 preprocess_image function (MobileNetV2 normalization
        scaling to [-1.0, 1.0]), eliminating train-serve skew.

        Args:
            crop: Input image array (OpenCV BGR crop).

        Returns:
            Tuple[str, float]: (predicted_class_name, confidence_score)
                               Returns ("unknown", 0.0) if crop is invalid or empty.
        """
        if crop is None or crop.size == 0:
            return "unknown", 0.0

        if len(crop.shape) != 3 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return "unknown", 0.0

        try:
            # Preprocess crop using unified Phase 2 pipeline (BGR -> RGB, resize 224x224, [-1, 1], expand dims)
            img_tensor = preprocess_image(
                crop,
                target_size=(224, 224),
                is_bgr=True,
                preserve_aspect_ratio=False,
                expand_dims=True,
            )

            preds = self.model.predict(img_tensor, verbose=0)[0]
            class_id = int(np.argmax(preds))
            confidence = float(np.max(preds))

            class_name = self.idx_to_class.get(class_id, "unknown")
            return class_name, confidence

        except Exception as e:
            logger.warning(f"Error classifying vehicle crop: {e}")
            return "unknown", 0.0
