"""Two-stage prediction pipeline coordinating detection, cropping, and classification."""

from typing import List, Optional
import numpy as np

from src.models.classifier import VehicleClassifier
from src.models.detector import VehicleDetector
from src.pipeline.schema import DetectionResult
from src.utils.logger import get_logger

logger = get_logger("vehicle_pipeline")


class VehiclePipeline:
    """Orchestrates Stage 1 (YOLOv8 Detection) and Stage 2 (MobileNetV2 Classification)."""

    def __init__(self, detector: VehicleDetector, classifier: VehicleClassifier):
        """Initialize pipeline with detector and classifier instances.

        Args:
            detector: Initialized VehicleDetector instance.
            classifier: Initialized VehicleClassifier instance.
        """
        self.detector = detector
        self.classifier = classifier

    @classmethod
    def from_config(cls, config=None) -> "VehiclePipeline":
        """Factory method to construct the pipeline from centralized AppConfig.

        Args:
            config: Optional AppConfig instance. If None, loaded via load_config().

        Returns:
            VehiclePipeline: Ready-to-use pipeline instance.
        """
        from src.config import load_config
        cfg = config or load_config()

        detector = VehicleDetector(
            model_path=cfg.detector_model_path,
            conf=cfg.detector.confidence_threshold,
            vehicle_classes=cfg.detector.vehicle_classes,
        )

        classifier = VehicleClassifier(
            model_path=cfg.classifier_model_path,
            class_indices_path=cfg.class_indices_path,
        )

        return cls(detector=detector, classifier=classifier)

    def process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        """Process a single frame through the complete two-stage pipeline.

        1. Runs YOLOv8 vehicle detection.
        2. Clamps coordinates safely within frame boundaries.
        3. Extracts each valid vehicle crop.
        4. Classifies each crop using MobileNetV2 with unified Phase 2 preprocessing.
        5. Associates detection and classification results into DetectionResult objects.

        Args:
            frame: Input video/camera frame array (OpenCV BGR).

        Returns:
            List[DetectionResult]: List of structured vehicle predictions.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        raw_detections = self.detector.detect(frame)
        results: List[DetectionResult] = []

        for det in raw_detections:
            x1, y1, x2, y2 = det["box"]
            yolo_class = det["yolo_class"]
            yolo_conf = det["confidence"]

            # Clamp coordinates to frame boundaries
            x1_clamped = max(0, min(x1, w - 1))
            y1_clamped = max(0, min(y1, h - 1))
            x2_clamped = max(0, min(x2, w))
            y2_clamped = max(0, min(y2, h))

            # Validate non-degenerate crop geometry
            if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
                continue

            crop = frame[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
            if crop is None or crop.size == 0:
                continue

            # Stage 2: Fine-grained classification
            cnn_class, cnn_conf = self.classifier.predict(crop)

            results.append(
                DetectionResult(
                    box=(x1_clamped, y1_clamped, x2_clamped, y2_clamped),
                    yolo_class=yolo_class,
                    yolo_confidence=yolo_conf,
                    cnn_class=cnn_class,
                    cnn_confidence=cnn_conf,
                )
            )

        return results
