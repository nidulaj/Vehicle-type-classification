"""Vehicle detector wrapper using Ultralytics YOLOv8.

Responsible for loading YOLOv8 weights, running detection on image frames,
filtering for vehicle classes, and clamping coordinates to frame boundaries.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ultralytics import YOLO

from src.utils.logger import get_logger
from src.utils.path_helpers import resolve_path

logger = get_logger("vehicle_detector")


class VehicleDetector:
    """YOLOv8 vehicle detection wrapper."""

    # Default COCO vehicle classes mapping
    DEFAULT_VEHICLE_CLASSES: Dict[int, str] = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_path: Union[str, Path] = "yolov8n.pt",
        conf: float = 0.4,
        vehicle_classes: Optional[Dict[int, str]] = None,
    ):
        """Initialize the YOLOv8 vehicle detector.

        Args:
            model_path: Path to YOLO model weights (resolved against project root).
            conf: Minimum detection confidence threshold (0.0 to 1.0).
            vehicle_classes: Dictionary mapping class IDs to label names. Defaults
                             to COCO vehicle classes (car, motorcycle, bus, truck).
        """
        self.model_path = resolve_path(model_path)
        self.conf = conf
        self.vehicle_classes = vehicle_classes or self.DEFAULT_VEHICLE_CLASSES

        if not self.model_path.exists():
            raise FileNotFoundError(f"Detector model weights not found at: {self.model_path}")

        logger.info(f"Loading YOLO detector from {self.model_path} (conf={self.conf})")
        self.model = YOLO(str(self.model_path))

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run vehicle detection on a single frame.

        Args:
            frame: Input image array of shape (H, W, 3).

        Returns:
            List[Dict[str, Any]]: List of detection dictionaries matching the legacy schema:
                [
                    {
                        "box": (x1, y1, x2, y2),
                        "confidence": float,
                        "yolo_class": str
                    },
                    ...
                ]
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        results = self.model(
            frame,
            conf=self.conf,
            verbose=False,
        )[0]

        detections: List[Dict[str, Any]] = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls[0])

            if cls_id in self.vehicle_classes:
                raw_x1, raw_y1, raw_x2, raw_y2 = map(int, box.xyxy[0])

                # Clamp bounding box coordinates to frame dimensions
                x1 = max(0, min(raw_x1, w - 1))
                y1 = max(0, min(raw_y1, h - 1))
                x2 = max(0, min(raw_x2, w))
                y2 = max(0, min(raw_y2, h))

                # Validate valid positive area
                if x2 > x1 and y2 > y1:
                    detections.append({
                        "box": (x1, y1, x2, y2),
                        "confidence": float(box.conf[0]),
                        "yolo_class": self.vehicle_classes[cls_id],
                    })

        return detections
