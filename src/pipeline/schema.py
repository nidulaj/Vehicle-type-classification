"""Structured schema for detection and classification results."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DetectionResult:
    """Represents a single detected and classified vehicle in a frame."""

    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    yolo_class: str
    yolo_confidence: float
    cnn_class: str
    cnn_confidence: float

    @property
    def display_label(self) -> str:
        """Formatted label matching original visualization format."""
        return f"YOLO:{self.yolo_class} ({self.yolo_confidence:.2f}) | CNN:{self.cnn_class} ({self.cnn_confidence:.2f})"
