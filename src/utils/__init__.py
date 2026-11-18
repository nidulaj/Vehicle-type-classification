"""Shared utilities package."""

from src.utils.logger import get_logger, setup_logger
from src.utils.path_helpers import PROJECT_ROOT, get_project_root, resolve_path

# Backward compatibility for existing code/notebooks that do:
# from src.utils import draw_detection
try:
    import cv2

    def draw_detection(frame, detection):
        x1, y1, x2, y2 = detection["box"]
        class_name = detection.get("class_name", detection.get("yolo_class", "vehicle"))
        confidence = detection["confidence"]
        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        return frame
except ImportError:
    draw_detection = None  # cv2 not available in environment

__all__ = [
    "PROJECT_ROOT",
    "get_project_root",
    "resolve_path",
    "setup_logger",
    "get_logger",
    "draw_detection",
]

