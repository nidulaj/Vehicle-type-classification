"""Visualizer module for vehicle detection and classification results."""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.pipeline.schema import DetectionResult


class Visualizer:
    """Renders bounding boxes and dual-stage labels on image/video frames."""

    def __init__(
        self,
        default_color: Tuple[int, int, int] = (0, 255, 0),
        box_thickness: int = 2,
        font_scale: float = 0.6,
        font_thickness: int = 2,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
    ):
        """Initialize the visualizer.

        Args:
            default_color: Default BGR color tuple for bounding boxes and labels (default: green).
            box_thickness: Bounding box border thickness in pixels.
            font_scale: Scale factor for label text font.
            font_thickness: Font stroke thickness in pixels.
            font: OpenCV font type.
        """
        self.default_color = default_color
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.font = font

    def draw_detection(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Draw a single detection result on the frame.

        Preserves existing project formatting:
        - Rectangle: (x1, y1) to (x2, y2)
        - Label: result.display_label ('YOLO:<class> (<conf>) | CNN:<class> (<conf>)')

        Safely adjusts text position if the box is near the top image boundary
        to prevent text from clipping off-screen. Does NOT modify the result object.

        Args:
            frame: OpenCV BGR image array.
            result: DetectionResult object.
            color: Optional BGR color override.

        Returns:
            np.ndarray: Frame with bounding box and text drawn.
        """
        box_color = color or self.default_color
        x1, y1, x2, y2 = result.box

        # 1. Draw bounding box rectangle
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            box_color,
            self.box_thickness,
        )

        # 2. Compute text dimensions for safe boundary positioning
        label = result.display_label
        (text_w, text_h), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, self.font_thickness
        )

        # If y1 is too close to top edge (y1 - 10 < text_h), draw text below y1 inside box
        if y1 - 10 < text_h + 4:
            text_y = int(y1 + text_h + 6)
        else:
            text_y = int(y1 - 10)

        text_x = int(max(0, x1))

        # 3. Draw text label
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            self.font,
            self.font_scale,
            box_color,
            self.font_thickness,
        )

        return frame

    def draw_detections(
        self,
        frame: np.ndarray,
        results: List[DetectionResult],
        copy: bool = True,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Draw multiple detection results onto a frame.

        Args:
            frame: Input OpenCV BGR image array.
            results: List of DetectionResult objects.
            copy: If True, operates on a copy of the frame to keep original unmodified.
            color: Optional BGR color override.

        Returns:
            np.ndarray: Annotated frame.
        """
        if frame is None or frame.size == 0:
            return frame

        out_frame = frame.copy() if copy else frame

        for res in results:
            self.draw_detection(out_frame, res, color=color)

        return out_frame
