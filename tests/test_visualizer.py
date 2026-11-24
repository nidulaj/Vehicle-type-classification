"""Unit tests for the Visualizer component."""

import numpy as np
import pytest

from src.inference.visualizer import Visualizer
from src.pipeline.schema import DetectionResult


@pytest.fixture
def sample_detection() -> DetectionResult:
    """Create a sample DetectionResult."""
    return DetectionResult(
        box=(50, 60, 200, 220),
        yolo_class="car",
        yolo_confidence=0.88,
        cnn_class="Car",
        cnn_confidence=0.95,
    )


def test_visualizer_initialization():
    """Verify default visualizer configuration."""
    vis = Visualizer()
    assert vis.default_color == (0, 255, 0)
    assert vis.box_thickness == 2
    assert vis.font_scale == 0.6
    assert vis.font_thickness == 2


def test_draw_detection_preserves_result(sample_detection: DetectionResult):
    """Verify drawing does not modify the underlying DetectionResult dataclass."""
    vis = Visualizer()
    frame = np.zeros((300, 400, 3), dtype=np.uint8)

    original_box = sample_detection.box
    original_yolo = sample_detection.yolo_class
    original_cnn = sample_detection.cnn_class

    vis.draw_detection(frame, sample_detection)

    assert sample_detection.box == original_box
    assert sample_detection.yolo_class == original_yolo
    assert sample_detection.cnn_class == original_cnn
    # Ensure pixels were modified in the frame (drawn on)
    assert np.any(frame > 0)


def test_draw_detections_copy_behavior(sample_detection: DetectionResult):
    """Verify copy=True leaves the input frame untouched."""
    vis = Visualizer()
    frame = np.zeros((300, 400, 3), dtype=np.uint8)

    annotated = vis.draw_detections(frame, [sample_detection], copy=True)

    # Original frame must still be all zeros
    assert np.all(frame == 0)
    # Returned frame has bounding box drawn
    assert np.any(annotated > 0)
    assert annotated.shape == frame.shape


def test_draw_detections_near_top_boundary():
    """Verify safe label positioning when bounding box touches top edge (y1 < 10)."""
    vis = Visualizer()
    frame = np.zeros((300, 400, 3), dtype=np.uint8)

    # Box right at y1 = 2 (near top boundary)
    top_detection = DetectionResult(
        box=(30, 2, 150, 100),
        yolo_class="truck",
        yolo_confidence=0.82,
        cnn_class="Truck",
        cnn_confidence=0.91,
    )

    # Drawing should succeed without error or exception
    annotated = vis.draw_detection(frame, top_detection)
    assert np.any(annotated > 0)


def test_draw_detections_empty_list():
    """Verify empty detection list returns unchanged frame."""
    vis = Visualizer()
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
    original = frame.copy()

    annotated = vis.draw_detections(frame, [], copy=True)
    np.testing.assert_array_equal(annotated, original)


def test_draw_detections_empty_frame():
    """Verify empty/None frame handling."""
    vis = Visualizer()
    assert vis.draw_detections(None, []) is None
    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    assert vis.draw_detections(empty_frame, []).size == 0
