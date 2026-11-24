"""Unit tests for VehicleDetector wrapper."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.models.detector import VehicleDetector


class DummyBox:
    def __init__(self, xyxy, conf, cls_id):
        self.xyxy = [xyxy]
        self.conf = [conf]
        self.cls = [cls_id]


class DummyResults:
    def __init__(self, boxes):
        self.boxes = boxes


def test_detector_initialization_missing_weights():
    with pytest.raises(FileNotFoundError):
        VehicleDetector(model_path="non_existent_weights.pt")


@patch("src.models.detector.YOLO")
def test_detector_clamping_and_filtering(mock_yolo):
    # Setup mock YOLO
    mock_instance = MagicMock()
    mock_yolo.return_value = mock_instance

    # Mock detection:
    # Box 1: Car (class 2) with negative coordinate (-10, -5, 100, 150)
    # Box 2: Person (class 0) - should be filtered out!
    # Box 3: Bus (class 5) exceeding frame boundaries (50, 50, 800, 600) on 640x480 frame
    boxes = [
        DummyBox(xyxy=[-10, -5, 100, 150], conf=0.88, cls_id=2),
        DummyBox(xyxy=[20, 20, 50, 80], conf=0.95, cls_id=0),
        DummyBox(xyxy=[50, 50, 800, 600], conf=0.75, cls_id=5),
    ]
    mock_instance.return_value = [DummyResults(boxes=boxes)]

    detector = VehicleDetector(model_path="yolov8n.pt", conf=0.4)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(dummy_frame)

    # Person should be filtered out, leaving 2 detections (Car, Bus)
    assert len(detections) == 2, f"Expected 2 vehicle detections, got {len(detections)}"

    # Check Car clamping: -10, -5 should be clamped to 0, 0
    car_det = detections[0]
    assert car_det["yolo_class"] == "car"
    assert car_det["confidence"] == 0.88
    assert car_det["box"][0] >= 0
    assert car_det["box"][1] >= 0
    assert car_det["box"] == (0, 0, 100, 150)

    # Check Bus clamping: 800, 600 should be clamped to width 640, height 480
    bus_det = detections[1]
    assert bus_det["yolo_class"] == "bus"
    assert bus_det["box"][2] <= 640
    assert bus_det["box"][3] <= 480
    assert bus_det["box"] == (50, 50, 640, 480)


def test_detector_empty_frame():
    # Calling detect on None or empty array returns empty list without error
    detector = VehicleDetector.__new__(VehicleDetector)
    detector.conf = 0.4
    detector.vehicle_classes = {2: "car"}

    assert detector.detect(None) == []
    assert detector.detect(np.array([])) == []
