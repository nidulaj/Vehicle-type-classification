"""Unit tests for VehiclePipeline orchestrator."""

from unittest.mock import MagicMock
import numpy as np

from src.pipeline.schema import DetectionResult
from src.pipeline.vehicle_pipeline import VehiclePipeline


def test_pipeline_empty_frame():
    mock_detector = MagicMock()
    mock_classifier = MagicMock()
    pipeline = VehiclePipeline(detector=mock_detector, classifier=mock_classifier)

    assert pipeline.process_frame(None) == []
    assert pipeline.process_frame(np.array([])) == []
    mock_detector.detect.assert_not_called()


def test_pipeline_no_detections():
    mock_detector = MagicMock()
    mock_detector.detect.return_value = []
    mock_classifier = MagicMock()

    pipeline = VehiclePipeline(detector=mock_detector, classifier=mock_classifier)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    results = pipeline.process_frame(dummy_frame)
    assert results == []
    mock_detector.detect.assert_called_once()
    mock_classifier.predict.assert_not_called()


def test_pipeline_multi_vehicle_flow():
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [
        {"box": (10, 10, 100, 100), "confidence": 0.85, "yolo_class": "car"},
        {"box": (150, 150, 300, 300), "confidence": 0.90, "yolo_class": "truck"},
    ]

    mock_classifier = MagicMock()
    # First crop -> Car (0.95), Second crop -> Truck (0.88)
    mock_classifier.predict.side_effect = [
        ("Car", 0.95),
        ("Truck", 0.88),
    ]

    pipeline = VehiclePipeline(detector=mock_detector, classifier=mock_classifier)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    results = pipeline.process_frame(dummy_frame)

    assert len(results) == 2
    assert isinstance(results[0], DetectionResult)
    assert results[0].box == (10, 10, 100, 100)
    assert results[0].yolo_class == "car"
    assert results[0].cnn_class == "Car"
    assert results[0].cnn_confidence == 0.95
    assert results[0].display_label == "YOLO:car (0.85) | CNN:Car (0.95)"

    assert results[1].box == (150, 150, 300, 300)
    assert results[1].yolo_class == "truck"
    assert results[1].cnn_class == "Truck"
    assert results[1].cnn_confidence == 0.88
    assert results[1].display_label == "YOLO:truck (0.90) | CNN:Truck (0.88)"


def test_pipeline_invalid_crops_skipped():
    mock_detector = MagicMock()
    # Detection with invalid degenerate coordinates where x2 <= x1
    mock_detector.detect.return_value = [
        {"box": (50, 50, 50, 50), "confidence": 0.80, "yolo_class": "car"},  # Zero area
        {"box": (100, 100, 50, 50), "confidence": 0.80, "yolo_class": "car"}, # Inverted
    ]

    mock_classifier = MagicMock()
    pipeline = VehiclePipeline(detector=mock_detector, classifier=mock_classifier)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    results = pipeline.process_frame(dummy_frame)
    # Both invalid boxes must be safely skipped
    assert results == []
    mock_classifier.predict.assert_not_called()
