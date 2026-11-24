"""Unit tests for VehicleClassifier wrapper."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.models.classifier import VehicleClassifier


def test_classifier_initialization_missing_model():
    with pytest.raises(FileNotFoundError):
        VehicleClassifier(model_path="non_existent_model.keras")


def test_classifier_invalid_and_empty_crops():
    classifier = VehicleClassifier.__new__(VehicleClassifier)
    classifier.idx_to_class = {0: "Bus", 1: "Car"}

    # None crop
    label, conf = classifier.predict(None)
    assert label == "unknown" and conf == 0.0

    # Empty array
    label, conf = classifier.predict(np.array([]))
    assert label == "unknown" and conf == 0.0

    # Zero height or width
    zero_dim_crop = np.zeros((0, 100, 3), dtype=np.uint8)
    label, conf = classifier.predict(zero_dim_crop)
    assert label == "unknown" and conf == 0.0


@patch("src.models.classifier.tf.keras.models.load_model")
@patch("src.models.classifier.preprocess_image")
def test_classifier_predict_valid_crop(mock_preprocess, mock_load_model):
    # Setup mock Keras model
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    # Probabilities for 5 classes: [Bus, Car, Three Wheeler, Truck, motorcycle]
    # Set "Three Wheeler" (index 2) as winner with 0.92
    mock_model.predict.return_value = np.array([[0.02, 0.03, 0.92, 0.02, 0.01]], dtype=np.float32)

    # Setup mock preprocessor returning (1, 224, 224, 3)
    mock_preprocess.return_value = np.zeros((1, 224, 224, 3), dtype=np.float32)

    class_mapping = {0: "Bus", 1: "Car", 2: "Three Wheeler", 3: "Truck", 4: "motorcycle"}
    classifier = VehicleClassifier(
        model_path="models/vehicle_classifier_best.keras",
        idx_to_class=class_mapping,
    )

    dummy_crop = np.full((120, 100, 3), 150, dtype=np.uint8)
    label, confidence = classifier.predict(dummy_crop)

    # Verify preprocess_image was called
    mock_preprocess.assert_called_once()
    assert label == "Three Wheeler"
    assert np.isclose(confidence, 0.92, atol=1e-4)
