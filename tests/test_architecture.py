"""Unit tests for MobileNetV2 architecture builder and layer freezing logic."""

from pathlib import Path
import numpy as np
import tensorflow as tf

from src.models.architecture import (
    build_mobilenetv2_classifier,
    unfreeze_top_layers,
)
from src.utils.path_helpers import PROJECT_ROOT


def test_architecture_construction():
    # Build model without downloading ImageNet weights for quick unit test
    model, base_model = build_mobilenetv2_classifier(
        num_classes=5,
        input_shape=(224, 224, 3),
        weights=None,
        freeze_base=True,
    )

    # 1. Output shape verification
    dummy_input = np.zeros((2, 224, 224, 3), dtype=np.float32)
    output = model(dummy_input)
    assert output.shape == (2, 5), f"Expected output shape (2, 5), got {output.shape}"

    # 2. Verify layer sequence
    layer_types = [layer.__class__.__name__ for layer in model.layers]
    expected_types = [
        "Functional",  # MobileNetV2 base
        "GlobalAveragePooling2D",
        "BatchNormalization",
        "Dense",
        "Dropout",
        "Dense",
    ]
    assert layer_types == expected_types, f"Layer mismatch! Got {layer_types}"

    # 3. Verify Stage 1 base model is frozen
    assert base_model.trainable is False, "Base model should be frozen when freeze_base=True"
    # When base is frozen, base model weights should not be trainable
    assert len(base_model.trainable_weights) == 0, "Frozen base model must have 0 trainable weights"


def test_unfreeze_top_layers():
    model, base_model = build_mobilenetv2_classifier(
        num_classes=5,
        input_shape=(224, 224, 3),
        weights=None,
        freeze_base=True,
    )

    # Unfreeze top 30 layers
    unfreeze_top_layers(base_model, num_layers=30)

    assert base_model.trainable is True, "Base model should be trainable after unfreezing"

    total_base_layers = len(base_model.layers)
    frozen_layers = base_model.layers[:-30]
    unfrozen_layers = base_model.layers[-30:]

    # Verify bottom layers are frozen
    for layer in frozen_layers:
        assert layer.trainable is False, f"Layer {layer.name} should be frozen!"

    # Verify top 30 layers are trainable
    for layer in unfrozen_layers:
        assert layer.trainable is True, f"Layer {layer.name} should be trainable!"


def test_existing_trained_model_loadable():
    # Verify the existing trained checkpoint matches the architecture and can be loaded
    weights_path = PROJECT_ROOT / "models" / "vehicle_classifier_best.keras"
    assert weights_path.exists(), f"Weights file missing: {weights_path}"

    loaded_model = tf.keras.models.load_model(str(weights_path), compile=False)
    assert loaded_model.output_shape == (None, 5), f"Expected (None, 5), got {loaded_model.output_shape}"

    # Run dummy prediction
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    preds = loaded_model.predict(dummy_input, verbose=0)
    assert preds.shape == (1, 5)
    assert np.isclose(np.sum(preds), 1.0, atol=1e-4), "Softmax output must sum to 1.0"
