"""Model architecture definition for vehicle classification.

Extracts the exact MobileNetV2 transfer learning architecture from notebook 03,
separating model topology construction from training and inference routines.
"""

from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


def build_mobilenetv2_classifier(
    num_classes: int = 5,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    weights: Optional[str] = "imagenet",
    freeze_base: bool = True,
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Construct the two-stage MobileNetV2 classification model.

    Preserves the exact layer structure, activations, and initialization
    from notebook 03:
    1. MobileNetV2 base (pre-trained on ImageNet, without top classification head)
    2. GlobalAveragePooling2D
    3. BatchNormalization
    4. Dense(128, activation="relu")
    5. Dropout(0.5)
    6. Dense(num_classes, activation="softmax")

    Args:
        num_classes: Number of classification categories (default: 5).
        input_shape: Input image tensor dimensions (H, W, C).
        weights: Pre-trained weights to load for base model ("imagenet" or None).
        freeze_base: If True, freezes all base model layers for Stage 1 feature extraction.

    Returns:
        Tuple[tf.keras.Model, tf.keras.Model]: (sequential_model, base_model)
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )

    if freeze_base:
        base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    return model, base_model


def unfreeze_top_layers(base_model: tf.keras.Model, num_layers: int = 30) -> None:
    """Unfreeze the top N layers of the MobileNetV2 base model for Stage 2 fine-tuning.

    Preserves the exact layer freezing logic from notebook 03 cell 12:
    - Sets base_model.trainable = True
    - Freezes all bottom layers up to the last `num_layers` (i.e. layers[:-30])

    Args:
        base_model: MobileNetV2 base model instance.
        num_layers: Number of top layers to leave trainable (default: 30).
    """
    base_model.trainable = True

    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
