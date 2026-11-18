"""Unified image preprocessing module for vehicle classification.

Ensures that training, validation, and real-time inference use identical,
compatible mathematical transformations, eliminating train-serve skew.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def resize_if_smaller(img: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Conditionally resize and pad an image only if its dimensions are smaller than target.

    Preserves the exact legacy behavior from the original training pipeline.

    Args:
        img: Input image array of shape (H, W, C).
        target_size: Target (height, width) tuple.

    Returns:
        np.ndarray: Resized and padded image array.
    """
    h, w, _ = img.shape
    if h < target_size[0] or w < target_size[1]:
        img_padded = tf.image.resize_with_pad(img, target_size[0], target_size[1])
        return img_padded.numpy()
    return img


def resize_with_pad(img: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize an image to target size while preserving aspect ratio using zero padding.

    Args:
        img: Input image array of shape (H, W, C).
        target_size: Target (height, width) tuple.

    Returns:
        np.ndarray: Padded image of shape (target_height, target_width, C).
    """
    img_padded = tf.image.resize_with_pad(img, target_size[0], target_size[1])
    return img_padded.numpy()


def full_preprocessing(img: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline used by training and validation ImageDataGenerators.

    Preserved verbatim from the original training pipeline to maintain reproducible
    generator behavior.

    Args:
        img: Image array passed by ImageDataGenerator.

    Returns:
        np.ndarray: Preprocessed image normalized to [-1.0, 1.0] via MobileNetV2 preprocess_input.
    """
    img = resize_if_smaller(img, (224, 224))
    img = preprocess_input(img)
    return img


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    is_bgr: bool = True,
    preserve_aspect_ratio: bool = False,
    expand_dims: bool = True,
) -> np.ndarray:
    """Unified preprocessing function for inference crops and standalone evaluation.

    Takes a raw vehicle crop (e.g. from OpenCV VideoCapture or cv2.imread),
    converts color space, resizes, and applies official MobileNetV2 normalization.

    Args:
        image: Input image array (e.g., from camera crop).
        target_size: Target dimensions as (height, width). Defaults to (224, 224).
        is_bgr: Set to True if input is BGR (standard OpenCV format).
        preserve_aspect_ratio: If True, uses zero-padding to prevent distortion.
                               If False, uses standard bilinear resizing (matching original prototype).
        expand_dims: If True, prepends a batch dimension of size 1 (shape: 1, H, W, C).

    Returns:
        np.ndarray: Preprocessed float32 array in range [-1.0, 1.0].

    Raises:
        ValueError: If input image is None or has empty dimensions.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be None or empty.")

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3-channel color image, got shape: {image.shape}")

    # Color space conversion: OpenCV reads BGR, MobileNetV2 expects RGB
    if is_bgr:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image.copy()

    # Resizing
    if preserve_aspect_ratio:
        img_resized = resize_with_pad(img_rgb, target_size=target_size)
    else:
        # Standard bilinear resize (width, height order in OpenCV)
        img_resized = cv2.resize(img_rgb, (target_size[1], target_size[0]))

    # Ensure float32 representation
    img_float = img_resized.astype(np.float32)

    # MobileNetV2 normalization: scales [0, 255] to [-1.0, 1.0]
    img_normalized = preprocess_input(img_float)

    # Add batch dimension for model input: (1, H, W, C)
    if expand_dims:
        return np.expand_dims(img_normalized, axis=0)

    return img_normalized

