"""Data augmentations for vehicle image processing.

Provides domain-specific augmentations including Gaussian noise injection (simulating
low-light camera sensor noise) and random CutOut (simulating vehicle occlusions).
"""

from typing import Tuple
import numpy as np


def add_gaussian_noise(img: np.ndarray, mean: float = 0.0, std: float = 0.02) -> np.ndarray:
    """Inject Gaussian noise into an image to simulate sensor grain and low-light artifacts.

    Args:
        img: Input image array.
        mean: Mean of the Gaussian noise distribution.
        std: Standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: Noisy image array with values clipped to valid ranges.
    """
    noise = np.random.normal(mean, std, img.shape)
    img_noisy = img + noise
    # If image is float in [0, 1] range, clip to [0.0, 1.0]; if uint8 [0, 255], clip to [0, 255]
    if img.max() <= 1.0:
        return np.clip(img_noisy, 0.0, 1.0)
    return np.clip(img_noisy, 0.0, 255.0)


def random_cutout(
    img: np.ndarray,
    min_mask_size: int = 20,
    max_mask_size: int = 50,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Apply random rectangular CutOut masking to simulate vehicle occlusions.

    Args:
        img: Input image array of shape (H, W, C).
        min_mask_size: Minimum square mask width/height in pixels.
        max_mask_size: Maximum square mask width/height in pixels.
        fill_value: Value used to fill the masked patch (default: 0.0 for black).

    Returns:
        np.ndarray: Image copy with a randomized rectangular occlusion.
    """
    img_copy = img.copy()
    h, w, _ = img_copy.shape

    # Constrain mask size to image boundaries
    effective_max = min(max_mask_size, h, w)
    effective_min = min(min_mask_size, effective_max)

    if effective_min >= effective_max:
        mask_size = effective_min
    else:
        mask_size = np.random.randint(effective_min, effective_max)

    x = np.random.randint(0, w)
    y = np.random.randint(0, h)

    x1 = max(0, x - mask_size // 2)
    y1 = max(0, y - mask_size // 2)
    x2 = min(w, x + mask_size // 2)
    y2 = min(h, y + mask_size // 2)

    img_copy[y1:y2, x1:x2, :] = fill_value
    return img_copy


def combined_augmentation(
    img: np.ndarray,
    noise_std: float = 0.02,
    max_mask_size: int = 50,
) -> np.ndarray:
    """Apply composite Gaussian noise and CutOut occlusion to an image.

    Args:
        img: Input image array.
        noise_std: Standard deviation of Gaussian noise.
        max_mask_size: Maximum mask dimension for CutOut.

    Returns:
        np.ndarray: Augmented image array.
    """
    img_aug = add_gaussian_noise(img, std=noise_std)
    img_aug = random_cutout(img_aug, max_mask_size=max_mask_size)
    return img_aug

