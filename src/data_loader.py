"""Backward-compatibility bridge for legacy code and notebooks.

Re-exports data loading and preprocessing functions from the modular src.data package.
New code should import directly from `src.data`.
"""

from src.data.augmentations import (
    add_gaussian_noise,
    combined_augmentation,
    random_cutout,
)
from src.data.data_loader import create_generators
from src.data.preprocessor import (
    full_preprocessing,
    resize_if_smaller,
)

__all__ = [
    "add_gaussian_noise",
    "random_cutout",
    "combined_augmentation",
    "resize_if_smaller",
    "full_preprocessing",
    "create_generators",
]
