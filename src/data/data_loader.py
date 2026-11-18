"""Data loader and batch generator factory for model training and evaluation.

Creates training and validation ImageDataGenerators from directory structures,
supporting both modern AppConfig objects and legacy configuration dictionaries.
"""

from typing import Any, Dict, Tuple, Union
from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator

from src.data.preprocessor import full_preprocessing


def create_generators(
    config: Union[Dict[str, Any], Any],
) -> Tuple[DirectoryIterator, DirectoryIterator]:
    """Create training and validation data generators from a directory structure.

    Supports both Pydantic AppConfig / DataConfig instances and legacy dictionaries.

    Args:
        config: Configuration dictionary or AppConfig instance containing:
                - dataset_dir (or config.dataset_path)
                - validation_split (default: 0.2)
                - image_height (default: 224)
                - image_width (default: 224)
                - batch_size (default: 32)

    Returns:
        Tuple[DirectoryIterator, DirectoryIterator]: (train_generator, val_generator)
    """
    # Extract values dynamically to support both dict and AppConfig
    if hasattr(config, "data"):
        # AppConfig instance
        dataset_dir = str(config.dataset_path)
        val_split = config.data.validation_split
        target_size = (config.data.image_height, config.data.image_width)
        batch_size = config.data.batch_size
        aug = config.training.augmentations
        rot_range = aug.rotation_range
        w_shift = aug.width_shift_range
        h_shift = aug.height_shift_range
        s_range = aug.shear_range
        z_range = aug.zoom_range
        h_flip = aug.horizontal_flip
    elif isinstance(config, dict):
        # Legacy dictionary
        dataset_dir = config["dataset_dir"]
        val_split = config.get("validation_split", 0.2)
        target_size = (config.get("image_height", 224), config.get("image_width", 224))
        batch_size = config.get("batch_size", 32)
        rot_range = 20
        w_shift = 0.15
        h_shift = 0.15
        s_range = 0.15
        z_range = 0.15
        h_flip = True
    else:
        raise TypeError(f"Unsupported config type: {type(config)}")

    train_datagen = ImageDataGenerator(
        preprocessing_function=full_preprocessing,
        validation_split=val_split,
        rotation_range=rot_range,
        width_shift_range=w_shift,
        height_shift_range=h_shift,
        shear_range=s_range,
        zoom_range=z_range,
        horizontal_flip=h_flip,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=full_preprocessing,
        validation_split=val_split,
    )

    train_gen = train_datagen.flow_from_directory(
        directory=dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        directory=dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_gen, val_gen

