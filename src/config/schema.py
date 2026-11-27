"""Pydantic configuration models and loader for centralized validation.

Provides strongly typed, validated configuration schemas for all components:
project metadata, paths, dataset parameters, YOLO detector, MobileNetV2 classifier,
training/fine-tuning hyperparameters, live inference, and logging.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field, field_validator

from src.utils.path_helpers import PROJECT_ROOT, resolve_path


class ProjectConfig(BaseModel):
    """Project metadata."""
    name: str = "vehicle-type-classification"
    version: str = "1.0.0"
    description: str = ""


class PathsConfig(BaseModel):
    """Core directory and file paths relative to project root."""
    artifacts_dir: str = "artifacts"
    models_dir: str = "models"
    data_dir: str = "data"
    outputs_dir: str = "outputs"
    class_indices_path: str = "artifacts/class_indices.json"


class DataConfig(BaseModel):
    """Dataset and preprocessing specifications."""
    dataset_dir: str = "data/classification-dataset"
    image_height: int = Field(default=224, gt=0)
    image_width: int = Field(default=224, gt=0)
    image_channels: int = Field(default=3, gt=0)
    batch_size: int = Field(default=32, gt=0)
    validation_split: float = Field(default=0.2, ge=0.0, lt=1.0)


class DetectorConfig(BaseModel):
    """Stage 1: YOLOv8 vehicle detector settings."""
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    vehicle_classes: Dict[int, str] = Field(
        default_factory=lambda: {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    )


class ClassifierConfig(BaseModel):
    """Stage 2: MobileNetV2 fine-grained classifier settings."""
    model_path: str = "models/vehicle_classifier_best.keras"
    input_height: int = Field(default=224, gt=0)
    input_width: int = Field(default=224, gt=0)
    num_classes: int = Field(default=5, gt=0)
    class_names: List[str] = Field(
        default_factory=lambda: ["Bus", "Car", "Three Wheeler", "Truck", "motorcycle"]
    )


class TrainingStage1Config(BaseModel):
    """Stage 1 training (frozen base model)."""
    epochs: int = Field(default=10, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    early_stopping_patience: int = Field(default=5, ge=1)


class TrainingStage2Config(BaseModel):
    """Stage 2 fine-tuning (unfrozen top layers)."""
    epochs: int = Field(default=20, gt=0)
    learning_rate: float = Field(default=1e-5, gt=0.0)
    unfreeze_last_n_layers: int = Field(default=30, ge=1)
    early_stopping_patience: int = Field(default=5, ge=1)
    reduce_lr_patience: int = Field(default=3, ge=1)
    reduce_lr_factor: float = Field(default=0.3, gt=0.0, lt=1.0)
    reduce_lr_min: float = Field(default=1e-6, gt=0.0)


class AugmentationsConfig(BaseModel):
    """Image data augmentation parameters."""
    rotation_range: int = 20
    width_shift_range: float = 0.15
    height_shift_range: float = 0.15
    shear_range: float = 0.15
    zoom_range: float = 0.15
    horizontal_flip: bool = True
    gaussian_noise_std: float = 0.02
    cutout_mask_size: int = 50


class TrainingConfig(BaseModel):
    """Composite training hyperparameters."""
    stage1: TrainingStage1Config = Field(default_factory=TrainingStage1Config)
    stage2: TrainingStage2Config = Field(default_factory=TrainingStage2Config)
    augmentations: AugmentationsConfig = Field(default_factory=AugmentationsConfig)


class InferenceConfig(BaseModel):
    """Live and video inference settings."""
    camera_index: int = 1
    frame_width: int = 1920
    frame_height: int = 1080
    window_name: str = "Real-Time Vehicle Classification"
    fullscreen: bool = True


class LoggingConfig(BaseModel):
    """Application logging settings."""
    level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "outputs/logs/app.log"

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"Invalid log level '{v}'. Must be one of {valid_levels}")
        return upper_v


class TrackingConfig(BaseModel):
    """MLflow experiment tracking configuration."""
    enabled: bool = True
    experiment_name: str = Field(default="vehicle-type-classification", min_length=1)
    tracking_uri: str = Field(default="sqlite:///mlflow.db", min_length=1)
    log_models: bool = True

    @field_validator("experiment_name", "tracking_uri")
    @classmethod
    def validate_non_empty(cls, v: str, info) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"'{info.field_name}' must be a non-empty string.")
        return v.strip()


class AppConfig(BaseModel):
    """Master application configuration model."""
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    # Convenience properties that return fully resolved absolute Path objects
    @property
    def artifacts_path(self) -> Path:
        return resolve_path(self.paths.artifacts_dir)

    @property
    def models_path(self) -> Path:
        return resolve_path(self.paths.models_dir)

    @property
    def data_path(self) -> Path:
        return resolve_path(self.paths.data_dir)

    @property
    def outputs_path(self) -> Path:
        return resolve_path(self.paths.outputs_dir)

    @property
    def class_indices_path(self) -> Path:
        return resolve_path(self.paths.class_indices_path)

    @property
    def dataset_path(self) -> Path:
        return resolve_path(self.data.dataset_dir)

    @property
    def detector_model_path(self) -> Path:
        return resolve_path(self.detector.model_path)

    @property
    def classifier_model_path(self) -> Path:
        return resolve_path(self.classifier.model_path)

    @property
    def log_file_path(self) -> Optional[Path]:
        if self.logging.log_to_file and self.logging.log_file:
            return resolve_path(self.logging.log_file)
        return None


def load_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Load, parse, and validate the application configuration YAML file.

    Args:
        config_path: Path to the YAML configuration file. If omitted, defaults
                     to PROJECT_ROOT / 'configs' / 'config.yaml'.

    Returns:
        AppConfig: Validated strongly typed configuration instance.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        ValidationError: If the configuration violates the schema.
    """
    if config_path is None:
        target_path = PROJECT_ROOT / "configs" / "config.yaml"
    else:
        target_path = resolve_path(config_path)

    if not target_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at: {target_path}. "
            "Please ensure configs/config.yaml exists."
        )

    with open(target_path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    return AppConfig.model_validate(raw_dict)

