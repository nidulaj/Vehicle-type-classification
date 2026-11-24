"""Model training orchestrator for two-stage transfer learning.

Extracts the training workflow from notebook 03 into a reusable component:
- Computes balanced class weights
- Stage 1: Feature extraction with frozen MobileNetV2 base (Adam lr=1e-4)
- Stage 2: Fine-tuning unfreezing top 30 layers (Adam lr=1e-5, ReduceLROnPlateau)
- Callbacks, checkpointing, and model persistence
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.config import AppConfig, load_config
from src.data.data_loader import create_generators
from src.models.architecture import (
    build_mobilenetv2_classifier,
    unfreeze_top_layers,
)
from src.utils.logger import get_logger
from src.utils.path_helpers import resolve_path

logger = get_logger("vehicle_trainer")


def compute_balanced_class_weights(classes: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights for training imbalanced datasets.

    Preserves exact formula from notebook 03 cell 10.

    Args:
        classes: 1D array of class indices for all training samples.

    Returns:
        Dict[int, float]: Mapping from integer class index to balanced weight.
    """
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(classes),
        y=classes,
    )
    return dict(enumerate(weights))


class VehicleTrainer:
    """Orchestrates Stage 1 and Stage 2 transfer learning for vehicle classification."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize trainer with configuration.

        Args:
            config: Optional AppConfig instance. If None, loaded via load_config().
            checkpoint_dir: Directory where model checkpoints will be saved.
                            Defaults to configured models_dir.
        """
        self.config = config or load_config()
        self.models_dir = resolve_path(checkpoint_dir or self.config.paths.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.best_checkpoint_path = self.models_dir / "vehicle_classifier_best.keras"
        self.stage2_save_path = self.models_dir / "vehicle_mobilenetv2_stage2.keras"

        self.model: Optional[tf.keras.Model] = None
        self.base_model: Optional[tf.keras.Model] = None
        self.class_weights: Optional[Dict[int, float]] = None

    def setup_model(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """Construct the two-stage model architecture."""
        logger.info("Constructing MobileNetV2 architecture with frozen base...")
        self.model, self.base_model = build_mobilenetv2_classifier(
            num_classes=self.config.classifier.num_classes,
            input_shape=(
                self.config.data.image_height,
                self.config.data.image_width,
                self.config.data.image_channels,
            ),
            weights="imagenet",
            freeze_base=True,
        )
        return self.model, self.base_model

    def get_stage1_callbacks(self) -> list:
        """Construct Stage 1 callbacks (EarlyStopping)."""
        patience = self.config.training.stage1.early_stopping_patience
        return [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        ]

    def get_stage2_callbacks(self, checkpoint_path: Optional[Path] = None) -> list:
        """Construct Stage 2 callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)."""
        save_path = str(checkpoint_path or self.best_checkpoint_path)
        t2 = self.config.training.stage2

        return [
            EarlyStopping(
                monitor="val_loss",
                patience=t2.early_stopping_patience,
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                filepath=save_path,
                monitor="val_loss",
                save_best_only=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=t2.reduce_lr_factor,
                patience=t2.reduce_lr_patience,
                min_lr=t2.reduce_lr_min,
            ),
        ]

    def compile_stage1(self) -> None:
        """Compile model for Stage 1 (Adam lr=1e-4, categorical_crossentropy)."""
        lr = self.config.training.stage1.learning_rate
        logger.info(f"Compiling Stage 1: Adam(lr={lr}), loss='categorical_crossentropy'")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def compile_stage2(self) -> None:
        """Unfreeze top 30 layers and recompile for Stage 2 (Adam lr=1e-5)."""
        num_unfreeze = self.config.training.stage2.unfreeze_last_n_layers
        lr = self.config.training.stage2.learning_rate

        logger.info(f"Unfreezing top {num_unfreeze} layers of MobileNetV2 base model...")
        unfreeze_top_layers(self.base_model, num_layers=num_unfreeze)

        logger.info(f"Compiling Stage 2: Adam(lr={lr}), loss='categorical_crossentropy'")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train(
        self,
        train_gen=None,
        val_gen=None,
        stage1_epochs: Optional[int] = None,
        stage2_epochs: Optional[int] = None,
        best_checkpoint_path: Optional[Path] = None,
        stage2_save_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Execute the complete two-stage training sequence.

        Args:
            train_gen: Training data generator. If None, created via create_generators.
            val_gen: Validation data generator. If None, created via create_generators.
            stage1_epochs: Override Stage 1 epoch count (defaults to config: 10).
            stage2_epochs: Override Stage 2 epoch count (defaults to config: 20).
            best_checkpoint_path: Custom path for saving best checkpoint.
            stage2_save_path: Custom path for saving final stage2 weights.

        Returns:
            Dict[str, Any]: Dictionary containing 'history_stage1' and 'history_stage2'.
        """
        # 1. Setup generators if not provided
        if train_gen is None or val_gen is None:
            logger.info("Initializing data generators from configuration...")
            train_gen, val_gen = create_generators(self.config)

        # 2. Compute balanced class weights
        self.class_weights = compute_balanced_class_weights(train_gen.classes)
        logger.info(f"Computed balanced class weights: {self.class_weights}")

        # 3. Build architecture
        if self.model is None:
            self.setup_model()

        # 4. Stage 1: Feature Extraction
        s1_epochs = stage1_epochs or self.config.training.stage1.epochs
        self.compile_stage1()
        callbacks_s1 = self.get_stage1_callbacks()

        logger.info(f"Starting Stage 1 training ({s1_epochs} epochs)...")
        history_stage1 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=s1_epochs,
            callbacks=callbacks_s1,
            class_weight=self.class_weights,
        )

        # 5. Stage 2: Fine-Tuning
        s2_epochs = stage2_epochs or self.config.training.stage2.epochs
        ckpt_path = best_checkpoint_path or self.best_checkpoint_path
        self.compile_stage2()
        callbacks_s2 = self.get_stage2_callbacks(checkpoint_path=ckpt_path)

        logger.info(f"Starting Stage 2 fine-tuning ({s2_epochs} epochs)...")
        history_stage2 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=s2_epochs,
            callbacks=callbacks_s2,
            class_weight=self.class_weights,
        )

        # 6. Save final stage 2 model
        final_save = stage2_save_path or self.stage2_save_path
        logger.info(f"Saving final Stage 2 model state to {final_save}")
        self.model.save(str(final_save))

        return {
            "history_stage1": history_stage1.history if hasattr(history_stage1, "history") else {},
            "history_stage2": history_stage2.history if hasattr(history_stage2, "history") else {},
        }
