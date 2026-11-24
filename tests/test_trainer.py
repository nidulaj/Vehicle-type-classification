"""Unit tests for VehicleTrainer and training utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import tensorflow as tf

from src.config.schema import (
    AppConfig,
    ClassifierConfig,
    DataConfig,
    DetectorConfig,
    InferenceConfig,
    LoggingConfig,
    PathsConfig,
    TrainingConfig,
    TrainingStage1Config,
    TrainingStage2Config,
)
from src.training.trainer import (
    VehicleTrainer,
    compute_balanced_class_weights,
)


@pytest.fixture
def mock_config(tmp_path: Path) -> AppConfig:
    """Create a test AppConfig with temporary model directory."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return AppConfig(
        paths=PathsConfig(
            data_dir="data",
            models_dir=str(models_dir),
            logs_dir="logs",
            configs_dir="configs",
            notebooks_dir="notebooks",
        ),
        data=DataConfig(
            image_height=224,
            image_width=224,
            image_channels=3,
            batch_size=16,
            val_split=0.2,
            random_seed=42,
            class_names=["Bus", "Car", "Three Wheeler", "Truck", "motorcycle"],
        ),
        detector=DetectorConfig(),
        classifier=ClassifierConfig(
            model_path=str(models_dir / "vehicle_classifier_best.keras"),
            input_height=224,
            input_width=224,
            num_classes=5,
            confidence_threshold=0.5,
            class_mapping={
                "0": "Bus",
                "1": "Car",
                "2": "Three Wheeler",
                "3": "Truck",
                "4": "motorcycle",
            },
        ),
        training=TrainingConfig(
            stage1=TrainingStage1Config(
                epochs=2,
                learning_rate=1e-4,
                early_stopping_patience=3,
            ),
            stage2=TrainingStage2Config(
                epochs=4,
                learning_rate=1e-5,
                unfreeze_last_n_layers=10,
                early_stopping_patience=3,
                reduce_lr_factor=0.3,
                reduce_lr_patience=2,
                reduce_lr_min=1e-6,
            ),
        ),
        inference=InferenceConfig(),
        logging=LoggingConfig(),
    )


def test_compute_balanced_class_weights():
    """Verify class weight calculation preserves sklearn balanced weighting."""
    # 5 classes with imbalanced frequencies
    classes = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 4])
    weights = compute_balanced_class_weights(classes)

    assert isinstance(weights, dict)
    assert len(weights) == 5
    for c in range(5):
        assert c in weights
        assert weights[c] > 0.0

    # Class 3 has 1 sample (least frequent), so it must have the highest weight
    # Class 2 has 4 samples (most frequent), so it must have the lowest weight
    assert weights[3] > weights[2]


def test_trainer_initialization(mock_config: AppConfig, tmp_path: Path):
    """Verify trainer initializes paths and properties from config."""
    trainer = VehicleTrainer(config=mock_config)

    assert trainer.config == mock_config
    assert trainer.models_dir.exists()
    assert trainer.best_checkpoint_path.name == "vehicle_classifier_best.keras"
    assert trainer.stage2_save_path.name == "vehicle_mobilenetv2_stage2.keras"
    assert trainer.model is None
    assert trainer.base_model is None


def test_get_callbacks(mock_config: AppConfig):
    """Verify callback configuration for Stage 1 and Stage 2."""
    trainer = VehicleTrainer(config=mock_config)

    callbacks_s1 = trainer.get_stage1_callbacks()
    assert len(callbacks_s1) == 1
    assert isinstance(callbacks_s1[0], tf.keras.callbacks.EarlyStopping)
    assert callbacks_s1[0].patience == 3
    assert callbacks_s1[0].restore_best_weights is True

    callbacks_s2 = trainer.get_stage2_callbacks()
    assert len(callbacks_s2) == 3
    cb_types = [type(cb) for cb in callbacks_s2]
    assert tf.keras.callbacks.EarlyStopping in cb_types
    assert tf.keras.callbacks.ModelCheckpoint in cb_types
    assert tf.keras.callbacks.ReduceLROnPlateau in cb_types


def test_setup_and_compilation(mock_config: AppConfig):
    """Verify model setup and two-stage compilation using lightweight layers."""
    trainer = VehicleTrainer(config=mock_config)

    # Use a dummy base model and full model to test compile methods without downloading ImageNet weights
    base_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(8, (3, 3), name="conv1"),
        tf.keras.layers.Conv2D(8, (3, 3), name="conv2"),
    ])
    full_model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(5, activation="softmax"),
    ])

    trainer.model = full_model
    trainer.base_model = base_model

    # Stage 1 compile
    trainer.compile_stage1()
    assert trainer.model.optimizer is not None
    # lr is float32 variable in modern Keras
    np.testing.assert_allclose(float(trainer.model.optimizer.learning_rate.numpy()), 1e-4)

    # Stage 2 compile
    trainer.compile_stage2()
    np.testing.assert_allclose(float(trainer.model.optimizer.learning_rate.numpy()), 1e-5)


def test_train_mocked_execution(mock_config: AppConfig, tmp_path: Path):
    """Verify complete two-stage training sequence using mocked generator and fit."""
    trainer = VehicleTrainer(config=mock_config)

    # Mock generators
    mock_train_gen = MagicMock()
    mock_train_gen.classes = np.array([0, 1, 2, 3, 4] * 10)
    mock_val_gen = MagicMock()

    # Mock model
    mock_model = MagicMock(spec=tf.keras.Model)
    mock_history_s1 = MagicMock()
    mock_history_s1.history = {"loss": [0.5, 0.4], "val_loss": [0.6, 0.5]}
    mock_history_s2 = MagicMock()
    mock_history_s2.history = {"loss": [0.3, 0.2], "val_loss": [0.4, 0.3]}

    mock_model.fit.side_effect = [mock_history_s1, mock_history_s2]
    trainer.model = mock_model
    trainer.base_model = MagicMock(spec=tf.keras.Model)

    custom_ckpt = tmp_path / "custom_best.keras"
    custom_stage2 = tmp_path / "custom_stage2.keras"

    result = trainer.train(
        train_gen=mock_train_gen,
        val_gen=mock_val_gen,
        stage1_epochs=2,
        stage2_epochs=3,
        best_checkpoint_path=custom_ckpt,
        stage2_save_path=custom_stage2,
    )

    # Verify fit was called twice (Stage 1 and Stage 2)
    assert mock_model.fit.call_count == 2
    # Verify save was called for stage 2 final weights
    mock_model.save.assert_called_once_with(str(custom_stage2))

    # Verify result structure
    assert "history_stage1" in result
    assert "history_stage2" in result
    assert result["history_stage1"]["loss"] == [0.5, 0.4]
    assert result["history_stage2"]["val_loss"] == [0.4, 0.3]
