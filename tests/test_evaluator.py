"""Unit tests for VehicleEvaluator and evaluation utilities."""

import json
from pathlib import Path
from unittest.mock import MagicMock
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
from src.evaluation.evaluator import VehicleEvaluator


@pytest.fixture
def test_config(tmp_path: Path) -> AppConfig:
    """Create test AppConfig."""
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
            validation_split=0.2,
        ),
        detector=DetectorConfig(),
        classifier=ClassifierConfig(
            model_path=str(models_dir / "vehicle_classifier_best.keras"),
            input_height=224,
            input_width=224,
            num_classes=5,
            class_names=["Bus", "Car", "Three Wheeler", "Truck", "motorcycle"],
        ),
        training=TrainingConfig(
            stage1=TrainingStage1Config(epochs=2),
            stage2=TrainingStage2Config(epochs=4),
        ),
        inference=InferenceConfig(),
        logging=LoggingConfig(),
    )


def test_evaluator_initialization(test_config: AppConfig):
    """Verify evaluator initializes with configured class names."""
    evaluator = VehicleEvaluator(config=test_config)
    assert evaluator.class_names == ["Bus", "Car", "Three Wheeler", "Truck", "motorcycle"]
    assert evaluator.last_results is None


def test_evaluate_with_mock_model(test_config: AppConfig):
    """Verify metrics computation against known predictions and ground truth."""
    evaluator = VehicleEvaluator(config=test_config)

    # 10 samples: 2 of each class
    # Classes: [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    mock_val_gen = MagicMock()
    mock_val_gen.classes = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

    # Predict: 9 correct, 1 mistake (sample 3 predicted as class 0 instead of 1)
    pred_classes = [0, 0, 1, 0, 2, 2, 3, 3, 4, 4]
    pred_probs = np.zeros((10, 5), dtype=np.float32)
    for i, c in enumerate(pred_classes):
        pred_probs[i, c] = 0.95

    mock_model = MagicMock(spec=tf.keras.Model)
    mock_model.predict.return_value = pred_probs

    results = evaluator.evaluate(val_gen=mock_val_gen, model=mock_model)

    assert "accuracy" in results
    assert "classification_report" in results
    assert "confusion_matrix" in results
    assert "y_true" in results
    assert "y_pred" in results

    # 9 / 10 = 0.90 accuracy
    assert results["accuracy"] == pytest.approx(0.90)

    # Confusion matrix shape (5, 5)
    cm = results["confusion_matrix"]
    assert cm.shape == (5, 5)
    assert cm[0, 0] == 2  # class 0: 2 correct
    assert cm[1, 1] == 1  # class 1: 1 correct
    assert cm[1, 0] == 1  # class 1: 1 predicted as 0

    # Classification report includes all classes
    report = results["classification_report"]
    for cname in ["Bus", "Car", "Three Wheeler", "Truck", "motorcycle"]:
        assert cname in report
        assert "precision" in report[cname]
        assert "recall" in report[cname]
        assert "f1-score" in report[cname]


def test_save_plots_and_metrics(test_config: AppConfig, tmp_path: Path):
    """Verify saving confusion matrix plot and metrics JSON."""
    evaluator = VehicleEvaluator(config=test_config)

    # Synthetic results
    cm = np.array([
        [10, 1, 0, 0, 0],
        [0, 15, 0, 0, 0],
        [0, 0, 12, 1, 0],
        [0, 1, 0, 8, 0],
        [0, 0, 0, 0, 14],
    ])
    evaluator.last_results = {
        "accuracy": 0.9667,
        "classification_report": {
            "Bus": {"precision": 1.0, "recall": 0.91, "f1-score": 0.95, "support": 11},
            "accuracy": 0.9667,
        },
        "confusion_matrix": cm,
        "y_true": np.array([0, 1]),
        "y_pred": np.array([0, 1]),
    }

    # Test save confusion matrix plot
    plot_path = tmp_path / "reports" / "confusion_matrix.png"
    saved_plot = evaluator.save_confusion_matrix_plot(plot_path)
    assert saved_plot.exists()
    assert saved_plot.stat().st_size > 0

    # Test save metrics JSON
    json_path = tmp_path / "reports" / "metrics.json"
    saved_json = evaluator.save_metrics_json(json_path)
    assert saved_json.exists()

    with open(saved_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["accuracy"] == pytest.approx(0.9667)
    assert "confusion_matrix" in loaded
    assert len(loaded["confusion_matrix"]) == 5
