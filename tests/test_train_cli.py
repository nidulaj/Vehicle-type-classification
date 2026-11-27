"""Unit tests for training CLI script and MLflow integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import mlflow

from scripts.train import extract_training_params, main, parse_args
from src.config.schema import AppConfig, TrackingConfig


@pytest.fixture
def test_tracking_config(tmp_path):
    """Provides a temporary SQLite tracking config."""
    db_file = tmp_path / "test_train_cli.db"
    return TrackingConfig(
        enabled=True,
        experiment_name="train_cli_experiment",
        tracking_uri=f"sqlite:///{db_file.as_posix()}",
        log_models=True,
    )


def test_parse_args_defaults_and_flags():
    """Verify parse_args handles default and custom arguments."""
    args = parse_args([])
    assert args.config is None
    assert args.stage1_epochs is None
    assert args.stage2_epochs is None
    assert args.checkpoint_dir is None
    assert args.dry_run is False
    assert args.no_tracking is False
    assert args.run_name is None

    args_custom = parse_args([
        "--stage1-epochs", "3",
        "--stage2-epochs", "7",
        "--checkpoint-dir", "custom_models",
        "--dry-run",
        "--no-tracking",
        "--run-name", "my_custom_run",
    ])
    assert args_custom.stage1_epochs == 3
    assert args_custom.stage2_epochs == 7
    assert args_custom.checkpoint_dir == "custom_models"
    assert args_custom.dry_run is True
    assert args_custom.no_tracking is True
    assert args_custom.run_name == "my_custom_run"


def test_extract_training_params():
    """Verify parameter extraction captures all critical hyperparameters."""
    config = AppConfig()
    params = extract_training_params(config=config, stage1_epochs=5, stage2_epochs=12)

    assert params["model_architecture"] == "MobileNetV2"
    assert params["num_classes"] == 5
    assert params["stage1_epochs"] == 5
    assert params["stage2_epochs"] == 12
    assert params["image_height"] == 224
    assert params["batch_size"] == 32
    assert params["stage1_learning_rate"] == 1e-4
    assert params["stage2_learning_rate"] == 1e-5


@patch("scripts.train.VehicleTrainer")
def test_train_cli_dry_run_no_tracking(mock_trainer_cls, test_tracking_config):
    """Verify --dry-run exits without training and without creating any MLflow runs."""
    mock_trainer = MagicMock()
    mock_model = MagicMock()
    mock_model.name = "mock_mobilenetv2"
    mock_model.layers = [1, 2, 3]
    mock_base = MagicMock()
    mock_base.name = "mock_base"
    mock_base.layers = [1, 2]
    mock_trainer.setup_model.return_value = (mock_model, mock_base)
    mock_trainer_cls.return_value = mock_trainer

    with patch("scripts.train.load_config") as mock_load_cfg:
        mock_cfg = AppConfig(tracking=test_tracking_config)
        mock_load_cfg.return_value = mock_cfg

        exit_code = main(["--dry-run"])

        assert exit_code == 0
        mock_trainer.setup_model.assert_called_once()
        mock_trainer.compile_stage1.assert_called_once()
        mock_trainer.compile_stage2.assert_called_once()
        mock_trainer.train.assert_not_called()

        # No experiment or run should be created in the tracking DB
        mlflow.set_tracking_uri(test_tracking_config.tracking_uri)
        exp = mlflow.get_experiment_by_name(test_tracking_config.experiment_name)
        assert exp is None


@patch("scripts.train.VehicleTrainer")
def test_train_cli_with_tracking_success(mock_trainer_cls, test_tracking_config, tmp_path):
    """Verify successful training run logs params, metrics, artifacts, and finishes cleanly."""
    mock_trainer = MagicMock()
    dummy_checkpoint = tmp_path / "vehicle_classifier_best.keras"
    dummy_checkpoint.write_text("model_bytes", encoding="utf-8")
    mock_trainer.best_checkpoint_path = dummy_checkpoint
    mock_trainer.stage2_save_path = tmp_path / "vehicle_mobilenetv2_stage2.keras"
    mock_trainer.class_weights = {0: 1.5, 1: 0.8}

    mock_trainer.train.return_value = {
        "history_stage1": {"loss": [0.5, 0.4], "accuracy": [0.8, 0.9]},
        "history_stage2": {"loss": [0.3, 0.2], "accuracy": [0.92, 0.96], "lr": [1e-5, 3e-6]},
    }
    mock_trainer_cls.return_value = mock_trainer

    with patch("scripts.train.load_config") as mock_load_cfg:
        mock_cfg = AppConfig(tracking=test_tracking_config)
        mock_load_cfg.return_value = mock_cfg

        exit_code = main(["--run-name", "successful_run_test"])

        assert exit_code == 0
        mock_trainer.train.assert_called_once()

        # Verify run in MLflow database
        mlflow.set_tracking_uri(test_tracking_config.tracking_uri)
        exp = mlflow.get_experiment_by_name(test_tracking_config.experiment_name)
        assert exp is not None

        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        assert len(runs) == 1
        run = runs[0]
        assert run.info.status == "FINISHED"
        assert run.data.params["model_architecture"] == "MobileNetV2"
        assert "{0: 1.5, 1: 0.8}" in run.data.params["class_weights"]

        # Check metrics
        s1_loss_history = client.get_metric_history(run.info.run_id, "stage1_loss")
        assert len(s1_loss_history) == 2
        s2_lr_history = client.get_metric_history(run.info.run_id, "stage2_lr")
        assert len(s2_lr_history) == 2

        # Check model checkpoint artifact
        artifacts = client.list_artifacts(run.info.run_id, path="model_checkpoint")
        assert any("vehicle_classifier_best.keras" in a.path for a in artifacts)


@patch("scripts.train.VehicleTrainer")
def test_train_cli_no_tracking_flag(mock_trainer_cls, test_tracking_config, tmp_path):
    """Verify --no-tracking flag suppresses MLflow logging even when config enabled=True."""
    mock_trainer = MagicMock()
    mock_trainer.best_checkpoint_path = tmp_path / "best.keras"
    mock_trainer.stage2_save_path = tmp_path / "stage2.keras"
    mock_trainer.class_weights = None
    mock_trainer.train.return_value = {"history_stage1": {}, "history_stage2": {}}
    mock_trainer_cls.return_value = mock_trainer

    with patch("scripts.train.load_config") as mock_load_cfg:
        mock_cfg = AppConfig(tracking=test_tracking_config)
        mock_load_cfg.return_value = mock_cfg

        exit_code = main(["--no-tracking"])

        assert exit_code == 0
        mock_trainer.train.assert_called_once()

        # No experiment should exist
        mlflow.set_tracking_uri(test_tracking_config.tracking_uri)
        exp = mlflow.get_experiment_by_name(test_tracking_config.experiment_name)
        assert exp is None


@patch("scripts.train.VehicleTrainer")
def test_train_cli_failure_marks_run_failed(mock_trainer_cls, test_tracking_config):
    """Verify that unhandled exception during training marks MLflow run as FAILED."""
    mock_trainer = MagicMock()
    mock_trainer.train.side_effect = RuntimeError("Simulated OOM or training error")
    mock_trainer_cls.return_value = mock_trainer

    with patch("scripts.train.load_config") as mock_load_cfg:
        mock_cfg = AppConfig(tracking=test_tracking_config)
        mock_load_cfg.return_value = mock_cfg

        with pytest.raises(RuntimeError, match="Simulated OOM or training error"):
            main(["--run-name", "failing_run_test"])

        mlflow.set_tracking_uri(test_tracking_config.tracking_uri)
        exp = mlflow.get_experiment_by_name(test_tracking_config.experiment_name)
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        assert len(runs) == 1
        assert runs[0].info.status == "FAILED"
