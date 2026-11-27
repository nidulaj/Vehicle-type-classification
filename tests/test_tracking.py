"""Unit tests for MLflowTracker module."""

import os
from pathlib import Path
import pytest
import mlflow

from src.config.schema import AppConfig, TrackingConfig
from src.tracking.tracker import MLflowTracker


@pytest.fixture
def temp_tracking_config(tmp_path):
    """Provides a TrackingConfig configured with an isolated temporary SQLite database."""
    db_file = tmp_path / "test_tracking.db"
    return TrackingConfig(
        enabled=True,
        experiment_name="unit_test_experiment",
        tracking_uri=f"sqlite:///{db_file.as_posix()}",
        log_models=True,
    )


def test_tracker_initialization(temp_tracking_config):
    """Verify tracker initialization with TrackingConfig, AppConfig, and invalid types."""
    # TrackingConfig
    tracker1 = MLflowTracker(config=temp_tracking_config)
    assert tracker1.is_enabled is True
    assert tracker1.config.experiment_name == "unit_test_experiment"

    # AppConfig
    app_cfg = AppConfig(tracking=temp_tracking_config)
    tracker2 = MLflowTracker(config=app_cfg)
    assert tracker2.is_enabled is True

    # Invalid type
    with pytest.raises(TypeError):
        MLflowTracker(config="invalid_config_type")


def test_tracker_disabled_behavior():
    """Verify that disabled tracking no-ops gracefully without raising errors."""
    disabled_config = TrackingConfig(enabled=False)
    tracker = MLflowTracker(config=disabled_config)

    assert tracker.is_enabled is False
    assert tracker.setup_experiment() is None
    assert tracker.start_run() is None

    # Operations should cleanly no-op
    tracker.log_params({"param1": "val1"})
    tracker.log_metric("metric1", 0.95)
    tracker.log_metrics({"metric2": 0.90})
    tracker.log_training_history({"loss": [0.5, 0.4]})
    tracker.log_artifact("non_existent_file.txt")
    tracker.log_model_checkpoint("non_existent_model.keras")
    tracker.end_run()


def test_experiment_setup_and_reuse(temp_tracking_config):
    """Verify experiment creation and reusing existing experiment."""
    tracker = MLflowTracker(config=temp_tracking_config)
    exp_id1 = tracker.setup_experiment()
    assert exp_id1 is not None

    # Setup again with same experiment name should reuse existing ID
    exp_id2 = tracker.setup_experiment()
    assert exp_id1 == exp_id2


def test_run_lifecycle_and_context_manager(temp_tracking_config):
    """Verify start_run, end_run, and context manager lifecycle."""
    tracker = MLflowTracker(config=temp_tracking_config)

    # Context manager normal exit
    with tracker as active:
        assert active.active_run is not None
        run_id = active.active_run.info.run_id

    # Active run should be cleared
    assert tracker.active_run is None

    client = mlflow.tracking.MlflowClient()
    run_info = client.get_run(run_id).info
    assert run_info.status == "FINISHED"

    # Context manager failure exit
    try:
        with tracker as active:
            failed_run_id = active.active_run.info.run_id
            raise ValueError("Intentional failure for testing")
    except ValueError:
        pass

    assert tracker.active_run is None
    failed_run_info = client.get_run(failed_run_id).info
    assert failed_run_info.status == "FAILED"


def test_parameter_logging(temp_tracking_config):
    """Verify logging parameters including non-primitive serialization."""
    tracker = MLflowTracker(config=temp_tracking_config)

    with tracker as active:
        active.log_params({
            "learning_rate": 0.001,
            "batch_size": 32,
            "class_weights": {0: 1.2, 1: 0.8},
            "class_names": ["Bus", "Car"],
        })
        run_id = active.active_run.info.run_id

    client = mlflow.tracking.MlflowClient()
    params = client.get_run(run_id).data.params
    assert params["learning_rate"] == "0.001"
    assert params["batch_size"] == "32"
    assert "{0: 1.2, 1: 0.8}" in params["class_weights"]
    assert "['Bus', 'Car']" in params["class_names"]


def test_metric_logging(temp_tracking_config):
    """Verify logging single and multiple metrics."""
    tracker = MLflowTracker(config=temp_tracking_config)

    with tracker as active:
        active.log_metric("accuracy", 0.949, step=1)
        active.log_metrics({"loss": 0.15, "val_loss": 0.22}, step=1)
        run_id = active.active_run.info.run_id

    client = mlflow.tracking.MlflowClient()
    metrics = client.get_run(run_id).data.metrics
    assert abs(metrics["accuracy"] - 0.949) < 1e-4
    assert abs(metrics["loss"] - 0.15) < 1e-4
    assert abs(metrics["val_loss"] - 0.22) < 1e-4


def test_training_history_logging(temp_tracking_config):
    """Verify logging Stage 1 and Stage 2 histories with epoch steps and optional metrics."""
    tracker = MLflowTracker(config=temp_tracking_config)

    history_stage1 = {
        "loss": [0.6, 0.4, 0.3],
        "accuracy": [0.75, 0.85, 0.90],
        "val_loss": [0.55, 0.38, 0.28],
        "val_accuracy": [0.78, 0.88, 0.92],
    }

    history_stage2 = {
        "loss": [0.25, 0.18],
        "accuracy": [0.91, 0.95],
        "val_loss": [0.24, 0.19],
        "val_accuracy": [0.93, 0.96],
        "lr": [1e-5, 3e-6],  # Optional metric
    }

    with tracker as active:
        active.log_training_history(history_stage1, history_stage2)
        run_id = active.active_run.info.run_id

    client = mlflow.tracking.MlflowClient()
    # Check metric history at step 3 for Stage 1
    s1_acc_history = client.get_metric_history(run_id, "stage1_accuracy")
    assert len(s1_acc_history) == 3
    assert s1_acc_history[2].step == 3
    assert abs(s1_acc_history[2].value - 0.90) < 1e-4

    # Check metric history for Stage 2 optional metric (lr)
    s2_lr_history = client.get_metric_history(run_id, "stage2_lr")
    assert len(s2_lr_history) == 2
    assert s2_lr_history[1].step == 2
    assert abs(s2_lr_history[1].value - 3e-6) < 1e-7


def test_artifact_logging(temp_tracking_config, tmp_path):
    """Verify logging single file artifacts and directory artifacts."""
    tracker = MLflowTracker(config=temp_tracking_config)

    # Create dummy artifact files
    artifact_file = tmp_path / "test_report.txt"
    artifact_file.write_text("Test classification report content", encoding="utf-8")

    artifact_dir = tmp_path / "plots"
    artifact_dir.mkdir()
    (artifact_dir / "plot1.png").write_text("fake png bytes", encoding="utf-8")

    with tracker as active:
        active.log_artifact(artifact_file, artifact_path="reports")
        active.log_artifact(artifact_dir, artifact_path="figures")
        run_id = active.active_run.info.run_id

    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    artifact_paths = [a.path for a in artifacts]
    assert "reports" in artifact_paths
    assert "figures" in artifact_paths


def test_artifact_missing_file_raises_error(temp_tracking_config, tmp_path):
    """Verify attempting to log a non-existent artifact raises FileNotFoundError."""
    tracker = MLflowTracker(config=temp_tracking_config)
    missing_file = tmp_path / "missing_file.png"

    with tracker as active:
        with pytest.raises(FileNotFoundError):
            active.log_artifact(missing_file)


def test_model_checkpoint_logging(temp_tracking_config, tmp_path):
    """Verify logging a .keras model checkpoint as an artifact."""
    tracker = MLflowTracker(config=temp_tracking_config)

    dummy_model = tmp_path / "vehicle_classifier_best.keras"
    dummy_model.write_text("dummy keras model weights", encoding="utf-8")

    with tracker as active:
        active.log_model_checkpoint(dummy_model, artifact_path="model_checkpoint")
        run_id = active.active_run.info.run_id

    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id, path="model_checkpoint")
    filenames = [a.path for a in artifacts]
    assert "model_checkpoint/vehicle_classifier_best.keras" in filenames


def test_model_checkpoint_logging_disabled(tmp_path):
    """Verify model checkpoint logging respects log_models=False."""
    db_file = tmp_path / "test_no_models.db"
    config = TrackingConfig(
        enabled=True,
        experiment_name="no_models_exp",
        tracking_uri=f"sqlite:///{db_file.as_posix()}",
        log_models=False,
    )
    tracker = MLflowTracker(config=config)

    dummy_model = tmp_path / "vehicle_classifier_best.keras"
    dummy_model.write_text("dummy keras model weights", encoding="utf-8")

    with tracker as active:
        active.log_model_checkpoint(dummy_model, artifact_path="model_checkpoint")
        run_id = active.active_run.info.run_id

    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    assert len(artifacts) == 0


def test_logging_without_active_run_raises_error(temp_tracking_config):
    """Verify attempting to log without an active run raises RuntimeError."""
    tracker = MLflowTracker(config=temp_tracking_config)

    with pytest.raises(RuntimeError):
        tracker.log_params({"param": "val"})

    with pytest.raises(RuntimeError):
        tracker.log_metric("metric", 1.0)

    with pytest.raises(RuntimeError):
        tracker.log_metrics({"metric": 1.0})

    with pytest.raises(RuntimeError):
        tracker.log_training_history({"loss": [0.1]})

    with pytest.raises(RuntimeError):
        tracker.log_artifact("some_file.txt")
