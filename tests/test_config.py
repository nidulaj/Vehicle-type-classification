"""Unit tests for configuration schema, tracking settings, and backward compatibility."""

import pytest
from pydantic import ValidationError

from src.config.schema import AppConfig, TrackingConfig, load_config


def test_default_tracking_config():
    """Verify default values for TrackingConfig."""
    cfg = TrackingConfig()
    assert cfg.enabled is True
    assert cfg.experiment_name == "vehicle-type-classification"
    assert cfg.tracking_uri == "sqlite:///mlflow.db"
    assert cfg.log_models is True


def test_explicit_tracking_config():
    """Verify custom tracking configuration overrides."""
    cfg = TrackingConfig(
        enabled=False,
        experiment_name="custom-experiment",
        tracking_uri="file:./custom_mlruns",
        log_models=False,
    )
    assert cfg.enabled is False
    assert cfg.experiment_name == "custom-experiment"
    assert cfg.tracking_uri == "file:./custom_mlruns"
    assert cfg.log_models is False


def test_invalid_tracking_config():
    """Verify rejection of invalid tracking configuration parameters."""
    # Empty experiment_name
    with pytest.raises(ValidationError):
        TrackingConfig(experiment_name="")

    # Whitespace-only experiment_name
    with pytest.raises(ValidationError):
        TrackingConfig(experiment_name="   ")

    # Empty tracking_uri
    with pytest.raises(ValidationError):
        TrackingConfig(tracking_uri="")

    # Whitespace-only tracking_uri
    with pytest.raises(ValidationError):
        TrackingConfig(tracking_uri="   ")

    # Invalid boolean for enabled
    with pytest.raises(ValidationError):
        TrackingConfig(enabled="invalid_bool")

    # Invalid boolean for log_models
    with pytest.raises(ValidationError):
        TrackingConfig(log_models="invalid_bool")


def test_backward_compatibility_when_tracking_absent():
    """Verify AppConfig remains fully compatible when tracking section is omitted."""
    raw_config = {
        "project": {"name": "legacy-test"},
        "data": {"image_height": 224, "image_width": 224},
    }
    app_cfg = AppConfig.model_validate(raw_config)
    assert hasattr(app_cfg, "tracking")
    assert isinstance(app_cfg.tracking, TrackingConfig)
    # Defaults must be automatically populated
    assert app_cfg.tracking.enabled is True
    assert app_cfg.tracking.experiment_name == "vehicle-type-classification"
    assert app_cfg.tracking.tracking_uri == "sqlite:///mlflow.db"
    assert app_cfg.tracking.log_models is True


def test_app_config_with_explicit_tracking():
    """Verify AppConfig parses explicit tracking section correctly."""
    raw_config = {
        "tracking": {
            "enabled": True,
            "experiment_name": "vehicle-benchmark",
            "tracking_uri": "sqlite:///test.db",
            "log_models": False,
        }
    }
    app_cfg = AppConfig.model_validate(raw_config)
    assert app_cfg.tracking.enabled is True
    assert app_cfg.tracking.experiment_name == "vehicle-benchmark"
    assert app_cfg.tracking.tracking_uri == "sqlite:///test.db"
    assert app_cfg.tracking.log_models is False


def test_existing_load_config_integration():
    """Verify load_config() loads configs/config.yaml including the tracking section."""
    cfg = load_config()
    assert cfg.tracking.enabled is True
    assert cfg.tracking.experiment_name == "vehicle-type-classification"
    assert cfg.tracking.tracking_uri == "sqlite:///mlflow.db"
    assert cfg.tracking.log_models is True

    # Check that other existing configurations are intact
    assert cfg.project.name == "vehicle-type-classification"
    assert cfg.detector.confidence_threshold == 0.4
    assert cfg.data.image_height == 224
    assert cfg.classifier.num_classes == 5
