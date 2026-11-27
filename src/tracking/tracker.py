"""MLflow experiment tracking orchestrator for model training and evaluation."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import mlflow

from src.config.schema import AppConfig, TrackingConfig, load_config
from src.utils.logger import get_logger
from src.utils.path_helpers import resolve_path

logger = get_logger("mlflow_tracker")


class MLflowTracker:
    """Manages experiment tracking lifecycle, metrics, parameters, and artifacts with MLflow."""

    def __init__(
        self,
        config: Optional[Union[TrackingConfig, AppConfig]] = None,
    ):
        """Initialize MLflowTracker.

        Args:
            config: Optional TrackingConfig or AppConfig instance. If None,
                    loaded via load_config().
        """
        if config is None:
            app_cfg = load_config()
            self.config = app_cfg.tracking
        elif isinstance(config, AppConfig):
            self.config = config.tracking
        elif isinstance(config, TrackingConfig):
            self.config = config
        else:
            raise TypeError(f"Unsupported config type for MLflowTracker: {type(config)}")

        self.active_run: Optional[mlflow.ActiveRun] = None
        self.experiment_id: Optional[str] = None

    @property
    def is_enabled(self) -> bool:
        """Check if experiment tracking is enabled."""
        return bool(self.config.enabled)

    def setup_experiment(self) -> Optional[str]:
        """Configure tracking URI and retrieve or create the target experiment.

        Returns:
            Optional[str]: Experiment ID if tracking enabled, None if disabled.
        """
        if not self.is_enabled:
            logger.info("MLflow tracking is disabled. Skipping experiment setup.")
            return None

        # Allow local filesystem stores if specified
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

        tracking_uri = self.config.tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")

        exp_name = self.config.experiment_name
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(exp_name)
            logger.info(f"Created new MLflow experiment '{exp_name}' with ID: {self.experiment_id}")
        else:
            self.experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment '{exp_name}' with ID: {self.experiment_id}")

        return self.experiment_id

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> Optional[mlflow.ActiveRun]:
        """Start a new MLflow run under the configured experiment.

        Args:
            run_name: Optional name for the run.
            tags: Optional dictionary of tags.

        Returns:
            Optional[mlflow.ActiveRun]: Active run if enabled, None if disabled.
        """
        if not self.is_enabled:
            return None

        if self.experiment_id is None:
            self.setup_experiment()

        if self.active_run is not None:
            logger.warning("Active run already exists. Ending previous run before starting a new one.")
            self.end_run()

        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags,
        )
        logger.info(f"Started MLflow run '{self.active_run.info.run_name}' (ID: {self.active_run.info.run_id})")
        return self.active_run

    def end_run(self, status: str = "FINISHED") -> None:
        """End the active MLflow run safely.

        Args:
            status: Terminal status for the run ('FINISHED', 'FAILED', 'KILLED').
        """
        if not self.is_enabled or self.active_run is None:
            return

        run_id = self.active_run.info.run_id
        mlflow.end_run(status=status)
        logger.info(f"Ended MLflow run {run_id} with status: {status}")
        self.active_run = None

    def __enter__(self) -> "MLflowTracker":
        """Context manager entry."""
        if self.is_enabled and self.active_run is None:
            self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if self.is_enabled and self.active_run is not None:
            status = "FAILED" if exc_type is not None else "FINISHED"
            self.end_run(status=status)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameter dictionary to the active run.

        Args:
            params: Dictionary of parameter names and values.
        """
        if not self.is_enabled:
            return

        if self.active_run is None:
            raise RuntimeError("Cannot log parameters: No active MLflow run. Call start_run() first.")

        safe_params = {
            k: (str(v) if not isinstance(v, (int, float, str, bool)) else v)
            for k, v in params.items()
        }
        mlflow.log_params(safe_params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single scalar metric.

        Args:
            key: Metric name.
            value: Float metric value.
            step: Optional epoch or step index.
        """
        if not self.is_enabled:
            return

        if self.active_run is None:
            raise RuntimeError(f"Cannot log metric '{key}': No active MLflow run. Call start_run() first.")

        mlflow.log_metric(key=key, value=float(value), step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log a dictionary of scalar metrics.

        Args:
            metrics: Dictionary of metric names and float values.
            step: Optional epoch or step index.
        """
        if not self.is_enabled:
            return

        if self.active_run is None:
            raise RuntimeError("Cannot log metrics: No active MLflow run. Call start_run() first.")

        safe_metrics = {k: float(v) for k, v in metrics.items()}
        mlflow.log_metrics(safe_metrics, step=step)

    def log_training_history(
        self,
        history_stage1: Dict[str, Any],
        history_stage2: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log Stage 1 and Stage 2 Keras training history metrics epoch-by-epoch.

        Args:
            history_stage1: Dictionary of metric lists from Stage 1 fit().
            history_stage2: Optional dictionary of metric lists from Stage 2 fit().
        """
        if not self.is_enabled:
            return

        if self.active_run is None:
            raise RuntimeError("Cannot log training history: No active MLflow run. Call start_run() first.")

        # Log Stage 1 metrics
        if history_stage1:
            epoch_counts = [len(v) for v in history_stage1.values() if isinstance(v, list)]
            num_epochs_s1 = max(epoch_counts) if epoch_counts else 0
            for epoch in range(num_epochs_s1):
                epoch_metrics = {}
                for key, values in history_stage1.items():
                    if isinstance(values, list) and epoch < len(values):
                        epoch_metrics[f"stage1_{key}"] = float(values[epoch])
                if epoch_metrics:
                    self.log_metrics(epoch_metrics, step=epoch + 1)

        # Log Stage 2 metrics
        if history_stage2:
            epoch_counts = [len(v) for v in history_stage2.values() if isinstance(v, list)]
            num_epochs_s2 = max(epoch_counts) if epoch_counts else 0
            for epoch in range(num_epochs_s2):
                epoch_metrics = {}
                for key, values in history_stage2.items():
                    if isinstance(values, list) and epoch < len(values):
                        epoch_metrics[f"stage2_{key}"] = float(values[epoch])
                if epoch_metrics:
                    self.log_metrics(epoch_metrics, step=epoch + 1)

    def log_artifact(
        self,
        local_path: Union[str, Path],
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log a local file or directory as an MLflow artifact.

        Args:
            local_path: Path to the local file or directory.
            artifact_path: Optional destination directory inside the artifact repository.
        """
        if not self.is_enabled:
            return

        if self.active_run is None:
            raise RuntimeError(f"Cannot log artifact '{local_path}': No active MLflow run. Call start_run() first.")

        path = resolve_path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact path does not exist: {path}")

        if path.is_file():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
            logger.info(f"Logged artifact file: {path.name} -> {artifact_path or '/'}")
        elif path.is_dir():
            mlflow.log_artifacts(str(path), artifact_path=artifact_path)
            logger.info(f"Logged artifact directory: {path.name} -> {artifact_path or '/'}")

    def log_model_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        artifact_path: str = "model_checkpoint",
    ) -> None:
        """Log an existing .keras checkpoint as a regular artifact.

        Args:
            checkpoint_path: Path to the .keras model checkpoint file.
            artifact_path: Subdirectory in the run's artifact repository.
        """
        if not self.is_enabled:
            return

        if not self.config.log_models:
            logger.info("Model logging disabled in config (log_models=False). Skipping checkpoint logging.")
            return

        path = resolve_path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint file not found: {path}")

        self.log_artifact(local_path=path, artifact_path=artifact_path)
        logger.info(f"Logged model checkpoint artifact: {path.name} -> {artifact_path}")
