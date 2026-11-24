"""Model evaluation module for vehicle classification.

Extracts the evaluation workflow from notebook 04 into a reusable component:
- Computes predictions on validation data
- Calculates accuracy, classification report, and confusion matrix
- Saves confusion matrix heatmap visualizations (using Agg backend)
- Persists structured evaluation metrics as JSON
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf

# Use non-interactive backend for server/CLI compatibility
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import AppConfig, load_config
from src.data.data_loader import create_generators
from src.utils.logger import get_logger
from src.utils.path_helpers import resolve_path

logger = get_logger("vehicle_evaluator")


class VehicleEvaluator:
    """Evaluates trained vehicle classification models against validation data."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        class_names: Optional[List[str]] = None,
    ):
        """Initialize evaluator with configuration and class mapping.

        Args:
            config: Optional AppConfig instance. If None, loaded via load_config().
            class_names: Optional list of class names in index order.
                         Defaults to configured data.class_names.
        """
        self.config = config or load_config()
        self.class_names = class_names or list(self.config.classifier.class_names)

        self.last_results: Optional[Dict[str, Any]] = None

    def evaluate(
        self,
        val_gen=None,
        model: Optional[tf.keras.Model] = None,
        model_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Run evaluation on validation generator and compute full metric suite.

        Preserves exact metrics computation and structure from notebook 04:
        - y_true extracted from val_generator.classes
        - y_pred computed via argmax of model predictions
        - confusion_matrix and classification_report (output_dict=True)

        Args:
            val_gen: Validation data generator. If None, created via create_generators.
            model: Optional pre-loaded tf.keras.Model.
            model_path: Optional path to .keras model file. If model is None,
                        loads from model_path or config.classifier.model_path.

        Returns:
            Dict[str, Any]: Dictionary containing 'accuracy', 'classification_report',
                            'confusion_matrix', 'y_true', and 'y_pred'.
        """
        # 1. Setup validation generator if not provided
        if val_gen is None:
            logger.info("Initializing validation generator from configuration...")
            _, val_gen = create_generators(self.config)

        # 2. Setup model
        if model is None:
            resolved_model_path = resolve_path(
                model_path or self.config.classifier.model_path
            )
            logger.info(f"Loading model from {resolved_model_path} for evaluation...")
            model = tf.keras.models.load_model(str(resolved_model_path), compile=False)

        # 3. Ground truth
        y_true = np.array(val_gen.classes)

        # 4. Predict
        logger.info(f"Running inference on {len(y_true)} validation samples...")
        val_gen.reset()
        y_pred_probs = model.predict(val_gen)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # 5. Compute metrics
        acc = float(accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
        )

        logger.info(f"Evaluation complete. Validation Accuracy: {acc:.4f}")

        self.last_results = {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm,
            "y_true": y_true,
            "y_pred": y_pred,
        }

        return self.last_results

    def save_confusion_matrix_plot(
        self,
        output_path: Union[str, Path],
        confusion_mat: Optional[np.ndarray] = None,
    ) -> Path:
        """Plot and save confusion matrix heatmap.

        Preserves exact styling and labels from notebook 04 cell 18:
        - 8x6 figure
        - seaborn heatmap with annot=True, fmt='d', cmap='Blues'
        - X/Y tick labels from class mapping

        Args:
            output_path: Destination path for image (.png, .jpg, etc.).
            confusion_mat: Optional confusion matrix. If None, uses last evaluation result.

        Returns:
            Path: Resolved output image path.
        """
        if confusion_mat is None:
            if self.last_results is None or "confusion_matrix" not in self.last_results:
                raise ValueError("No confusion matrix available. Call evaluate() first.")
            confusion_mat = self.last_results["confusion_matrix"]

        save_path = resolve_path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            confusion_mat,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix – Vehicle Classification")
        fig.tight_layout()

        fig.savefig(str(save_path), dpi=300)
        plt.close(fig)
        logger.info(f"Saved confusion matrix plot to {save_path}")
        return save_path

    def save_metrics_json(
        self,
        output_path: Union[str, Path],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save evaluation metrics summary to JSON.

        Args:
            output_path: Destination path for .json file.
            metrics: Optional metrics dictionary. If None, uses last evaluation result.

        Returns:
            Path: Resolved output JSON path.
        """
        if metrics is None:
            if self.last_results is None:
                raise ValueError("No evaluation results available. Call evaluate() first.")
            metrics = self.last_results

        save_path = resolve_path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare JSON-serializable dictionary
        data_to_save = {
            "accuracy": float(metrics["accuracy"]),
            "classification_report": metrics["classification_report"],
            "confusion_matrix": (
                metrics["confusion_matrix"].tolist()
                if isinstance(metrics["confusion_matrix"], np.ndarray)
                else metrics["confusion_matrix"]
            ),
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2)

        logger.info(f"Saved metrics JSON to {save_path}")
        return save_path
