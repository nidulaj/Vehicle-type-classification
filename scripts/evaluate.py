#!/usr/bin/env python
"""CLI entry point for vehicle classifier evaluation.

Evaluates a trained MobileNetV2 classification model against validation data,
generating accuracy, a classification report, and a confusion matrix heatmap.

Usage:
    python scripts/evaluate.py [--config configs/config.yaml]
    python scripts/evaluate.py --model-path models/vehicle_classifier_best.keras --output-dir outputs/evaluation
"""

import argparse
import sys
from pathlib import Path
from sklearn.metrics import classification_report

# Add project root to sys.path so scripts can run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.evaluation.evaluator import VehicleEvaluator
from src.utils.logger import get_logger
from src.utils.path_helpers import resolve_path

logger = get_logger("evaluate_cli")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained vehicle classification model on validation dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (defaults to configs/config.yaml).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to .keras model file to evaluate (defaults to config classifier.model_path).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation reports and plots (default: outputs/evaluation).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing vehicle classification evaluation CLI...")

    config = load_config(args.config) if args.config else load_config()
    evaluator = VehicleEvaluator(config=config)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path or config.classifier.model_path
    logger.info(f"Evaluating model: {model_path}")
    logger.info(f"Outputs will be saved to: {output_dir}")

    results = evaluator.evaluate(model_path=model_path)

    # Print summary report to console
    print("\n" + "=" * 65)
    print("           VEHICLE CLASSIFICATION EVALUATION REPORT")
    print("=" * 65)
    print(f"Overall Accuracy: {results['accuracy']:.4f}\n")
    print(classification_report(
        results["y_true"],
        results["y_pred"],
        target_names=evaluator.class_names,
    ))
    print("=" * 65 + "\n")

    # Save artifacts
    plot_path = output_dir / "confusion_matrix.png"
    json_path = output_dir / "metrics.json"

    evaluator.save_confusion_matrix_plot(plot_path)
    evaluator.save_metrics_json(json_path)

    logger.info(f"Evaluation artifacts saved successfully:")
    logger.info(f"  - Confusion Matrix Heatmap: {plot_path}")
    logger.info(f"  - Detailed Metrics (JSON):  {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
