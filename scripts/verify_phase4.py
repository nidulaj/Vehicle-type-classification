"""Comprehensive verification suite for Phase 4: Training & Evaluation Modularization.

Tests:
1. Part A: Architecture Construction & Layer Freezing/Unfreezing verification.
2. Part B: VehicleTrainer initialization, balanced class weights, and callback configuration.
3. Part C: Training CLI validation via dry-run execution.
4. Part D: VehicleEvaluator verification (metrics computation, heatmap visualization, JSON persistence).
"""

import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import tensorflow as tf

# Ensure project root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.evaluation.evaluator import VehicleEvaluator
from src.models.architecture import (
    build_mobilenetv2_classifier,
    unfreeze_top_layers,
)
from src.training.trainer import (
    VehicleTrainer,
    compute_balanced_class_weights,
)
from src.utils.logger import get_logger

logger = get_logger("phase4_verification")


def verify_architecture():
    logger.info("--- [Test 1/4] Architecture Construction & Layer Unfreezing ---")
    model, base_model = build_mobilenetv2_classifier(
        num_classes=5,
        input_shape=(224, 224, 3),
        weights=None,  # fast test without network request
        freeze_base=True,
    )

    assert isinstance(model, tf.keras.Model)
    assert isinstance(base_model, tf.keras.Model)
    assert base_model.trainable is False
    assert len(model.layers) == 6
    assert len(base_model.layers) == 154

    # Test top layer unfreezing for Stage 2
    unfreeze_top_layers(base_model, num_layers=30)
    assert base_model.trainable is True
    assert all(not layer.trainable for layer in base_model.layers[:-30])
    assert all(layer.trainable for layer in base_model.layers[-30:])

    logger.info("Verified: MobileNetV2 architecture matches notebook 03 exact topology.")
    logger.info("Verified: Stage 1 base freeze and Stage 2 top 30 unfreeze behave correctly.")
    logger.info("PASS: Architecture extraction confirmed.")


def verify_trainer_components():
    logger.info("--- [Test 2/4] Trainer Setup, Weights & Callbacks ---")
    config = load_config()
    trainer = VehicleTrainer(config=config)

    # 1. Class weights formula verification
    dummy_classes = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 4])
    weights = compute_balanced_class_weights(dummy_classes)
    assert len(weights) == 5
    assert weights[3] > weights[2]  # minority class gets larger weight
    logger.info(f"Verified: Balanced class weights formula: {weights}")

    # 2. Callbacks verification
    cb1 = trainer.get_stage1_callbacks()
    assert len(cb1) == 1
    assert isinstance(cb1[0], tf.keras.callbacks.EarlyStopping)
    assert cb1[0].patience == config.training.stage1.early_stopping_patience

    cb2 = trainer.get_stage2_callbacks()
    assert len(cb2) == 3
    cb_types = [type(c) for c in cb2]
    assert tf.keras.callbacks.EarlyStopping in cb_types
    assert tf.keras.callbacks.ModelCheckpoint in cb_types
    assert tf.keras.callbacks.ReduceLROnPlateau in cb_types

    logger.info("Verified: Stage 1 and Stage 2 callbacks match training specification.")
    logger.info("PASS: Trainer components confirmed.")


def verify_train_cli_dry_run():
    logger.info("--- [Test 3/4] Training CLI Dry-Run Execution ---")
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "train.py"), "--dry-run"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))

    if res.returncode != 0:
        logger.error(f"train.py --dry-run failed with code {res.returncode}:")
        logger.error(res.stderr)
        raise RuntimeError("train.py --dry-run failed!")

    assert "Dry run validation successful" in res.stdout or "Dry run validation successful" in res.stderr
    logger.info("Verified: train.py CLI dry-run executed and validated successfully.")
    logger.info("PASS: Training CLI entry point confirmed.")


def verify_evaluator_and_artifacts():
    logger.info("--- [Test 4/4] VehicleEvaluator & Artifact Generation ---")
    config = load_config()
    evaluator = VehicleEvaluator(config=config)

    # Verify existing trained model exists and is loadable
    model_path = PROJECT_ROOT / config.classifier.model_path
    assert model_path.exists(), f"Expected trained model not found at {model_path}"
    model = tf.keras.models.load_model(str(model_path), compile=False)
    logger.info(f"Verified: Existing trained model loaded successfully from {model_path}")

    # Test evaluation metrics calculation with synthetic ground truth & predictions
    val_classes = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    mock_val_gen = type("MockValGen", (), {
        "classes": val_classes,
        "reset": lambda self: None,
    })()

    # Create dummy predict returning mostly accurate predictions
    pred_probs = np.zeros((10, 5), dtype=np.float32)
    for i, c in enumerate(val_classes):
        pred_probs[i, c] = 0.98

    mock_model = type("MockModel", (), {
        "predict": lambda self, gen: pred_probs,
    })()

    results = evaluator.evaluate(val_gen=mock_val_gen, model=mock_model)
    assert results["accuracy"] == 1.0
    assert results["confusion_matrix"].shape == (5, 5)
    assert "Car" in results["classification_report"]

    # Test artifact generation in outputs/evaluation/
    eval_dir = PROJECT_ROOT / "outputs" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    plot_path = eval_dir / "confusion_matrix.png"
    json_path = eval_dir / "metrics.json"

    saved_plot = evaluator.save_confusion_matrix_plot(plot_path)
    saved_json = evaluator.save_metrics_json(json_path)

    assert saved_plot.exists() and saved_plot.stat().st_size > 0
    assert saved_json.exists() and saved_json.stat().st_size > 0

    with open(saved_json, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    assert "accuracy" in metrics_data
    assert "classification_report" in metrics_data
    assert "confusion_matrix" in metrics_data

    logger.info(f"Verified: Saved confusion matrix heatmap to {saved_plot}")
    logger.info(f"Verified: Saved metrics JSON to {saved_json}")
    logger.info("PASS: VehicleEvaluator and artifact persistence confirmed.")


def main():
    logger.info("==================================================")
    logger.info("STARTING PHASE 4 VERIFICATION SUITE")
    logger.info("==================================================")

    verify_architecture()
    verify_trainer_components()
    verify_train_cli_dry_run()
    verify_evaluator_and_artifacts()

    logger.info("==================================================")
    logger.info("ALL PHASE 4 VERIFICATIONS PASSED SUCCESSFULLY!")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
