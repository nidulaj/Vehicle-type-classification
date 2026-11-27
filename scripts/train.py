#!/usr/bin/env python
"""CLI entry point for two-stage vehicle classifier training.

Executes Stage 1 (feature extraction) and Stage 2 (fine-tuning) transfer learning
using the modular VehicleTrainer, with optional MLflow experiment tracking.

Usage:
    python scripts/train.py [--config configs/config.yaml] [--dry-run]
    python scripts/train.py --stage1-epochs 5 --stage2-epochs 10
    python scripts/train.py --no-tracking
"""

import argparse
from datetime import datetime
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path so scripts can run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, load_config
from src.tracking import MLflowTracker
from src.training.trainer import VehicleTrainer
from src.utils.logger import get_logger

logger = get_logger("train_cli")


def parse_args(args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 vehicle classifier using two-stage transfer learning."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (defaults to configs/config.yaml).",
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=None,
        help="Number of epochs for Stage 1 (frozen base). Overrides config.",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=None,
        help="Number of epochs for Stage 2 (fine-tuning). Overrides config.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints. Overrides config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and construct model architecture without fitting.",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable MLflow experiment tracking for this run.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom name for the MLflow run.",
    )
    return parser.parse_args(args)


def extract_training_params(
    config: AppConfig,
    stage1_epochs: Optional[int],
    stage2_epochs: Optional[int],
) -> dict:
    """Extract full hyperparameter dictionary from config for MLflow tracking."""
    return {
        "model_architecture": "MobileNetV2",
        "num_classes": config.classifier.num_classes,
        "class_names": config.classifier.class_names,
        "image_height": config.data.image_height,
        "image_width": config.data.image_width,
        "image_channels": config.data.image_channels,
        "batch_size": config.data.batch_size,
        "validation_split": config.data.validation_split,
        "stage1_epochs": stage1_epochs or config.training.stage1.epochs,
        "stage1_learning_rate": config.training.stage1.learning_rate,
        "stage1_early_stopping_patience": config.training.stage1.early_stopping_patience,
        "stage2_epochs": stage2_epochs or config.training.stage2.epochs,
        "stage2_learning_rate": config.training.stage2.learning_rate,
        "stage2_unfreeze_last_n_layers": config.training.stage2.unfreeze_last_n_layers,
        "stage2_early_stopping_patience": config.training.stage2.early_stopping_patience,
        "stage2_reduce_lr_patience": config.training.stage2.reduce_lr_patience,
        "stage2_reduce_lr_factor": config.training.stage2.reduce_lr_factor,
        "stage2_reduce_lr_min": config.training.stage2.reduce_lr_min,
        "rotation_range": config.training.augmentations.rotation_range,
        "width_shift_range": config.training.augmentations.width_shift_range,
        "height_shift_range": config.training.augmentations.height_shift_range,
        "shear_range": config.training.augmentations.shear_range,
        "zoom_range": config.training.augmentations.zoom_range,
        "horizontal_flip": config.training.augmentations.horizontal_flip,
        "gaussian_noise_std": config.training.augmentations.gaussian_noise_std,
        "cutout_mask_size": config.training.augmentations.cutout_mask_size,
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logger.info("Initializing vehicle classification training CLI...")

    config = load_config(args.config) if args.config else load_config()

    # Override tracking configuration if --no-tracking flag is passed
    if args.no_tracking:
        config.tracking.enabled = False

    trainer = VehicleTrainer(config=config, checkpoint_dir=args.checkpoint_dir)

    if args.dry_run:
        logger.info("[Dry Run] Setting up model architecture...")
        model, base_model = trainer.setup_model()
        logger.info(f"[Dry Run] Model constructed: {model.name} with {len(model.layers)} layers.")
        logger.info(f"[Dry Run] Base model: {base_model.name} with {len(base_model.layers)} layers.")
        logger.info("[Dry Run] Stage 1 callbacks and compile:")
        trainer.compile_stage1()
        logger.info("[Dry Run] Stage 2 callbacks and compile:")
        trainer.compile_stage2()
        logger.info("[Dry Run] Dry run validation successful! Exiting without training or tracking.")
        return 0

    tracker = MLflowTracker(config=config.tracking)

    if tracker.is_enabled:
        run_name = args.run_name or f"mobilenetv2_two_stage_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Setting up MLflow tracking (experiment: '{config.tracking.experiment_name}', run: '{run_name}')...")
        tracker.setup_experiment()
        tracker.start_run(run_name=run_name)
        params = extract_training_params(
            config=config,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
        )
        tracker.log_params(params)

    try:
        logger.info("Starting complete two-stage training sequence...")
        results = trainer.train(
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
        )

        if tracker.is_enabled:
            # Log balanced class weights if computed by trainer
            if trainer.class_weights:
                tracker.log_params({"class_weights": trainer.class_weights})

            # Log Stage 1 and Stage 2 epoch metrics
            h_s1 = results.get("history_stage1", {})
            h_s2 = results.get("history_stage2", {})
            tracker.log_training_history(h_s1, h_s2)

            # Log best checkpoint as artifact if present on disk
            if trainer.best_checkpoint_path.exists():
                tracker.log_model_checkpoint(trainer.best_checkpoint_path)

            tracker.end_run(status="FINISHED")

        logger.info("Training complete!")
        logger.info(f"Best checkpoint saved to: {trainer.best_checkpoint_path}")
        logger.info(f"Final Stage 2 model saved to: {trainer.stage2_save_path}")
        return 0

    except Exception as e:
        logger.error(f"Training sequence failed: {e}", exc_info=True)
        if tracker.is_enabled:
            tracker.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    sys.exit(main())
