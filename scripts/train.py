#!/usr/bin/env python
"""CLI entry point for two-stage vehicle classifier training.

Executes Stage 1 (feature extraction) and Stage 2 (fine-tuning) transfer learning
using the modular VehicleTrainer.

Usage:
    python scripts/train.py [--config configs/config.yaml] [--dry-run]
    python scripts/train.py --stage1-epochs 5 --stage2-epochs 10
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so scripts can run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.training.trainer import VehicleTrainer
from src.utils.logger import get_logger

logger = get_logger("train_cli")


def parse_args():
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
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing vehicle classification training CLI...")

    config = load_config(args.config) if args.config else load_config()
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
        logger.info("[Dry Run] Dry run validation successful! Exiting without training.")
        return 0

    logger.info("Starting complete two-stage training sequence...")
    results = trainer.train(
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
    )

    logger.info("Training complete!")
    logger.info(f"Best checkpoint saved to: {trainer.best_checkpoint_path}")
    logger.info(f"Final Stage 2 model saved to: {trainer.stage2_save_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
