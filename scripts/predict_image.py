#!/usr/bin/env python
"""CLI entry point for single image vehicle detection and classification.

Usage:
    python scripts/predict_image.py --input path/to/image.jpg
    python scripts/predict_image.py --input data/classification-dataset/Car/2_1.jpg --output outputs/car_pred.jpg
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so scripts can run standalone
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.inference.inference_runner import InferenceRunner
from src.utils.logger import get_logger
from src.utils.path_helpers import resolve_path

logger = get_logger("predict_image_cli")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run two-stage vehicle detection and classification on an image."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image file (.jpg, .png, etc.).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated image (default: outputs/predicted_<input_name>).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (defaults to configs/config.yaml).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the annotated image to disk.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated image in an OpenCV window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = resolve_path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    # Default output path
    output_path = None
    if not args.no_save:
        if args.output:
            output_path = resolve_path(args.output)
        else:
            outputs_dir = resolve_path("outputs")
            outputs_dir.mkdir(parents=True, exist_ok=True)
            output_path = outputs_dir / f"predicted_{input_path.name}"

    logger.info(f"Initializing InferenceRunner for image: {input_path.name}")
    config = load_config(args.config) if args.config else load_config()
    runner = InferenceRunner(config=config)

    predictions, _ = runner.predict_image(
        input_path=input_path,
        output_path=output_path,
        annotate=True,
        show=args.show,
    )

    print("\n" + "=" * 60)
    print(f"  IMAGE PREDICTION RESULTS: {input_path.name}")
    print("=" * 60)
    print(f"Total Vehicles Detected: {len(predictions)}\n")
    for i, pred in enumerate(predictions, 1):
        x1, y1, x2, y2 = pred.box
        print(f"  [{i}] Box: ({x1}, {y1}, {x2}, {y2})")
        print(f"      YOLO: {pred.yolo_class} ({pred.yolo_confidence:.2f})")
        print(f"      CNN:  {pred.cnn_class} ({pred.cnn_confidence:.2f})")
    print("=" * 60)
    if output_path:
        print(f"Annotated image saved to: {output_path}")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
