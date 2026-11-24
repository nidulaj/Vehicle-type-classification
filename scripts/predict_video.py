#!/usr/bin/env python
"""CLI entry point for video file vehicle detection and classification.

Usage:
    python scripts/predict_video.py --input data/test-video/sample.mp4
    python scripts/predict_video.py --input data/test-video/sample.mp4 --output outputs/detected.mp4 --max-frames 150
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

logger = get_logger("predict_video_cli")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run two-stage vehicle detection and classification on a video file."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input video file (.mp4, .avi, etc.).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated output video (default: outputs/predicted_<input_name>).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (defaults to configs/config.yaml).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (optional, useful for fast testing).",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Render preview window during video processing (press 'q' or ESC to stop).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Process video without saving output file to disk.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = resolve_path(args.input)
    if not input_path.exists():
        logger.error(f"Input video file not found: {input_path}")
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

    logger.info(f"Initializing InferenceRunner for video: {input_path.name}")
    config = load_config(args.config) if args.config else load_config()
    runner = InferenceRunner(config=config)

    summary = runner.predict_video(
        input_path=input_path,
        output_path=output_path,
        show_preview=args.show_preview,
        max_frames=args.max_frames,
    )

    print("\n" + "=" * 60)
    print(f"  VIDEO PREDICTION SUMMARY: {input_path.name}")
    print("=" * 60)
    print(f"Total Video Frames:       {summary['total_frames']}")
    print(f"Processed Frames:         {summary['processed_frames']}")
    print(f"Total Vehicles Detected:  {summary['total_vehicles_detected']}")
    print(f"Resolution & FPS:         {summary['width']}x{summary['height']} @ {summary['fps']:.2f} FPS")
    if summary["output_path"]:
        print(f"Annotated Video Output:   {summary['output_path']}")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
