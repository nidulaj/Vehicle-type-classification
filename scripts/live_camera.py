#!/usr/bin/env python
"""CLI entry point for live camera vehicle detection and classification.

Usage:
    python scripts/live_camera.py
    python scripts/live_camera.py --camera-index 0 --no-fullscreen
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

logger = get_logger("live_camera_cli")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run real-time vehicle detection and classification on live camera feed."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Camera device index (defaults to config: 1 with fallback to 0).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (defaults to configs/config.yaml).",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default=None,
        help="OpenCV display window title.",
    )
    parser.add_argument(
        "--no-fullscreen",
        action="store_true",
        help="Disable fullscreen display mode.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Requested frame width in pixels (default: 1920).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Requested frame height in pixels (default: 1080).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing live camera inference CLI...")

    config = load_config(args.config) if args.config else load_config()
    runner = InferenceRunner(config=config)

    fullscreen = False if args.no_fullscreen else None

    runner.run_camera(
        camera_index=args.camera_index,
        window_name=args.window_name,
        fullscreen=fullscreen,
        frame_width=args.width,
        frame_height=args.height,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
