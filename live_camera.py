"""Live camera vehicle detection and classification entry point.

Preserves full backward compatibility for running 'python live_camera.py' directly.
Delegates to the modular InferenceRunner while maintaining identical camera selection,
fallback logic, 1080p resolution, fullscreen display, keyboard controls, and cleanup.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import InferenceRunner


def main():
    runner = InferenceRunner()
    runner.run_camera()


if __name__ == "__main__":
    main()
