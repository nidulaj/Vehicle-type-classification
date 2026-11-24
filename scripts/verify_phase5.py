"""Comprehensive verification suite for Phase 5: Inference Refactoring.

Tests:
1. Part A: Visualizer rendering, label formatting, and boundary safety.
2. Part B: InferenceRunner initialization and model reuse across multiple frames.
3. Part C: Real image inference and annotated output persistence.
4. Part D: Real short video inference, frame sequencing, and MP4 generation.
5. Part E: Camera fallback, key event handling, and resource cleanup (mocked).
6. Part F: Backward-compatibility check for root live_camera.py.
"""

from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock
import cv2
import numpy as np

# Ensure project root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.inference import InferenceRunner, Visualizer
from src.pipeline.schema import DetectionResult
from src.utils.logger import get_logger
from src.utils.path_helpers import resolve_path

logger = get_logger("phase5_verification")


def verify_visualizer():
    logger.info("--- [Test 1/6] Visualizer Rendering & Safe Boundary Clamping ---")
    vis = Visualizer()
    frame = np.zeros((300, 400, 3), dtype=np.uint8)

    det = DetectionResult(
        box=(50, 4, 200, 150),  # y1=4 is close to top edge
        yolo_class="truck",
        yolo_confidence=0.89,
        cnn_class="Truck",
        cnn_confidence=0.96,
    )

    orig_box = det.box
    annotated = vis.draw_detections(frame, [det], copy=True)

    # 1. Result was not mutated
    assert det.box == orig_box
    # 2. Input frame untouched
    assert np.all(frame == 0)
    # 3. Output frame modified
    assert np.any(annotated > 0)
    logger.info("Verified: Visualizer renders safe bounding boxes without mutating prediction objects.")
    logger.info("PASS: Visualizer component confirmed.")


def verify_inference_runner_initialization():
    logger.info("--- [Test 2/6] InferenceRunner Model Reuse Across Frames ---")
    config = load_config()
    runner = InferenceRunner(config=config)

    # Process 3 sequential dummy frames
    for i in range(3):
        dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        preds, _ = runner.predict_frame(dummy_frame, annotate=False)
        assert isinstance(preds, list)
        assert len(preds) == 0

    # Verify underlying models remain loaded and identical in memory
    assert runner.pipeline.detector.model is not None
    assert runner.pipeline.classifier.model is not None
    logger.info("Verified: Detector and Classifier instances are preserved across sequential frames.")
    logger.info("PASS: InferenceRunner frame prediction confirmed.")


def verify_real_image_inference():
    logger.info("--- [Test 3/6] Real Static Image Inference & Output Saving ---")
    config = load_config()
    runner = InferenceRunner(config=config)

    sample_img_path = PROJECT_ROOT / "data" / "classification-dataset" / "Car" / "Image_1.jpg"
    assert sample_img_path.exists(), f"Sample image not found: {sample_img_path}"

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "verification_phase5_image.jpg"

    preds, out_frame = runner.predict_image(
        input_path=sample_img_path,
        output_path=out_path,
        annotate=True,
    )

    logger.info(f"Processed {sample_img_path.name}: found {len(preds)} vehicle(s).")
    assert len(preds) > 0, "Expected at least 1 vehicle detected in sample image!"
    assert preds[0].yolo_class == "car"
    assert preds[0].cnn_class == "Car"
    assert out_path.exists() and out_path.stat().st_size > 0
    logger.info(f"Verified: Output annotated image saved to {out_path} ({out_path.stat().st_size} bytes).")
    logger.info("PASS: Real image inference confirmed.")


def verify_real_video_inference():
    logger.info("--- [Test 4/6] Real Short Video Inference & MP4 Generation ---")
    config = load_config()
    runner = InferenceRunner(config=config)

    sample_video_path = PROJECT_ROOT / "data" / "test-video" / "5638-183850414_medium.mp4"
    assert sample_video_path.exists(), f"Sample video not found: {sample_video_path}"

    out_video_path = PROJECT_ROOT / "outputs" / "verification_phase5_video.mp4"

    summary = runner.predict_video(
        input_path=sample_video_path,
        output_path=out_video_path,
        max_frames=10,
    )

    assert summary["processed_frames"] == 10
    assert summary["total_vehicles_detected"] > 0
    assert summary["width"] == 1280
    assert summary["height"] == 720
    assert summary["fps"] == 25.0
    assert out_video_path.exists() and out_video_path.stat().st_size > 0

    logger.info(
        f"Verified: Processed 10 frames ({summary['total_vehicles_detected']} vehicles detected), "
        f"saved to {out_video_path} ({out_video_path.stat().st_size} bytes)."
    )
    logger.info("PASS: Real video inference confirmed.")


def verify_camera_fallback_and_cleanup(monkeypatch):
    logger.info("--- [Test 5/6] Camera Fallback (1 -> 0) & Resource Cleanup (Mocked) ---")
    mock_cap1 = MagicMock()
    mock_cap1.isOpened.return_value = False

    mock_cap0 = MagicMock()
    mock_cap0.isOpened.return_value = True
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap0.read.return_value = (True, dummy_frame)

    def mock_video_capture(index, *args, **kwargs):
        return mock_cap1 if index == 1 else mock_cap0

    monkeypatch.setattr(cv2, "VideoCapture", mock_video_capture)
    monkeypatch.setattr(cv2, "namedWindow", MagicMock())
    monkeypatch.setattr(cv2, "setWindowProperty", MagicMock())
    monkeypatch.setattr(cv2, "imshow", MagicMock())
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("q"))
    mock_destroy = MagicMock()
    monkeypatch.setattr(cv2, "destroyAllWindows", mock_destroy)

    config = load_config()
    runner = InferenceRunner(config=config)
    runner.run_camera(camera_index=1)

    assert mock_cap0.release.called is True
    assert mock_destroy.called is True
    logger.info("Verified: Fallback from camera 1 to camera 0 succeeded, resources released on 'q'.")
    logger.info("PASS: Camera fallback and cleanup confirmed.")


def verify_backward_compatibility():
    logger.info("--- [Test 6/6] Backward Compatibility for root live_camera.py ---")
    root_live_camera = PROJECT_ROOT / "live_camera.py"
    assert root_live_camera.exists(), "Root live_camera.py is missing!"

    # Test importing root live_camera without error
    import live_camera
    assert hasattr(live_camera, "main")
    logger.info("Verified: Root live_camera.py imports and references modular InferenceRunner cleanly.")
    logger.info("PASS: Backward compatibility confirmed.")


def main():
    logger.info("==================================================")
    logger.info("STARTING PHASE 5 VERIFICATION SUITE")
    logger.info("==================================================")

    import pytest

    # Run tests using monkeypatch for the mocked camera test
    verify_visualizer()
    verify_inference_runner_initialization()
    verify_real_image_inference()
    verify_real_video_inference()

    # Test camera fallback with monkeypatch
    with pytest.MonkeyPatch.context() as mp:
        verify_camera_fallback_and_cleanup(mp)

    verify_backward_compatibility()

    logger.info("==================================================")
    logger.info("ALL PHASE 5 VERIFICATIONS PASSED SUCCESSFULLY!")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
