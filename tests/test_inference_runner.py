"""Unit tests for InferenceRunner frame processing."""

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.inference.inference_runner import InferenceRunner
from src.inference.visualizer import Visualizer
from src.pipeline.schema import DetectionResult
from src.pipeline.vehicle_pipeline import VehiclePipeline


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Create a mock VehiclePipeline."""
    mock = MagicMock(spec=VehiclePipeline)
    mock.process_frame.return_value = [
        DetectionResult(
            box=(10, 20, 100, 150),
            yolo_class="car",
            yolo_confidence=0.85,
            cnn_class="Car",
            cnn_confidence=0.92,
        )
    ]
    return mock


def test_runner_initialization_with_mock_pipeline(mock_pipeline: MagicMock):
    """Verify InferenceRunner accepts pre-initialized pipeline and visualizer."""
    vis = Visualizer()
    runner = InferenceRunner(pipeline=mock_pipeline, visualizer=vis)

    assert runner.pipeline is mock_pipeline
    assert runner.visualizer is vis


def test_predict_frame_reuses_pipeline_instance(mock_pipeline: MagicMock):
    """Verify that multiple frames reuse the exact same pipeline instance."""
    runner = InferenceRunner(pipeline=mock_pipeline)
    frame1 = np.zeros((200, 200, 3), dtype=np.uint8)
    frame2 = np.ones((200, 200, 3), dtype=np.uint8)

    dets1, out1 = runner.predict_frame(frame1, annotate=False)
    dets2, out2 = runner.predict_frame(frame2, annotate=False)

    # Pipeline process_frame called exactly twice
    assert mock_pipeline.process_frame.call_count == 2
    assert len(dets1) == 1
    assert len(dets2) == 1
    assert dets1[0].yolo_class == "car"


def test_predict_frame_annotation_flag(mock_pipeline: MagicMock):
    """Verify annotate=True invokes visualizer while annotate=False does not."""
    mock_vis = MagicMock(spec=Visualizer)
    mock_vis.draw_detections.return_value = np.ones((100, 100, 3), dtype=np.uint8)

    runner = InferenceRunner(pipeline=mock_pipeline, visualizer=mock_vis)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Without annotation
    _, out_no_ann = runner.predict_frame(frame, annotate=False)
    assert mock_vis.draw_detections.call_count == 0
    assert np.all(out_no_ann == 0)

    # With annotation
    _, out_ann = runner.predict_frame(frame, annotate=True)
    assert mock_vis.draw_detections.call_count == 1
    assert np.all(out_ann == 1)


def test_predict_frame_invalid_and_empty_frames(mock_pipeline: MagicMock):
    """Verify safe handling of None and 0-dimension frames."""
    runner = InferenceRunner(pipeline=mock_pipeline)

    dets_none, out_none = runner.predict_frame(None)
    assert dets_none == []
    assert out_none is None

    empty_frame = np.zeros((0, 50, 3), dtype=np.uint8)
    dets_empty, out_empty = runner.predict_frame(empty_frame)
    assert dets_empty == []
    assert out_empty.size == 0

    # Ensure pipeline process_frame was NOT called for invalid frames
    assert mock_pipeline.process_frame.call_count == 0


def test_predict_image_success(mock_pipeline: MagicMock, tmp_path):
    """Verify predict_image runs on valid image and writes output."""
    import cv2

    img_path = tmp_path / "test_car.jpg"
    dummy_img = np.ones((200, 300, 3), dtype=np.uint8) * 150
    cv2.imwrite(str(img_path), dummy_img)

    output_path = tmp_path / "out_pred.jpg"
    runner = InferenceRunner(pipeline=mock_pipeline)

    preds, out_frame = runner.predict_image(
        input_path=img_path,
        output_path=output_path,
        annotate=True,
    )

    assert len(preds) == 1
    assert preds[0].yolo_class == "car"
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_predict_image_missing_file(mock_pipeline: MagicMock, tmp_path):
    """Verify predict_image raises FileNotFoundError for missing path."""
    runner = InferenceRunner(pipeline=mock_pipeline)
    missing = tmp_path / "non_existent.jpg"

    with pytest.raises(FileNotFoundError):
        runner.predict_image(missing)


def test_predict_image_corrupt_file(mock_pipeline: MagicMock, tmp_path):
    """Verify predict_image raises ValueError for unreadable/corrupt files."""
    runner = InferenceRunner(pipeline=mock_pipeline)
    corrupt = tmp_path / "corrupt.jpg"
    # Write empty bytes which cv2.imread cannot decode
    corrupt.write_bytes(b"")

    with pytest.raises(ValueError):
        runner.predict_image(corrupt)


def test_predict_video_missing_file(mock_pipeline: MagicMock, tmp_path):
    """Verify predict_video raises FileNotFoundError for missing path."""
    runner = InferenceRunner(pipeline=mock_pipeline)
    missing = tmp_path / "non_existent.mp4"

    with pytest.raises(FileNotFoundError):
        runner.predict_video(missing)


def test_predict_video_failed_capture(mock_pipeline: MagicMock, tmp_path, monkeypatch):
    """Verify predict_video raises RuntimeError when VideoCapture cannot open."""
    import cv2

    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy video data")

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: mock_cap)

    runner = InferenceRunner(pipeline=mock_pipeline)
    with pytest.raises(RuntimeError, match="Could not open video file"):
        runner.predict_video(dummy_video)


def test_predict_video_sequential_mock_execution(mock_pipeline: MagicMock, tmp_path, monkeypatch):
    """Verify sequential frame processing and guaranteed resource cleanup."""
    import cv2

    dummy_video = tmp_path / "sample.mp4"
    dummy_video.write_bytes(b"sample bytes")

    # Mock VideoCapture with 4 valid frames
    frame_shape = (200, 300, 3)
    dummy_frames = [np.zeros(frame_shape, dtype=np.uint8) for _ in range(4)]
    frame_queue = [(True, f) for f in dummy_frames] + [(False, None)]

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: 300,
        cv2.CAP_PROP_FRAME_HEIGHT: 200,
        cv2.CAP_PROP_FPS: 25.0,
        cv2.CAP_PROP_FRAME_COUNT: 4,
    }.get(prop, 0.0)
    mock_cap.read.side_effect = frame_queue

    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = True

    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: mock_cap)
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args, **kwargs: mock_writer)

    out_path = tmp_path / "output.mp4"
    runner = InferenceRunner(pipeline=mock_pipeline)

    summary = runner.predict_video(
        input_path=dummy_video,
        output_path=out_path,
        max_frames=3,
    )

    # 1. Stopped at max_frames=3
    assert summary["processed_frames"] == 3
    assert summary["width"] == 300
    assert summary["height"] == 200
    assert summary["fps"] == 25.0

    # 2. Pipeline was called exactly 3 times (reusing the same pipeline instance!)
    assert mock_pipeline.process_frame.call_count == 3

    # 3. Writer was called 3 times
    assert mock_writer.write.call_count == 3

    # 4. Resources were properly released
    assert mock_cap.release.called is True
    assert mock_writer.release.called is True


def test_predict_video_releases_resources_on_error(mock_pipeline: MagicMock, tmp_path, monkeypatch):
    """Verify resources are released even if frame processing raises an exception."""
    import cv2

    dummy_video = tmp_path / "error_sample.mp4"
    dummy_video.write_bytes(b"sample bytes")

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 30.0
    mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))

    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = True

    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: mock_cap)
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args, **kwargs: mock_writer)

    # Force pipeline to fail during process_frame
    mock_pipeline.process_frame.side_effect = RuntimeError("Simulated processing error")

    runner = InferenceRunner(pipeline=mock_pipeline)
    with pytest.raises(RuntimeError, match="Simulated processing error"):
        runner.predict_video(dummy_video, output_path=tmp_path / "out.mp4")

    # Resources MUST be released in finally block
    assert mock_cap.release.called is True
    assert mock_writer.release.called is True


def test_run_camera_fallback_and_exit_on_q(mock_pipeline: MagicMock, monkeypatch):
    """Verify camera fallback from index 1 to 0 and exit on 'q' key."""
    import cv2

    mock_cap1 = MagicMock()
    mock_cap1.isOpened.return_value = False

    mock_cap0 = MagicMock()
    mock_cap0.isOpened.return_value = True
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap0.read.return_value = (True, dummy_frame)

    def mock_video_capture(index, *args, **kwargs):
        if index == 1:
            return mock_cap1
        return mock_cap0

    monkeypatch.setattr(cv2, "VideoCapture", mock_video_capture)
    monkeypatch.setattr(cv2, "namedWindow", MagicMock())
    monkeypatch.setattr(cv2, "setWindowProperty", MagicMock())
    monkeypatch.setattr(cv2, "imshow", MagicMock())
    # Return 'q' on first frame
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("q"))
    mock_destroy = MagicMock()
    monkeypatch.setattr(cv2, "destroyAllWindows", mock_destroy)

    runner = InferenceRunner(pipeline=mock_pipeline)
    runner.run_camera(camera_index=1)

    # 1. Verified fallback to camera 0 was called
    assert mock_cap0.set.called is True
    # 2. Pipeline was invoked for frame
    assert mock_pipeline.process_frame.called is True
    # 3. Resources cleaned up
    assert mock_cap0.release.called is True
    assert mock_destroy.called is True


def test_run_camera_failed_all_devices(mock_pipeline: MagicMock, monkeypatch):
    """Verify RuntimeError is raised when no camera device opens."""
    import cv2

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: mock_cap)

    runner = InferenceRunner(pipeline=mock_pipeline)
    with pytest.raises(RuntimeError, match="Could not open camera"):
        runner.run_camera(camera_index=1)


def test_run_camera_exit_on_esc(mock_pipeline: MagicMock, monkeypatch):
    """Verify camera loop exits cleanly when ESC (27) is pressed."""
    import cv2

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))

    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: mock_cap)
    monkeypatch.setattr(cv2, "namedWindow", MagicMock())
    monkeypatch.setattr(cv2, "setWindowProperty", MagicMock())
    monkeypatch.setattr(cv2, "imshow", MagicMock())
    # Return ESC (key code 27)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: 27)
    mock_destroy = MagicMock()
    monkeypatch.setattr(cv2, "destroyAllWindows", mock_destroy)

    runner = InferenceRunner(pipeline=mock_pipeline)
    runner.run_camera(camera_index=0)

    assert mock_cap.release.called is True
    assert mock_destroy.called is True



