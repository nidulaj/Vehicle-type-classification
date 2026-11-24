"""Inference runner for two-stage vehicle detection and classification.

Coordinates predictions across image files, video streams, and live camera feeds
while ensuring model weights are loaded once upon initialization and reused.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.config import AppConfig, load_config
from src.inference.visualizer import Visualizer
from src.pipeline.schema import DetectionResult
from src.pipeline.vehicle_pipeline import VehiclePipeline
from src.utils.logger import get_logger
from src.utils.path_helpers import resolve_path

logger = get_logger("inference_runner")


class InferenceRunner:
    """Manages inference across static images, video files, and camera streams."""

    def __init__(
        self,
        pipeline: Optional[VehiclePipeline] = None,
        config: Optional[AppConfig] = None,
        visualizer: Optional[Visualizer] = None,
    ):
        """Initialize the inference runner.

        Models are loaded ONCE here and reused for all subsequent frame predictions.

        Args:
            pipeline: Optional pre-configured VehiclePipeline. If None, created from config.
            config: Optional AppConfig instance. If None, loaded via load_config().
            visualizer: Optional Visualizer instance for rendering detections.
        """
        self.config = config or load_config()
        self.pipeline = pipeline or VehiclePipeline.from_config(self.config)
        self.visualizer = visualizer or Visualizer()

    def predict_frame(
        self,
        frame: np.ndarray,
        annotate: bool = False,
    ) -> Tuple[List[DetectionResult], np.ndarray]:
        """Process a single frame through the two-stage prediction pipeline.

        Args:
            frame: OpenCV BGR image array.
            annotate: If True, draws bounding boxes and labels on a copy of the frame.

        Returns:
            Tuple[List[DetectionResult], np.ndarray]: List of predictions and output frame.
        """
        if frame is None or frame.size == 0:
            return [], frame

        predictions = self.pipeline.process_frame(frame)

        if annotate:
            output_frame = self.visualizer.draw_detections(frame, predictions, copy=True)
        else:
            output_frame = frame

        return predictions, output_frame

    def predict_image(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        annotate: bool = True,
        show: bool = False,
    ) -> Tuple[List[DetectionResult], np.ndarray]:
        """Run vehicle detection and classification on a single image file.

        Args:
            input_path: Path to input image file (.jpg, .png, etc.).
            output_path: Optional path to save the annotated output image.
            annotate: If True, draws bounding boxes and labels on the image.
            show: If True, opens an OpenCV display window until a key is pressed.

        Returns:
            Tuple[List[DetectionResult], np.ndarray]: Predictions and output image array.
        """
        path = resolve_path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input image file not found: {path}")

        frame = None
        if path.stat().st_size > 0:
            try:
                frame = cv2.imread(str(path))
            except Exception as e:
                logger.warning(f"Failed to read image {path}: {e}")
                frame = None

        if frame is None:
            raise ValueError(f"Could not read/decode image from: {path}. File may be corrupted or invalid format.")

        logger.info(f"Processing image: {path.name} ({frame.shape[1]}x{frame.shape[0]})")
        predictions, annotated_frame = self.predict_frame(frame, annotate=annotate)
        logger.info(f"Detected {len(predictions)} vehicle(s) in {path.name}")

        if output_path:
            save_path = resolve_path(output_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), annotated_frame)
            logger.info(f"Saved annotated image to: {save_path}")

        if show:
            cv2.imshow("Vehicle Prediction - " + path.name, annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return predictions, annotated_frame

    def predict_video(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        show_preview: bool = False,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process a video file through the two-stage vehicle prediction pipeline.

        Reuses the same initialized pipeline and detector/classifier models across
        all frames without reloading weights. Preserves original dimensions and FPS.
        Guarantees that video capture and writer resources are released via try/finally.

        Args:
            input_path: Path to input video file (.mp4, .avi, etc.).
            output_path: Optional path to save the annotated output video.
            show_preview: If True, renders frames in an OpenCV preview window.
            max_frames: Optional limit on the number of frames to process.

        Returns:
            Dict[str, Any]: Summary dictionary containing processing statistics.
        """
        path = resolve_path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input video file not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file at: {path}. Corrupted or unsupported format.")

        out = None
        save_path = None
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0 or np.isnan(fps):
                fps = 30.0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Opened video: {path.name} ({width}x{height} @ {fps:.2f} FPS, {total_frames} frames)")

            if output_path:
                save_path = resolve_path(output_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
                if not out.isOpened():
                    raise RuntimeError(f"Failed to initialize VideoWriter for: {save_path}")
                logger.info(f"Writing annotated video to: {save_path}")

            processed_frames = 0
            total_vehicles = 0

            while True:
                if max_frames is not None and processed_frames >= max_frames:
                    logger.info(f"Reached max_frames limit ({max_frames}). Stopping video processing.")
                    break

                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    break

                predictions, annotated_frame = self.predict_frame(frame, annotate=True)
                processed_frames += 1
                total_vehicles += len(predictions)

                if out is not None:
                    out.write(annotated_frame)

                if show_preview:
                    cv2.imshow("Video Prediction Preview - " + path.name, annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        logger.info("Video preview interrupted by user.")
                        break

            logger.info(
                f"Video processing complete: {processed_frames} frames processed, "
                f"{total_vehicles} total vehicles detected."
            )

            return {
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "total_vehicles_detected": total_vehicles,
                "fps": fps,
                "width": width,
                "height": height,
                "output_path": save_path,
            }
        finally:
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
            if show_preview:
                cv2.destroyAllWindows()

    def run_camera(
        self,
        camera_index: Optional[int] = None,
        window_name: Optional[str] = None,
        fullscreen: Optional[bool] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> None:
        """Run real-time vehicle detection and classification on live camera feed.

        Preserves existing project camera experience:
        - Camera index selection with automatic fallback (1 -> 0)
        - 1080p frame resolution configuration (1920x1080)
        - Fullscreen display window
        - Keyboard controls ('q' and ESC to exit)
        - Guaranteed OpenCV and camera resource release via try/finally

        Args:
            camera_index: Camera device index. If None, uses configured value (default: 1).
            window_name: OpenCV window title. If None, uses configured value.
            fullscreen: Whether to open window in fullscreen mode.
            frame_width: Requested frame width in pixels (default: 1920).
            frame_height: Requested frame height in pixels (default: 1080).
        """
        cam_idx = camera_index if camera_index is not None else self.config.inference.camera_index
        win_name = window_name or self.config.inference.window_name
        is_fullscreen = fullscreen if fullscreen is not None else self.config.inference.fullscreen
        width = frame_width or self.config.inference.frame_width
        height = frame_height or self.config.inference.frame_height

        logger.info(f"Connecting to camera (index={cam_idx})...")
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened() and cam_idx != 0:
            logger.warning(f"Could not open camera {cam_idx}. Falling back to default camera 0...")
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            logger.error("Could not open any camera device. Exiting live camera inference.")
            raise RuntimeError("Could not open camera (tried requested index and fallback 0).")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        if is_fullscreen:
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        logger.info(f"Live camera started. Displaying window '{win_name}'. Press 'q' or ESC to exit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    logger.warning("Failed to read frame from camera feed.")
                    break

                predictions, annotated_frame = self.predict_frame(frame, annotate=True)
                cv2.imshow(win_name, annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    logger.info("Exit key pressed. Stopping live camera.")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera device released and windows closed.")
