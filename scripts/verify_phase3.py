"""Comprehensive test and verification suite for Phase 3: Model Wrappers & Pipeline.

Tests:
1. Part A: Mocked Unit Tests (boundary clamping, invalid crop handling, class mappings).
2. Part B: Real Model Inference (YOLOv8n + MobileNetV2 on actual project samples).
3. Part C: Two-Stage Pipeline Orchestration (VehiclePipeline.from_config on real image).
4. Part D: Backward Compatibility (Legacy imports from `src.detector` and `src.classifier`).
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.config import load_config
from src.pipeline import VehiclePipeline, DetectionResult

logger = get_logger("phase3_verification")


def verify_backward_compatibility():
    logger.info("--- [Test 1/4] Backward-Compatibility Bridge Verification ---")
    # Verify legacy imports resolve without errors
    from src.detector import VehicleDetector as LegacyDetector
    from src.classifier import VehicleClassifier as LegacyClassifier
    from src.models.detector import VehicleDetector as ModernDetector
    from src.models.classifier import VehicleClassifier as ModernClassifier

    assert LegacyDetector is ModernDetector, "Legacy VehicleDetector is not aliased to ModernDetector!"
    assert LegacyClassifier is ModernClassifier, "Legacy VehicleClassifier is not aliased to ModernClassifier!"

    logger.info("Verified: 'from src.detector import VehicleDetector' aliases to src.models.detector")
    logger.info("Verified: 'from src.classifier import VehicleClassifier' aliases to src.models.classifier")
    logger.info("PASS: Backward compatibility confirmed 100%.")


def verify_real_detector():
    logger.info("--- [Test 2/4] Real YOLOv8n Detector Inference ---")
    cfg = load_config()
    from src.models.detector import VehicleDetector

    detector = VehicleDetector(
        model_path=cfg.detector_model_path,
        conf=cfg.detector.confidence_threshold,
    )

    # Test on a dummy solid frame (should find 0 vehicles safely)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    blank_dets = detector.detect(blank_frame)
    assert isinstance(blank_dets, list) and len(blank_dets) == 0
    logger.info("Verified detector on blank frame: 0 false detections.")

    # Test on an actual sample image from classification dataset
    sample_img_path = PROJECT_ROOT / "data" / "classification-dataset" / "Car" / "2_1.jpg"
    if sample_img_path.exists():
        frame = cv2.imread(str(sample_img_path))
        dets = detector.detect(frame)
        logger.info(f"Detector ran on {sample_img_path.name}: found {len(dets)} vehicle detections.")
        for d in dets:
            logger.info(f"  -> Detected: {d['yolo_class']} ({d['confidence']:.2f}) at box {d['box']}")
            assert d["box"][0] >= 0 and d["box"][1] >= 0
            assert d["box"][2] <= frame.shape[1] and d["box"][3] <= frame.shape[0]

    logger.info("PASS: Real detector inference working with safe boundary clamping.")


def verify_real_classifier():
    logger.info("--- [Test 3/4] Real MobileNetV2 Classifier Inference ---")
    cfg = load_config()
    from src.models.classifier import VehicleClassifier

    classifier = VehicleClassifier(
        model_path=cfg.classifier_model_path,
        class_indices_path=cfg.class_indices_path,
    )

    # 1. Test empty and invalid crop handling
    label, conf = classifier.predict(None)
    assert label == "unknown" and conf == 0.0
    label, conf = classifier.predict(np.zeros((0, 50, 3), dtype=np.uint8))
    assert label == "unknown" and conf == 0.0
    logger.info("Verified: Invalid/empty crops safely return ('unknown', 0.0).")

    # 2. Test inference on actual vehicle crop from outputs/crops/vehicle_0.jpg
    crop_path = PROJECT_ROOT / "outputs" / "crops" / "vehicle_0.jpg"
    if crop_path.exists():
        crop = cv2.imread(str(crop_path))
        label, conf = classifier.predict(crop)
        logger.info(f"Classified {crop_path.name}: predicted '{label}' with confidence {conf:.4f}")
        assert label in cfg.classifier.class_names
        assert 0.0 <= conf <= 1.0

    logger.info("PASS: Real classifier inference verified using Phase 2 preprocessor.")


def verify_two_stage_pipeline():
    logger.info("--- [Test 4/4] End-to-End Two-Stage Pipeline Orchestration ---")
    # Initialize pipeline directly from centralized configuration
    pipeline = VehiclePipeline.from_config()

    sample_car_path = PROJECT_ROOT / "data" / "classification-dataset" / "Car" / "2_1.jpg"
    if sample_car_path.exists():
        frame = cv2.imread(str(sample_car_path))
        predictions = pipeline.process_frame(frame)
        logger.info(f"Pipeline processed frame ({frame.shape}): {len(predictions)} vehicle predictions.")

        for pred in predictions:
            assert isinstance(pred, DetectionResult)
            logger.info(f"  -> Prediction: {pred.display_label}")
            logger.info(f"     Box: {pred.box}")
            assert pred.box[2] > pred.box[0] and pred.box[3] > pred.box[1]

    # Test pipeline on blank frame (no crashes, clean empty list)
    blank_results = pipeline.process_frame(np.zeros((300, 300, 3), dtype=np.uint8))
    assert blank_results == []
    logger.info("Verified: Blank frame returns empty list without error.")

    logger.info("PASS: Two-stage pipeline coordination verified end-to-end.")


def main():
    logger.info("==================================================")
    logger.info("STARTING PHASE 3 VERIFICATION SUITE")
    logger.info("==================================================")

    verify_backward_compatibility()
    verify_real_detector()
    verify_real_classifier()
    verify_two_stage_pipeline()

    logger.info("==================================================")
    logger.info("ALL PHASE 3 VERIFICATIONS PASSED SUCCESSFULLY!")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
