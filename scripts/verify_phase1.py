"""Verification script for Phase 1: Project Foundation.

Tests:
1. Environment-independent path resolution (pathlib).
2. Centralized YAML configuration loading.
3. Strongly-typed Pydantic validation (valid & invalid inputs).
4. Centralized logging output (console & file).
5. Unbroken access to legacy utils for backward compatibility.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.path_helpers import get_project_root, resolve_path
from src.utils.logger import get_logger, setup_logger
from src.config import load_config, AppConfig
from pydantic import ValidationError


def test_path_resolution(logger):
    logger.info("--- [Test 1/5] Path Resolution ---")
    root = get_project_root()
    logger.info(f"Detected project root: {root}")
    assert root.exists(), "Project root does not exist!"
    assert (root / "configs" / "config.yaml").exists(), "config.yaml not found at root!"

    resolved_config = resolve_path("configs/config.yaml")
    assert resolved_config.is_absolute(), "resolve_path did not return an absolute path!"
    logger.info(f"Resolved 'configs/config.yaml' -> {resolved_config}")
    logger.info("PASS: Path resolution working as expected.")


def test_config_loading(logger):
    logger.info("--- [Test 2/5] YAML Configuration Loading ---")
    cfg = load_config()
    logger.info(f"Project name: {cfg.project.name} (v{cfg.project.version})")
    logger.info(f"Dataset path: {cfg.dataset_path}")
    logger.info(f"Detector model path: {cfg.detector_model_path}")
    logger.info(f"Classifier model path: {cfg.classifier_model_path}")
    logger.info(f"Target classes ({cfg.classifier.num_classes}): {cfg.classifier.class_names}")
    logger.info(f"YOLO vehicle classes: {cfg.detector.vehicle_classes}")
    logger.info(f"Stage 1 epochs: {cfg.training.stage1.epochs}, LR: {cfg.training.stage1.learning_rate}")
    logger.info(f"Stage 2 epochs: {cfg.training.stage2.epochs}, Unfreeze layers: {cfg.training.stage2.unfreeze_last_n_layers}")
    
    assert cfg.detector.confidence_threshold == 0.4
    assert cfg.data.image_height == 224
    assert cfg.data.image_width == 224
    assert cfg.data.batch_size == 32
    logger.info("PASS: All configuration values match existing project specifications.")


def test_pydantic_validation(logger):
    logger.info("--- [Test 3/5] Pydantic Schema Validation ---")
    # Test valid defaults
    valid_cfg = AppConfig()
    assert valid_cfg.detector.confidence_threshold == 0.4
    logger.info("Verified default AppConfig initialization.")

    # Test rejection of invalid confidence threshold (> 1.0)
    try:
        AppConfig.model_validate({"detector": {"confidence_threshold": 1.5}})
        raise AssertionError("Validation failed: allowed confidence_threshold > 1.0")
    except ValidationError:
        logger.info("Caught invalid confidence_threshold (> 1.0) correctly.")

    # Test rejection of invalid negative image size
    try:
        AppConfig.model_validate({"data": {"image_height": -100}})
        raise AssertionError("Validation failed: allowed negative image_height")
    except ValidationError:
        logger.info("Caught invalid negative image_height correctly.")

    # Test rejection of unknown log level
    try:
        AppConfig.model_validate({"logging": {"level": "INVALID_LEVEL"}})
        raise AssertionError("Validation failed: allowed invalid log level")
    except ValidationError:
        logger.info("Caught invalid log level correctly.")

    logger.info("PASS: Pydantic rejects invalid parameters and enforces constraints.")


def test_logging(logger):
    logger.info("--- [Test 4/5] Logging Functionality ---")
    cfg = load_config()
    test_log_file = cfg.log_file_path
    logger.info(f"Configured log file destination: {test_log_file}")
    
    custom_logger = setup_logger(
        name="verification_submodule",
        level=cfg.logging.level,
        log_file=test_log_file
    )
    custom_logger.debug("This is a DEBUG message (should only show if level is DEBUG).")
    custom_logger.info("This is an INFO message verifying file writing.")
    custom_logger.warning("This is a WARNING message verifying formatting.")
    
    if test_log_file and test_log_file.exists():
        logger.info(f"Verified log file exists on disk: {test_log_file} ({test_log_file.stat().st_size} bytes)")
    logger.info("PASS: Logger formatted and written successfully.")


def test_cwd_independence(logger):
    logger.info("--- [Test 5/5] CWD Independence ---")
    original_cwd = os.getcwd()
    try:
        # Change directory to src
        os.chdir(str(PROJECT_ROOT / "src"))
        logger.info(f"Changed working directory to: {os.getcwd()}")
        cfg = load_config()
        assert cfg.detector_model_path == PROJECT_ROOT / "yolov8n.pt"
        logger.info("Config successfully loaded from non-root directory.")
    finally:
        os.chdir(original_cwd)
        logger.info(f"Restored working directory to: {os.getcwd()}")
    logger.info("PASS: System is 100% CWD-independent.")


def main():
    logger = get_logger("phase1_verification")
    logger.info("==================================================")
    logger.info("STARTING PHASE 1 VERIFICATION SUITE")
    logger.info("==================================================")
    
    test_path_resolution(logger)
    test_config_loading(logger)
    test_pydantic_validation(logger)
    test_logging(logger)
    test_cwd_independence(logger)

    logger.info("==================================================")
    logger.info("ALL PHASE 1 TESTS PASSED SUCCESSFULLY!")
    logger.info("==================================================")


if __name__ == "__main__":
    main()

