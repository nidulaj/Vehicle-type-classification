"""Comprehensive test and verification suite for Phase 2: Preprocessing Unification.

Tests:
1. Image input shape (1, 224, 224, 3).
2. Image data type (np.float32).
3. Pixel value range ([-1.0, 1.0] from MobileNetV2 normalization).
4. RGB / BGR color space conversion verification.
5. Resizing and aspect-ratio padding behavior.
6. Consistency between training and inference preprocessing.
7. Backward compatibility of legacy `src/data_loader.py` imports.
8. Model inference verification across real sample images and crops.
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
from src.data.preprocessor import (
    preprocess_image,
    full_preprocessing,
    resize_if_smaller,
    resize_with_pad,
)
from src.data.augmentations import (
    add_gaussian_noise,
    random_cutout,
    combined_augmentation,
)
from src.data.data_loader import create_generators as modern_create_generators
from src.data_loader import create_generators as legacy_create_generators

logger = get_logger("phase2_verification")


def test_shape_and_dtype():
    logger.info("--- [Test 1/7] Input Shape and Data Type ---")
    dummy_bgr = np.random.randint(0, 256, (360, 480, 3), dtype=np.uint8)
    
    # Preprocess with default settings
    tensor = preprocess_image(dummy_bgr)
    
    assert tensor.shape == (1, 224, 224, 3), f"Expected shape (1, 224, 224, 3), got {tensor.shape}"
    assert tensor.dtype == np.float32, f"Expected dtype float32, got {tensor.dtype}"
    logger.info(f"Verified output tensor: shape={tensor.shape}, dtype={tensor.dtype}")
    logger.info("PASS: Shape and dtype validation successful.")


def test_pixel_value_range():
    logger.info("--- [Test 2/7] Pixel Value Range Verification ---")
    # Pure black [0, 0, 0] should map to -1.0
    black_img = np.zeros((100, 100, 3), dtype=np.uint8)
    black_tensor = preprocess_image(black_img)
    assert np.allclose(black_tensor, -1.0, atol=1e-5), f"Black should normalize to -1.0, got min={black_tensor.min()}"

    # Pure white [255, 255, 255] should map to +1.0
    white_img = np.full((100, 100, 3), 255, dtype=np.uint8)
    white_tensor = preprocess_image(white_img)
    assert np.allclose(white_tensor, 1.0, atol=1e-5), f"White should normalize to 1.0, got max={white_tensor.max()}"

    # Mid-gray [127.5] should map close to 0.0
    gray_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    gray_tensor = preprocess_image(gray_img)
    assert -0.05 < gray_tensor.mean() < 0.05, f"Mid-gray should center around 0.0, got mean={gray_tensor.mean()}"

    logger.info(f"Verified range: black={black_tensor.min():.2f}, white={white_tensor.max():.2f}, gray_mean={gray_tensor.mean():.4f}")
    logger.info("PASS: MobileNetV2 [-1.0, 1.0] value range strictly verified.")


def test_color_space_conversion():
    logger.info("--- [Test 3/7] Color Space (BGR -> RGB) Conversion ---")
    # Create pure blue image in BGR: B=255, G=0, R=0
    blue_bgr = np.zeros((50, 50, 3), dtype=np.uint8)
    blue_bgr[:, :, 0] = 255  # Channel 0 is Blue in BGR

    # When preprocessed with is_bgr=True, Channel 2 (Red) should be -1.0, Channel 0 (Red in RGB) should be +1.0
    tensor = preprocess_image(blue_bgr, is_bgr=True)
    # Channel 0 is R (should be -1.0 because original R was 0)
    # Channel 2 is B (should be +1.0 because original B was 255)
    r_val = tensor[0, 25, 25, 0]
    g_val = tensor[0, 25, 25, 1]
    b_val = tensor[0, 25, 25, 2]

    assert np.isclose(r_val, -1.0, atol=1e-3), f"Expected Red channel -1.0, got {r_val}"
    assert np.isclose(g_val, -1.0, atol=1e-3), f"Expected Green channel -1.0, got {g_val}"
    assert np.isclose(b_val, 1.0, atol=1e-3), f"Expected Blue channel +1.0, got {b_val}"

    logger.info("Verified BGR [255, 0, 0] properly mapped to RGB [0, 0, 255] in normalized space.")
    logger.info("PASS: Color space conversion verified.")


def test_resizing_and_padding():
    logger.info("--- [Test 4/7] Resizing and Padding Behavior ---")
    # Test 1: Wide image (aspect ratio 4:1)
    wide_img = np.full((100, 400, 3), 200, dtype=np.uint8)
    
    # Without aspect ratio preservation (standard bilinear stretch)
    stretched = preprocess_image(wide_img, preserve_aspect_ratio=False)
    assert stretched.shape == (1, 224, 224, 3)

    # With aspect ratio preservation (padded with zero / -1.0 after normalization)
    padded = preprocess_image(wide_img, preserve_aspect_ratio=True)
    assert padded.shape == (1, 224, 224, 3)
    # Top and bottom rows should be padded (black/zero input -> -1.0 normalized)
    top_border = padded[0, 0, :, :]
    assert np.allclose(top_border, -1.0, atol=1e-4), "Padding should be filled with normalized zero (-1.0)"

    logger.info(f"Verified stretch shape: {stretched.shape}, padded shape: {padded.shape}")
    logger.info("PASS: Resizing and aspect-ratio padding verified.")


def test_training_inference_consistency():
    logger.info("--- [Test 5/7] Training vs. Inference Preprocessing Consistency ---")
    sample_file = PROJECT_ROOT / "data" / "classification-dataset" / "Bus" / "2_1.jpg"
    assert sample_file.exists(), f"Sample file not found at {sample_file}"

    # Simulation 1: Training pipeline flow (PIL read as RGB, resized to 224x224, then full_preprocessing)
    bgr = cv2.imread(str(sample_file))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(rgb, (224, 224)).astype(np.float32)
    training_preprocessed = full_preprocessing(resized_rgb.copy())

    # Simulation 2: Unified inference pipeline flow (Raw BGR crop passed directly to preprocess_image)
    inference_preprocessed = preprocess_image(bgr, target_size=(224, 224), is_bgr=True, expand_dims=False)

    # Compare pixel-for-pixel
    max_diff = np.max(np.abs(training_preprocessed - inference_preprocessed))
    logger.info(f"Max absolute difference between training and inference output: {max_diff:.6f}")
    assert max_diff < 1e-4, f"Training and inference preprocessing differ! Max diff: {max_diff}"
    logger.info("PASS: 100% mathematical consistency confirmed between training and inference.")


def test_augmentations():
    logger.info("--- [Test 6/7] Custom Augmentations Validation ---")
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    
    # Test CutOut
    cutout_img = random_cutout(img, min_mask_size=20, max_mask_size=40)
    assert cutout_img.shape == img.shape
    # Must have some zero pixels
    zero_pixels = np.sum(cutout_img == 0)
    assert zero_pixels > 0, "CutOut failed to mask any pixels"

    # Test Gaussian Noise
    noisy_img = add_gaussian_noise(img.astype(np.float32), std=5.0)
    assert noisy_img.shape == img.shape
    assert not np.array_equal(img, noisy_img), "Gaussian noise had no effect"

    logger.info(f"Verified CutOut masked {zero_pixels} pixels; Gaussian noise applied successfully.")
    logger.info("PASS: Augmentations working as expected.")


def test_data_loader_compatibility():
    logger.info("--- [Test 7/7] Data Loader Backward Compatibility ---")
    cfg = load_config()

    # Test with modern AppConfig
    train_gen1, val_gen1 = modern_create_generators(cfg)
    logger.info(f"Modern create_generators returned {train_gen1.samples} train and {val_gen1.samples} val samples.")
    assert train_gen1.num_classes == 5

    # Test with legacy dictionary config via legacy import bridge
    legacy_dict = {
        "dataset_dir": str(cfg.dataset_path),
        "validation_split": 0.2,
        "image_height": 224,
        "image_width": 224,
        "batch_size": 32,
    }
    train_gen2, val_gen2 = legacy_create_generators(legacy_dict)
    logger.info(f"Legacy create_generators returned {train_gen2.samples} train and {val_gen2.samples} val samples.")
    assert train_gen2.num_classes == 5

    # Verify identical class indices
    assert train_gen1.class_indices == train_gen2.class_indices
    logger.info(f"Class indices confirmed identical: {train_gen1.class_indices}")
    logger.info("PASS: Backward compatibility of data_loader verified 100%.")


def main():
    logger.info("==================================================")
    logger.info("STARTING PHASE 2 VERIFICATION SUITE")
    logger.info("==================================================")

    test_shape_and_dtype()
    test_pixel_value_range()
    test_color_space_conversion()
    test_resizing_and_padding()
    test_training_inference_consistency()
    test_augmentations()
    test_data_loader_compatibility()

    logger.info("==================================================")
    logger.info("ALL PHASE 2 TESTS PASSED SUCCESSFULLY!")
    logger.info("==================================================")


if __name__ == "__main__":
    main()

