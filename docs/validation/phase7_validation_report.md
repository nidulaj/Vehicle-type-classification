# Phase 7: Final End-to-End Validation Report
**Project:** Two-Stage Vehicle Detection and Classification Pipeline  
**Architecture:** YOLOv8n (Localization) + Fine-Tuned MobileNetV2 (5-Class Classification)  
**Date of Validation:** November 26, 2026  
**Status:** Validated (Steps A–E Completed, Step F Deferred)  

---

## 1. Executive Summary

### 1.1 Project Purpose
The **Vehicle-type-classification** project implements a modular, production-grade two-stage computer vision system designed to detect and classify road vehicles into granular categories. Originally structured as monolithic, tightly coupled scripts, the codebase underwent a comprehensive six-phase architectural refactoring into a clean Python package (`src/`) with centralized Pydantic configuration, isolated data pipelines, decoupled model wrappers, standardized training/evaluation routines, and unified inference workflows.

### 1.2 Two-Stage Architecture
1. **Stage 1 — Vehicle Localization (YOLOv8n):** Scans the full input image or video frame to detect candidate vehicles (`car`, `motorcycle`, `bus`, `truck`) and outputs bounding box coordinates.
2. **Stage 2 — Fine-Grained Classification (MobileNetV2):** Extracts and normalizes each detected bounding box crop to $224 \times 224 \times 3$, then classifies it into one of 5 target vehicle categories (`Bus`, `Car`, `Three Wheeler`, `Truck`, `motorcycle`).

```
Input Frame (Image / Video / Camera)
         │
         ▼
┌─────────────────────────────────┐
│ Stage 1: YOLOv8n Vehicle Detector│  ──▶ Detects generic vehicle bounding boxes
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Dynamic Cropping & Clamping     │  ──▶ Clamps coordinates & extracts ROI crops
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Stage 2: MobileNetV2 Classifier │  ──▶ Fine-grained 5-class categorization
└─────────────────────────────────┘
         │
         ▼
Annotated Visual Output (BBox + Dual Labels + Confidence Scores)
```

### 1.3 Scope of Phase 7 Validation
Phase 7 validates the refactored repository end-to-end to ensure zero functional regressions, verify classifier quantitative performance, and confirm runtime pipeline stability across real data modalities without modifying source code, configurations, or model weights:
- **Step A:** Full automated pytest suite (42 tests) and all 5 phase verification scripts.
- **Step B:** Quantitative evaluation of MobileNetV2 across all 863 validation images.
- **Step C:** Inspection and audit of evaluation artifacts (`metrics.json`, `confusion_matrix.png`).
- **Step D:** End-to-end pipeline inference on a real static image (`Car/Image_1.jpg`).
- **Step E:** End-to-end pipeline inference on real multi-frame video (`5638-183850414_medium.mp4`).
- **Step F:** Live camera verification (**DEFERRED** due to lack of physical hardware).
- **Step G:** Final synthesis and validation documentation.

### 1.4 Overall Conclusion & Limitations
The refactored architecture passed all active validation gates with 100% test coverage compliance and zero regressions. However, the system is an engineering portfolio/prototype pipeline and **is not certified for safety-critical production deployment**. Key operational constraints include CPU-bound inference throughput (~2.7 FPS on video), mutual confusion between visually similar commercial classes (`Bus` vs. `Truck`), and the explicit distinction between standalone classifier accuracy (94.90%) and complete two-stage pipeline accuracy (which is bounded by detector recall and bounding box quality).

---

## 2. Project and Model Details

### 2.1 Model Specifications
- **Vehicle Detector (Stage 1):**
  - Architecture: Ultralytics YOLOv8 Nano (`YOLOv8n`)
  - Checkpoint: `yolov8n.pt` (3.2M parameters)
  - Pretraining: COCO Dataset
  - Target Vehicle Class Indices: `[2, 3, 5, 7]` (`car`, `motorcycle`, `bus`, `truck`)
  - Operational Confidence Threshold: `0.40`
  - Coordinate Clamping: Automated border clamping preventing negative coordinates or out-of-bounds indexing.
- **Vehicle Classifier (Stage 2):**
  - Architecture: MobileNetV2 with pre-trained ImageNet backbone
  - Checkpoint: `models/vehicle_classifier_best.keras` (2.26M parameters)
  - Head Architecture: `GlobalAveragePooling2D` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(5, activation='softmax')`
  - Input Resolution: $224 \times 224 \times 3$
  - Normalization: `tensorflow.keras.applications.mobilenet_v2.preprocess_input` (pixel scaling to $[-1.0, +1.0]$)
  - Target Classes: `['Bus', 'Car', 'Three Wheeler', 'Truck', 'motorcycle']` (mapped via `models/class_indices.json`)

### 2.2 Dataset and Data Splits
- **Source Dataset:** `data/classification-dataset/`
- **Total Dataset Size:** 4,309 images organized in class-specific subfolders.
- **Splits (Deterministic Seed = 42):**
  - **Training Split (80%):** 3,446 images
  - **Validation Split (20%):** 863 images (evaluated across 27 batches of batch size 32)
- **Data Pipeline:** Implemented in `src/data/loader.py` with decoupled data augmentation (`src/data/augmentations.py`) applied strictly to training sets while validation sets undergo deterministic resizing and scaling.

---

## 3. Automated Testing Results (Step A)

### 3.1 Pytest Suite Execution
The automated test suite was executed in an isolated environment without cache pollution:
```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v -p no:cacheprovider
```

- **Total Tests:** 42
- **Passed:** 42
- **Failed:** 0
- **Execution Duration:** 34.01 seconds
- **Subsystem Breakdown:**
  - `tests/test_config.py`: 5 passed (schema validation, defaults, custom overrides, path resolution)
  - `tests/test_data.py`: 7 passed (image loading, corrupt image resilience, batch tensor shapes, split reproducibility)
  - `tests/test_models.py`: 10 passed (detector wrapper, classifier wrapper, boundary box clamping, fallback handling)
  - `tests/test_training.py`: 7 passed (two-stage training loop, frozen backbone transitions, metric logging)
  - `tests/test_evaluation.py`: 6 passed (confusion matrix generation, per-class report computation, serialization)
  - `tests/test_inference.py`: 7 passed (image runner, video runner, graceful camera handling, mock visualizer)

### 3.2 Phase Verification Scripts
Each milestone verification script was executed sequentially to ensure architectural contracts remain satisfied:
- `scripts/verify_phase1.py` (Configuration & Logging): **PASSED** (100% assertions verified)
- `scripts/verify_phase2.py` (Data Pipeline & Augmentations): **PASSED** (100% assertions verified)
- `scripts/verify_phase3.py` (Pipeline Wrappers & Coordinate Clamping): **PASSED** (100% assertions verified)
- `scripts/verify_phase4.py` (Model Architecture, Trainer & Evaluator): **PASSED** (100% assertions verified)
- `scripts/verify_phase5.py` (Visualizer & Multi-modal Inference Runners): **PASSED** (100% assertions verified)

### 3.3 System Warnings Observed
- Standard TensorFlow runtime logs: `oneDNN custom operations are on`, informing user of floating-point CPU acceleration.
- Informational warning: `TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11`. All TensorFlow operations ran as expected on CPU without functional degradation.

---

## 4. Classifier Evaluation Results (Step B)

### 4.1 Evaluation Protocol
The standalone fine-tuned MobileNetV2 classifier was evaluated against the entire validation split (863 images) using the evaluation entrypoint:
```powershell
.\venv\Scripts\python.exe scripts/evaluate.py --model-path models/vehicle_classifier_best.keras --output-dir outputs/evaluation
```
- **Evaluated Samples:** Exactly 863 images across 27 batches.
- **Evaluation Time:** ~18 seconds on CPU (~48 images/sec).
- **Correct Predictions:** 819 / 863 images.

### 4.2 Global Performance Metrics
- **Overall Accuracy:** **94.90%** (0.949015)
- **Macro Average:**
  - Precision: **0.9372** (93.72%)
  - Recall: **0.9470** (94.70%)
  - F1-Score: **0.9416** (94.16%)
- **Weighted Average:**
  - Precision: **0.9509** (95.09%)
  - Recall: **0.9490** (94.90%)
  - F1-Score: **0.9495** (94.95%)

### 4.3 Per-Class Performance Breakdown

| Class Name | Precision | Recall | F1-Score | Support (Samples) | Correct Predictions |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bus** | 0.9365 | 0.9568 | 0.9465 | 185 | 177 |
| **Car** | 0.9957 | 0.9355 | 0.9647 | 248 | 232 |
| **Three Wheeler** | 0.9321 | 0.9497 | 0.9408 | 159 | 151 |
| **Truck** | 0.8447 | 0.9158 | 0.8788 | 95 | 87 |
| **motorcycle** | 0.9773 | 0.9773 | 0.9773 | 176 | 172 |
| **Total / Summary** | **0.9509** | **0.9490** | **0.9495** | **863** | **819** |

### 4.4 Confusion Matrix Analysis
The $5 \times 5$ confusion matrix (rows = true classes, columns = predicted classes):

| True \ Pred | Bus | Car | Three Wheeler | Truck | motorcycle |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bus** | **177** | 0 | 0 | 7 | 1 |
| **Car** | 2 | **232** | 8 | 4 | 2 |
| **Three Wheeler** | 3 | 1 | **151** | 3 | 1 |
| **Truck** | 7 | 0 | 1 | **87** | 0 |
| **motorcycle** | 0 | 0 | 2 | 2 | **172** |

#### Matrix Audit & Key Patterns:
1. **Trace & Total Check:** Diagonal elements sum to $177 + 232 + 151 + 87 + 172 = 819$. Total matrix elements sum to $863$.
2. **Weakest Performing Class:** **Truck** achieved the lowest F1-score (0.8788) and lowest precision (0.8447).
3. **Primary Misclassification Clusters:**
   - **Bus $\leftrightarrow$ Truck Mutual Confusion:** 7 true Buses were misclassified as Trucks; 7 true Trucks were misclassified as Buses (14 errors total). This stems from visual similarity in chassis height, flat frontals, and large boxy dimensions.
   - **Car $\rightarrow$ Three Wheeler Confusion:** 8 true Cars were misclassified as Three Wheelers, predominantly on compact or steep rear-quarter perspectives.
   - **Car $\rightarrow$ Truck Confusion:** 4 true Cars were misclassified as Trucks (e.g., SUVs/Pickups).

---

## 5. Evaluation Artifacts Review (Step C)

The evaluation artifacts generated in Step B were verified for mathematical consistency, format compliance, and structural integrity:

1. **`outputs/evaluation/metrics.json`:**
   - Serialized format: UTF-8 JSON.
   - Accuracy, classification report dictionary, and raw $5 \times 5$ confusion matrix verified.
   - Exact numerical identity confirmed between JSON values and runtime evaluation logs.
2. **`outputs/evaluation/confusion_matrix.png`:**
   - Image Format: PNG, 4-channel RGBA.
   - Image Dimensions: $2400 \times 1800$ pixels.
   - File Size: 109,910 bytes.
   - Visual Verification: High-contrast Blues heatmap, fully legible class labels on both axes, numeric count overlay in each cell, and normalized colorbar present.

---

## 6. Static Image Pipeline Inference (Step D)

### 6.1 Inference Setup
- **Command:**
  ```powershell
  .\venv\Scripts\python.exe scripts/predict_image.py --input data/classification-dataset/Car/Image_1.jpg --output outputs/predictions/real_car_prediction.jpg
  ```
- **Input File:** `data/classification-dataset/Car/Image_1.jpg` ($1200 \times 800 \times 3$)
- **Output File:** `outputs/predictions/real_car_prediction.jpg` ($1200 \times 800 \times 3$, 227,701 bytes)

### 6.2 Inference Results
- **Detections:** 1 vehicle detected.
- **Stage 1 (YOLOv8n):** Detected `car` with confidence **`0.87`** (0.8659) at bounding box coordinates `(x1=0, y1=235, x2=1188, y2=644)`.
- **Stage 2 (MobileNetV2):** Classified crop as **`Car`** with confidence **`1.00`** (0.9999).
- **Visualization Check:** The output image confirmed accurate bounding box overlay, contrasting text background, and correctly formatted composite label (`YOLO:car (0.87) | CNN:Car (1.00)`).
- **Execution Time:** ~1.5 seconds.

---

## 7. Multi-Frame Video Pipeline Inference (Step E)

### 7.1 Inference Setup
- **Command:**
  ```powershell
  .\venv\Scripts\python.exe scripts/predict_video.py --input data/test-video/5638-183850414_medium.mp4 --output outputs/predictions/real_video_prediction.mp4 --max-frames 50
  ```
- **Input File:** `data/test-video/5638-183850414_medium.mp4` (1280x720 @ 25.0 FPS, 362 total frames)
- **Output File:** `outputs/predictions/real_video_prediction.mp4` (1280x720 @ 25.0 FPS, 3,568,405 bytes)
- **Processed Limit:** 50 frames.

### 7.2 Performance & Throughput
- **Processed Frames:** Exactly 50 frames.
- **Total Vehicles Detected:** 235 detections across all frames (average: 4.7 vehicles per frame).
- **Processing Duration:** 18.2 seconds.
- **Inference Throughput:** Approximately **2.74 FPS on CPU**.
- **Resource Management:** `cv2.VideoCapture` and `cv2.VideoWriter` instances were released cleanly via `try ... finally` constructs with zero dangling file handles.

### 7.3 Detection & Confidence Distribution

| Predicted Vehicle Class | Total Detections | Percentage of Detections |
| :--- | :---: | :---: |
| **`motorcycle`** | 161 | 68.5% |
| **`Three Wheeler`** | 53 | 22.6% |
| **`Car`** | 21 | 8.9% |
| **`Bus`** | 0 | 0.0% |
| **`Truck`** | 0 | 0.0% |

- **YOLOv8n Detector Confidence:**
  - Min: `0.4043`, Max: `0.7910`, Mean: **`0.5520`**
- **MobileNetV2 Classifier Confidence:**
  - Min: `0.3924`, Max: `1.0000`, Mean: **`0.8676`**

### 7.4 Qualitative Pipeline Observation
A prominent strength of the two-stage architecture was observed in congested traffic frames:
COCO-trained YOLOv8n frequently detects localized motorized three-wheelers (tuk-tuks) under its generic `motorcycle` or `car` classes due to COCO's lack of an explicit `three-wheeler` category. The fine-tuned MobileNetV2 classifier in Stage 2 consistently re-categorized these crops to **`Three Wheeler`** with high confidence ($0.91$ to $1.00$), fulfilling the primary design objective of the two-stage system.

---

## 8. Live-Camera Verification (Step F)

- **Status:** **DEFERRED / NOT TESTED**
- **Rationale:** Physical camera / webcam hardware was not connected or accessible in the current execution environment.
- **Audit Compliance:** In accordance with the validation plan, Step F was **not marked as passed**.
- **Code Readiness:** The CLI script `scripts/live_camera.py`, the root convenience wrapper `live_camera.py`, and the underlying `InferenceRunner.run_camera()` implementation are fully covered by unit tests with mocked video captures and graceful failure assertions (`SystemExit(1)` upon device disconnection). Physical end-to-end verification remains a post-deployment item.

---

## 9. Known Limitations and Future Improvements

### 9.1 Known Limitations
1. **CPU Inference Latency:** At ~2.7 FPS on video, CPU-only execution cannot maintain real-time 25–30 FPS processing without dropping frames or queuing.
2. **Independence of Validation Metric:** The reported **94.90% accuracy** applies **exclusively to the standalone MobileNetV2 classifier** evaluated on clean cropped validation images. It does **not** represent end-to-end pipeline accuracy, which is constrained by YOLOv8n bounding box localization precision, IoU thresholds, and occlusion.
3. **Class Confusion (Bus vs. Truck):** 14 total mutual misclassifications occurred between Bus and Truck. Heavy commercial vehicles with box-like silhouettes represent the most ambiguous category pair for MobileNetV2.
4. **Dataset Imbalance:** Support for `Truck` (95 validation samples) is substantially lower than `Car` (248 samples), contributing to Truck's lower precision (84.47%).
5. **No Built-in Temporal Tracking:** In video processing, detections across consecutive frames are computed independently without object tracking (e.g., DeepSORT / ByteTrack), resulting in potential label flickering on marginal crops.

### 9.2 Recommended Future Enhancements
- **Model Quantization & Acceleration:** Export MobileNetV2 and YOLOv8n to ONNX / OpenVINO / TensorRT or INT8 TFLite to achieve 30+ FPS real-time throughput on edge devices and CPUs.
- **Temporal Tracking Integration:** Integrate ByteTrack to maintain vehicle IDs and apply temporal smoothing to classification confidences over time.
- **Commercial Vehicle Data Expansion:** Augment the training dataset with additional commercial truck and bus variants under varying angles and lighting conditions.
- **Edge Deployment Verification:** Execute Step F on dedicated hardware equipped with an active webcam or RTSP IP camera feed.

---

## 10. Final Validation Summary

| Step | Phase 7 Milestone | Target / Artifact | Result |
| :---: | :--- | :--- | :---: |
| **A** | Automated Test Suite & Regressions | 42 pytest tests + 5 verification scripts | **PASSED** (100%) |
| **B** | Full Validation Dataset Evaluation | 863 validation images, MobileNetV2 | **PASSED** (94.90% Acc) |
| **C** | Evaluation Artifact Review | `metrics.json` & `confusion_matrix.png` | **PASSED** (Verified) |
| **D** | Real Static Image Pipeline Inference | `Car/Image_1.jpg` $\rightarrow$ `real_car_prediction.jpg` | **PASSED** |
| **E** | Real Multi-Frame Video Inference | 50 frames $\rightarrow$ `real_video_prediction.mp4` | **PASSED** |
| **F** | Live Camera Hardware Inference | Webcam device stream | **DEFERRED** |
| **G** | Final Comprehensive Validation Report | `docs/validation/phase7_validation_report.md` | **COMPLETED** |

### Final Conclusion
The vehicle detection and classification system has been successfully verified across Steps A through E without modifying source code, configurations, or model weights during testing. All components adhere to the modular architecture established during Phases 1–6. The project is verified as structurally sound, repeatable, and ready for portfolio presentation or further edge optimization.
