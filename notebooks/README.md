# Experimental and Educational Notebooks

This directory contains the original Jupyter notebooks developed during the exploratory and prototyping phases of the **Vehicle Type Classification** project.

---

## Overview and Purpose

These notebooks document the progression of the project from raw data exploration to two-stage transfer learning, evaluation, and video inference experiments.

> [!NOTE]
> **Preserved for Research and Education:**
> The notebooks are preserved in their original state as historical research logs and educational references.
> For production use, automated pipelines, testing, and command-line execution, use the modular implementation in [`src/`](file:///c:/Users/Nidula/Documents/Projects/Vehicle-type-classification/src/) and the operational scripts in [`scripts/`](file:///c:/Users/Nidula/Documents/Projects/Vehicle-type-classification/scripts/).

---

## Notebook Guide

The notebooks are organized in logical development sequence:

### 1. `01_data_understanding_eda.ipynb`
- **Stage:** Exploratory Data Analysis (EDA)
- **Purpose:** Inspects raw image files from `data/classification-dataset`, checks class folder distributions, counts available images per vehicle class, and visualizes sample vehicle images with Matplotlib.
- **What You Learn:**
  - Identifying class imbalances (e.g., Cars and Buses have more samples than Trucks and Three Wheelers).
  - Inspecting image dimension variability across classes before resizing.
- **Modular Replacement:** Serves purely as an exploratory and visual record. Dataset paths and class structures are codified in `configs/config.yaml`.

---

### 2. `02_data_loading_preprocessing.ipynb`
- **Stage:** Data Preparation & Metadata Export
- **Purpose:** Scans the 5-class dataset directory structure, defines baseline image parameters ($224 \times 224 \times 3$, batch size 32, validation split 0.2), and generates configuration and class index artifacts.
- **What You Learn:**
  - Creating mapping dictionaries between class string labels and integer indices (`class_indices.json`).
  - Serializing dataset configurations for reproducible experimentation (`config.json`).
- **Modular Replacement:** Replaced by the centralized configuration system in [`src/config/`](file:///c:/Users/Nidula/Documents/Projects/Vehicle-type-classification/src/config/) (`configs/config.yaml`) and the image preprocessor in [`src/data/preprocessor.py`](file:///c:/Users/Nidula/Documents/Projects/Vehicle-type-classification/src/data/preprocessor.py).

---

### 3. `03_model_training.ipynb`
- **Stage:** Model Architecture & Transfer Learning
- **Purpose:** Constructs the custom classification head on top of a pretrained MobileNetV2 base (GlobalAveragePooling2D, BatchNormalization, Dense(128, ReLU), Dropout(0.5), Dense(5, Softmax)), trains the frozen base model (Stage 1), unfreezes the top 30 layers for fine-tuning (Stage 2), and computes balanced class weights.
- **What You Learn:**
  - Designing a two-stage transfer learning strategy for fine-grained image classification.
  - Applying balanced class weights to mitigate dataset class imbalance.
  - Using EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau callbacks to prevent overfitting.
- **Modular Replacement:** Fully modularized into [`src/models/architecture.py`](file:///c:/Users/Nidula/Documents/Projects/Vehicle-type-classification/src/models/architecture.py), [`src/training/trainer.py`](file:///c:/Users/Nidula/Documents/Projects/Vehicle-type-classification/src/training/trainer.py), and the CLI tool [`scripts/train.py`](file:///c:/Users/Nidula/Documents/Projects/Vehicle-type-classification/scripts/train.py).

---

### 4. `04_evaluation.ipynb`
- **Stage:** Validation & Metrics Analysis
- **Purpose:** Loads the best trained MobileNetV2 checkpoint (`models/vehicle_classifier_best.keras`), computes predictions over the validation dataset, generates a Scikit-Learn classification report, and visualizes a confusion matrix heatmap using Seaborn.
- **What You Learn:**
  - Analyzing per-class precision, recall, and F1-scores.
  - Identifying visual classification confusion (e.g., distinguishing Trucks vs. Buses).
- **Modular Replacement:** Modularized into [`src/evaluation/evaluator.py`](file:///c:/Users/Nidula/Documents/Projects/Vehicle-type-classification/src/evaluation/evaluator.py) and the CLI tool [`scripts/evaluate.py`](file:///c:/Users/Nidula/Documents/Projects/Vehicle-type-classification/scripts/evaluate.py).

---

### 5. `05_test_vehicle_cropping.ipynb`
- **Stage:** Object Detection & Bounding Box Extraction
- **Purpose:** Runs YOLO vehicle detection on frames from sample traffic video footage (`data/test-video/`), extracts cropped sub-images of detected vehicles, and displays the cropped regions.
- **What You Learn:**
  - Interfacing with YOLO to extract bounding box coordinates.
  - Extracting region-of-interest (ROI) crops from full image frames.
- **Modular Replacement:** Handled in the two-stage pipeline [`src/pipeline/vehicle_pipeline.py`](file:///c:/Users/Nidula/Documents/Projects/Vehicle-type-classification/src/pipeline/vehicle_pipeline.py) with coordinate boundary clamping and safe crop extraction.

---

### 6. `06_test_classifier_on_crop.ipynb`
- **Stage:** Classifier Crop Validation
- **Purpose:** Feeds extracted vehicle crops directly into the MobileNetV2 classifier to verify that the classification model correctly predicts cropped objects.
- **What You Learn:**
  - Preprocessing cropped vehicle images ($224 \times 224$ normalization) for MobileNetV2 inference.
  - Mapping output softmax probabilities to human-readable class names.
- **Modular Replacement:** Incorporated directly into [`src/models/classifier.py`](file:///c:/Users/Nidula/Documents/Projects/Vehicle-type-classification/src/models/classifier.py) and the unified prediction pipeline [`src/pipeline/vehicle_pipeline.py`](file:///c:/Users/Nidula/Documents/Projects/Vehicle-type-classification/src/pipeline/vehicle_pipeline.py).

---

### 7. `yolo.ipynb`
- **Stage:** Standalone Video Detection Experiment
- **Purpose:** An experimental notebook running YOLO vehicle detection directly on recorded video, annotating boxes and class labels onto video frames using the legacy `draw_detection` helper, and displaying the annotated video clip.
- **What You Learn:**
  - Reading and writing video streams with OpenCV `VideoCapture` and `VideoWriter`.
  - Visualizing detection results directly onto video frames.
- **Modular Replacement:** Fully superseded by the unified inference runner [`src/inference/inference_runner.py`](file:///c:/Users/Nidula/Documents/Projects/Vehicle-type-classification/src/inference/inference_runner.py) and the CLI tool [`scripts/predict_video.py`](file:///c:/Users/Nidula/Documents/Projects/Vehicle-type-classification/scripts/predict_video.py).

---

## Mapping Summary: Notebooks to Modular Code

| Notebook | Focus | Modular Location | Operational CLI Script |
| :--- | :--- | :--- | :--- |
| `01_data_understanding_eda.ipynb` | EDA & Distributions | `configs/config.yaml` | N/A (Exploratory) |
| `02_data_loading_preprocessing.ipynb` | Metadata & Preprocessing | `src/data/`, `src/config/` | N/A (Automated on import) |
| `03_model_training.ipynb` | Transfer Learning & Training | `src/models/`, `src/training/` | `python scripts/train.py` |
| `04_evaluation.ipynb` | Validation & Metrics | `src/evaluation/` | `python scripts/evaluate.py` |
| `05_test_vehicle_cropping.ipynb` | Detection & Cropping | `src/models/detector.py`, `src/pipeline/` | `python scripts/predict_image.py` |
| `06_test_classifier_on_crop.ipynb` | Crop Classification | `src/models/classifier.py`, `src/pipeline/` | `python scripts/predict_image.py` |
| `yolo.ipynb` | Video Detection | `src/inference/` | `python scripts/predict_video.py` |

---

## Running Notebooks Locally or in Google Colab

The notebooks were originally authored in Google Colab and contain cells mounting Google Drive (`drive.mount('/content/drive')`). 

When running them locally in VS Code or JupyterLab:
1. Ensure your virtual environment is active: `.\venv\Scripts\Activate.ps1`.
2. Skip or comment out the Google Colab mount cell.
3. The backward-compatibility bridges ([`src/data_loader.py`](file:///c:/Users/Nidula/Documents/Projects/Vehicle-type-classification/src/data_loader.py) and [`src/detector.py`](file:///c:/Users/Nidula/Documents/Projects/Vehicle-type-classification/src/detector.py)) guarantee that all notebook import statements execute seamlessly.
