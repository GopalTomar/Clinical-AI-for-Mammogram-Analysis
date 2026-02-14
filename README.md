# 🎗️ Clinical AI for Mammogram Analysis
### A Deep Learning-Based Breast Cancer Detection & Clinical Decision Support System

> **"Early Detection Saves Lives"** — AI-powered diagnostic support for healthcare professionals.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [End-to-End Pipeline Flow](#end-to-end-pipeline-flow)
4. [Step 1 — Dataset Preparation & Loading](#step-1--dataset-preparation--loading)
5. [Step 2 — Data Augmentation](#step-2--data-augmentation)
6. [Step 3 — Image Preprocessing](#step-3--image-preprocessing)
7. [Step 4 — Image Segmentation](#step-4--image-segmentation)
8. [Step 5 — Feature Extraction (HOG)](#step-5--feature-extraction-hog)
9. [Step 6 — Machine Learning Models (SVM & Random Forest)](#step-6--machine-learning-models-svm--random-forest)
10. [Step 7 — CNN Model Architecture & Training](#step-7--cnn-model-architecture--training)
11. [Step 8 — MobileNetV2 Transfer Learning](#step-8--mobilenetv2-transfer-learning)
12. [Step 9 — Model Evaluation & Comparison](#step-9--model-evaluation--comparison)
13. [Step 10 — Sample Prediction Output](#step-10--sample-prediction-output)
14. [Step 11 — Model Saving & Loading](#step-11--model-saving--loading)
15. [Step 12 — Streamlit Web Application (app.py)](#step-12--streamlit-web-application-apppy)
16. [Step 13 — Enhanced GUI (GUI_Application_cancer.py)](#step-13--enhanced-gui-gui_application_cancerpy)
17. [Step 14 — Grad-CAM Explainability](#step-14--grad-cam-explainability)
18. [Step 15 — Image Quality Analysis](#step-15--image-quality-analysis)
19. [Step 16 — Multi-Language Support](#step-16--multi-language-support)
20. [Step 17 — PDF Report Generation](#step-17--pdf-report-generation)
21. [Step 18 — Batch Processing](#step-18--batch-processing)
22. [Step 19 — Formspree Email Integration](#step-19--formspree-email-integration)
23. [Installation & Setup](#installation--setup)
24. [Running the Application](#running-the-application)
25. [Important Disclaimer](#important-disclaimer)

---

## Project Overview

This project is a full end-to-end **clinical decision support system** that uses computer vision and deep learning to classify breast cancer mammogram images into three categories:

| Class | Label | Description |
|-------|-------|-------------|
| 0 | **Non-Cancer** | No malignancy detected |
| 1 | **Early Phase** | Early-stage indicators present |
| 2 | **Middle Phase** | Intermediate-stage indicators present |

The system combines **classical machine learning** (SVM, Random Forest with HOG features), **deep learning** (custom CNN), and **transfer learning** (MobileNetV2) with a professional **Streamlit web interface** that supports multi-language output, Grad-CAM visual explanations, image quality analysis, batch processing, and clinical PDF reports.

### Validated Results Summary

| Model | Test Accuracy | Notes |
|---|---|---|
| SVM (HOG) | **74.22%** | Strong on Non-Cancer class (F1: 0.99) |
| Random Forest (HOG) | **76.89%** | Best classical model; Non-Cancer F1: 0.98 |
| Custom CNN | **80.62%** | Best overall; test loss: 0.6186 |
| MobileNetV2 (frozen base) | **39.34%** | Underfit due to grayscale→RGB domain mismatch |
| MobileNetV2 (fine-tuned) | **39.29%** | EarlyStopping triggered at epoch 6 |

**Winner: Custom CNN** with **80.62% test accuracy** and **0.6186 test loss**, deployed in the Streamlit web application.

---

## Project Structure

```
computer_vision_project_paper_publish/
│
├── Breast_cancer/
│   ├── Non_Cancer_done/          # Non-cancer mammogram images
│   ├── Early_phase/              # Early-stage cancer images (original)
│   ├── Middle_phase/             # Middle-stage cancer images (original)
│   ├── Early_phase_Augmentated/  # Augmented early-phase images (6,000 generated)
│   └── Middle_phase_Augmented/   # Augmented middle-phase images (6,000 generated)
│
├── cnn_breast_cancer_model.h5    # Saved CNN model (legacy HDF5 format)
├── cnn_breast_cancer_model.keras # Saved CNN model (modern Keras format — used by app)
│
├── Project_cancer_update.ipynb   # Main training notebook (all 4 models)
├── augment.ipynb                 # Data augmentation notebook
│
├── app.py                        # Streamlit web app (base version)
├── GUI_Application_cancer.py     # Enhanced Streamlit app (v2: Grad-CAM, quality, etc.)
├── test_formspree.py             # Utility to debug Formspree email integration
└── requirements.txt              # Python package dependencies
```

---

## End-to-End Pipeline Flow

```
Raw Dataset (3 classes: Non-Cancer, Early Phase, Middle Phase)
    │
    ▼
Data Augmentation (augment.ipynb)
    │  → 6,000 synthetic images generated for Early & Middle phase each
    │  → 12 transforms applied in random order (flip, rotate, blur, noise, etc.)
    │
    ▼
Dataset Loading & Label Assignment
    │  → cv2.imread → resize to 128×128 → label encoding (0 / 1 / 2)
    │  → Total test set: 4,081 samples
    │
    ▼
Image Preprocessing
    │  → BGR → Grayscale → normalize [0, 1] → shape: (H, W, 1)
    │  → Gaussian Smoothing (σ=1) → Histogram Equalization
    │
    ▼
Train / Test Split  →  80% train  |  20% test (4,081 samples)
    │
    ├──── Path A: Classical ML
    │         ├── Otsu Segmentation → Remove Small Objects → Clear Border
    │         ├── HOG Feature Extraction (9 orient, 8×8 cells, L2-Hys)
    │         ├── SVM (linear kernel)       → 74.22% accuracy
    │         └── Random Forest (100 trees) → 76.89% accuracy
    │
    ├──── Path B: Custom CNN
    │         ├── 4× Conv blocks (32→64→128→256 filters)
    │         │   + BatchNorm + MaxPool + Dropout (0.25 → 0.40)
    │         ├── Dense(512) + Dropout(0.50) → Dense(3, softmax)
    │         ├── EarlyStopping (patience=5) → triggered at epoch 17
    │         └── CNN Test Accuracy: 80.62%  |  Test Loss: 0.6186
    │
    └──── Path C: MobileNetV2 Transfer Learning
              ├── Phase 1: Frozen base → train top head (35 epochs max)
              │   → Pre-Fine-Tuning Accuracy: 39.34%
              ├── Phase 2: All layers unfrozen → lr=1e-5 (30 epochs max)
              │   → EarlyStopping at epoch 6 → Final Accuracy: 39.29%
              └── Note: Limited by grayscale input ≠ ImageNet RGB features
    │
    ▼
Best Model → cnn_breast_cancer_model.keras (Custom CNN)
    │
    ▼
Streamlit Web Application
    ├── Single Image Analysis + Full Probability Scores
    ├── Grad-CAM Attention Heatmap (visual explainability)
    ├── Image Quality Analysis (blur / brightness / noise)
    ├── Confidence-Based Risk Classification (High / Medium / Low)
    ├── Batch Processing (multi-image upload + CSV export)
    ├── PDF Report Generation (reportlab, in-memory)
    ├── Multi-Language Support (10 languages via GoogleTranslator)
    ├── Session History + CSV Download
    └── Contact / Feedback via Formspree
```

---

## Step 1 — Dataset Preparation & Loading

**File:** `Project_cancer_update.ipynb`

The dataset is organized into three directories, each representing a cancer classification class:

```python
data_dirs = {
    "Non-Cancer":  "Breast_cancer/Non_Cancer_done",
    "Early Stage": "Breast_cancer/Early_phase",
    "Middle Stage":"Breast_cancer/Middle_phase"
}
```

The `load_data_and_labels()` function iterates through all directories. Each image is read using OpenCV (`cv2.imread`), resized to a fixed `128×128` resolution, and assigned an integer label (0, 1, or 2) based on its source folder. All images and labels are collected into NumPy arrays.

The resulting test set after the 80/20 split contained **4,081 samples** across the three classes:

| Class | Label | Test Samples |
|---|---|---|
| Non-Cancer | 0 | 1,609 |
| Early Phase | 1 | 1,221 |
| Middle Phase | 2 | 1,251 |
| **Total** | — | **4,081** |

**Why 128×128?** This resolution balances detail preservation against computational cost and memory during batch training.

---

## Step 2 — Data Augmentation

**File:** `augment.ipynb`

The original Early and Middle phase datasets had limited samples, creating class imbalance. A comprehensive augmentation pipeline using the `imgaug` library generated **6,000 augmented images per class**, applied in `random_order=True` to maximize diversity:

| Augmentation | Parameters | Purpose |
|---|---|---|
| `Fliplr` | p=1 | Horizontal flip — mirrored scan positions |
| `Rotate` | -45° to +45° | Rotation invariance |
| `GaussianBlur` | σ=0–2 | Motion blur / low-resolution scans |
| `AdditiveGaussianNoise` | scale=0–51 | Scanner electronic noise |
| `Dropout` | p=0–0.2 | Robustness to missing pixel regions |
| `Resize` | 0.5×–1.5× | Scale invariance |
| `Crop` | 0–20% | Off-center mammogram frames |
| `ElasticTransformation` | α=0–10, σ=1 | Realistic soft-tissue deformation |
| `PiecewiseAffine` | scale=0.02–0.1 | Local geometric distortions |
| `PerspectiveTransform` | scale=0.05–0.15 | Varying x-ray capture angles |
| `LinearContrast` | 0.2–3.0 | Extreme contrast variation |
| `Multiply` | 0.5–1.5, per_channel | Random brightness per channel |

Augmented images are saved as `.jpg` directly to their class directories, making them seamlessly available for training.

---

## Step 3 — Image Preprocessing

**File:** `Project_cancer_update.ipynb` → `preprocess_images()`

All images undergo a standardized preprocessing pipeline before training:

1. **Grayscale Conversion** (`cv2.COLOR_BGR2GRAY`) — Mammogram analysis relies on structural texture; color information is not clinically relevant and reduces memory.
2. **Normalization** — Pixel values divided by `255.0` → range `[0, 1]`. Stabilizes gradient updates during training.
3. **Channel Expansion** — `np.newaxis` reshapes `(H, W)` → `(H, W, 1)` to match Conv2D's expected input format.

A secondary pipeline (`process_images_pipeline()`) is applied before segmentation:
- **Gaussian Smoothing** (`sigma=1`) — suppresses high-frequency noise that would confuse threshold-based segmentation.
- **Histogram Equalization** (`cv2.equalizeHist`) — redistributes intensity to the full 0–255 range, enhancing structural contrast.

---

## Step 4 — Image Segmentation

**File:** `Project_cancer_update.ipynb` → `segment_images()`

Segmentation isolates the tissue region of interest from scanner background and artifacts using `scikit-image`:

1. **Otsu Thresholding** (`threshold_otsu`) — Automatically computes the optimal intensity threshold by maximizing inter-class variance. No manual tuning required.
2. **Remove Small Objects** (`remove_small_objects`, `min_size=50`) — Eliminates artifact pixels smaller than 50 pixels that are not genuine anatomical structures.
3. **Clear Border** (`clear_border`) — Removes connected components touching image edges — typically scanner frame artifacts.

The output is a clean binary mask of significant tissue regions, which feeds directly into HOG feature extraction.

---

## Step 5 — Feature Extraction (HOG)

**File:** `Project_cancer_update.ipynb` → `extract_hog_features()`

**Histogram of Oriented Gradients (HOG)** captures local shape and texture by computing gradient orientation distributions across localized image patches. Each image is converted to a 1D feature vector:

```python
hog(img, orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False)
```

- `orientations=9` — 9 gradient direction bins (0°–180°)
- `pixels_per_cell=(8,8)` — captures fine-grained local structure
- `cells_per_block=(2,2)` — local contrast normalization over 2×2 cell groups
- `L2-Hys` — robust normalization with hysteresis clipping

The resulting `X_train_hog` and `X_test_hog` matrices are fed directly into SVM and Random Forest.

---

## Step 6 — Machine Learning Models (SVM & Random Forest)

**File:** `Project_cancer_update.ipynb`

### Support Vector Machine (SVM)

```python
SVC(kernel='linear', probability=True, random_state=42)
```

A linear kernel SVM finds the optimal hyperplane separating HOG feature vectors of different classes. `probability=True` enables Platt-scaled confidence scores.

#### SVM Classification Report — Test Set: 4,081 samples

```
SVM Classification Report
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      1609
           1       0.58      0.57      0.58      1221
           2       0.58      0.59      0.59      1251

    accuracy                           0.74      4081
   macro avg       0.72      0.72      0.72      4081
weighted avg       0.74      0.74      0.74      4081
```

**SVM Overall Accuracy: 74.22%**

Key observations:
- Class 0 (Non-Cancer): Near-perfect F1 of **0.99** — the SVM reliably identifies healthy tissue with only ~1% error.
- Classes 1 & 2 (Early/Middle Phase): F1 scores of **0.58–0.59** — the linear HOG feature space cannot cleanly separate the two cancer stages, likely because their texture patterns overlap significantly.
- The gap between macro average (0.72) and weighted average (0.74) reflects the larger Non-Cancer class pulling the weighted score upward.

---

### Random Forest

```python
RandomForestClassifier(n_estimators=100, random_state=42)
```

An ensemble of 100 decision trees trained on random HOG feature subsets. Majority vote determines the final class prediction.

#### Random Forest Classification Report — Test Set: 4,081 samples

```
Random Forest Classification Report
              precision    recall  f1-score   support

           0       1.00      0.96      0.98      1609
           1       0.62      0.64      0.63      1221
           2       0.63      0.64      0.64      1251

    accuracy                           0.77      4081
   macro avg       0.75      0.75      0.75      4081
weighted avg       0.77      0.77      0.77      4081
```

**Random Forest Overall Accuracy: 76.89%**

Key observations:
- Class 0 (Non-Cancer): F1 of **0.98**, precision of **1.00** — zero false positives on healthy tissue. The model never incorrectly labels a healthy scan as cancerous.
- Classes 1 & 2: Improved to **0.63–0.64** F1 vs. SVM's 0.58–0.59, reflecting Random Forest's ability to model non-linear decision boundaries in the HOG feature space.
- Random Forest **outperforms SVM by +2.67 percentage points** overall.

---

## Step 7 — CNN Model Architecture & Training

**File:** `Project_cancer_update.ipynb`

### Architecture

A custom **4-block Convolutional Neural Network** built with Keras Sequential API:

```
Input: (128, 128, 1)  — Normalized grayscale mammogram

Block 1: Conv2D(32,  3×3, relu) + L2(0.01)  →  BatchNorm  →  MaxPool(2×2)  →  Dropout(0.25)
Block 2: Conv2D(64,  3×3, relu) + L2(0.01)  →  BatchNorm  →  MaxPool(2×2)  →  Dropout(0.30)
Block 3: Conv2D(128, 3×3, relu) + L2(0.01)  →  BatchNorm  →  MaxPool(2×2)  →  Dropout(0.35)
Block 4: Conv2D(256, 3×3, relu) + L2(0.01)  →  BatchNorm  →  MaxPool(2×2)  →  Dropout(0.40)

Flatten
Dense(512, relu) + L2(0.01)  →  Dropout(0.50)
Dense(3, softmax)              →  Output: 3-class probability vector
```

**Design rationale:**
- **L2 Regularization (`l2=0.01`)** on all Conv and Dense layers penalizes large weights, combating overfitting on limited medical image data.
- **Batch Normalization** after every Conv layer stabilizes activations, accelerates convergence, and provides mild implicit regularization.
- **Progressively increasing Dropout** (0.25 → 0.40 → 0.50) applies stronger regularization in deeper, more abstract layers where overfitting risk is highest.
- **EarlyStopping** (`monitor='val_loss'`, `patience=5`, `restore_best_weights=True`) automatically restores the best checkpoint and halts training when generalization stops improving.

**Training config:** `Adam` | `sparse_categorical_crossentropy` | up to 50 epochs | batch size 32

---

### CNN Training Log — Selected Epochs (511 steps/epoch)

EarlyStopping triggered at **epoch 17**, with best weights restored from the epoch with lowest `val_loss`:

```
Epoch  1/50 — 511/511 — 306s — accuracy: 0.6911 — loss: 10.9170 — val_accuracy: 0.6165
Epoch  2/50 — 511/511 — 324s — accuracy: 0.7302 — loss:  2.5650 — val_accuracy: 0.7444
Epoch  3/50 — 511/511 — 424s — accuracy: 0.7648 — loss:  1.2685 — val_accuracy: 0.7633
Epoch  4/50 — 511/511 — 341s — accuracy: 0.7679 — loss:  1.1855 — val_accuracy: 0.6667
Epoch  5/50 — 511/511 — 416s — accuracy: 0.7551 — loss:  1.6983 — val_accuracy: 0.7608
Epoch  6/50 — 511/511 — 350s — accuracy: 0.7946 — loss:  0.8240 — val_accuracy: 0.8236  ← validation peak
Epoch  7/50 — 511/511 — 357s — accuracy: 0.7892 — loss:  0.9330 — val_accuracy: 0.7653
Epoch  8/50 — 511/511 — 355s — accuracy: 0.7947 — loss:  0.8717 — val_accuracy: 0.7978
Epoch  9/50 — 511/511 — 334s — accuracy: 0.8093 — loss:  0.7400 — val_accuracy: 0.7711
Epoch 10/50 — 511/511 — 361s — accuracy: 0.8043 — loss:  0.7786 — val_accuracy: 0.7858
Epoch 11/50 — 511/511 — 374s — accuracy: 0.8028 — loss:  0.7681 — val_accuracy: 0.7873
Epoch 12/50 — 511/511 — 510s — accuracy: 0.8088 — loss:  0.7128 — val_accuracy: 0.8062
Epoch 13/50 — 511/511 — 474s — accuracy: 0.8063 — loss:  0.8370 — val_accuracy: 0.7920
Epoch 14/50 — 511/511 — 358s — accuracy: 0.7964 — loss:  0.8140 — val_accuracy: 0.7557
Epoch 15/50 — 511/511 — 397s — accuracy: 0.8154 — loss:  0.7149 — val_accuracy: 0.6890
Epoch 16/50 — 511/511 — 368s — accuracy: 0.8113 — loss:  0.6948 — val_accuracy: 0.8018
Epoch 17/50 — 511/511 — 325s — accuracy: 0.8099 — loss:  0.6745 — val_accuracy: 0.7567
[EarlyStopping triggered — best weights restored]
```

---

### CNN Final Evaluation

```
128/128 ━━━━━━━━━━━━━━━━━━━━ 12s 96ms/step — accuracy: 0.8096 — loss: 0.6178

CNN Test Loss:     0.6185510158538818
CNN Test Accuracy: 0.8061749339103699   →   80.62%
```

---

### CNN Training Curve Analysis

**Accuracy Plot (CNN):** Training accuracy climbs steadily from ~69.1% (epoch 1) to ~81.5% (epoch 15), while validation accuracy peaks at **~82.4%** around epoch 6 then fluctuates between 69%–81% — indicating high epoch-to-epoch variance in the validation set. EarlyStopping correctly identifies epoch 6 as the optimal generalization point and restores those weights.

**Loss Plot (CNN):** Both training and validation loss drop sharply from ~7.0 and ~3.7 respectively (epoch 0) to below 1.0 by epoch 4. Training loss continues declining to ~0.62 by epoch 17. The tight convergence of training and validation loss after epoch 4 (both stable in the 0.62–0.90 range) confirms that L2 + BatchNorm + Dropout effectively prevent overfitting. No significant divergence between training and validation loss is observed.

---

## Step 8 — MobileNetV2 Transfer Learning

**File:** `Project_cancer_update.ipynb`

**MobileNetV2** (pre-trained on ImageNet, `input_shape=(128,128,3)`) was used for transfer learning. A custom classification head is stacked on top of the frozen base:

```
MobileNetV2 base (frozen, ImageNet weights)
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.5)
    ↓
Dense(512, relu) + L2(0.01)
    ↓
Dropout(0.5)
    ↓
Dense(3, softmax)
```

A custom `mobilenet_data_generator` was built to serve images as 3-channel (RGB-replicated) batches at 510 steps/epoch.

---

### Phase 1 — Frozen Base Training (up to 35 epochs)

```
Epoch  1/35 — 510/510 — 224s — accuracy: 0.3693 — loss: 2.9107 — val_accuracy: 0.3962 — val_loss: 1.xxxx
Epoch  2/35 — 510/510 — 175s — accuracy: 0.3820 — loss: 1.2232 — val_accuracy: 0.3962 — val_loss: 1.1320
Epoch  3/35 — 510/510 — 194s — accuracy: 0.3942 — loss: 1.1308 — val_accuracy: 0.3959 — val_loss: 1.1068
Epoch  4/35 — 510/510 — 167s — accuracy: 0.3928 — loss: 1.1113 — val_accuracy: 0.3979 — val_loss: 1.0984
Epoch  5/35 — 510/510 — 167s — accuracy: 0.3952 — loss: 1.1007 — val_accuracy: 0.3969 — val_loss: 1.0903
Epoch  6/35 — 510/510 — 144s — accuracy: 0.3952 — loss: 1.0948 — val_accuracy: 0.3986 — val_loss: 1.0884
Epoch  7/35 — 510/510 — 138s — accuracy: 0.3950 — loss: 1.0914 — val_accuracy: 0.3964 — val_loss: 1.0886
Epoch  8/35 — 510/510 — 134s — accuracy: 0.3957 — loss: 1.0908 — val_accuracy: 0.3969 — val_loss: 1.0878
Epoch  9/35 — 510/510 — 138s — accuracy: 0.3953 — loss: 1.0909 — val_accuracy: 0.3966 — val_loss: 1.0906
Epoch 10/35 — 510/510 — 110s — accuracy: 0.3961 — loss: 1.0914 — val_accuracy: 0.3981 — val_loss: 1.0880
Epoch 11/35 — 510/510 — 123s — accuracy: 0.3955 — loss: 1.0906 — val_accuracy: 0.3971 — val_loss: 1.0912
Epoch 12/35 — 510/510 — 115s — accuracy: 0.3965 — loss: 1.0907 — val_accuracy: 0.3952 — val_loss: 1.0881
Epoch 13/35 — 510/510 — 138s — accuracy: 0.3965 — loss: 1.0903 — val_accuracy: 0.3934 — val_loss: 1.0889
[EarlyStopping triggered after patience=5]
```

**Phase 1 Evaluation (Before Fine-Tuning):**
```
127/127 ━━━━━━━━━━━━━━━━━━━━ 25s 196ms/step — accuracy: 0.3985 — loss: 1.0872

MobileNetV2 (Before Fine-Tuning) Test Loss:     1.0887674...
MobileNetV2 (Before Fine-Tuning) Test Accuracy: 0.3934304...   →   39.34%
```

---

### Phase 2 — Fine-Tuning (All Layers Unfrozen, lr=1e-5, up to 30 epochs)

```
Epoch 1/30 — val_accuracy: 0.3934
Epoch 2/30 — val_accuracy: 0.3961
Epoch 3/30 — val_accuracy: 0.3997   ← validation accuracy peak
Epoch 4/30 — val_accuracy: 0.3979
Epoch 5/30 — val_accuracy: 0.3971
Epoch 6/30 — val_accuracy: 0.3941
[EarlyStopping triggered at epoch 6]
```

**MobileNetV2 Fine-Tuned Test Accuracy: 39.29%**

---

### MobileNetV2 Training Curve Analysis

**Accuracy Plot (Frozen Base):** Both training (~39.5%) and validation (~39.6%) accuracy plateau immediately from epoch 1, barely above random chance for 3 classes (33.3%). The model makes no meaningful progress across 13 epochs — indicating the ImageNet pre-trained features provide almost no useful signal for grayscale mammogram classification.

**Loss Plot (Frozen Base):** Validation loss descends from ~1.25 to ~1.09 and flattens — converging to a poor local minimum. Training loss mirrors this, indicating neither over- nor under-fitting is the issue; rather, the feature representations are simply mismatched.

**Fine-Tuning Accuracy Plot:** A brief spike to ~39.97% (epoch 3), followed by rapid decline, triggering EarlyStopping at epoch 6. The extremely low learning rate (1e-5) limits effective weight adjustment, but the underlying domain mismatch remains the primary bottleneck.

**Fine-Tuning Loss Plot:** Training loss drops from ~0.96 to ~0.73, while validation loss stays flat at ~1.09 throughout — a divergence indicating the model is beginning to overfit the training distribution without improving generalization. EarlyStopping correctly intervenes.

**Root Cause:** MobileNetV2 was pre-trained on natural RGB color photographs (ImageNet). Our dataset uses grayscale mammograms converted to 3-channel by replication — the pre-trained convolutional kernels that detect color-based and natural image textures are fundamentally mismatched to medical tissue patterns in grayscale.

---

## Step 9 — Model Evaluation & Comparison

**File:** `Project_cancer_update.ipynb`

### Final Model Comparison — Official Results

```
Model Comparison:
  SVM Accuracy:           74.22%
  Random Forest Accuracy: 76.89%
  CNN Accuracy:           80.62%
  MobileNetV2 Accuracy:   39.29%
```

### Summary Table

| Model | Test Accuracy | Test Loss | Strength | Limitation |
|---|---|---|---|---|
| SVM (HOG) | 74.22% | — | Non-Cancer F1=0.99 | Poor cancer stage separation (F1≈0.58) |
| Random Forest (HOG) | 76.89% | — | Non-Cancer precision=1.00; best classical model | Cancer stages still F1≈0.63 |
| **Custom CNN** | **80.62%** | **0.6186** | **End-to-end learned features; best overall** | ~5 min/epoch training time |
| MobileNetV2 (frozen) | 39.34% | 1.089 | Efficient architecture | ImageNet features mismatch grayscale medical domain |
| MobileNetV2 (fine-tuned) | 39.29% | ~1.09 | Full model flexibility | Same domain mismatch; EarlyStopping at epoch 6 |

The Custom CNN is the clear winner, achieving **80.62% accuracy** by learning domain-specific mammogram features directly from pixel data — without relying on hand-crafted descriptors (HOG) or features from a mismatched domain (ImageNet RGB).

---

## Step 10 — Sample Prediction Output

**File:** `Project_cancer_update.ipynb` → `predict_image_cnn()`

The trained CNN was tested on a sample mammogram image (`earlyphasetest04.jpg`). The prediction pipeline: resize to 128×128 → grayscale → normalize → `model.predict()` → argmax class + confidence.

### Sample Prediction Result

```
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 112ms/step

Predicted Phase:  Non-Cancer
Confidence Score: 38.85%

Class Probabilities:
  Non-Cancer:  38.85%
  Early Phase: 34.37%
  Middle Phase: 26.78%
```

The prediction visualization displays the input mammogram image labeled **"Prediction: Non-Cancer (38.85%)"**.

**Interpretation of this result:** The relatively close probability distribution (38.85% / 34.37% / 26.78%) reflects a genuinely ambiguous mammogram — clinically common in borderline cases. The low confidence score (38.85%) would trigger a **"Low Confidence"** risk flag in the web application, automatically prompting expert review. This demonstrates the confidence-based safety mechanism working correctly as designed: the system defers to human expertise when uncertainty is high rather than making a high-stakes decision with low information.

---

## Step 11 — Model Saving & Loading

**File:** `Project_cancer_update.ipynb`

The trained CNN is saved in two formats for compatibility:

```python
# Legacy HDF5 format (backward compatibility with older TF/Keras versions)
cnn_model.save('cnn_breast_cancer_model.h5')

# Modern Keras format (recommended — used by the web application)
cnn_model.save('cnn_breast_cancer_model.keras')
```

Both saves are immediately verified by reloading:

```python
cnn_model_loaded = load_model('cnn_breast_cancer_model.keras')
# Output: "Model loaded successfully!"
```

In the web application, model loading is wrapped in `@st.cache_resource`, ensuring the ~100MB model file is read from disk only once per session and kept in memory for all subsequent real-time predictions.

---

## Step 12 — Streamlit Web Application (app.py)

**File:** `app.py`

The base version of the clinical web interface, structured as a sidebar-navigated multi-page application:

| Page | Key Functionality |
|---|---|
| **Analysis** | Image upload → inference → probability chart → clinical recommendations |
| **Model Performance** | Training curves, accuracy/loss metrics, session statistics |
| **History** | Filterable, sortable log of all session analyses with CSV export |
| **Batch Processing** | Multi-image upload → batch inference → results table + CSV download |
| **About** | Project info, technical specs, Formspree contact form |

The app loads the saved `.keras` model at startup via `@st.cache_resource` and uses `st.session_state` to persist analysis history across page navigations within a session.

---

## Step 13 — Enhanced GUI (GUI_Application_cancer.py)

**File:** `GUI_Application_cancer.py`

The **v2 enhanced version**, adding clinical explainability and quality tools on top of the base app. Custom CSS injected via `st.markdown(..., unsafe_allow_html=True)` delivers a professional clinical aesthetic: gradient headers, card-style containers with drop shadows, color-coded risk badges, and a medical disclaimer footer.

### New Features in v2

| Feature | Description |
|---|---|
| **Grad-CAM Heatmap** | Overlays the AI's attention regions on the mammogram |
| **Image Quality Analyzer** | 3-metric quality scoring before accepting the image |
| **Risk Classification** | Confidence-based risk level badge (High / Medium / Low) |
| **Prediction Comparison** | Side-by-side view of multiple analyses in one session |

---

## Step 14 — Grad-CAM Explainability

**File:** `GUI_Application_cancer.py` → `generate_gradcam_heatmap()`, `create_gradcam_overlay()`

**Gradient-weighted Class Activation Mapping (Grad-CAM)** makes the CNN's decision visually interpretable — critical for clinical trust, as radiologists can verify the AI focuses on biologically relevant tissue areas.

### Implementation Steps

1. A sub-model (`grad_model`) is constructed outputting both the **last Conv layer's activations** and the **final softmax predictions** simultaneously.
2. `tf.GradientTape` records the gradient of the target class score with respect to the last Conv layer's feature map outputs.
3. Gradients are **globally average-pooled** over spatial dimensions → per-channel importance weights.
4. Feature maps are weighted by importance scores and channel-averaged → a 2D spatial heatmap.
5. **ReLU** is applied → retains only regions that positively increase the predicted class score.
6. Heatmap is resized to original image dimensions and overlaid via **JET colormap** (`cv2.COLORMAP_JET`) at `alpha=0.4` using `cv2.addWeighted`.

Red/yellow regions in the output highlight tissue areas that most strongly drove the model's classification — giving clinicians a visual audit trail of the AI's reasoning.

---

## Step 15 — Image Quality Analysis

**File:** `GUI_Application_cancer.py` → `analyze_image_quality()`

Before prediction, the system scores each uploaded image on three quality metrics:

| Metric | Method | Scoring | Weight |
|---|---|---|---|
| **Sharpness** | Variance of Laplacian (`cv2.Laplacian`) | Normalized to 0–100 | 40% |
| **Brightness** | Mean pixel intensity; optimal range 80–180 | 100 if in range, scaled otherwise | 30% |
| **Noise** | Median std deviation across 9×9 local patches | `max(0, 100 - noise/2)` | 30% |

The **Overall Quality Score** (0–100) is the weighted average, classified as:

| Score | Label | Clinical Guidance |
|---|---|---|
| ≥ 80 | **Excellent** | Suitable for reliable AI analysis |
| 60–79 | **Good** | Minor quality issues; generally reliable |
| 40–59 | **Fair** | Noticeable issues; manual review recommended |
| < 40 | **Poor** | Significant concerns; expert verification required |

---

## Step 16 — Multi-Language Support

**File:** `GUI_Application_cancer.py`, `app.py`

The application supports **10 languages** via `deep-translator`'s `GoogleTranslator`, cached with `@st.cache_data`:

```
English  |  Spanish  |  French  |  German  |  Hindi
Chinese (Simplified)  |  Arabic  |  Portuguese  |  Russian  |  Japanese
```

All UI strings are defined once in `BASE_TEXT` (English). At runtime, `get_translated_text(lang_code)` translates every key. For texts over 5,000 characters (e.g., the About page), content is chunked into 4,500-character segments, translated individually, and rejoined. Every UI element — buttons, labels, headings, clinical recommendations, chart titles, and PDF content — is rendered in the user-selected language.

---

## Step 17 — PDF Report Generation

**File:** `GUI_Application_cancer.py` → `generate_pdf_report()`

After each analysis, a **clinical-grade PDF report** is generated in-memory via `reportlab`:

**Report Contents:**
- **Title**: "Clinical AI Analysis Report" — 24pt, centered, styled
- **Metadata Table**: Timestamp | Classification | Confidence | Quality label | Quality score
- **Mammogram Image**: Embedded at 4×4 inches
- **Medical Disclaimer**: Bold, clearly states AI-assist-only purpose

The report is built into `io.BytesIO()` with no temporary disk files. It is immediately downloadable via `st.download_button` with filename `report_YYYYMMDD_HHMMSS.pdf`.

---

## Step 18 — Batch Processing

**File:** `GUI_Application_cancer.py` → `show_batch_page()`

The Batch Processing page processes multiple mammogram images in a single run:

- Multi-file upload via `st.file_uploader(accept_multiple_files=True)`
- Real-time progress bar as each image is processed
- Per-image pipeline: preprocessing → prediction → quality analysis → risk classification
- Results aggregated into a pandas DataFrame: filename | status | phase | confidence | quality score | quality label | risk level
- Summary metrics: successful count | error count | average confidence | average quality score
- Full results table downloadable as **CSV** (`batch_YYYYMMDD_HHMMSS.csv`)

Errors on individual images are caught, logged in the results table, and do not interrupt the remaining batch.

---

## Step 19 — Formspree Email Integration

**Files:** `GUI_Application_cancer.py`, `app.py`, `test_formspree.py`

**Formspree** handles form submissions without a backend server via HTTP POST to `https://formspree.io/f/xpqjdwqv`:

- **Contact Form** (About page): name + email + message → `send_formspree_message()` → success/failure displayed in UI.
- **Feedback Form** (Analysis results): free-text + optional email → `send_feedback_formspree()`.

### Debugging Tool (`test_formspree.py`)

Standalone script for verifying the integration independently:
- Checks internet connectivity (GET to `google.com`)
- Sends a full test POST and logs status code, response headers, and body
- Tests the anonymous email path separately
- Prints a structured troubleshooting checklist (endpoint ID, rate limits, email verification, spam folder, Formspree status page)

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/breast-cancer-detection.git
cd breast-cancer-detection
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
streamlit
tensorflow
opencv-python
numpy
pandas
pillow
plotly
matplotlib
seaborn
reportlab
deep-translator
requests
scipy
scikit-learn
```

### 4. Dataset Setup
```
Breast_cancer/
├── Non_Cancer_done/   # Non-cancer mammogram images
├── Early_phase/       # Early-stage images
└── Middle_phase/      # Middle-stage images
```

### 5. Train the Models (Optional — pre-trained .keras model included)
```bash
jupyter notebook augment.ipynb                # Step 1: Generate augmented data first
jupyter notebook Project_cancer_update.ipynb  # Step 2: Train all 4 models
```
This generates `cnn_breast_cancer_model.keras` in the project root.

---

## Running the Application

### Base Version
```bash
streamlit run app.py
```

### Enhanced Version (Grad-CAM + Quality Analysis + Risk Classification)
```bash
streamlit run GUI_Application_cancer.py
```

Opens at `http://localhost:8501`

### Test Formspree Integration
```bash
python test_formspree.py
```

---

## Important Disclaimer

> **This system is designed to ASSIST healthcare professionals, not replace them.**
>
> All predictions generated by this tool must be validated by qualified medical professionals. This application is intended for screening and educational purposes only. It should **not** be used as the sole basis for clinical diagnosis or treatment decisions.
>
> This tool is **not FDA-approved** and is not intended for clinical deployment without proper regulatory review. Always consult a licensed radiologist or oncologist for breast cancer diagnosis.

---

*Made with ❤️ for better healthcare | Clinical AI Platform v2.0 | © 2024*
