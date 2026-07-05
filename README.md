# WESAD Stress Detection

Machine learning pipeline that classifies a subject's physiological state as **Baseline (relaxed)** or **Stress**, using multimodal wearable sensor data — ECG, EDA (electrodermal activity), and skin temperature — recorded via a chest-mounted RespiBAN sensor.

## Problem Statement

Given raw physiological signals from wearable sensors, the goal is to build a pipeline that:
- Extracts meaningful features (HRV metrics, EDA statistics, temperature trends) from windowed signal segments
- Trains and compares multiple ML models to classify the subject's state
- Validates generalization across *unseen* subjects using Leave-One-Subject-Out (LOSO) cross-validation

## Dataset

This project uses the **[WESAD (Wearable Stress and Affect Detection) dataset](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)**.

> **Note:** The raw dataset is not included in this repo (multi-GB, publicly available). Download it from the link above and update the `RAW_EXTRACT` / `EXTRACT_PATH` variables in the notebook to point to your local copy.

## Approach

1. **Signal processing** — Clean ECG signals and extract R-peaks using `neurokit2`; compute HRV time-domain features (meanNN, RMSSD, SDNN), EDA statistics (mean, std), and temperature statistics per window.
2. **Feature engineering** — Aggregate features across fixed-size windows, labeled as Baseline or Stress per WESAD's protocol.
3. **Model training** — Compare three classifiers:
   - Random Forest
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
4. **Validation** — Leave-One-Group-Out (LOGO) cross-validation, grouped by subject, to ensure the model generalizes to new, unseen individuals rather than overfitting to subject-specific patterns.

## Results

Best-performing model achieved **~92% classification accuracy** (Random Forest), evaluated under subject-independent (LOSO) validation.

## Tech Stack

- Python, NumPy, Pandas
- `neurokit2` for physiological signal processing
- `scikit-learn` for modeling (Random Forest, SVM, KNN) and LOGO cross-validation
- Matplotlib / Seaborn for visualization

## How to Run

1. Download the WESAD dataset and place/extract it locally.
2. Open `wesad_stress_classification.ipynb` in Jupyter or Google Colab.
3. Update the dataset path variables at the top of the notebook.
4. Run all cells — the notebook handles feature extraction, model training, and evaluation end-to-end.

## Author

Shubh Dhariwal
