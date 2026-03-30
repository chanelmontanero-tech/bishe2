# 基于机器学习的网络流量异常检测

ML-Based Network Traffic Anomaly Detection

## Project Overview

This project uses multiple machine learning algorithms to detect anomalous
(attack) traffic in network flows. It uses the **NSL-KDD** benchmark dataset
and compares 5 classifiers:

| Model | Description |
|-------|-------------|
| Random Forest | Ensemble of 100 decision trees |
| Decision Tree | Simple, interpretable baseline |
| XGBoost | Gradient boosting (typically best) |
| KNN | K-Nearest Neighbors (k=5) |
| SVM | Support Vector Machine (RBF kernel) |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py

# Or run individual steps:
python main.py --step 1   # Download dataset
python main.py --step 2   # Preprocess data
python main.py --step 3   # Feature engineering
python main.py --step 4   # Train models
python main.py --step 5   # Evaluate & generate plots
```

## Project Structure

```
bishe2/
├── main.py                 # Entry point — runs full pipeline
├── requirements.txt        # Python dependencies
├── src/
│   ├── download_data.py    # Step 1: Download NSL-KDD dataset
│   ├── preprocess.py       # Step 2: Clean & encode data
│   ├── features.py         # Step 3: Scale, select, SMOTE
│   ├── train.py            # Step 4: Train 5 ML models
│   └── evaluate.py         # Step 5: Metrics & visualizations
├── data/
│   ├── raw/                # Original dataset files
│   └── processed/          # Cleaned feature matrices
├── models/                 # Saved trained models (.pkl)
└── output/figures/         # Generated charts and plots
```

## Output

After running the pipeline, you will find in `output/figures/`:

- `confusion_matrix_*.png` — Confusion matrix for each model
- `roc_curves_all.png` — ROC curves comparing all models
- `model_comparison.png` — Bar chart of accuracy/precision/recall/F1
- `feature_importance.png` — Top features from Random Forest
- `results.csv` — Summary table of all metrics
