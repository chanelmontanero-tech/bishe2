"""
Step 4: Feature engineering — scale features and select the most important ones.

This module handles:
  1. Loading processed data
  2. Standardizing features (zero mean, unit variance)
  3. Selecting top-K most relevant features
  4. Applying SMOTE to handle class imbalance
  5. Saving the final feature matrices

Usage:
    python -m src.features
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE


PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"

# Columns that are NOT features (they are labels)
LABEL_COLS = ["label", "label_binary", "label_category"]
TOP_K_FEATURES = 20


def load_processed_data():
    """Load processed train and test CSVs."""
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    # Separate features and labels
    feature_cols = [c for c in train_df.columns if c not in LABEL_COLS]

    X_train = train_df[feature_cols].values
    y_train = train_df["label_binary"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["label_binary"].values

    # Also save category labels for multi-class evaluation
    y_train_cat = train_df["label_category"].values
    y_test_cat = test_df["label_category"].values

    print(f"  Training features shape: {X_train.shape}")
    print(f"  Test features shape:     {X_test.shape}")

    return (X_train, y_train, y_train_cat,
            X_test, y_test, y_test_cat,
            feature_cols)


def scale_features(X_train, X_test):
    """Standardize features using StandardScaler (fit on train only)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler for future use
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print("  Features scaled (StandardScaler)")
    print(f"  Scaler saved to {MODELS_DIR}/scaler.pkl")

    return X_train_scaled, X_test_scaled


def select_features(X_train, y_train, X_test, feature_names, k=TOP_K_FEATURES):
    """Select the top-K most important features using ANOVA F-test."""
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Get selected feature names
    mask = selector.get_support()
    selected_names = [feature_names[i] for i in range(len(feature_names)) if mask[i]]

    # Get feature scores
    scores = selector.scores_
    feature_scores = sorted(
        zip(feature_names, scores), key=lambda x: x[1], reverse=True
    )

    print(f"\n  Top {k} features selected (ANOVA F-test):")
    for i, (name, score) in enumerate(feature_scores[:k]):
        print(f"    {i+1:2d}. {name:35s} (F-score: {score:.2f})")

    # Save selector
    joblib.dump(selector, os.path.join(MODELS_DIR, "selector.pkl"))
    joblib.dump(selected_names, os.path.join(MODELS_DIR, "selected_features.pkl"))

    return X_train_selected, X_test_selected, selected_names


def apply_smote(X_train, y_train):
    """Apply SMOTE to balance the training data."""
    print(f"\n  Before SMOTE:")
    print(f"    Normal:  {(y_train == 0).sum()}")
    print(f"    Attack:  {(y_train == 1).sum()}")

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"  After SMOTE:")
    print(f"    Normal:  {(y_resampled == 0).sum()}")
    print(f"    Attack:  {(y_resampled == 1).sum()}")

    return X_resampled, y_resampled


def build_features():
    """Full feature engineering pipeline."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print("Feature Engineering")
    print("=" * 60)

    # Load
    (X_train, y_train, y_train_cat,
     X_test, y_test, y_test_cat,
     feature_cols) = load_processed_data()

    # Scale
    print()
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Select top features
    X_train_selected, X_test_selected, selected_names = select_features(
        X_train_scaled, y_train, X_test_scaled, feature_cols
    )

    # Apply SMOTE for balanced training
    X_train_balanced, y_train_balanced = apply_smote(X_train_selected, y_train)

    # Save all processed feature matrices
    np.savez(
        os.path.join(PROCESSED_DIR, "features.npz"),
        X_train=X_train_balanced,
        y_train=y_train_balanced,
        X_test=X_test_selected,
        y_test=y_test,
        y_test_cat=y_test_cat,
    )

    print(f"\n  Features saved to {PROCESSED_DIR}/features.npz")
    print(f"  Final training shape: {X_train_balanced.shape}")
    print(f"  Final test shape:     {X_test_selected.shape}")
    print("=" * 60)

    return X_train_balanced, y_train_balanced, X_test_selected, y_test


if __name__ == "__main__":
    build_features()
