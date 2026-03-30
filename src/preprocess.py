"""
Step 3: Data preprocessing — load, clean, encode, and split the data.

This module handles:
  1. Loading raw NSL-KDD text files into pandas DataFrames
  2. Cleaning invalid/missing values
  3. Encoding categorical features (protocol_type, service, flag)
  4. Creating binary labels (normal=0, attack=1) and multi-class labels
  5. Saving processed data to data/processed/

Usage:
    python -m src.preprocess
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.download_data import COLUMN_NAMES, ATTACK_CATEGORY


RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")


def load_raw_data(filepath):
    """Load a raw NSL-KDD text file into a DataFrame."""
    print(f"  Loading {filepath} ...")
    df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def clean_data(df):
    """Clean the data: remove duplicates, handle missing/infinite values."""
    original_len = len(df)

    # Replace infinite values with NaN, then drop NaN rows
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Drop the difficulty_level column (not a network feature)
    if "difficulty_level" in df.columns:
        df.drop(columns=["difficulty_level"], inplace=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    removed = original_len - len(df)
    if removed > 0:
        print(f"  Cleaned: removed {removed} rows ({removed/original_len*100:.1f}%)")
    else:
        print(f"  Cleaned: no rows removed")

    return df


def add_labels(df):
    """
    Create two label columns:
      - label_binary: 0 = normal, 1 = attack
      - label_category: normal / DoS / Probe / R2L / U2R
    """
    # Map specific attack names to categories
    df["label_category"] = df["label"].map(
        lambda x: ATTACK_CATEGORY.get(x, "unknown")
    )

    # Binary label
    df["label_binary"] = (df["label_category"] != "normal").astype(int)

    # Remove rows with unknown attack types (if any)
    unknown_count = (df["label_category"] == "unknown").sum()
    if unknown_count > 0:
        print(f"  Warning: {unknown_count} rows with unknown attack type, keeping as attack")
        df.loc[df["label_category"] == "unknown", "label_category"] = "unknown_attack"
        df["label_binary"] = (df["label_category"] != "normal").astype(int)

    # Print distribution
    print("\n  Binary label distribution:")
    for label, count in df["label_binary"].value_counts().items():
        name = "Normal" if label == 0 else "Attack"
        print(f"    {name}: {count} ({count/len(df)*100:.1f}%)")

    print("\n  Category distribution:")
    for cat, count in df["label_category"].value_counts().items():
        print(f"    {cat}: {count} ({count/len(df)*100:.1f}%)")

    return df


def encode_features(df):
    """Encode categorical features using LabelEncoder."""
    categorical_cols = ["protocol_type", "service", "flag"]
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"  Encoded '{col}': {len(le.classes_)} unique values")

    return df, encoders


def process_and_save():
    """Full preprocessing pipeline: load, clean, encode, save."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Process training data
    print("\n" + "=" * 60)
    print("Processing TRAINING data")
    print("=" * 60)
    train_df = load_raw_data(os.path.join(RAW_DIR, "KDDTrain+.txt"))
    train_df = clean_data(train_df)
    train_df = add_labels(train_df)
    train_df, encoders = encode_features(train_df)

    # Process test data
    print("\n" + "=" * 60)
    print("Processing TEST data")
    print("=" * 60)
    test_df = load_raw_data(os.path.join(RAW_DIR, "KDDTest+.txt"))
    test_df = clean_data(test_df)
    test_df = add_labels(test_df)
    # Use the same encoder categories for test data
    for col in ["protocol_type", "service", "flag"]:
        le = encoders[col]
        # Handle unseen categories in test data
        test_df[col] = test_df[col].map(
            lambda x, le=le: le.transform([x])[0] if x in le.classes_
            else -1
        )
    print("  Encoded test data using training encoders")

    # Save processed data
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"  Training data: {train_path} ({len(train_df)} rows)")
    print(f"  Test data:     {test_path} ({len(test_df)} rows)")
    print("=" * 60)


if __name__ == "__main__":
    process_and_save()
