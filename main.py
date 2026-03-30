"""
Machine Learning-Based Network Traffic Anomaly Detection
基于机器学习的网络流量异常检测

Main entry point — runs the full pipeline:
  Step 1: Download the NSL-KDD dataset
  Step 2: Preprocess and clean the data
  Step 3: Feature engineering (scaling, selection, SMOTE)
  Step 4: Train 5 ML models
  Step 5: Evaluate and generate visualizations

Usage:
    python main.py           # Run the full pipeline
    python main.py --step 1  # Run only a specific step (1-5)
"""

import argparse
import sys


def step1_download():
    """Download the NSL-KDD dataset."""
    print("\n" + "#" * 60)
    print("# STEP 1: Download Dataset")
    print("#" * 60)
    from src.download_data import main as download_main
    download_main()


def step2_preprocess():
    """Preprocess the raw data."""
    print("\n" + "#" * 60)
    print("# STEP 2: Data Preprocessing")
    print("#" * 60)
    from src.preprocess import process_and_save
    process_and_save()


def step3_features():
    """Feature engineering."""
    print("\n" + "#" * 60)
    print("# STEP 3: Feature Engineering")
    print("#" * 60)
    from src.features import build_features
    build_features()


def step4_train():
    """Train all models."""
    print("\n" + "#" * 60)
    print("# STEP 4: Model Training")
    print("#" * 60)
    from src.train import train_all_models
    train_all_models()


def step5_evaluate():
    """Evaluate and visualize results."""
    print("\n" + "#" * 60)
    print("# STEP 5: Evaluation & Visualization")
    print("#" * 60)
    from src.evaluate import evaluate_all
    evaluate_all()


def main():
    parser = argparse.ArgumentParser(
        description="ML-Based Network Traffic Anomaly Detection"
    )
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3, 4, 5],
        help="Run only a specific step (1=download, 2=preprocess, "
             "3=features, 4=train, 5=evaluate)"
    )
    args = parser.parse_args()

    steps = {
        1: step1_download,
        2: step2_preprocess,
        3: step3_features,
        4: step4_train,
        5: step5_evaluate,
    }

    print("=" * 60)
    print("  ML-Based Network Traffic Anomaly Detection")
    print("  基于机器学习的网络流量异常检测")
    print("=" * 60)

    if args.step:
        # Run single step
        steps[args.step]()
    else:
        # Run all steps
        for step_num in sorted(steps.keys()):
            steps[step_num]()

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
