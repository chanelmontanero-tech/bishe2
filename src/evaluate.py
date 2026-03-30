"""
Step 6: Model evaluation — generate metrics, comparisons, and visualizations.

Generates:
  1. Classification report (precision, recall, F1-score)
  2. Confusion matrix heatmaps
  3. ROC curves (all models on one plot)
  4. Model comparison bar chart
  5. Feature importance chart
  6. Summary table saved as CSV

Usage:
    python -m src.evaluate
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)


PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
OUTPUT_DIR = os.path.join("output", "figures")

# Model display names for charts
MODEL_NAMES = ["RandomForest", "DecisionTree", "XGBoost", "KNN", "SVM"]

# Use Chinese-compatible font if available, otherwise fallback
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_test_data():
    """Load the test feature matrix and labels."""
    data = np.load(os.path.join(PROCESSED_DIR, "features.npz"))
    return data["X_test"], data["y_test"]


def load_models():
    """Load all trained models."""
    models = {}
    for name in MODEL_NAMES:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"  Loaded {name}")
        else:
            print(f"  [skip] {name} not found at {path}")
    return models


def evaluate_model(model, X_test, y_test):
    """Compute evaluation metrics for a single model."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    # Get probability predictions for ROC curve
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        metrics["auc"] = auc(fpr, tpr)
        metrics["fpr"] = fpr
        metrics["tpr"] = tpr
        metrics["y_proba"] = y_proba

    metrics["y_pred"] = y_pred
    return metrics


def plot_confusion_matrix(y_test, y_pred, model_name, output_dir):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix - {model_name}")

    path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_all_roc_curves(all_metrics, output_dir):
    """Plot ROC curves for all models on one figure."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, metrics in all_metrics.items():
        if "fpr" in metrics:
            ax.plot(
                metrics["fpr"], metrics["tpr"],
                label=f'{name} (AUC={metrics["auc"]:.4f})',
                linewidth=2
            )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Guess")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curves - All Models")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "roc_curves_all.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_model_comparison(results_df, output_dir):
    """Bar chart comparing all models on key metrics."""
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    colors = sns.color_palette("Set2", n_colors=len(results_df))

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        bars = ax.bar(results_df["model"], results_df[metric], color=colors)
        ax.set_title(metric.upper(), fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(results_df["model"], rotation=45, ha="right")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height + 0.01,
                f"{height:.3f}", ha="center", va="bottom", fontsize=9
            )

    fig.suptitle("Model Performance Comparison", fontsize=16, y=1.02)
    path = os.path.join(output_dir, "model_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(output_dir):
    """Plot feature importance from the Random Forest model."""
    rf_path = os.path.join(MODELS_DIR, "RandomForest.pkl")
    feat_path = os.path.join(MODELS_DIR, "selected_features.pkl")

    if not os.path.exists(rf_path) or not os.path.exists(feat_path):
        print("  [skip] Feature importance plot (model or features not found)")
        return None

    rf = joblib.load(rf_path)
    feature_names = joblib.load(feat_path)
    importances = rf.feature_importances_

    # Sort by importance
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(sorted_names)), sorted_importances[::-1], color="steelblue")
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names[::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Feature Importance (Random Forest)")
    ax.grid(True, alpha=0.3, axis="x")

    path = os.path.join(output_dir, "feature_importance.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def evaluate_all():
    """Full evaluation pipeline for all models."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    # Load data and models
    X_test, y_test = load_test_data()
    models = load_models()

    if not models:
        print("  No trained models found. Run training first.")
        return

    # Evaluate each model
    all_metrics = {}
    results = []

    for name, model in models.items():
        print(f"\n  Evaluating {name}:")
        metrics = evaluate_model(model, X_test, y_test)
        all_metrics[name] = metrics

        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1-Score:  {metrics['f1']:.4f}")
        if "auc" in metrics:
            print(f"    AUC:       {metrics['auc']:.4f}")

        # Print full classification report
        print(f"\n    Classification Report:")
        report = classification_report(
            y_test, metrics["y_pred"],
            target_names=["Normal", "Attack"]
        )
        for line in report.split("\n"):
            print(f"    {line}")

        # Confusion matrix plot
        cm_path = plot_confusion_matrix(
            y_test, metrics["y_pred"], name, OUTPUT_DIR
        )
        print(f"    Confusion matrix saved: {cm_path}")

        results.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auc": metrics.get("auc", None),
        })

    # Save results table
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results table saved: {csv_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False, float_format="%.4f"))

    # Generate comparison plots
    print()
    roc_path = plot_all_roc_curves(all_metrics, OUTPUT_DIR)
    print(f"  ROC curves saved: {roc_path}")

    comp_path = plot_model_comparison(results_df, OUTPUT_DIR)
    print(f"  Model comparison chart saved: {comp_path}")

    fi_path = plot_feature_importance(OUTPUT_DIR)
    if fi_path:
        print(f"  Feature importance chart saved: {fi_path}")

    print("\n" + "=" * 60)
    print("Evaluation complete! All figures saved to output/figures/")
    print("=" * 60)

    return results_df


if __name__ == "__main__":
    evaluate_all()
