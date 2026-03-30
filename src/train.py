"""
Step 5: Model training — train multiple ML models and save them.

Models trained:
  1. Random Forest      — ensemble of decision trees, robust and accurate
  2. Decision Tree      — simple, interpretable baseline
  3. XGBoost            — gradient boosting, typically best performance
  4. K-Nearest Neighbors — distance-based classifier
  5. Support Vector Machine — finds optimal decision boundary

Usage:
    python -m src.train
"""

import os
import time
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"


def get_models():
    """Return a dictionary of models to train."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=20,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,    # needed for ROC curve
            random_state=42,
        ),
    }


def train_all_models():
    """Train all models on the processed feature data."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load features
    data = np.load(os.path.join(PROCESSED_DIR, "features.npz"))
    X_train = data["X_train"]
    y_train = data["y_train"]

    print("=" * 60)
    print("Model Training")
    print("=" * 60)
    print(f"  Training data shape: {X_train.shape}")
    print(f"  Label distribution:  Normal={int((y_train==0).sum())}, "
          f"Attack={int((y_train==1).sum())}")
    print()

    models = get_models()
    trained = {}

    for name, model in models.items():
        print(f"  Training {name} ...")
        start = time.time()

        model.fit(X_train, y_train)

        elapsed = time.time() - start
        print(f"  [done] {name} trained in {elapsed:.1f}s")

        # Save model
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"         Saved to {model_path}")
        print()

        trained[name] = model

    print("=" * 60)
    print(f"All {len(trained)} models trained and saved!")
    print("=" * 60)

    return trained


if __name__ == "__main__":
    train_all_models()
