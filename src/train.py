"""SecureML-CI: Banking Customer Churn Training Module.

Trains a classification model to predict bank customer churn,
evaluates metrics on a held-out test split, and serializes both
the model artifact and evaluation metrics.
"""

import argparse
import json
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split

FEATURE_COLS = [
    "credit_score",
    "gender",
    "age",
    "tenure",
    "balance",
    "num_of_products",
    "has_cr_card",
    "is_active_member",
    "estimated_salary",
    "complain",
    "satisfaction_score",
    "point_earned",
]
TARGET_COL = "target"


def train(data_path: str, model_path: str, metrics_path: str) -> dict:
    """Train the churn classification model, evaluate performance, and save artifacts."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset not found at {data_path}")

    # 1. Load data
    df = pd.read_csv(data_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing '{TARGET_COL}' target column in dataset")

    x = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # 2. Train / Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Model Training: Random Forest
    model = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42
    )
    model.fit(x_train, y_train)

    # 4. Evaluation & 5-Fold Cross-Validation
    y_pred = model.predict(x_test)
    cv_scores = cross_val_score(model, x, y, cv=5)

    # Identify top churn risk predictor
    top_feature_idx = int(model.feature_importances_.argmax())
    top_feature_name = FEATURE_COLS[top_feature_idx]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std_accuracy": round(float(cv_scores.std()), 4),
        "top_churn_factor": top_feature_name,
        "test_samples": int(len(y_test)),
        "feature_names": FEATURE_COLS,
    }

    # 5. Persist artifacts
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[+] Model artifact successfully saved to: {model_path}")

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[+] Training metrics successfully saved to: {metrics_path}")
    print(f"[+] Evaluation Results: {json.dumps(metrics, indent=2)}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SecureML-CI Banking Training")
    parser.add_argument(
        "--data-path",
        default="data/customer_churn.csv",
        help="Path to training CSV file",
    )
    parser.add_argument(
        "--model-path",
        default="models/model.joblib",
        help="Path to save trained model artifact",
    )
    parser.add_argument(
        "--metrics-path",
        default="models/metrics.json",
        help="Path to save evaluation metrics",
    )

    args = parser.parse_args()
    train(args.data_path, args.model_path, args.metrics_path)
