"""SecureML-CI: Validation & Quality Gate Module.

Enforces two levels of quality gates:
1. Data Quality Gate: Schema, missing values, column integrity, and range assertions.
2. Model Quality Gate: Accuracy and F1-score threshold assertions.

If any check fails, the script exits with status code 1, which automatically
terminates the CI/CD pipeline and prevents deployment.
"""

import argparse
import json
import os
import sys
import pandas as pd

EXPECTED_COLUMNS = [
    "age",
    "sex",
    "resting_bp",
    "cholesterol",
    "fasting_bs",
    "max_heart_rate",
    "exercise_angina",
    "oldpeak",
    "target",
]

DEFAULT_MIN_ACCURACY = 0.75
DEFAULT_MIN_F1 = 0.70


def validate_data(data_path: str) -> bool:
    """Validate data integrity, required schema, nulls, and value ranges."""
    print("=" * 60)
    print("STAGE 1: DATA QUALITY GATE")
    print("=" * 60)

    if not os.path.exists(data_path):
        print(f"[FAIL] Dataset file does not exist: {data_path}")
        return False

    df = pd.read_csv(data_path)

    # 1. Row count check
    if len(df) < 20:
        print(f"[FAIL] Dataset is too small ({len(df)} rows). Minimum required is 20 rows.")
        return False

    # 2. Schema check
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"[FAIL] Missing required columns in dataset: {missing_cols}")
        return False

    # 3. Null values check
    null_counts = df[EXPECTED_COLUMNS].isnull().sum()
    if null_counts.sum() > 0:
        print(f"[FAIL] Missing/NaN values detected in dataset:\n{null_counts[null_counts > 0]}")
        return False

    # 4. Domain value ranges
    if not df["age"].between(1, 120).all():
        print("[FAIL] 'age' contains values outside valid domain [1, 120].")
        return False

    if not df["target"].isin([0, 1]).all():
        print("[FAIL] 'target' contains invalid labels. Must be binary {0, 1}.")
        return False

    print(f"[PASS] Data quality gate passed. Verified {len(df)} rows and {len(df.columns)} columns.")
    return True


def validate_model_performance(metrics_path: str, min_accuracy: float, min_f1: float) -> bool:
    """Enforce minimum model quality gates."""
    print("\n" + "=" * 60)
    print("STAGE 2: MODEL QUALITY GATE")
    print("=" * 60)

    if not os.path.exists(metrics_path):
        print(f"[FAIL] Metrics file not found at: {metrics_path}")
        return False

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    accuracy = metrics.get("accuracy", 0.0)
    f1 = metrics.get("f1_score", 0.0)

    print(f"[*] Evaluated Accuracy: {accuracy:.4f} | Minimum Threshold: {min_accuracy:.4f}")
    print(f"[*] Evaluated F1-Score: {f1:.4f} | Minimum Threshold: {min_f1:.4f}")

    if accuracy < min_accuracy:
        print(f"[FAIL] Accuracy gate failed: {accuracy:.4f} < {min_accuracy:.4f}")
        return False

    if f1 < min_f1:
        print(f"[FAIL] F1-Score gate failed: {f1:.4f} < {min_f1:.4f}")
        return False

    print("[PASS] Model performance gate passed! Model meets release criteria.")
    return True


def run_quality_gates(data_path: str, metrics_path: str, min_accuracy: float, min_f1: float) -> int:
    """Run full validation suite and return exit code."""
    data_valid = validate_data(data_path)
    if not data_valid:
        print("\n[CRITICAL] Data Quality Gate failed. Pipeline halted.")
        return 1

    model_valid = validate_model_performance(metrics_path, min_accuracy, min_f1)
    if not model_valid:
        print("\n[CRITICAL] Model Quality Gate failed. Model rejected for deployment.")
        return 1

    print("\n[SUCCESS] ALL QUALITY GATES PASSED. Ready for security scan & container build.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SecureML-CI Quality Gates")
    parser.add_argument("--data-path", default="data/heart_disease.csv", help="Path to CSV dataset")
    parser.add_argument("--metrics-path", default="models/metrics.json", help="Path to metrics JSON")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=DEFAULT_MIN_ACCURACY,
        help="Minimum required accuracy threshold",
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        default=DEFAULT_MIN_F1,
        help="Minimum required F1 score threshold",
    )

    args = parser.parse_args()
    exit_code = run_quality_gates(
        args.data_path, args.metrics_path, args.min_accuracy, args.min_f1
    )
    sys.exit(exit_code)
