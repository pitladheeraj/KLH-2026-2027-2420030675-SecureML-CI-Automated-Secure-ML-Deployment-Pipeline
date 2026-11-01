"""Tests for ML Training and Validation Gates."""

import json
import os
import pytest
from src.train import train
from src.validate import validate_data, validate_model_performance


DATA_PATH = "data/heart_disease.csv"
TMP_MODEL_PATH = "models/test_model.joblib"
TMP_METRICS_PATH = "models/test_metrics.json"


def test_data_validation():
    """Verify data quality checks on the provided clinical dataset."""
    assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"
    is_valid = validate_data(DATA_PATH)
    assert is_valid is True, "Dataset should pass all quality gate assertions."


def test_model_training_and_artifacts():
    """Verify training executes and writes expected artifacts."""
    metrics = train(DATA_PATH, TMP_MODEL_PATH, TMP_METRICS_PATH)
    
    assert os.path.exists(TMP_MODEL_PATH), "Model artifact should be generated."
    assert os.path.exists(TMP_METRICS_PATH), "Metrics file should be generated."
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert metrics["accuracy"] > 0.60, "Model baseline accuracy should exceed 60%."


def test_quality_gate_threshold_pass():
    """Quality gate should pass when thresholds are reasonable."""
    is_valid = validate_model_performance(TMP_METRICS_PATH, min_accuracy=0.60, min_f1=0.60)
    assert is_valid is True


def test_quality_gate_threshold_fail():
    """Quality gate must fail when threshold is unrealistically high (ensuring gate works)."""
    is_valid = validate_model_performance(TMP_METRICS_PATH, min_accuracy=0.9999, min_f1=0.9999)
    assert is_valid is False, "Quality gate should strictly reject models below threshold."
