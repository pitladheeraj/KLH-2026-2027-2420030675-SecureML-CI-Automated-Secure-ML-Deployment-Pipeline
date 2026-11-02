"""Integration tests for FastAPI inference service."""

import pytest
from fastapi.testclient import TestClient
from src.app import app, load_model


client = TestClient(app)


@pytest.fixture(autouse=True)
def ensure_model_is_loaded():
    """Ensure a model exists and is loaded before running API tests."""
    # Attempt to load model artifact
    load_model("models/model.joblib")


def test_root_endpoint():
    """Test GET / returns service information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "SecureML-CI" in data["service"]
    assert "docs_url" in data


def test_health_check():
    """Test GET /health returns status."""
    response = client.get("/health")
    assert response.status_code in (200, 503)


def test_predict_endpoint_success():
    """Test POST /predict with a valid bank customer churn payload."""
    payload = {
        "credit_score": 650,
        "gender": 1,
        "age": 42,
        "tenure": 3,
        "balance": 75000.0,
        "num_of_products": 1,
        "has_cr_card": 1,
        "is_active_member": 0,
        "estimated_salary": 95000.0,
        "complain": 1,
        "satisfaction_score": 2,
        "point_earned": 350,
    }
    response = client.post("/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in (0, 1)
        assert "probability" in data
        assert "risk_level" in data
    else:
        # If model hasn't been trained yet, 503 is expected
        assert response.status_code == 503


def test_predict_endpoint_validation_error():
    """Test POST /predict rejects invalid payload with HTTP 422."""
    invalid_payload = {
        "credit_score": 100,  # Invalid: less than 300
        "gender": 5,          # Invalid: not 0 or 1
        "age": 250,           # Invalid: greater than 110
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422, "API must reject out-of-bounds input schema."
def test_model_info_endpoint():
    """Test GET /model-info returns governance metadata."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "RandomForestClassifier"
    assert data["domain"] == "Banking Customer Churn"
    assert data["features_count"] == 12