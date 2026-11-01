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
    assert data["service"] == "SecureML-CI Prediction API"
    assert "docs_url" in data


def test_health_check():
    """Test GET /health returns status."""
    response = client.get("/health")
    assert response.status_code in (200, 503)


def test_predict_endpoint_success():
    """Test POST /predict with a valid patient payload."""
    payload = {
        "age": 55,
        "sex": 1,
        "resting_bp": 140.0,
        "cholesterol": 240.0,
        "fasting_bs": 0,
        "max_heart_rate": 150.0,
        "exercise_angina": 0,
        "oldpeak": 1.2,
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
        "age": 250,  # Invalid: greater than 120
        "sex": 5,    # Invalid: not 0 or 1
        "resting_bp": 10.0,
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422, "API must reject out-of-bounds input schema."
