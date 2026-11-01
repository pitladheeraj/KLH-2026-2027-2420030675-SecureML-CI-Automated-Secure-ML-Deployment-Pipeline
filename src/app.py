"""SecureML-CI: Deployment & Serving Module.

FastAPI REST API serving the machine learning model.
Provides:
  - GET  /         : Service details and documentation links
  - GET  /health   : Health & readiness check for orchestrators
  - POST /predict  : Pydantic-validated prediction endpoint with confidence scores
"""

import os
from typing import Optional
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

app = FastAPI(
    title="SecureML-CI Heart Disease Prediction Service",
    description="Containerized, secure REST API for clinical risk classification.",
    version="1.0.0",
)

# Global model holder
model = None


def load_model(path: str = MODEL_PATH):
    """Load model artifact into memory."""
    global model
    if os.path.exists(path):
        model = joblib.load(path)
        print(f"[+] Loaded ML model from: {path}")
    else:
        print(f"[!] Warning: Model artifact not found at {path}")
        model = None
    return model


@app.on_event("startup")
def startup_event():
    """Load model during application bootstrap."""
    load_model()


class PredictionInput(BaseModel):
    """Schema for prediction input with strict validation bounds."""
    age: int = Field(..., ge=1, le=120, description="Age in years (1 - 120)")
    sex: int = Field(..., ge=0, le=1, description="Biological sex (1 = male, 0 = female)")
    resting_bp: float = Field(..., ge=50.0, le=260.0, description="Resting blood pressure in mm Hg")
    cholesterol: float = Field(..., ge=50.0, le=600.0, description="Serum cholesterol in mg/dl")
    fasting_bs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)")
    max_heart_rate: float = Field(..., ge=50.0, le=250.0, description="Maximum heart rate achieved")
    exercise_angina: int = Field(..., ge=0, le=1, description="Exercise induced angina (1 = yes, 0 = no)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise relative to rest")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 58,
                "sex": 1,
                "resting_bp": 130.0,
                "cholesterol": 240.0,
                "fasting_bs": 0,
                "max_heart_rate": 160.0,
                "exercise_angina": 0,
                "oldpeak": 1.2,
            }
        }


class PredictionOutput(BaseModel):
    """Response payload for inference predictions."""
    prediction: int = Field(..., description="Binary classification (1 = Disease, 0 = Normal)")
    probability: float = Field(..., description="Confidence probability for positive class")
    risk_level: str = Field(..., description="Categorical risk assessment (Low / High)")
    model_version: str = "1.0.0"


@app.get("/", tags=["General"])
def root():
    """Service metadata and documentation links."""
    return {
        "service": "SecureML-CI Prediction API",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
def health():
    """Health check endpoint for Docker / Kubernetes probes."""
    is_ready = model is not None
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "unhealthy", "model_loaded": False},
        )
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict(payload: PredictionInput):
    """Run inference against the validated, containerized model."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavailable.",
        )

    features = np.array([
        [
            payload.age,
            payload.sex,
            payload.resting_bp,
            payload.cholesterol,
            payload.fasting_bs,
            payload.max_heart_rate,
            payload.exercise_angina,
            payload.oldpeak,
        ]
    ])

    raw_pred = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    pos_prob = round(float(probabilities[1]), 4)

    risk_level = "High Risk" if raw_pred == 1 else "Low Risk"

    return PredictionOutput(
        prediction=raw_pred,
        probability=pos_prob,
        risk_level=risk_level,
        model_version="1.0.0",
    )
