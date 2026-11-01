"""SecureML-CI: Banking Serving Module.

FastAPI REST API serving the machine learning model for bank customer churn.
Provides:
  - GET  /         : Service details and documentation links
  - GET  /health   : Health & readiness check for orchestrators
  - POST /predict  : Pydantic-validated prediction endpoint with confidence scores
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

app = FastAPI(
    title="SecureML-CI Banking Churn Prediction Service",
    description="Containerized, secure REST API for bank customer churn risk classification.",
    version="2.0.0",
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
    """Schema for bank customer churn prediction with validation bounds."""
    credit_score: int = Field(..., ge=300, le=900, description="Credit score (300-900)")
    gender: int = Field(..., ge=0, le=1, description="Gender (1 = Male, 0 = Female)")
    age: int = Field(..., ge=18, le=110, description="Customer age (18-110)")
    tenure: int = Field(..., ge=0, le=20, description="Years with the bank (0-20)")
    balance: float = Field(..., ge=0.0, description="Current account balance in USD")
    num_of_products: int = Field(..., ge=1, le=5, description="Number of bank products held (1-5)")
    has_cr_card: int = Field(..., ge=0, le=1, description="Has bank credit card (1 = Yes, 0 = No)")
    is_active_member: int = Field(..., ge=0, le=1, description="Active banking user (1 = Yes, 0 = No)")
    estimated_salary: float = Field(..., ge=0.0, description="Estimated annual salary in USD")
    complain: int = Field(..., ge=0, le=1, description="Has customer filed a complaint (1 = Yes, 0 = No)")
    satisfaction_score: int = Field(..., ge=1, le=5, description="Customer satisfaction score (1-5)")
    point_earned: int = Field(..., ge=0, description="Reward points accumulated")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionOutput(BaseModel):
    """Response payload for bank churn predictions."""
    prediction: int = Field(..., description="Binary classification (1 = Churn Risk, 0 = Retained)")
    probability: float = Field(..., description="Confidence probability for positive class")
    risk_level: str = Field(..., description="Categorical risk assessment (High Churn Risk / Loyal Customer)")
    recommended_action: str = Field(..., description="Actionable bank retention policy")
    model_version: str = "2.0.0"


@app.get("/", tags=["General"])
def root():
    """Service metadata and documentation links."""
    return {
        "service": "SecureML-CI Banking Churn Prediction API",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
        "version": "2.0.0",
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
            payload.credit_score,
            payload.gender,
            payload.age,
            payload.tenure,
            payload.balance,
            payload.num_of_products,
            payload.has_cr_card,
            payload.is_active_member,
            payload.estimated_salary,
            payload.complain,
            payload.satisfaction_score,
            payload.point_earned,
        ]
    ])

    raw_pred = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    pos_prob = round(float(probabilities[1]), 4)

    if raw_pred == 1:
        risk_level = "High Churn Risk"
        recommended_action = "Trigger retention offer: assign dedicated relationship manager."
    else:
        risk_level = "Loyal Customer"
        recommended_action = "Standard engagement: recommend premium rewards program."

    return PredictionOutput(
        prediction=raw_pred,
        probability=pos_prob,
        risk_level=risk_level,
        recommended_action=recommended_action,
        model_version="2.0.0",
    )
