"""
Risk Prediction API
--------------------
Wraps the trained XGBoost model behind a simple REST endpoint so that:
  1. The Gemini Enterprise Agent can call it as a "tool" (via OpenAPI schema)
  2. The HTML dashboard can call it directly from JavaScript

Deploy target: Cloud Run (scales to zero -> free tier friendly).
"""

import json
import os
from typing import Optional

import xgboost as xgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from pydantic import BaseModel, Field

BUCKET_NAME = os.environ.get("MODEL_BUCKET", "alzheimer-501509-models")
MODEL_PATH_IN_BUCKET = os.environ.get("MODEL_PATH", "v1/xgb_alzheimers_model.json")
SCHEMA_PATH_IN_BUCKET = os.environ.get("SCHEMA_PATH", "v1/feature_schema.json")
LOCAL_MODEL_PATH = "/tmp/model.json"
LOCAL_SCHEMA_PATH = "/tmp/feature_schema.json"

app = FastAPI(title="Alzheimer's Risk Prediction API", version="1.0")

# Allow the dashboard (any origin, tighten this to your real domain in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model: Optional[xgb.Booster] = None
feature_schema: Optional[dict] = None


def download_from_gcs():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    bucket.blob(MODEL_PATH_IN_BUCKET).download_to_filename(LOCAL_MODEL_PATH)
    bucket.blob(SCHEMA_PATH_IN_BUCKET).download_to_filename(LOCAL_SCHEMA_PATH)


@app.on_event("startup")
def load_model():
    global model, feature_schema
    download_from_gcs()
    booster = xgb.Booster()
    booster.load_model(LOCAL_MODEL_PATH)
    model = booster
    with open(LOCAL_SCHEMA_PATH) as f:
        feature_schema = json.load(f)
    print("Model + schema loaded.")


class PatientInput(BaseModel):
    Age: int
    Gender: int = Field(..., description="0 = female, 1 = male (matches training encoding)")
    Ethnicity: int
    EducationLevel: int
    BMI: float
    Smoking: int
    AlcoholConsumption: float
    PhysicalActivity: float
    DietQuality: float
    SleepQuality: float
    FamilyHistoryAlzheimers: int
    CardiovascularDisease: int
    Diabetes: int
    Depression: int
    HeadInjury: int
    Hypertension: int
    SystolicBP: float
    DiastolicBP: float
    CholesterolTotal: float
    CholesterolLDL: float
    CholesterolHDL: float
    CholesterolTriglycerides: float
    MMSE: float
    FunctionalAssessment: float
    MemoryComplaints: int
    BehavioralProblems: int
    ADL: float
    Confusion: int
    Disorientation: int
    PersonalityChanges: int
    DifficultyCompletingTasks: int
    Forgetfulness: int


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_category: str
    top_contributing_factors: list


def build_feature_row(patient: PatientInput) -> pd.DataFrame:
    """Recreate the exact same feature engineering used in training."""
    row = patient.dict()
    row["age_bucket"] = (row["Age"] // 10) * 10
    row["Gender_0"] = 1 if row["Gender"] == 0 else 0
    row["Gender_1"] = 1 if row["Gender"] == 1 else 0
    del row["Gender"]

    df = pd.DataFrame([row])
    # Reindex to the exact training column order; fill anything missing with 0
    df = df.reindex(columns=feature_schema["feature_names"], fill_value=0)
    return df


def risk_category(prob: float) -> str:
    if prob < 0.33:
        return "Low"
    if prob < 0.66:
        return "Moderate"
    return "High"

@app.get("/")
def read_root():
    return {
        "status": "healthy", 
        "message": "Alzheimer's Risk Prediction API is running. Append /docs to the URL to view interactive documentation."
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    if model is None:
        raise HTTPException(503, "Model not loaded yet")

    df = build_feature_row(patient)
    dmatrix = xgb.DMatrix(df, feature_names=feature_schema["feature_names"])
    prob = float(model.predict(dmatrix)[0])

    # Simple, explainable "top factors" using the model's built-in importance
    importance = model.get_score(importance_type="gain")
    top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    top_factors = [name for name, _ in top]

    return PredictionResponse(
        risk_probability=round(prob, 4),
        risk_category=risk_category(prob),
        top_contributing_factors=top_factors,
    )