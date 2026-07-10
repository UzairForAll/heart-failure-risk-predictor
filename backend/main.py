import json
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from explain import explain_prediction
from features import FEATURES

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT.parent / "frontend"
MODEL_PATH = ROOT / "model.pkl"
META_PATH = ROOT / "model_meta.json"


class PredictRequest(BaseModel):
    age: float = Field(..., ge=0, le=120)
    anaemia: int = Field(..., ge=0, le=1)
    creatinine_phosphokinase: float = Field(..., ge=0)
    diabetes: int = Field(..., ge=0, le=1)
    ejection_fraction: float = Field(..., ge=0, le=100)
    high_blood_pressure: int = Field(..., ge=0, le=1)
    platelets: float = Field(..., ge=0)
    serum_creatinine: float = Field(..., ge=0)
    serum_sodium: float = Field(..., ge=0)
    sex: int = Field(..., ge=0, le=1)
    smoking: int = Field(..., ge=0, le=1)
    time: float = Field(..., ge=0)


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model files not found. Run `python train.py` in the backend folder first."
        )

    model = joblib.load(MODEL_PATH)
    metadata = {}
    if META_PATH.exists():
        metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, metadata


app = FastAPI(title="Heart Failure Risk API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, metadata = load_artifacts()


@app.get("/")
def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


@app.get("/health")
def health():
    tuned = metadata.get("tuned_test_metrics", {})
    return {
        "status": "ok",
        "selected_model": metadata.get("selected_model", "unknown"),
        "models_compared": len(metadata.get("models", [])),
        "decision_threshold": metadata.get("decision_threshold", 0.5),
        "test_f1": tuned.get("f1"),
        "test_recall": tuned.get("recall"),
        "test_accuracy": tuned.get("accuracy"),
    }


@app.get("/models")
def models():
    if not metadata:
        raise HTTPException(status_code=404, detail="Model metadata not found")
    return metadata


@app.post("/predict")
def predict(payload: PredictRequest):
    payload_dict = payload.model_dump()
    row = np.array([[payload_dict[feature] for feature in FEATURES]])
    probabilities = model.predict_proba(row)[0]
    probability = float(probabilities[1])
    threshold = float(metadata.get("decision_threshold", 0.5))
    risk = "high" if probability >= threshold else "low"
    factors = explain_prediction(model, metadata, payload_dict)

    return {
        "risk": risk,
        "probability": round(probability, 4),
        "probability_percent": round(probability * 100, 1),
        "threshold": round(threshold, 4),
        "model": metadata.get("selected_model", "unknown"),
        "message": (
            "Higher risk of adverse outcome based on the provided clinical parameters."
            if risk == "high"
            else "Lower risk of adverse outcome based on the provided clinical parameters."
        ),
        "why": factors,
    }
