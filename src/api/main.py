"""FastAPI REST Service for California Housing Decision Tree Inference & CQR.

Lead Architect: Guillen Concepcion (Senior Data Scientist & MLOps Engineer)
"""
import time
from pathlib import Path
from typing import Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import joblib
import pandas as pd

from src.api.schemas import CaliforniaHousingFeatures, HousingPredictionResponse, ConformalIntervalDTO
from src.config import settings

app = FastAPI(
    title="California Housing — Decision Tree & CQR Serving API",
    description="Microservicio de inferencia de baja latencia con cuantificación de incertidumbre conforme (CQR 95%).",
    version="1.0.0",
)

MODEL = None
CALIBRATOR = None


@app.on_event("startup")
def load_artifacts():
    global MODEL, CALIBRATOR
    model_path = settings.MODELS_DIR / "champion_model.joblib"
    calib_path = settings.MODELS_DIR / "conformal_calibrator.joblib"
    
    if model_path.exists():
        MODEL = joblib.load(model_path)
    if calib_path.exists():
        CALIBRATOR = joblib.load(calib_path)


@app.get("/", include_in_schema=False)
def root():
    """Redirige automáticamente la raíz hacia la documentación interactiva Swagger."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check():
    """Endpoint de estado del microservicio y estado de calibración conforme."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "conformal_calibrator_loaded": CALIBRATOR is not None,
        "model_type": type(MODEL).__name__ if MODEL else None,
        "conformal_guarantee": "95.0%" if CALIBRATOR else None,
        "environment": "production",
    }


@app.post("/predict", response_model=HousingPredictionResponse)
def predict(features: CaliforniaHousingFeatures):
    """Endpoint de inferencia en tiempo real con intervalos conformes de incertidumbre."""
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo campeón no se encuentra cargado. Ejecute scripts/train.py primero."
        )

    t0 = time.perf_counter()
    df_input = pd.DataFrame([features.dict()])
    
    # 1. Predicción puntual
    pred_raw = float(MODEL.predict(df_input)[0])
    pred_usd = pred_raw * 100_000.0 # En dólares reales
    
    # 2. Intervalo Conforme (CQR 95%)
    interval_dto = None
    if CALIBRATOR is not None and CALIBRATOR.q_val is not None:
        low_raw, up_raw = CALIBRATOR.predict_interval(pred_raw)
        low_usd = float(low_raw) * 100_000.0
        up_usd = float(up_raw) * 100_000.0
        interval_dto = ConformalIntervalDTO(
            lower_usd=round(low_usd, 2),
            upper_usd=round(up_usd, 2),
            coverage_guarantee="95.0%",
            interval_width_usd=round(up_usd - low_usd, 2),
        )

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return HousingPredictionResponse(
        predicted_value_usd=round(pred_usd, 2),
        predicted_value_raw=round(pred_raw, 4),
        conformal_interval_95=interval_dto,
        model_version="1.0.0-champion",
        latency_ms=round(latency_ms, 3),
        status="success",
    )
