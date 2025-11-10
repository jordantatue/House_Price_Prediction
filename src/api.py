from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
import joblib
import numpy as np
import os
import hashlib
import logging
from datetime import datetime
from src.settings import settings


logger = logging.getLogger("uvicorn")

# App FastAPI configurée via settings
app = FastAPI(title=settings.app_name, version=settings.app_version)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Limiter la taille de payload
@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    max_bytes = settings.request_body_max_mb * 1024 * 1024
    body = await request.body()
    if len(body) > max_bytes:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Payload too large (> {settings.request_body_max_mb} MB)"},
        )
    return await call_next(request)


# Chargement du modèle/scaler
MODEL_PATH = settings.model_path
SCALER_PATH = settings.scaler_path
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError("Modèle ou scaler manquant. Entraînez d'abord le modèle.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


class HouseFeatures(BaseModel):
    MedInc: float = Field(..., gt=0)
    HouseAge: float = Field(..., ge=0)
    AveRooms: float = Field(..., ge=0)
    AveBedrms: float = Field(..., ge=0)
    Population: float = Field(..., ge=1)
    AveOccup: float = Field(..., ge=0)
    Latitude: float = Field(..., ge=32, le=42)
    Longitude: float = Field(..., ge=-125, le=-114)


# Prometheus metrics
REQUEST_COUNT = Counter("prediction_requests_total", "Nombre total de requêtes /predict")
LAST_PREDICTION = Gauge("last_prediction_timestamp", "Horodatage de la dernière prédiction")


def _file_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _model_info():
    return {
        "model_version": settings.app_version,
        "model_hash": _file_md5(MODEL_PATH),
        "scaler_hash": _file_md5(SCALER_PATH),
        "last_updated": datetime.utcfromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat()
        + "Z",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/version")
def version():
    # Tests attendent la clé "model_version"
    return {"model_version": settings.app_version}


@app.get("/model_info")
def model_info():
    return _model_info()


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        X = np.array(
            [
                [
                    features.MedInc,
                    features.HouseAge,
                    features.AveRooms,
                    features.AveBedrms,
                    features.Population,
                    features.AveOccup,
                    features.Latitude,
                    features.Longitude,
                ]
            ]
        )
        X_scaled = scaler.transform(X)
        y = model.predict(X_scaled)[0]
        REQUEST_COUNT.inc()
        LAST_PREDICTION.set_to_current_time()
        logger.info("predict ok | lat=%.4f lon=%.4f", features.Latitude, features.Longitude)
        return {"predicted_price": float(y)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
