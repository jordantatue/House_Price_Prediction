# … le haut du fichier reste identique …
from pyexpat import features
from src.settings import settings
app = FastAPI(title=settings.app_name, version=settings.app_version)


from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
import joblib, numpy as np, os, hashlib
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("uvicorn")  # réutilise logger uvicorn

# dans /predict après calcul :
logger.info("predict ok | lat=%.4f lon=%.4f", features.Latitude, features.Longitude)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET","POST"],
    allow_headers=["*"],
)

# Limiter la taille (simple middleware)
@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    max_bytes = settings.request_body_max_mb * 1024 * 1024
    body = await request.body()
    if len(body) > max_bytes:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Payload too large (> {settings.request_body_max_mb} MB)"}
        )
    return await call_next(request)

# Erreurs contrôlées (fallback)
@app.exception_handler(Exception)
async def generic_error_handler(_, exc: Exception):
    return JSONResponse(status_code=400, content={"detail": str(exc)})



MODEL_PATH = "models/random_forest.pkl"
SCALER_PATH = "models/scaler.pkl"
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError("Modèle/scaler manquant. Entraîne d'abord le modèle.")

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
REQUEST_COUNT = Counter("prediction_requests_total", "Total /predict")
LAST_PRED_TS   = Gauge("last_prediction_timestamp", "Dernière prédiction (timestamp)")

def _file_md5(p: str) -> str:
    with open(p, "rb") as f: return hashlib.md5(f.read()).hexdigest()

def _model_info():
    return {
        "api_version": app.version,
        "model_hash": _file_md5(MODEL_PATH),
        "scaler_hash": _file_md5(SCALER_PATH),
        "model_last_updated": datetime.utcfromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat()+"Z"
    }

@app.get("/health")
def health(): return {"status": "ok", "model_loaded": True}

@app.get("/version")
def version(): return {"api_version": app.version}

@app.get("/model_info")
def model_info(): return _model_info()

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        X = np.array([[features.MedInc, features.HouseAge, features.AveRooms,
                       features.AveBedrms, features.Population, features.AveOccup,
                       features.Latitude, features.Longitude]])
        Xs = scaler.transform(X)
        y  = model.predict(Xs)[0]
        REQUEST_COUNT.inc(); LAST_PRED_TS.set_to_current_time()
        return {"predicted_price": float(y)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
