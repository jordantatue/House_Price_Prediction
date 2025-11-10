import logging, logging.config, yaml, time
from fastapi import FastAPI, Request
import joblib, numpy as np
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

PREDICTIONS = Counter("predictions_total", "Total de prédictions retournées")
PREDICTION_ERRORS = Counter("prediction_errors_total", "Total d'erreurs de prédiction")
LATENCY = Histogram("prediction_latency_seconds", "Latence des prédictions en secondes")
IN_PROGRESS = Gauge("prediction_in_progress", "Prédictions en cours")

# --- Logging config ---
with open("configs/logging.yaml", "r") as f:
    logging.config.dictConfig(yaml.safe_load(f))
logger = logging.getLogger("app")

app = FastAPI(title="Housing Price Predictor")

# --- Versioning simple (affiché dans les logs et /health) ---
MODEL_VERSION = "rf-1.0.0"
DATA_VERSION = "california-housing-2025.11"
model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

# --- Middleware: mesure de latence + logs ---
@app.middleware("http")
async def add_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(f"path={request.url.path} status={response.status_code} elapsed_ms={elapsed_ms:.2f}")
    return response

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "data_version": DATA_VERSION}

@app.get("/ready")
def ready():
    ok = model is not None and scaler is not None
    return {"ready": ok}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
@LATENCY.time()
def predict(data: dict):
    IN_PROGRESS.inc()
    try:
        logger.info(f"predict_received keys={list(data.keys())} model={MODEL_VERSION}")
        X = np.array([list(data.values())], dtype=float)
        X_scaled = scaler.transform(X)
        y = model.predict(X_scaled)
        PREDICTIONS.inc()
        logger.info(f"predict_done value={float(y[0])} model={MODEL_VERSION}")
        return {"predicted_price": float(y[0]), "model_version": MODEL_VERSION}
    except Exception as e:
        PREDICTION_ERRORS.inc()
        logger.exception("prediction_failed")
        return {"error": str(e)}
    finally:
        IN_PROGRESS.dec()