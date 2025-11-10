"""
API FastAPI simple pour la prédiction du prix des maisons
Inclut :
- Endpoint /predict : prédictions
- Endpoint /health : état de santé de l'API
- Endpoint /version : infos sur le modèle
- Endpoint /metrics : intégration Prometheus
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import joblib
import numpy as np
import os
import hashlib
from datetime import datetime

# =====================================================
# 1️⃣ Initialisation de l'app FastAPI
# =====================================================
app = FastAPI(title="Housing Price Prediction API", version="1.0.0")

# =====================================================
# 2️⃣ Chargement du modèle et du scaler
# =====================================================
MODEL_PATH = "models/random_forest.pkl"
SCALER_PATH = "models/scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError("Modèle ou scaler manquant. Entraîne d'abord le modèle.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# =====================================================
# 3️⃣ Schéma de données pour les requêtes
# =====================================================
class HouseFeatures(BaseModel):
    MedInc: float = Field(..., gt=0, description="Revenu médian du quartier")
    HouseAge: float = Field(..., ge=0)
    AveRooms: float = Field(..., ge=0)
    AveBedrms: float = Field(..., ge=0)
    Population: float = Field(..., ge=1)
    AveOccup: float = Field(..., ge=0)
    Latitude: float = Field(..., ge=32, le=42)
    Longitude: float = Field(..., ge=-125, le=-114)

# =====================================================
# 4️⃣ Métriques Prometheus
# =====================================================
REQUEST_COUNT = Counter("prediction_requests_total", "Nombre total de requêtes /predict")
LAST_PREDICTION = Gauge("last_prediction_timestamp", "Horodatage de la dernière prédiction")

# =====================================================
# 5️⃣ Fonctions utilitaires
# =====================================================
def compute_model_hash(file_path: str) -> str:
    """Retourne un hash unique du modèle pour traçabilité"""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def model_info():
    """Retourne les infos sur le modèle"""
    return {
        "model_version": "1.0.0",
        "model_hash": compute_model_hash(MODEL_PATH),
        "last_updated": datetime.utcfromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat() + "Z"
    }

# =====================================================
# 6️⃣ Endpoints de service
# =====================================================

@app.get("/health")
def health():
    """Vérifie que l’API et le modèle fonctionnent"""
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}

@app.get("/version")
def version():
    """Retourne la version et les métadonnées du modèle"""
    return model_info()

@app.get("/metrics")
def metrics():
    """Expose les métriques Prometheus"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# =====================================================
# 7️⃣ Endpoint principal de prédiction
# =====================================================
@app.post("/predict")
def predict(features: HouseFeatures):
    """
    Prend les caractéristiques d'une maison et renvoie le prix prédit.
    """
    try:
        data = np.array([[features.MedInc, features.HouseAge, features.AveRooms,
                          features.AveBedrms, features.Population, features.AveOccup,
                          features.Latitude, features.Longitude]])

        X_scaled = scaler.transform(data)
        prediction = model.predict(X_scaled)[0]

        REQUEST_COUNT.inc()
        LAST_PREDICTION.set_to_current_time()

        return {"predicted_price": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# =====================================================
# 8️⃣ Lancement local
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
