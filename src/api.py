from fastapi import FastAPI
import joblib
import numpy as np
import os
import requests

app = FastAPI(title="Housing Price Prediction API")

MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/random_forest.pkl"

# IDs Google Drive
MODEL_FILE_ID = "1BBC7dQXcMZ0suauHZ28CKWY8f5bL5Wg4"

def download_from_gdrive(file_id: str, destination: str):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    response.raise_for_status()
    with open(destination, "wb") as f:
        f.write(response.content)
    print(f"✅ Téléchargé : {destination}")

# Crée le dossier si nécessaire
os.makedirs(MODEL_DIR, exist_ok=True)

# Téléchargements si absents
if not os.path.exists(MODEL_PATH):
    print("📥 Téléchargement du modèle…")
    download_from_gdrive(MODEL_FILE_ID, MODEL_PATH)

# Chargement
model = joblib.load(MODEL_PATH)
scaler = joblib.load("models/scaler.pkl")

@app.post("/predict")
def predict(data: dict):
    X = np.array([list(data.values())])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    return {"predicted_price": float(pred[0])}
