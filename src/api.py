from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="Housing Price Prediction API")

model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.post("/predict")
def predict(data: dict):
    X = np.array([list(data.values())])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    return {"predicted_price": float(prediction[0])}
