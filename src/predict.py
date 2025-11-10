from .io import vectorize_payload
import numpy as np

# Fait une prédiction pour un seul échantillon.
def predict_one(payload: dict, scaler, model) -> float:
    X = vectorize_payload(payload)
    Xs = scaler.transform(X)
    y = model.predict(Xs)
    return float(y[0])