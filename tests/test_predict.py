import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def test_model_files_exist():
    """Vérifie la présence des fichiers modèles"""
    assert os.path.exists("models/random_forest.pkl")
    assert os.path.exists("models/scaler.pkl")

def test_model_prediction_shape():
    """Teste que le modèle prédit correctement"""
    model = joblib.load("models/random_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")

    X = np.array([[5.0, 20.0, 6.0, 1.0, 1000.0, 3.0, 34.0, -118.0]])
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    assert isinstance(model, RandomForestRegressor)
    assert y_pred.shape == (1,)
