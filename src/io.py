import joblib
from pathlib import Path
from typing import Tuple
import numpy as np
from .schemas import FEATURE_ORDER

MODELS_DIR = Path("models")

def load_scaler_and_model(
    scaler_path: Path = MODELS_DIR / "scaler.pkl",
    model_path: Path = MODELS_DIR / "random_forest.pkl",
) -> Tuple[object, object]:
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return scaler, model

def vectorize_payload(payload: dict) -> np.ndarray:
    # Respecte l'ordre CANONIQUE défini dans schemas.FEATURE_ORDER
    return np.array([[payload[name] for name in FEATURE_ORDER]], dtype=float)
