from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

# Ordre canonique des features utilisé par le scaler/modèle
FEATURE_ORDER = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

class PredictRequest(BaseModel):
    MedInc: float = Field(..., ge=0, description="Revenu médian (10k$)")
    HouseAge: float = Field(..., ge=0, le=100)
    AveRooms: float = Field(..., ge=0)
    AveBedrms: float = Field(..., ge=0)
    Population: float = Field(..., ge=0)
    AveOccup: float = Field(..., ge=0.1)
    Latitude: float = Field(..., ge=32.0, le=43.0)
    Longitude: float = Field(..., ge=-125.0, le=-114.0)

class PredictResponse(BaseModel):
    model_config = ConfigDict(ser_json_inf_nan="null")
    predicted_price: float
    model_version: str
