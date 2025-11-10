from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    app_name: str = "Housing Price Prediction API"
    app_version: str = "1.0.0"
    allowed_origins: List[str] = ["http://localhost:3000"]  # Grafana
    request_body_max_mb: int = 2
    log_level: str = "info"

    model_path: str = "models/random_forest.pkl"
    scaler_path: str = "models/scaler.pkl"

    class Config:
        env_file = ".env"

settings = Settings()
