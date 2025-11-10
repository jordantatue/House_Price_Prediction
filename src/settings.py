from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Pydantic v2 config: load .env and ignore unknown variables
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Housing Price Prediction API"
    app_version: str = "1.0.0"
    allowed_origins: List[str] = ["http://localhost:3000"]  # Grafana
    request_body_max_mb: int = 2
    log_level: str = "info"

    model_path: str = "models/random_forest.pkl"
    scaler_path: str = "models/scaler.pkl"


settings = Settings()
