"""Application settings for the acne AI engine.

Covers: RNF-03 (reproducibility) — the seed, the model backend and the dataset version all live
here and travel with every inference response.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ModelBackend = Literal["stub", "torch"]
TrackingDriver = Literal["file", "mlflow"]


class Settings(BaseSettings):
    """Environment-driven configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    ai_host: str = "0.0.0.0"
    ai_port: int = 8000
    log_level: str = "info"

    #: ``stub`` keeps the whole clinical pipeline demonstrable before a model exists;
    #: ``torch`` serves the real EfficientNet-B2. See ``app.models.registry``.
    model_backend: ModelBackend = "stub"
    model_checkpoint_path: str = "./checkpoints/efficientnet_b2_iga.pt"
    model_seed: int = 42
    model_input_size: int = 260

    enable_gradcam: bool = True

    tracking_driver: TrackingDriver = "file"
    tracking_dir: str = "./runs"
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"

    data_dir: str = "./data"
    dataset_version: str = "v1.0"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance, so the process reads the environment exactly once."""
    return Settings()
