from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="XLENS_",
        extra="ignore",
    )

    env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000

    max_image_size_mb: int = 10

    quality_threshold: float = 0.25
    quality_aggregation: Literal[
        "minimum",
        "geometric_mean",
        "weighted_average",
    ] = "minimum"

    min_width: int = 640
    min_height: int = 480

    # Image-quality hard limits.
    min_blur_score: float = 0.40
    max_overexposed_fraction: float = 0.30
    max_underexposed_fraction: float = 0.45

    enable_vlm: bool = False
    vlm_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vlm_timeout_seconds: int = 120

    enable_rag: bool = False
    rag_threshold: float = 0.35
    rag_top_k: int = 5

    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store_dir: Path = Path(
        "data/vector_store"
    )

    data_dir: Path = Path("data")


@lru_cache
def get_settings() -> Settings:
    return Settings()