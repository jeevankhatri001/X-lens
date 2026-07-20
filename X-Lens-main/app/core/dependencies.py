from __future__ import annotations

from functools import lru_cache

from fastapi import Request

from app.core.config import get_settings
from app.quality.analyzer import ImageQualityAnalyzer
from app.services.pipeline import XLensPipeline


@lru_cache
def get_quality_analyzer() -> ImageQualityAnalyzer:
    settings = get_settings()

    return ImageQualityAnalyzer(
        settings.quality_threshold,
        settings.quality_aggregation,
        settings.min_width,
        settings.min_height,
    )


def get_pipeline(request: Request) -> XLensPipeline:
    return request.app.state.pipeline