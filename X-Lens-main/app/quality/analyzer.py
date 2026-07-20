from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping

import cv2
import numpy as np
from PIL import Image

from app.quality import metrics


class AggregationMethod(str, Enum):
    MINIMUM = "minimum"
    GEOMETRIC_MEAN = "geometric_mean"
    WEIGHTED_AVERAGE = "weighted_average"


@dataclass
class QualityReport:
    brightness: float
    contrast: float
    blur: float
    noise: float
    resolution: float
    exposure: float
    colorfulness: float
    overall: float
    is_acceptable: bool
    rejection_reason: str | None
    underexposed_fraction: float
    overexposed_fraction: float
    blur_components: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


class ImageQualityAnalyzer:
    def __init__(
        self,
        threshold: float = 0.25,
        aggregation: AggregationMethod = AggregationMethod.MINIMUM,
        min_width: int = 640,
        min_height: int = 480,
        min_blur_score: float = 0.40,
        max_overexposed_fraction: float = 0.30,
        max_underexposed_fraction: float = 0.45,
    ) -> None:
        self.threshold = threshold
        self.aggregation = AggregationMethod(aggregation)
        self.min_width = min_width
        self.min_height = min_height

        # Hard quality limits.
        self.min_blur_score = min_blur_score
        self.max_overexposed_fraction = max_overexposed_fraction
        self.max_underexposed_fraction = max_underexposed_fraction

    def _aggregate(
        self,
        scores: Mapping[str, float],
    ) -> float:
        values = np.array(
            list(scores.values()),
            dtype=float,
        )

        if self.aggregation is AggregationMethod.MINIMUM:
            return float(values.min())

        if self.aggregation is AggregationMethod.GEOMETRIC_MEAN:
            safe_values = np.maximum(
                values,
                1e-8,
            )

            return float(
                np.prod(safe_values)
                ** (1 / len(safe_values))
            )

        weights = np.array(
            [0.15, 0.15, 0.25, 0.15, 0.10, 0.20],
            dtype=float,
        )

        return float(
            np.average(
                values,
                weights=weights,
            )
        )

    def analyze(
        self,
        image: Image.Image,
    ) -> QualityReport:
        """
        Analyse image quality and determine whether the image should
        be processed by the vision model or captured again.
        """

        rgb = np.asarray(
            image.convert("RGB")
        )

        bgr = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        )

        gray = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2GRAY,
        )

        blur, blur_parts = metrics.blur_score(
            gray
        )

        exposure, underexposed, overexposed = (
            metrics.exposure_score(bgr)
        )

        scores = {
            "brightness": metrics.brightness_score(
                bgr
            ),
            "contrast": metrics.contrast_score(
                gray
            ),
            "blur": blur,
            "noise": metrics.noise_score(
                gray
            ),
            "resolution": metrics.resolution_score(
                bgr,
                self.min_width,
                self.min_height,
            ),
            "exposure": exposure,
        }

        overall = self._aggregate(scores)

        failed_metrics = [
            name
            for name, score in scores.items()
            if score < self.threshold
        ]

        rejection_reasons: list[str] = []

        if failed_metrics:
            rejection_reasons.append(
                "Low quality: "
                + ", ".join(failed_metrics)
            )

        # Reject images that are too blurry or indistinct.
        if blur < self.min_blur_score:
            rejection_reasons.append(
                "The image is too blurry"
            )

        # Reject images with a large bright or washed-out region.
        if (
            overexposed
            > self.max_overexposed_fraction
        ):
            rejection_reasons.append(
                "Too much of the image is overexposed"
            )

        # Reject images that are mostly too dark.
        if (
            underexposed
            > self.max_underexposed_fraction
        ):
            rejection_reasons.append(
                "Too much of the image is underexposed"
            )

        is_acceptable = not rejection_reasons

        rejection_reason = (
            "; ".join(rejection_reasons)
            if rejection_reasons
            else None
        )

        return QualityReport(
            **scores,
            colorfulness=metrics.colorfulness_score(
                bgr
            ),
            overall=overall,
            is_acceptable=is_acceptable,
            rejection_reason=rejection_reason,
            underexposed_fraction=underexposed,
            overexposed_fraction=overexposed,
            blur_components=blur_parts,
        )