from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RequestMode(str, Enum):
    """
    Basic request types.

    TEXT:
        No image was supplied. The pipeline will first search RAG,
        then ask Qwen using retrieved context when available.

    VISION:
        An image was supplied. The pipeline will use Qwen vision,
        with optional RAG context when relevant.
    """

    TEXT = "text"
    VISION = "vision"


@dataclass(frozen=True)
class RouteDecision:
    """
    Result returned by the request router.
    """

    mode: RequestMode
    reason: str
    question: str | None
    image_provided: bool


class RequestRouter:
    """
    Route requests based only on available input.

    This router does not contain question-specific keywords.

    The pipeline is responsible for:

    1. Searching the college knowledge base.
    2. Measuring retrieval confidence.
    3. Supplying relevant RAG context to Qwen.
    4. Letting Qwen answer general text questions.
    5. Letting Qwen answer image questions.
    """

    def classify(
        self,
        question: str | None,
        image_provided: bool,
    ) -> RouteDecision:
        clean_question = self._clean_question(question)

        if image_provided:
            return RouteDecision(
                mode=RequestMode.VISION,
                reason=(
                    "An image is available, so use the vision-capable "
                    "Qwen processing path."
                ),
                question=clean_question,
                image_provided=True,
            )

        return RouteDecision(
            mode=RequestMode.TEXT,
            reason=(
                "No image is available, so use the text processing path "
                "with optional RAG context."
            ),
            question=clean_question,
            image_provided=False,
        )

    @staticmethod
    def _clean_question(
        question: str | None,
    ) -> str | None:
        """
        Remove surrounding whitespace without changing the user's wording.
        """

        if question is None:
            return None

        clean = question.strip()

        return clean or None