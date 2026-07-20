from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.dependencies import get_pipeline
from app.services.pipeline import XLensPipeline


router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(
        min_length=2,
        max_length=300,
        examples=["How do I apply to Sunway College?"],
    )


@router.post("/ask")
async def ask_question(
    payload: AskRequest,
    pipeline: XLensPipeline = Depends(get_pipeline),
):
    return pipeline.process(
        image=None,
        question=payload.question,
    )