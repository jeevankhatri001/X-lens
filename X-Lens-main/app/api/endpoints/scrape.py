from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.rag.scraper import scrape_url


router = APIRouter()


class ScrapeRequest(BaseModel):
    url: str = Field(
        min_length=8,
        examples=["https://sunway.edu.np/frequently-asked-questions"],
    )


@router.post("/scrape")
async def scrape_website(payload: ScrapeRequest):
    text = await scrape_url(payload.url)

    preview = " ".join(text.split())

    if len(preview) > 500:
        preview = preview[:500].rstrip() + "..."

    return {
        "status": "completed",
        "url": payload.url,
        "characters": len(text),
        "preview": preview,
    }