from fastapi import APIRouter

from app.api.endpoints import (
    analyze,
    ask,
    chat,
    health,
    index,
    knowledge,
    metrics,
    scrape,
)


api_router = APIRouter(prefix="/api/v1")

api_router.include_router(analyze.router)
api_router.include_router(ask.router)
api_router.include_router(chat.router)
api_router.include_router(health.router)
api_router.include_router(index.router)
api_router.include_router(knowledge.router)
api_router.include_router(metrics.router)
api_router.include_router(scrape.router)