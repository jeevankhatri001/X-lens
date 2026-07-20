from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.middleware import RequestIdMiddleware
from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging_config import configure_logging
from app.ml.embedder import SentenceEmbedder
from app.ml.qwen_vlm import QwenVLM
from app.quality.analyzer import ImageQualityAnalyzer
from app.rag.retriever import RAGRetriever
from app.rag.service import RAGService
from app.rag.vector_store import VectorStore
from app.services.pipeline import XLensPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    app.state.settings = settings
    app.state.ready = False
    app.state.vlm = None
    app.state.rag = None
    app.state.quality_analyzer = None
    app.state.pipeline = None

    quality_analyzer = ImageQualityAnalyzer(
        threshold=settings.quality_threshold,
        aggregation=settings.quality_aggregation,
        min_width=settings.min_width,
        min_height=settings.min_height,
        min_blur_score=settings.min_blur_score,
        max_overexposed_fraction=(
            settings.max_overexposed_fraction
        ),
        max_underexposed_fraction=(
            settings.max_underexposed_fraction
        ),
    )

    app.state.quality_analyzer = quality_analyzer

    if settings.enable_vlm:
        vlm = QwenVLM(
            settings.vlm_model_id
        )

        vlm.load()
        vlm.warmup()

        app.state.vlm = vlm

    if settings.enable_rag:
        vector_store_dir = Path(
            settings.vector_store_dir
        )

        index_path = (
            vector_store_dir
            / "index.faiss"
        )

        metadata_path = (
            vector_store_dir
            / "metadata.json"
        )

        if (
            index_path.exists()
            and metadata_path.exists()
        ):
            embedder = SentenceEmbedder(
                model_name=settings.embedding_model,
                device="cpu",
            )

            embedder.load()

            store = VectorStore(
                vector_store_dir
            )

            store.load()

            retriever = RAGRetriever(
                embedder=embedder,
                store=store,
                threshold=settings.rag_threshold,
                top_k=settings.rag_top_k,
            )

            app.state.rag = RAGService(
                retriever
            )

            print(
                "RAG loaded successfully with "
                f"{len(store.metadata)} chunks."
            )

        else:
            print(
                "RAG is enabled, but the "
                "vector-store files were not found."
            )

            print(
                f"Expected index: {index_path}"
            )

            print(
                f"Expected metadata: {metadata_path}"
            )

    app.state.pipeline = XLensPipeline(
        quality_analyzer=quality_analyzer,
        vlm=app.state.vlm,
        rag_service=app.state.rag,
    )

    app.state.ready = True

    try:
        yield

    finally:
        app.state.ready = False

        if app.state.vlm is not None:
            app.state.vlm.unload()

        app.state.vlm = None
        app.state.rag = None
        app.state.quality_analyzer = None
        app.state.pipeline = None


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(
        title="X-Lens API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        RequestIdMiddleware
    )

    app.include_router(
        api_router
    )

    return app


app = create_app()