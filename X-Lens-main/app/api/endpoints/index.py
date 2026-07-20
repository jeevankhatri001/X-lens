from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.ml.embedder import SentenceEmbedder
from app.rag.chunker import chunk_text
from app.rag.scraper import scrape_url
from app.rag.vector_store import VectorStore


router = APIRouter()


class IndexRequest(BaseModel):
    urls: list[str] = Field(
        min_length=1,
        examples=[
            [
                "https://sunway.edu.np/frequently-asked-questions",
            ]
        ],
    )


@router.post("/index")
async def build_index(
    payload: IndexRequest,
    request: Request,
):
    settings = request.app.state.settings

    all_chunks = []
    seen_hashes: set[str] = set()
    per_url_counts: dict[str, int] = {}

    for url in payload.urls:
        text = await scrape_url(url)

        chunks = chunk_text(
            text=text,
            source_url=url,
            size=180,
            overlap=30,
        )

        added_for_url = 0

        for chunk in chunks:
            if chunk.content_hash in seen_hashes:
                continue

            seen_hashes.add(chunk.content_hash)
            all_chunks.append(chunk)
            added_for_url += 1

        per_url_counts[url] = added_for_url

    if not all_chunks:
        return {
            "status": "failed",
            "message": "No usable content was found.",
            "chunks": 0,
            "per_url_counts": per_url_counts,
        }

    embedder = SentenceEmbedder(
        settings.embedding_model,
        device="cpu",
    )
    embedder.load()

    texts = [chunk.text for chunk in all_chunks]
    vectors = embedder.encode(texts)

    metadata = [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "source_url": chunk.source_url,
            "content_hash": chunk.content_hash,
        }
        for chunk in all_chunks
    ]

    output_dir = Path(settings.vector_store_dir)

    store = VectorStore(output_dir)
    store.build(
        vectors=vectors,
        metadata=metadata,
    )

    return {
        "status": "completed",
        "chunks": len(metadata),
        "per_url_counts": per_url_counts,
        "index_path": str(output_dir / "index.faiss"),
        "metadata_path": str(output_dir / "metadata.json"),
    }