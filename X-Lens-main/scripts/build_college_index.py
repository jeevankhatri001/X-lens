from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np

from app.ml.embedder import SentenceEmbedder
from app.rag.chunker import chunk_text
from app.rag.scraper import scrape_url
from app.rag.vector_store import VectorStore


async def build_index(urls: list[str], output_dir: Path) -> None:
    all_chunks = []
    seen_hashes: set[str] = set()

    for url in urls:
        print(f"Scraping: {url}")

        try:
            text = await scrape_url(url)
        except Exception as exc:
            print(f"Failed to scrape {url}: {exc}")
            continue

        if not text.strip():
            print(f"No usable text found: {url}")
            continue

        chunks = chunk_text(
            text=text,
            source_url=url,
            size=300,
            overlap=50,
        )

        added = 0

        for chunk in chunks:
            if chunk.content_hash in seen_hashes:
                continue

            seen_hashes.add(chunk.content_hash)
            all_chunks.append(chunk)
            added += 1

        print(f"Added {added} unique chunks from {url}")

    if not all_chunks:
        raise RuntimeError(
            "No usable text was collected from the supplied URLs."
        )

    print(f"Total unique chunks: {len(all_chunks)}")
    print("Loading embedding model...")

    embedder = SentenceEmbedder()
    embedder.load()

    texts = [chunk.text for chunk in all_chunks]

    vectors = embedder.encode(texts)
    vectors = np.asarray(vectors, dtype=np.float32)

    metadata = [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "source_url": chunk.source_url,
            "content_hash": chunk.content_hash,
        }
        for chunk in all_chunks
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    store = VectorStore(output_dir)

    store.build(
        vectors=vectors,
        metadata=metadata,
    )

    print()
    print("RAG index successfully created.")
    print(f"Chunks: {len(metadata)}")
    print(f"Index: {output_dir / 'index.faiss'}")
    print(f"Metadata: {output_dir / 'metadata.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape college pages and build the X-Lens FAISS index."
    )

    parser.add_argument(
        "--url",
        action="append",
        required=True,
        help="College webpage URL. Repeat --url for multiple pages.",
    )

    parser.add_argument(
        "--output",
        default="data/vector_store",
        help="Directory where the FAISS index will be saved.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    asyncio.run(
        build_index(
            urls=args.url,
            output_dir=Path(args.output),
        )
    )


if __name__ == "__main__":
    main()