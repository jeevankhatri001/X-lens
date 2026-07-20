from __future__ import annotations

from typing import Sequence

import numpy as np


class SentenceEmbedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.model = None

    def load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "RAG dependencies are not installed. "
                "Run: python -m pip install faiss-cpu sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.model is None:
            self.load()

        vectors = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return np.asarray(vectors, dtype=np.float32)