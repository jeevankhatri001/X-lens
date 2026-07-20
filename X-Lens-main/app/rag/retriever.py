from __future__ import annotations

from typing import Any

import numpy as np


class RAGRetriever:
    """
    Retrieve relevant but diverse knowledge chunks.

    Standard nearest-neighbour retrieval can return several chunks
    covering almost the same topic. This implementation first finds a
    larger candidate set, then selects chunks that balance:

    - relevance to the user's question
    - diversity from chunks already selected

    This improves broad questions without hardcoding any topic names.
    """

    def __init__(
        self,
        embedder: Any,
        store: Any,
        threshold: float = 0.35,
        top_k: int = 5,
        candidate_k: int = 12,
        diversity_weight: float = 0.35,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.threshold = threshold
        self.top_k = max(1, top_k)
        self.candidate_k = max(self.top_k, candidate_k)
        self.diversity_weight = min(
            max(diversity_weight, 0.0),
            1.0,
        )

    def retrieve(
        self,
        query: str,
    ) -> list[dict]:
        clean_query = query.strip()

        if not clean_query:
            return []

        query_vector = self.embedder.encode(
            [clean_query]
        )[0]

        candidates = self.store.search(
            query_vector=query_vector,
            k=self.candidate_k,
        )

        candidates = [
            candidate
            for candidate in candidates
            if candidate.get("similarity", 0.0)
            >= self.threshold
            and candidate.get("text", "").strip()
        ]

        if not candidates:
            return []

        if len(candidates) <= self.top_k:
            return candidates

        candidate_texts = [
            candidate["text"]
            for candidate in candidates
        ]

        candidate_vectors = np.asarray(
            self.embedder.encode(candidate_texts),
            dtype=np.float32,
        )

        candidate_vectors = self._normalize_vectors(
            candidate_vectors
        )

        selected_indices: list[int] = []
        remaining_indices = list(range(len(candidates)))

        # Always begin with the most relevant candidate.
        first_index = max(
            remaining_indices,
            key=lambda index: candidates[index].get(
                "similarity",
                0.0,
            ),
        )

        selected_indices.append(first_index)
        remaining_indices.remove(first_index)

        while (
            remaining_indices
            and len(selected_indices) < self.top_k
        ):
            best_index = None
            best_score = float("-inf")

            for candidate_index in remaining_indices:
                relevance = candidates[
                    candidate_index
                ].get("similarity", 0.0)

                maximum_similarity_to_selected = max(
                    float(
                        np.dot(
                            candidate_vectors[candidate_index],
                            candidate_vectors[selected_index],
                        )
                    )
                    for selected_index in selected_indices
                )

                score = (
                    (1.0 - self.diversity_weight)
                    * relevance
                    - self.diversity_weight
                    * maximum_similarity_to_selected
                )

                if score > best_score:
                    best_score = score
                    best_index = candidate_index

            if best_index is None:
                break

            selected_indices.append(best_index)
            remaining_indices.remove(best_index)

        return [
            candidates[index]
            for index in selected_indices
        ]

    @staticmethod
    def _normalize_vectors(
        vectors: np.ndarray,
    ) -> np.ndarray:
        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        norms = np.maximum(norms, 1e-12)

        return vectors / norms