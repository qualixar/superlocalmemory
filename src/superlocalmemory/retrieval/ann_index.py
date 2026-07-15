# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Approximate Nearest Neighbor Index.

Numpy-based cosine similarity with thread-safe operations. Supports
rebuilding from database on cold start and incremental add/remove.

No FAISS dependency — pure numpy brute-force is sufficient for the
scale we target (up to 100K facts). At that scale, a single cosine
scan takes ~5ms on CPU which is well within our latency budget.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)


class ANNIndex:
    """Thread-safe approximate nearest neighbor index using numpy.

    Stores (fact_id, embedding) pairs and supports top-k cosine
    similarity search. Vectors are L2-normalized on insertion for
    efficient dot-product scoring.

    Args:
        dimension: Embedding vector dimension (e.g. 768 for nomic-embed).
    """

    def __init__(self, dimension: int) -> None:
        self._dim = dimension
        self._ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._vectors: list[np.ndarray] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of indexed vectors."""
        with self._lock:
            return len(self._ids)

    @property
    def dimension(self) -> int:
        """Embedding dimension this index was created for."""
        return self._dim

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, fact_id: str, embedding: list[float]) -> None:
        """Add or update a fact embedding in the index.

        The vector is L2-normalized before storage so that dot-product
        equals cosine similarity at search time.

        Args:
            fact_id: Unique fact identifier.
            embedding: Raw embedding vector (will be normalized).
        """
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        if vec.shape[0] != self._dim:
            logger.warning(
                "Dimension mismatch: expected %d, got %d for %s",
                self._dim, vec.shape[0], fact_id,
            )
            return

        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm

        with self._lock:
            if fact_id in self._id_to_idx:
                # Update existing entry
                idx = self._id_to_idx[fact_id]
                self._vectors[idx] = vec
            else:
                # Append new entry
                self._id_to_idx[fact_id] = len(self._ids)
                self._ids.append(fact_id)
                self._vectors.append(vec)

    def remove(self, fact_id: str) -> None:
        """Remove a fact from the index.

        Uses swap-and-pop for O(1) removal: the last element fills
        the gap left by the removed element.

        Args:
            fact_id: Fact identifier to remove. No-op if not found.
        """
        with self._lock:
            if fact_id not in self._id_to_idx:
                return

            idx = self._id_to_idx.pop(fact_id)
            last_idx = len(self._ids) - 1

            if idx != last_idx:
                # Swap with last element
                last_id = self._ids[last_idx]
                self._ids[idx] = last_id
                self._vectors[idx] = self._vectors[last_idx]
                self._id_to_idx[last_id] = idx

            self._ids.pop()
            self._vectors.pop()

    def clear(self) -> None:
        """Remove all indexed vectors."""
        with self._lock:
            self._ids.clear()
            self._id_to_idx.clear()
            self._vectors.clear()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 30,
    ) -> list[tuple[str, float]]:
        """Find top-k most similar facts by cosine similarity.

        Args:
            query_embedding: Query vector (will be L2-normalized).
            top_k: Number of results to return.

        Returns:
            List of (fact_id, score) tuples sorted by score descending.
            Score is cosine similarity in [-1.0, 1.0].
        """
        q = np.asarray(query_embedding, dtype=np.float32).ravel()
        if q.shape[0] != self._dim:
            logger.warning(
                "Query dim mismatch: expected %d, got %d",
                self._dim, q.shape[0],
            )
            return []

        norm = np.linalg.norm(q)
        if norm < 1e-10:
            return []
        q_normed = q / norm

        with self._lock:
            if not self._vectors:
                return []

            # Stack into matrix for vectorized dot product
            mat = np.stack(self._vectors)        # shape: (N, dim)
            scores = mat @ q_normed              # shape: (N,)

            # Partial sort for top-k (faster than full sort for large N)
            k = min(top_k, len(scores))
            if k <= 0:
                return []

            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            return [
                (self._ids[i], float(scores[i]))
                for i in top_indices
            ]

    # ------------------------------------------------------------------
    # Bulk loading (cold start)
    # ------------------------------------------------------------------

    def rebuild(
        self,
        fact_ids: list[str],
        embeddings: list[list[float]],
    ) -> int:
        """Rebuild the entire index from database contents.

        Replaces all existing entries. Used on cold start to populate
        the index from persisted embeddings.

        Args:
            fact_ids: List of fact identifiers.
            embeddings: Corresponding embedding vectors.

        Returns:
            Number of vectors successfully indexed.
        """
        if len(fact_ids) != len(embeddings):
            logger.error(
                "rebuild: mismatched lengths — %d ids vs %d embeddings",
                len(fact_ids), len(embeddings),
            )
            return 0

        with self._lock:
            self._ids.clear()
            self._id_to_idx.clear()
            self._vectors.clear()

        indexed = 0
        for fid, emb in zip(fact_ids, embeddings):
            self.add(fid, emb)
            indexed += 1

        logger.info("ANN index rebuilt with %d vectors (dim=%d)", indexed, self._dim)
        return indexed
