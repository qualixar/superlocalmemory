# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""SuperLocalMemory V3.3 -- Hopfield Associative Memory (6th Retrieval Channel).

Modern Continuous Hopfield Network retrieval channel based on
Ramsauer et al. (2020): "Hopfield Networks is All You Need".

The Hopfield channel excels at pattern completion for vague/noisy queries.
It operates on the same embedding space as the semantic channel but uses
an energy-based attention mechanism instead of cosine similarity.

Key features:
  - Full memory matrix path for stores < 10K facts
  - ANN pre-filter path for stores 10K-100K (VectorStore KNN -> Hopfield refinement)
  - Skip path for stores > 100K (other 5 channels are sufficient)
  - TTL-based matrix cache to avoid rebuilding every query
  - Returns [] on any error (HR-06)

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: Elastic-2.0
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from superlocalmemory.math.hopfield import HopfieldConfig, ModernHopfieldNetwork

if TYPE_CHECKING:
    from superlocalmemory.retrieval.vector_store import VectorStore
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class HopfieldChannel:
    """6th retrieval channel: Modern Hopfield associative memory.

    Implements the RetrievalChannel protocol::

        def search(query, profile_id, top_k=50) -> list[tuple[str, float]]

    The channel builds an in-memory matrix from all fact embeddings,
    computes Hopfield attention scores (softmax of scaled dot products),
    then ranks facts by similarity to the completed pattern.

    Routing logic (per LLD Section 2.2):
      - n > skip_threshold (100K): return [] immediately
      - n > prefilter_threshold (10K): ANN pre-filter + Hopfield on subset
      - n <= prefilter_threshold: full matrix Hopfield
    """

    def __init__(
        self,
        db: Any,
        vector_store: Any,
        config: HopfieldConfig | None = None,
    ) -> None:
        """Initialize HopfieldChannel.

        Args:
            db: DatabaseManager with get_all_facts() and get_facts_by_ids().
            vector_store: VectorStore with search() and count().
            config: Hopfield configuration. Uses defaults if None.
        """
        self._db = db
        self._vector_store = vector_store
        self._config = config or HopfieldConfig()
        self._hopfield = ModernHopfieldNetwork(self._config)

        # Memory matrix cache (per LLD Section 2.2, HR-09)
        self._cached_matrix: np.ndarray | None = None
        self._cached_fact_ids: list[str] = []
        self._cached_profile: str = ""
        self._cached_count: int = 0
        self._cache_timestamp: float = 0.0

    # -- Public API (RetrievalChannel protocol) --------------------------------

    def search(
        self,
        query: Any,
        profile_id: str,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        """Search for facts using Hopfield associative retrieval.

        Args:
            query: Query embedding (list[float] or np.ndarray).
            profile_id: Scope search to this profile.
            top_k: Maximum results to return.

        Returns:
            List of (fact_id, score) sorted by score descending.
            Returns [] on any error (HR-06).
        """
        # Step 1: Check enabled
        if not self._config.enabled:
            return []

        try:
            return self._search_inner(query, profile_id, top_k)
        except Exception as exc:
            # HR-06: Return [] on any error
            logger.warning("Hopfield channel error: %s", exc)
            return []

    # -- Private implementation ------------------------------------------------

    def _search_inner(
        self,
        query: Any,
        profile_id: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Core search logic, separated for clean error handling."""
        # Step 2: Convert query to numpy
        q_vec = np.array(query, dtype=np.float32)

        # Step 3: Validate dimension
        if q_vec.shape != (self._config.dimension,):
            logger.debug(
                "Hopfield dimension mismatch: query %s, expected (%d,)",
                q_vec.shape, self._config.dimension,
            )
            return []

        # Step 3b (AUDIT FIX G-MEDIUM-02): Check skip_threshold BEFORE loading matrix
        total_count = (
            self._vector_store.count(profile_id)
            if self._vector_store and getattr(self._vector_store, "available", False)
            else 0
        )
        # Step 3c: Skip for very large stores
        if total_count > self._config.skip_threshold:
            logger.debug(
                "Hopfield skipped: %d facts exceeds skip_threshold %d",
                total_count, self._config.skip_threshold,
            )
            return []

        # Step 4 (v3.5.0 FIX): route by the cheap total_count BEFORE building
        # the full memory matrix. Previously _get_memory_matrix() loaded and
        # normalized ALL embeddings (~6s at 17.5k facts) even when the prefilter
        # path was taken — and that path builds its own ANN sub-matrix and never
        # uses the full one. So the entire 6s build was wasted on large stores.
        # Now the full matrix is built ONLY for small stores (<= prefilter_threshold).
        vs_ok = bool(
            self._vector_store
            and getattr(self._vector_store, "available", False)
        )
        # v3.5.0 FIX: prefer the bounded ANN-prefilter path whenever a vector
        # store is available and the corpus is larger than the candidate set.
        # The full-matrix path loads & L2-normalizes EVERY embedding from SQLite
        # (~6s at 17.5k facts), so it's only worth it for tiny stores or when no
        # VS exists. Routing on prefilter_candidates (not prefilter_threshold)
        # ensures the matrix is always bounded to ~prefilter_candidates rows.
        if vs_ok and total_count > self._config.prefilter_candidates:
            return self._search_with_prefilter(q_vec, profile_id, [], top_k)

        # Tiny store (or no VS): build (cached) full matrix.
        memory_matrix, fact_ids = self._get_memory_matrix(profile_id)
        if memory_matrix is None or len(fact_ids) == 0:
            return []
        if vs_ok and len(fact_ids) > self._config.prefilter_candidates:
            return self._search_with_prefilter(q_vec, profile_id, fact_ids, top_k)
        return self._search_full_matrix(
            q_vec, memory_matrix, fact_ids, top_k,
        )

    def _search_full_matrix(
        self,
        query: np.ndarray,
        memory_matrix: np.ndarray,
        fact_ids: list[str],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Full matrix Hopfield retrieval for stores <= prefilter_threshold.

        Algorithm (LLD Section 2.2):
        1. Compute Hopfield attention weights
        2. Compute retrieved (completed) pattern via weighted sum
        3. Normalize retrieved pattern
        4. Score all stored patterns against the completed pattern
        5. Return top-K by similarity
        """
        # Step 1: Hopfield attention
        attention = self._hopfield.attention_scores(query, memory_matrix)

        # Step 2: Pattern completion
        retrieved = memory_matrix.T @ attention  # shape (d,)

        # Step 3: Normalize
        norm = float(np.linalg.norm(retrieved))
        if norm > 1e-8:
            retrieved = retrieved / norm

        # Step 4: Similarity to all patterns
        similarities = memory_matrix @ retrieved  # shape (n,)

        # Step 5: Top-K selection
        top_indices = np.argsort(-similarities)[:top_k]
        results: list[tuple[str, float]] = [
            (fact_ids[int(i)], float(similarities[i]))
            for i in top_indices
            if similarities[i] > 0.0
        ]

        return results

    def _search_with_prefilter(
        self,
        query: np.ndarray,
        profile_id: str,
        all_fact_ids: list[str],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Two-stage retrieval for large stores (>prefilter_threshold facts).

        Stage 1: VectorStore KNN pre-filter to get candidate subset
        Stage 2: Hopfield refinement on the small candidate set

        Algorithm (LLD Section 2.2):
        1. Get KNN candidates from VectorStore
        2. Load candidate facts from DB
        3. Build sub-matrix from candidate embeddings
        4. Run full-matrix Hopfield on the sub-matrix
        """
        # Stage 1: KNN pre-filter
        if not self._vector_store or not getattr(self._vector_store, "available", False):
            # No vector store available; fall back to full matrix
            # (only reached if matrix was somehow loaded despite no VS)
            return []

        knn_results = self._vector_store.search(
            query.tolist(),
            top_k=self._config.prefilter_candidates,
            profile_id=profile_id,
        )
        if not knn_results:
            return []

        # Stage 2: Load candidate facts
        candidate_ids = [fid for fid, _ in knn_results]
        candidates = self._db.get_facts_by_ids(
            candidate_ids, profile_id,
            include_global=getattr(self, 'include_global', False),
            include_shared=getattr(self, 'include_shared', False),
        )
        if not candidates:
            return []

        # Stage 3: Build sub-matrix
        sub_embeddings: list[np.ndarray] = []
        sub_ids: list[str] = []
        for fact in candidates:
            emb = getattr(fact, "embedding", None)
            if emb is not None and len(emb) == self._config.dimension:
                sub_embeddings.append(np.array(emb, dtype=np.float32))
                sub_ids.append(fact.fact_id)

        if not sub_embeddings:
            return []

        sub_matrix = np.array(sub_embeddings, dtype=np.float32)

        # HR-03: L2-normalize sub-matrix rows
        norms = np.linalg.norm(sub_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        sub_matrix = sub_matrix / norms

        # Stage 4: Hopfield on subset
        return self._search_full_matrix(query, sub_matrix, sub_ids, top_k)

    def _get_memory_matrix(
        self, profile_id: str,
    ) -> tuple[np.ndarray | None, list[str]]:
        """Build or retrieve cached memory matrix X (n x d).

        The matrix is L2-normalized per row (HR-03) and cached
        with a TTL (HR-09, default 60s).

        Returns:
            (memory_matrix, fact_ids) or (None, []) if no valid facts.
        """
        # Step 1: Check cache validity
        current_count = (
            self._vector_store.count(profile_id)
            if self._vector_store and getattr(self._vector_store, "available", False)
            else 0
        )

        if (
            self._cached_profile == profile_id
            and self._cached_count == current_count
            and self._cached_matrix is not None
            and (time.monotonic() - self._cache_timestamp)
            < self._config.cache_ttl_seconds
        ):
            return (self._cached_matrix, self._cached_fact_ids)

        # Step 2: Load facts (V3.3.12: cap to most recent 5000 to bound memory)
        # memory-bounding-02: push the cap into SQL (LIMIT) so we don't
        # deserialize the whole table just to slice it.
        facts = self._db.get_all_facts(
            profile_id, limit=5000,
            include_global=getattr(self, 'include_global', False),
            include_shared=getattr(self, 'include_shared', False),
        )
        if not facts:
            return (None, [])

        # Step 4: Filter facts with valid embeddings
        valid: list[tuple[str, list[float]]] = []
        for f in facts:
            emb = getattr(f, "embedding", None)
            if emb is not None and len(emb) == self._config.dimension:
                valid.append((f.fact_id, emb))

        if not valid:
            return (None, [])

        # Step 6: Build matrix
        fact_ids = [fid for fid, _ in valid]
        matrix = np.array(
            [emb for _, emb in valid], dtype=np.float32,
        )  # shape (n, d)

        # Step 7: HR-03 — L2 normalize each row
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        matrix = matrix / norms

        # Step 8: Update cache
        self._cached_matrix = matrix
        self._cached_fact_ids = fact_ids
        self._cached_profile = profile_id
        self._cached_count = current_count
        self._cache_timestamp = time.monotonic()

        return (matrix, fact_ids)

    def invalidate_cache(self) -> None:
        """Clear the memory matrix cache.

        Called after fact insertion/deletion to ensure
        next search rebuilds with fresh data.
        """
        self._cached_matrix = None
        self._cached_fact_ids = []
        self._cached_count = 0
        self._cache_timestamp = 0.0
