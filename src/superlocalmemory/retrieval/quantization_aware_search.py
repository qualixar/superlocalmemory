# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Three-tier mixed-precision search.

Merges results from:
  Tier 1: float32 (VectorStore.search -- exact cosine)
  Tier 2: int8    (VectorStore.search_int8 -- sqlite-vec native)
  Tier 3: polar   (QuantizedEmbeddingStore.search -- PolarQuant)

Deduplicates by keeping the highest score per fact_id.
Applies precision-dependent score penalties:
  - float32: no penalty (1.0x)
  - int8:    0.98x
  - polar:   config.polar_search_penalty (default 0.95x)

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from numpy.typing import NDArray

from superlocalmemory.core.config import QuantizationConfig

if TYPE_CHECKING:
    from superlocalmemory.storage.quantized_store import QuantizedEmbeddingStore

logger = logging.getLogger(__name__)

# Penalty factor for int8 tier (fixed, not configurable)
_INT8_PENALTY: float = 0.98


class QuantizationAwareSearch:
    """Three-tier mixed-precision embedding search.

    Combines float32 + int8 + polar results, deduplicates,
    and returns top_k by score descending.
    """

    __slots__ = ("_vector_store", "_quantized_store", "_config")

    def __init__(
        self,
        vector_store,
        quantized_store: QuantizedEmbeddingStore,
        config: QuantizationConfig,
    ) -> None:
        self._vector_store = vector_store
        self._quantized_store = quantized_store
        self._config = config

    def search(
        self,
        query_embedding: NDArray,
        profile_id: str,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        """Execute three-tier mixed-precision search.

        Args:
            query_embedding: Query vector (float32/64).
            profile_id:      Scope to this profile.
            top_k:           Max results to return.

        Returns:
            [(fact_id, score)] sorted by score descending.
        """
        # Tier 1: float32 exact search
        results_f32 = self._search_float32(query_embedding, profile_id, top_k)

        # Tier 2: int8 approximate search
        results_int8 = self._search_int8(query_embedding, profile_id, top_k)

        # Tier 3: polar quantized search
        results_polar = self._search_polar(query_embedding, profile_id, top_k)

        # Merge + dedup (keep highest score per fact_id)
        seen: dict[str, float] = {}
        for fid, score in results_f32 + results_int8 + results_polar:
            if fid not in seen or score > seen[fid]:
                seen[fid] = score

        # Sort by score descending
        merged = sorted(seen.items(), key=lambda x: x[1], reverse=True)
        return merged[:top_k]

    # -- Tier helpers (encapsulate error handling) -------------------------

    def _search_float32(
        self, query: NDArray, profile_id: str, top_k: int,
    ) -> list[tuple[str, float]]:
        """Tier 1: float32 exact cosine via VectorStore."""
        try:
            return self._vector_store.search(
                query_embedding=list(query) if hasattr(query, 'tolist') else query,
                top_k=top_k,
                profile_id=profile_id,
            )
        except Exception as exc:
            logger.debug("float32 search failed: %s", exc)
            return []

    def _search_int8(
        self, query: NDArray, profile_id: str, top_k: int,
    ) -> list[tuple[str, float]]:
        """Tier 2: int8 approximate via VectorStore.search_int8.

        Applies 0.98x penalty to account for int8 quantization error.
        Gracefully returns [] if VectorStore lacks search_int8 method.
        """
        fn = getattr(self._vector_store, "search_int8", None)
        if fn is None:
            return []
        try:
            raw = fn(query, profile_id=profile_id, top_k=top_k)
            return [(fid, score * _INT8_PENALTY) for fid, score in raw]
        except Exception as exc:
            logger.debug("int8 search failed: %s", exc)
            return []

    def _search_polar(
        self, query: NDArray, profile_id: str, top_k: int,
    ) -> list[tuple[str, float]]:
        """Tier 3: polar quantized via QuantizedEmbeddingStore.

        Applies polar_search_penalty from config.
        """
        try:
            raw = self._quantized_store.search(query, profile_id, top_k)
            penalty = self._config.polar_search_penalty
            return [(fid, score * penalty) for fid, score in raw]
        except Exception as exc:
            logger.debug("polar search failed: %s", exc)
            return []
