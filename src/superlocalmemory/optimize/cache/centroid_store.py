# optimize/cache/centroid_store.py
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
#
# LLD-03 §4.3 — SAFE-CACHE centroid registry for adversarial collision
# detection.
#
# Source: SAFE-CACHE — Nature Scientific Reports 2026 (no arXiv).
# Defense: compare the incoming query vector to the cluster centroid. If
# the query is far from the centroid (cosine sim < 1 - distance_floor),
# it falls outside the natural distribution → likely an adversarial
# probe crafted to collide with a specific entry. Reject as miss.
#
# Design:
#   - One centroid per tenant_id (simplest partition, proven sufficient).
#   - Rebuilt from llmcache_semantic_vectors on startup.
#   - Updated incrementally (Welford running mean) on every new set().
#   - In-memory only — no separate SQL query needed (cache is O(N) anyway).

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from superlocalmemory.optimize.storage.db import CacheDB

logger = logging.getLogger(__name__)

_VARIANCE_FLOOR: float = 1e-6
_EMBED_DIM: int = 768


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 on zero vectors."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < _VARIANCE_FLOOR or norm_b < _VARIANCE_FLOOR:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class CentroidStore:
    """In-memory centroid registry for SAFE-CACHE adversarial defense.

    Thread-safe via a single RLock. All public methods are fail-open.

    Centroid update rule (Welford running mean — exact, O(1) per update):
        new_centroid = old_centroid * (n / (n+1)) + new_vec * (1 / (n+1))
    """
    def __init__(self) -> None:
        self._centroids: dict[str, np.ndarray] = {}  # tenant_id → float32 vec
        self._counts: dict[str, int] = {}            # tenant_id → count
        self._lock = threading.RLock()

    def rebuild_from_db(self, db: "CacheDB", tenant_id: str) -> None:
        """Rebuild centroid for a tenant from all stored vectors.

        Called at VCacheSemantic startup. O(N) — runs once per tenant.
        Fail-open: on DB error or empty vector set, centroid is reset.
        """
        try:
            rows = db.get_all_vectors(tenant_id=tenant_id)
            if not rows:
                with self._lock:
                    self._centroids.pop(tenant_id, None)
                    self._counts.pop(tenant_id, None)
                return
            vectors: list[np.ndarray] = []
            for _entry_id, blob, _ctx_fp in rows:
                try:
                    vec = np.frombuffer(blob, dtype=np.float32).copy()
                    if vec.shape[0] == _EMBED_DIM:
                        vectors.append(vec)
                except Exception:
                    continue
            if not vectors:
                return
            centroid = np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
            with self._lock:
                self._centroids[tenant_id] = centroid
                self._counts[tenant_id] = len(vectors)
            logger.debug(
                "CentroidStore: rebuilt tenant=%s centroid from %d vectors",
                tenant_id, len(vectors),
            )
        except Exception as exc:
            logger.warning("CentroidStore.rebuild_from_db failed (fail-open): %s", exc)

    def update(self, tenant_id: str, new_vector: np.ndarray) -> None:
        """Update centroid with a new vector (Welford incremental mean).

        Thread-safe. Fail-open.
        """
        try:
            vec = new_vector.astype(np.float32)
            with self._lock:
                if tenant_id not in self._centroids:
                    self._centroids[tenant_id] = vec.copy()
                    self._counts[tenant_id] = 1
                else:
                    n = self._counts[tenant_id]
                    old = self._centroids[tenant_id]
                    self._centroids[tenant_id] = (
                        old * (n / (n + 1)) + vec * (1.0 / (n + 1))
                    ).astype(np.float32)
                    self._counts[tenant_id] = n + 1
        except Exception as exc:
            logger.warning("CentroidStore.update failed (fail-open): %s", exc)

    def is_adversarial(
        self,
        tenant_id: str,
        query_vector: np.ndarray,
        distance_floor: float = 0.15,
    ) -> bool:
        """Return True if query_vector appears adversarially crafted.

        Defense: if the query is too far from the cluster centroid (cosine
        similarity < 1 - distance_floor), it is an outlier — likely an
        adversarial probe. Return True → reject (adversarial).

        Fail-open: returns False (accept) on any error or missing centroid.
        Skips defense if the cluster has < 5 entries (insufficient data).
        """
        try:
            with self._lock:
                centroid = self._centroids.get(tenant_id)
                count = self._counts.get(tenant_id, 0)
            if centroid is None or count < 5:
                return False
            q = query_vector.astype(np.float32)
            sim = _cosine_similarity(q, centroid)
            threshold = 1.0 - distance_floor
            if sim < threshold:
                logger.warning(
                    "CentroidStore: adversarial probe detected "
                    "(tenant=%s, centroid_sim=%.4f < %.4f) — returning miss.",
                    tenant_id, sim, threshold,
                )
                return True
            return False
        except Exception as exc:
            logger.warning("CentroidStore.is_adversarial failed (fail-open): %s", exc)
            return False

    def get_centroid(self, tenant_id: str) -> np.ndarray | None:
        """Return current centroid for a tenant, or None if not established."""
        with self._lock:
            return self._centroids.get(tenant_id)

    def count(self, tenant_id: str) -> int:
        """Return the number of vectors contributing to this tenant's centroid."""
        with self._lock:
            return self._counts.get(tenant_id, 0)
