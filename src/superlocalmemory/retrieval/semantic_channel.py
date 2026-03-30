# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Semantic Retrieval Channel.

Fisher-aware semantic search: uses Fisher-Rao geodesic distance when
variance data is available, falls back to cosine similarity otherwise.

The Fisher distance is meaningful when memories accumulate evidence
(repeated confirmation narrows variance). For fresh benchmark data
where all variances are identical, Fisher distance degenerates to a
monotonic transform of Euclidean distance — same ranking as cosine.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: MIT
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.models import AtomicFact

logger = logging.getLogger(__name__)

# Minimum variance floor to prevent division-by-zero in Fisher distance
_VARIANCE_FLOOR: float = 1e-6


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 on zero vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < _VARIANCE_FLOOR or norm_b < _VARIANCE_FLOOR:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _fisher_rao_similarity(
    mu_q: np.ndarray,
    mu_f: np.ndarray,
    var_f: np.ndarray,
    temperature: float = 15.0,
) -> float:
    """Fisher-Rao geodesic similarity on diagonal Gaussian manifold.

    d_FR^2 = sum_i [ (mu_q_i - mu_f_i)^2 / max(var_f_i, eps) ]
    similarity = exp(-d_FR^2 / temperature)

    When all variances are equal, this is equivalent to a scaled
    Euclidean distance — same ranking as cosine on normalized vectors.
    The advantage appears when variances differ across memories.
    """
    diff = mu_q - mu_f
    var_safe = np.maximum(var_f, _VARIANCE_FLOOR)
    d_sq = float(np.sum(diff * diff / var_safe))
    return math.exp(-d_sq / temperature)


class SemanticChannel:
    """Dense semantic retrieval via embedding similarity.

    Scans all facts for a profile. Uses a GRADUATED Fisher-Rao ramp:
    fresh facts (low access_count) use cosine, frequently-accessed facts
    transition to Fisher-Rao distance for uncertainty-aware similarity.

    V3.2: VectorStore KNN fast path when available, falls back to full scan.

    Graduated ramp: weight = min(1.2, access_count / 10 * 1.2)
    Final sim = fisher_weight * fisher_sim + (1 - fisher_weight) * cosine_sim
    """

    def __init__(
        self,
        db: DatabaseManager,
        fisher_temperature: float = 15.0,
        embedder: object | None = None,
        fisher_mode: str = "simplified",
        vector_store: Any | None = None,
    ) -> None:
        self._db = db
        self._temperature = fisher_temperature
        self._embedder = embedder
        self._fisher_mode = fisher_mode if fisher_mode in ("simplified", "full") else "simplified"
        # Lazily instantiated full metric (avoids import cost when not needed)
        self._full_metric: object | None = None
        self._vector_store = vector_store

    def search(
        self,
        query_embedding: list[float],
        profile_id: str,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        """Search for semantically similar facts.

        Uses VectorStore KNN if available, otherwise full-table scan.
        Fisher-Rao scoring preserved as post-KNN secondary signal.

        Args:
            query_embedding: Dense vector for the query.
            profile_id: Scope to this profile.
            top_k: Maximum results to return.

        Returns:
            List of (fact_id, score) sorted by score descending.
            Score is in [0, 1] range.
        """
        if not query_embedding:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)

        # --- FAST PATH: sqlite-vec KNN ---
        if self._vector_store and self._vector_store.available:
            results = self._search_via_vector_store(
                query_embedding, q_vec, profile_id, top_k,
            )
            if results:  # If vec0 returned results, use them
                return results
            # If vec0 is empty (cold start), fall through to full scan

        # --- FALLBACK: full-table scan (original code, unchanged) ---
        return self._search_full_scan(query_embedding, q_vec, profile_id, top_k)

    def _search_via_vector_store(
        self,
        query_embedding: list[float],
        q_vec: np.ndarray,
        profile_id: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """KNN via VectorStore, then Fisher-Rao re-scoring on top-K subset."""
        # Step 1: Fast KNN -- get 2x top_k candidates for Fisher re-ranking
        knn_results = self._vector_store.search(
            query_embedding, top_k=top_k * 2, profile_id=profile_id,
        )
        if not knn_results:
            return []  # Caller falls through to full scan

        # Step 2: Load only the candidate facts (NOT all facts)
        candidate_ids = [fid for fid, _ in knn_results]
        knn_scores = {fid: score for fid, score in knn_results}
        facts = self._db.get_facts_by_ids(candidate_ids, profile_id)

        if not facts:
            return [(fid, score) for fid, score in knn_results[:top_k]]

        # Step 3: Fisher-Rao re-scoring on the subset
        q_mean: np.ndarray | None = None
        q_var: np.ndarray | None = None
        if self._embedder and hasattr(self._embedder, 'compute_fisher_params'):
            qm, qv = self._embedder.compute_fisher_params(query_embedding)
            q_mean = np.array(qm, dtype=np.float32)
            q_var = np.array(qv, dtype=np.float32)

        scored: list[tuple[str, float]] = []
        for fact in facts:
            cos_sim = knn_scores.get(fact.fact_id, 0.0)

            # Graduated Fisher-Rao ramp (preserved from original)
            fisher_weight = min(1.2, (fact.access_count or 0) / 10.0 * 1.2)

            if (fisher_weight > 0.01
                    and fact.fisher_variance is not None
                    and fact.embedding is not None
                    and len(fact.fisher_variance) == len(q_vec)):
                f_vec = np.array(fact.embedding, dtype=np.float32)
                var_vec = np.array(fact.fisher_variance, dtype=np.float32)
                f_sim = self._compute_fisher_sim(
                    q_vec, f_vec, var_vec, fact, q_mean, q_var,
                )
                capped_w = min(1.0, fisher_weight)
                sim = capped_w * f_sim + (1.0 - capped_w) * cos_sim
            else:
                sim = cos_sim

            if sim > 0.3:
                scored.append((fact.fact_id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _search_full_scan(
        self,
        query_embedding: list[float],
        q_vec: np.ndarray,
        profile_id: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Original full-table-scan search. Used as fallback when VectorStore
        is unavailable or empty (cold start).
        """
        # Compute query Fisher params for Bayesian comparison (F45 fix)
        q_mean: np.ndarray | None = None
        q_var: np.ndarray | None = None
        if self._embedder and hasattr(self._embedder, 'compute_fisher_params'):
            qm, qv = self._embedder.compute_fisher_params(query_embedding)
            q_mean = np.array(qm, dtype=np.float32)
            q_var = np.array(qv, dtype=np.float32)

        facts = self._db.get_all_facts(profile_id)

        scored: list[tuple[str, float]] = []
        for fact in facts:
            if fact.embedding is None:
                continue

            f_vec = np.array(fact.embedding, dtype=np.float32)
            if f_vec.shape != q_vec.shape:
                continue

            # Cosine baseline (always computed)
            cos_sim = (_cosine_similarity(q_vec, f_vec) + 1.0) / 2.0

            # Graduated Fisher-Rao ramp (F37, F108)
            fisher_weight = min(1.2, (fact.access_count or 0) / 10.0 * 1.2)

            if (fisher_weight > 0.01
                    and fact.fisher_variance is not None
                    and len(fact.fisher_variance) == len(q_vec)):
                var_vec = np.array(fact.fisher_variance, dtype=np.float32)
                f_sim = self._compute_fisher_sim(
                    q_vec, f_vec, var_vec, fact, q_mean, q_var,
                )
                capped_w = min(1.0, fisher_weight)
                sim = capped_w * f_sim + (1.0 - capped_w) * cos_sim
            else:
                sim = cos_sim

            if sim > 0.3:
                scored.append((fact.fact_id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Fisher similarity dispatch
    # ------------------------------------------------------------------

    def _compute_fisher_sim(
        self,
        q_vec: np.ndarray,
        f_vec: np.ndarray,
        var_vec: np.ndarray,
        fact: AtomicFact,
        q_mean: np.ndarray | None,
        q_var: np.ndarray | None,
    ) -> float:
        """Compute Fisher-Rao similarity using simplified or full metric.

        Simplified (default): Mahalanobis-like distance using only fact variance.
        Full: Atkinson-Mitchell geodesic via FisherRaoMetric.similarity(),
              requires both query and fact (mean, variance) pairs.

        Falls back to simplified if full metric cannot be applied (e.g.
        missing fisher_mean on the fact, or missing query variance).
        """
        if self._fisher_mode == "full":
            return self._compute_full_fisher_sim(
                q_vec, f_vec, var_vec, fact, q_mean, q_var,
            )
        return _fisher_rao_similarity(q_vec, f_vec, var_vec, self._temperature)

    def _compute_full_fisher_sim(
        self,
        q_vec: np.ndarray,
        f_vec: np.ndarray,
        var_vec: np.ndarray,
        fact: AtomicFact,
        q_mean: np.ndarray | None,
        q_var: np.ndarray | None,
    ) -> float:
        """Full Atkinson-Mitchell geodesic via FisherRaoMetric.

        Requires fisher_mean on the fact AND query variance. If either is
        missing, falls back to the simplified local computation so that
        the graduated ramp still produces a score.
        """
        # Need fact fisher_mean for the full metric
        fact_mean = fact.fisher_mean
        if fact_mean is None or len(fact_mean) != len(f_vec):
            # No stored mean — fall back to simplified
            return _fisher_rao_similarity(q_vec, f_vec, var_vec, self._temperature)

        # Need query variance for the full metric
        if q_mean is None or q_var is None:
            return _fisher_rao_similarity(q_vec, f_vec, var_vec, self._temperature)

        if len(q_mean) != len(fact_mean) or len(q_var) != len(var_vec):
            return _fisher_rao_similarity(q_vec, f_vec, var_vec, self._temperature)

        metric = self._get_full_metric()
        try:
            return metric.similarity(
                q_mean.tolist(), q_var.tolist(),
                list(fact_mean), list(var_vec),
            )
        except (ValueError, FloatingPointError):
            # Numerical issue — fall back gracefully
            logger.debug("Full Fisher metric raised; falling back to simplified")
            return _fisher_rao_similarity(q_vec, f_vec, var_vec, self._temperature)

    def _get_full_metric(self) -> "FisherRaoMetric":  # noqa: F821
        """Lazy-load FisherRaoMetric to avoid import-time cost."""
        if self._full_metric is None:
            from superlocalmemory.math.fisher import FisherRaoMetric
            self._full_metric = FisherRaoMetric(temperature=self._temperature)
        return self._full_metric  # type: ignore[return-value]
