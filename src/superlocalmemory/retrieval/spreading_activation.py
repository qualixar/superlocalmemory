# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3

"""SYNAPSE spreading activation -- 5th retrieval channel.

SYNAPSE (arXiv 2601.02744) 5-step algorithm adapted for SLM.
Pure math -- no LLM calls at query time. With M=7, T=3 the
computation is ~21 neighbor lookups (<5ms on SQLite with indexes).

Reads BOTH graph_edges + association_edges via UNION query (Rule 13).
Registered as 5th channel via ChannelRegistry (needs_embedding=True).

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: MIT
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from superlocalmemory.storage.models import _new_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration (frozen dataclass, Rule 10)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpreadingActivationConfig:
    """Configuration for SYNAPSE spreading activation.

    All hyperparameters from the SYNAPSE paper (arXiv 2601.02744).
    SYNAPSE tuned on 384d (all-MiniLM-L6-v2). SLM uses 768d
    (nomic-embed-text). Phase 3 calibration test verifies convergence.
    """

    alpha: float = 1.0           # Seed scaling factor
    delta: float = 0.5           # Node retention / self-decay per iteration
    spreading_factor: float = 0.8  # S: energy diffusion rate
    theta: float = 0.5           # Activation threshold for sigmoid
    top_m: int = 7               # Lateral inhibition: max active nodes
    max_iterations: int = 3      # T: propagation depth
    tau_gate: float = 0.12       # FOK confidence gate
    enabled: bool = False        # Feature flag (Rule 12)


# ---------------------------------------------------------------------------
# SpreadingActivation Channel
# ---------------------------------------------------------------------------

class SpreadingActivation:
    """SYNAPSE 5-step spreading activation as 5th retrieval channel.

    Algorithm:
        Step 1: Initialization with ALPHA seed scaling
        Step 2: Propagation with fan effect (out-degree normalization)
        Step 3: Lateral inhibition (top-M=7 pruning)
        Step 4: Nonlinear sigmoid gating
        Step 5: Iterate T=3 times, then FOK gate

    Registered as 5th channel via ChannelRegistry (Rule 07).
    Reads BOTH graph_edges + association_edges via UNION query (Rule 13).
    """

    def __init__(
        self,
        db: Any,
        vector_store: Any,
        config: SpreadingActivationConfig | None = None,
    ) -> None:
        self._db = db
        self._vector_store = vector_store
        self._config = config or SpreadingActivationConfig()

    def search(
        self,
        query: Any,
        profile_id: str = "",
        top_k: int = 7,
    ) -> list[tuple[str, float]]:
        """Channel-compatible interface: (query, top_k) -> [(fact_id, score)].

        Matches ANNSearchable protocol (Rule 07).
        """
        if not self._config.enabled:
            return []

        try:
            # Step 0: Get seed nodes from VectorStore KNN
            seed_results = self._vector_store.search(
                query, top_k=self._config.top_m,
            )
            if not seed_results:
                return []

            # Check cache first
            query_hash = self._compute_query_hash(query, profile_id)
            cached = self._get_cached_results(query_hash, profile_id)
            if cached:
                return cached[:top_k]

            # Run 5-step spreading activation
            activations = self._propagate(seed_results, profile_id)

            # FOK gating
            if not self._fok_check(activations):
                return []

            # Cache results
            self._cache_results(query_hash, profile_id, activations)

            # Return top-K sorted by activation
            results = sorted(
                activations.items(), key=lambda x: x[1], reverse=True,
            )
            return results[:top_k]

        except Exception as exc:
            logger.debug(
                "SpreadingActivation.search failed for profile %s: %s",
                profile_id, exc,
            )
            return []

    def _propagate(
        self,
        seeds: list[tuple[str, float]],
        profile_id: str,
    ) -> dict[str, float]:
        """Execute the 5-step SYNAPSE algorithm.

        Step 1: a_i^(0) = alpha * sim(h_i, h_q) for seeds, 0 otherwise
        Step 2: u_i^(t+1) = delta * a_i^(t) + S * SUM(w_ji/deg(j) * a_j^(t))
        Step 3: Lateral inhibition -- keep top-M=7 only
        Step 4: sigmoid(u - theta)
        Step 5: Iterate T=3 times
        """
        cfg = self._config

        # Step 1: Initialization
        activations: dict[str, float] = {}
        for fact_id, similarity in seeds:
            activations[fact_id] = cfg.alpha * similarity

        # Precompute out-degrees for fan effect
        degree_cache: dict[str, int] = {}

        # Steps 2-4, repeated T times
        for _iteration in range(cfg.max_iterations):
            new_activations: dict[str, float] = {}

            for node_id, activation in activations.items():
                if activation < 0.001:
                    continue

                # Get neighbors from BOTH tables (Rule 13)
                neighbors = self._get_unified_neighbors(node_id, profile_id)

                # Out-degree for fan effect normalization
                if node_id not in degree_cache:
                    degree_cache[node_id] = max(len(neighbors), 1)
                out_degree = degree_cache[node_id]

                # Step 2: Propagation with fan effect
                for neighbor_id, edge_weight in neighbors:
                    spread = (
                        cfg.spreading_factor
                        * (edge_weight / out_degree)
                        * activation
                    )
                    new_activations[neighbor_id] = (
                        new_activations.get(neighbor_id, 0.0) + spread
                    )

            # Add self-retention (delta * current activation)
            for node_id, activation in activations.items():
                new_activations[node_id] = (
                    new_activations.get(node_id, 0.0) + cfg.delta * activation
                )

            # Step 3: Lateral inhibition -- keep only top-M
            sorted_nodes = sorted(
                new_activations.items(), key=lambda x: x[1], reverse=True,
            )
            top_m_nodes = sorted_nodes[: cfg.top_m]

            # Step 4: Nonlinear activation (sigmoid with threshold shift)
            activations = {}
            for node_id, raw_activation in top_m_nodes:
                gated = 1.0 / (1.0 + math.exp(-(raw_activation - cfg.theta)))
                activations[node_id] = gated

        return activations

    def _get_unified_neighbors(
        self, node_id: str, profile_id: str,
    ) -> list[tuple[str, float]]:
        """Get neighbors from BOTH graph_edges and association_edges.

        Uses bidirectional UNION query (Section 4 of LLD).
        """
        try:
            rows = self._db.execute(
                """
                SELECT target_id AS neighbor_id, weight FROM graph_edges
                    WHERE source_id = ? AND profile_id = ?
                UNION ALL
                SELECT target_fact_id AS neighbor_id, weight FROM association_edges
                    WHERE source_fact_id = ? AND profile_id = ?
                UNION ALL
                SELECT source_id AS neighbor_id, weight FROM graph_edges
                    WHERE target_id = ? AND profile_id = ?
                UNION ALL
                SELECT source_fact_id AS neighbor_id, weight FROM association_edges
                    WHERE target_fact_id = ? AND profile_id = ?
                """,
                (node_id, profile_id, node_id, profile_id,
                 node_id, profile_id, node_id, profile_id),
            )
            return [
                (dict(r)["neighbor_id"], dict(r)["weight"]) for r in rows
            ]
        except Exception as exc:
            logger.debug(
                "SpreadingActivation: UNION query failed for node %s "
                "profile %s: %s",
                node_id, profile_id, exc,
            )
            return []

    def _fok_check(self, activations: dict[str, float]) -> bool:
        """Feeling-of-Knowing gate.

        If max activation < tau_gate (0.12), reject results as noise.
        """
        if not activations:
            return False
        return max(activations.values()) >= self._config.tau_gate

    def _compute_query_hash(self, query: Any, profile_id: str) -> str:
        """Deterministic hash for cache key."""
        if isinstance(query, np.ndarray):
            data = query.tobytes() + profile_id.encode()
        elif isinstance(query, list):
            data = np.array(query, dtype=np.float32).tobytes() + profile_id.encode()
        else:
            data = str(query).encode() + profile_id.encode()
        return hashlib.sha256(data).hexdigest()[:16]

    def _get_cached_results(
        self, query_hash: str, profile_id: str,
    ) -> list[tuple[str, float]] | None:
        """Check activation_cache for recent results."""
        try:
            rows = self._db.execute(
                "SELECT node_id, activation_value FROM activation_cache "
                "WHERE profile_id = ? AND query_hash = ? "
                "AND expires_at > datetime('now') "
                "ORDER BY activation_value DESC",
                (profile_id, query_hash),
            )
            if not rows:
                return None
            return [
                (dict(r)["node_id"], dict(r)["activation_value"])
                for r in rows
            ]
        except Exception:
            return None

    def _cache_results(
        self,
        query_hash: str,
        profile_id: str,
        activations: dict[str, float],
    ) -> None:
        """Store results in activation_cache with 1-hour TTL."""
        try:
            for node_id, value in activations.items():
                self._db.execute(
                    "INSERT OR REPLACE INTO activation_cache "
                    "(cache_id, profile_id, query_hash, node_id, "
                    " activation_value, iteration, created_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, datetime('now'), "
                    "datetime('now', '+1 hour'))",
                    (_new_id(), profile_id, query_hash, node_id, value,
                     self._config.max_iterations),
                )
        except Exception as exc:
            logger.debug("Cache write failed: %s", exc)

    def cleanup_expired_cache(self) -> int:
        """Delete expired cache entries. Called by maintenance."""
        try:
            result = self._db.execute(
                "DELETE FROM activation_cache "
                "WHERE expires_at < datetime('now')",
                (),
            )
            return len(result) if result else 0
        except Exception:
            return 0
