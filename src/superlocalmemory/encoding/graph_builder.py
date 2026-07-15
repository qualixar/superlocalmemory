# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Knowledge graph construction — 4 edge types, zero memory limits.

Builds edges between AtomicFacts at encoding time. V1 only checked the
last 50 memories; this version searches ALL facts per canonical entity.

Edge types: ENTITY (shared entity, weight 1.0), TEMPORAL (exp-decay,
1-week window), SEMANTIC (ANN cosine > 0.7), CAUSAL (causal markers,
weight 0.8). CONTRADICTION exposed for external Sheaf module (Wave 4).

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""
from __future__ import annotations

import logging
import math
import re
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, EdgeType, GraphEdge

logger = logging.getLogger(__name__)

# Constants
_TEMPORAL_MAX_HOURS: float = 168.0   # 1 week
_TEMPORAL_DECAY_DAYS: float = 30.0   # exp(-delta_days / 30)
_SEMANTIC_THRESHOLD: float = 0.7
_SEMANTIC_TOP_K: int = 5
_CAUSAL_WEIGHT: float = 0.8
_ENTITY_WEIGHT: float = 1.0

# Causal cue patterns — longer phrases first to avoid partial matches.
_CAUSAL_CUES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bbecause of\b", re.I),
    re.compile(r"\bas a result\b", re.I),
    re.compile(r"\bresulted in\b", re.I),
    re.compile(r"\bin order to\b", re.I),
    re.compile(r"\bso that\b", re.I),
    re.compile(r"\bcaused by\b", re.I),
    re.compile(r"\bdue to\b", re.I),
    re.compile(r"\bled to\b", re.I),
    re.compile(r"\btherefore\b", re.I),
    re.compile(r"\bconsequently\b", re.I),
    re.compile(r"\bbecause\b", re.I),
)


@runtime_checkable
class ANNSearchable(Protocol):
    """Minimal protocol for approximate-nearest-neighbor search."""

    def search(self, query: Any, top_k: int = 5) -> list[tuple[str, float]]:
        """Return (fact_id, similarity_score) pairs."""
        ...  # pragma: no cover


class GraphBuilder:
    """Build knowledge-graph edges for newly stored facts.

    Searches the ENTIRE corpus per canonical entity (not just last 50).
    Adds semantic similarity edges via optional ANN index and detects
    causal language for directed cause/effect edges.
    """

    def __init__(self, db: DatabaseManager, ann_index: ANNSearchable | None = None) -> None:
        self._db = db
        self._ann = ann_index

    # -- Public API --------------------------------------------------------

    def build_edges(self, new_fact: AtomicFact, profile_id: str) -> list[GraphEdge]:
        """Create ALL relevant edges for *new_fact*. Persists and returns them."""
        hub_cache: dict[str, int] = {}  # shared across edge types to avoid redundant queries
        edges: list[GraphEdge] = []
        edges.extend(self._build_entity_edges(new_fact, profile_id, hub_cache))
        edges.extend(self._build_temporal_edges(new_fact, profile_id))
        edges.extend(self._build_semantic_edges(new_fact, profile_id))
        edges.extend(self._build_causal_edges(new_fact, profile_id, hub_cache))

        for edge in edges:
            self._db.store_edge(edge)

        if edges:
            logger.debug(
                "GraphBuilder: %d edges for %s (E=%d T=%d S=%d C=%d)",
                len(edges), new_fact.fact_id,
                sum(1 for e in edges if e.edge_type == EdgeType.ENTITY),
                sum(1 for e in edges if e.edge_type == EdgeType.TEMPORAL),
                sum(1 for e in edges if e.edge_type == EdgeType.SEMANTIC),
                sum(1 for e in edges if e.edge_type == EdgeType.CAUSAL),
            )
        return edges

    def add_contradiction_edge(
        self, fact_id_a: str, fact_id_b: str, profile_id: str,
        severity: float = 1.0,
    ) -> GraphEdge:
        """Add a contradiction edge. Called by Sheaf module (Wave 4)."""
        edge = GraphEdge(
            profile_id=profile_id,
            source_id=fact_id_a,
            target_id=fact_id_b,
            edge_type=EdgeType.CONTRADICTION,
            weight=max(0.0, min(1.0, severity)),
        )
        self._db.store_edge(edge)
        logger.info("Contradiction %s -> %s (%.2f)", fact_id_a, fact_id_b, severity)
        return edge

    def get_graph_stats(self, profile_id: str) -> dict[str, Any]:
        """Edge counts by type, node count, average degree."""
        rows = self._db.execute(
            "SELECT edge_type, COUNT(*) AS cnt FROM graph_edges "
            "WHERE profile_id = ? GROUP BY edge_type", (profile_id,),
        )
        edge_counts: dict[str, int] = {
            dict(r)["edge_type"]: int(dict(r)["cnt"]) for r in rows
        }
        total_edges = sum(edge_counts.values())

        node_rows = self._db.execute(
            "SELECT COUNT(DISTINCT n) AS c FROM ("
            "  SELECT source_id AS n FROM graph_edges WHERE profile_id = ? "
            "  UNION "
            "  SELECT target_id AS n FROM graph_edges WHERE profile_id = ?"
            ")", (profile_id, profile_id),
        )
        node_count = int(dict(node_rows[0])["c"]) if node_rows else 0
        avg_degree = (2.0 * total_edges / node_count) if node_count > 0 else 0.0

        return {
            "edge_counts": edge_counts,
            "total_edges": total_edges,
            "node_count": node_count,
            "avg_degree": round(avg_degree, 2),
        }

    # -- Edge builders (private) -------------------------------------------

    # v3.4.59: Lowered from 20→5. At 17K facts, 20 edges × many entities per fact
    # produces 1.4M+ entity edges and 13s recalls. 5 is sufficient graph signal.
    _MAX_ENTITY_EDGES_PER_ENTITY: int = 5
    # v3.4.59: Hub filter — nodes with > 200 total edges are "gravity wells"
    # (e.g. every fact mentions "SLM"). Skip adding more edges to them.
    _MAX_HUB_DEGREE: int = 200

    def _node_degree(self, fact_id: str, profile_id: str, cache: dict[str, int]) -> int:
        """Cached total degree (in + out) for a node."""
        if fact_id not in cache:
            rows = self._db.execute(
                "SELECT COUNT(*) as cnt FROM graph_edges "
                "WHERE profile_id = ? AND (source_id = ? OR target_id = ?)",
                (profile_id, fact_id, fact_id),
            )
            cache[fact_id] = int(dict(rows[0])["cnt"]) if rows else 0
        return cache[fact_id]

    def _build_entity_edges(
        self, new_fact: AtomicFact, profile_id: str,
        hub_cache: dict[str, int] | None = None,
    ) -> list[GraphEdge]:
        """ENTITY edges: shared canonical entity — capped to most recent per entity."""
        if not new_fact.canonical_entities:
            return []
        if hub_cache is None:
            hub_cache = {}
        edges: list[GraphEdge] = []
        seen: set[str] = set()

        for entity_id in new_fact.canonical_entities:
            entity_edge_count = 0
            for other in self._db.get_facts_by_entity(entity_id, profile_id):
                if entity_edge_count >= self._MAX_ENTITY_EDGES_PER_ENTITY:
                    break
                if other.fact_id == new_fact.fact_id or other.fact_id in seen:
                    continue
                if self._node_degree(other.fact_id, profile_id, hub_cache) >= self._MAX_HUB_DEGREE:
                    continue  # skip hub nodes — they link everything to everything
                if self._edge_exists(new_fact.fact_id, other.fact_id, EdgeType.ENTITY, profile_id):
                    continue
                seen.add(other.fact_id)
                edges.append(GraphEdge(
                    profile_id=profile_id, source_id=new_fact.fact_id,
                    target_id=other.fact_id, edge_type=EdgeType.ENTITY,
                    weight=_ENTITY_WEIGHT,
                ))
                entity_edge_count += 1
        return edges

    def _build_temporal_edges(
        self, new_fact: AtomicFact, profile_id: str,
    ) -> list[GraphEdge]:
        """TEMPORAL edges: bidirectional, exp-decay, 1-week window per entity.

        Only creates temporal edges when an explicit observation_date is set.
        Falling back to created_at would produce spurious temporal edges for
        facts that have no real temporal context.
        """
        if not new_fact.observation_date or not new_fact.canonical_entities:
            return []
        new_dt = _parse_date(new_fact.observation_date)
        if new_dt is None:
            return []

        edges: list[GraphEdge] = []
        seen_pairs: set[tuple[str, str]] = set()

        for entity_id in new_fact.canonical_entities:
            temporal_edge_count = 0
            for other in self._db.get_facts_by_entity(entity_id, profile_id):
                if temporal_edge_count >= self._MAX_ENTITY_EDGES_PER_ENTITY:
                    break  # V3.3.12: cap temporal edges like entity edges
                if other.fact_id == new_fact.fact_id:
                    continue
                other_dt = _parse_date(other.observation_date)
                if other_dt is None:
                    continue

                delta_hours = abs((new_dt - other_dt).total_seconds()) / 3600.0
                if delta_hours > _TEMPORAL_MAX_HOURS:
                    continue

                pair_key = (min(new_fact.fact_id, other.fact_id),
                            max(new_fact.fact_id, other.fact_id))
                if pair_key in seen_pairs:
                    continue
                if self._edge_exists(new_fact.fact_id, other.fact_id, EdgeType.TEMPORAL, profile_id):
                    seen_pairs.add(pair_key)
                    continue

                weight = round(max(math.exp(-(delta_hours / 24.0) / _TEMPORAL_DECAY_DAYS), 0.01), 4)
                seen_pairs.add(pair_key)

                # Forward: new -> other
                edges.append(GraphEdge(
                    profile_id=profile_id, source_id=new_fact.fact_id,
                    target_id=other.fact_id, edge_type=EdgeType.TEMPORAL,
                    weight=weight,
                ))
                temporal_edge_count += 1
                # Reverse: other -> new
                if not self._edge_exists(other.fact_id, new_fact.fact_id, EdgeType.TEMPORAL, profile_id):
                    edges.append(GraphEdge(
                        profile_id=profile_id, source_id=other.fact_id,
                        target_id=new_fact.fact_id, edge_type=EdgeType.TEMPORAL,
                        weight=weight,
                    ))
        return edges

    def _build_semantic_edges(
        self, new_fact: AtomicFact, profile_id: str,
    ) -> list[GraphEdge]:
        """SEMANTIC edges: ANN embedding similarity > 0.7 threshold."""
        if self._ann is None or new_fact.embedding is None:
            return []
        try:
            import numpy as np
            query_vec = np.asarray(new_fact.embedding, dtype=np.float32)
        except (ImportError, ValueError):
            return []

        edges: list[GraphEdge] = []
        for fact_id, score in self._ann.search(query_vec, top_k=_SEMANTIC_TOP_K + 1):
            if fact_id == new_fact.fact_id or score < _SEMANTIC_THRESHOLD:
                continue
            if self._edge_exists(new_fact.fact_id, fact_id, EdgeType.SEMANTIC, profile_id):
                continue
            edges.append(GraphEdge(
                profile_id=profile_id, source_id=new_fact.fact_id,
                target_id=fact_id, edge_type=EdgeType.SEMANTIC,
                weight=round(float(score), 4),
            ))
            if len(edges) >= _SEMANTIC_TOP_K:
                break
        return edges

    # v3.4.59: Lowered from 20→5, same reasoning as entity cap.
    _MAX_CAUSAL_EDGES_PER_ENTITY: int = 5

    def _build_causal_edges(
        self, new_fact: AtomicFact, profile_id: str,
        hub_cache: dict[str, int] | None = None,
    ) -> list[GraphEdge]:
        """CAUSAL edges: causal markers + shared entity. Direction: cause -> effect."""
        if not any(p.search(new_fact.content) for p in _CAUSAL_CUES):
            return []
        if not new_fact.canonical_entities:
            return []
        if hub_cache is None:
            hub_cache = {}

        edges: list[GraphEdge] = []
        seen: set[str] = set()
        for entity_id in new_fact.canonical_entities:
            causal_edge_count = 0
            for other in self._db.get_facts_by_entity(entity_id, profile_id):
                if causal_edge_count >= self._MAX_CAUSAL_EDGES_PER_ENTITY:
                    break
                if other.fact_id == new_fact.fact_id or other.fact_id in seen:
                    continue
                if self._node_degree(other.fact_id, profile_id, hub_cache) >= self._MAX_HUB_DEGREE:
                    continue
                if self._edge_exists(other.fact_id, new_fact.fact_id, EdgeType.CAUSAL, profile_id):
                    continue
                seen.add(other.fact_id)
                edges.append(GraphEdge(
                    profile_id=profile_id, source_id=other.fact_id,
                    target_id=new_fact.fact_id, edge_type=EdgeType.CAUSAL,
                    weight=_CAUSAL_WEIGHT,
                ))
                causal_edge_count += 1
        return edges

    # -- Helpers -----------------------------------------------------------

    def _edge_exists(
        self, source_id: str, target_id: str,
        edge_type: EdgeType, profile_id: str,
    ) -> bool:
        """Check if an edge already exists (prevents duplicates)."""
        rows = self._db.execute(
            "SELECT 1 FROM graph_edges "
            "WHERE profile_id = ? AND source_id = ? AND target_id = ? "
            "AND edge_type = ? LIMIT 1",
            (profile_id, source_id, target_id, edge_type.value),
        )
        return len(rows) > 0


def _parse_date(raw: str | None) -> datetime | None:
    """Best-effort ISO-8601 datetime parse."""
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None
