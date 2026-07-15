# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Bridge Discovery + Spreading Activation.

Connects disconnected retrieval results via intermediate facts.
Combines AriadneMem's 5-step bridging with Hindsight's TEMPR activation.

Algorithm:
1. Sort seed results chronologically
2. For consecutive pairs, check entity overlap + temporal proximity
3. If disconnected: try entity/keyword/proper-noun bridge strategies
4. Add bridge facts with inferred edges
5. Spreading activation from seeds through graph

Parameters:
- max_depth=3 hops
- node_budget=8-25
- time_window=1-168 hours
- decay=0.7 per hop
- typed mu: entity=1.2, causal=1.3, semantic=0.8, temporal=0.9

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from superlocalmemory.retrieval.scope_policy import (
    authorized_fact_ids,
    filter_authorized_results,
)

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)

# Spreading activation parameters (Hindsight TEMPR)
_DECAY: float = 0.7
_TYPED_MU: dict[str, float] = {
    "entity": 1.2,
    "causal": 1.3,
    "semantic": 0.8,
    "temporal": 0.9,
    "supersedes": 0.0,
}
_MAX_DEPTH: int = 4
_NODE_BUDGET: int = 50


class BridgeDiscovery:
    """Connect disconnected retrieval results via graph paths.

    Usage::
        bridge = BridgeDiscovery(db)
        expanded = bridge.discover(seed_fact_ids, profile_id)
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def discover(
        self,
        seed_ids: list[str],
        profile_id: str,
        max_bridges: int = 10,
        *,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> list[tuple[str, float]]:
        """Find bridge facts connecting seed results.

        Args:
            seed_ids: Fact IDs from initial retrieval.
            profile_id: Scope to this profile.
            max_bridges: Maximum bridge facts to return.

        Returns:
            List of (fact_id, bridge_score) for discovered bridges.
        """
        if len(seed_ids) < 2:
            return []

        allowed_seeds = authorized_fact_ids(
            self._db,
            seed_ids,
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        if any(seed_id not in allowed_seeds for seed_id in seed_ids):
            return []

        try:
            visible_seed_facts = self._db.get_facts_by_ids(
                seed_ids,
                profile_id,
                include_global=include_global,
                include_shared=include_shared,
            )
        except Exception:
            return []
        seed_facts = {fact.fact_id: fact for fact in visible_seed_facts}
        if any(seed_id not in seed_facts for seed_id in seed_ids):
            return []

        bridges: list[tuple[str, float]] = []
        seen = set(seed_ids)

        # Check consecutive pairs for entity overlap
        for i in range(len(seed_ids) - 1):
            fact_a = seed_facts.get(seed_ids[i])
            fact_b = seed_facts.get(seed_ids[i + 1])
            if not fact_a or not fact_b:
                continue

            entities_a = set(fact_a.canonical_entities)
            entities_b = set(fact_b.canonical_entities)

            # If they share entities, no bridge needed
            if entities_a & entities_b:
                continue

            # Strategy 1: Entity bridge (union minus intersection)
            bridge_entities = (entities_a | entities_b) - (entities_a & entities_b)
            for eid in bridge_entities:
                entity_facts = self._db.get_facts_by_entity(
                    eid,
                    profile_id,
                    include_global=include_global,
                    include_shared=include_shared,
                )
                for f in entity_facts[:5]:
                    if f.fact_id not in seen:
                        seen.add(f.fact_id)
                        overlap = (
                            len(set(f.canonical_entities) & entities_a)
                            + len(set(f.canonical_entities) & entities_b)
                        )
                        bridges.append((f.fact_id, min(1.0, 0.5 + overlap * 0.15)))

            if len(bridges) >= max_bridges:
                break

        bridges.sort(key=lambda x: x[1], reverse=True)
        return filter_authorized_results(
            self._db,
            bridges,
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )[:max_bridges]

    def spreading_activation(
        self,
        seed_ids: list[str],
        profile_id: str,
        max_depth: int = _MAX_DEPTH,
        budget: int = _NODE_BUDGET,
        *,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> list[tuple[str, float]]:
        """Spreading activation from seed facts through the graph.

        At each hop, activation decays by _DECAY and is modulated by
        edge type via _TYPED_MU.

        Args:
            seed_ids: Starting fact IDs.
            profile_id: Scope.
            max_depth: Maximum hops.
            budget: Maximum nodes to return.

        Returns:
            List of (fact_id, activation_score) for activated facts.
        """
        activations: dict[str, float] = {fid: 1.0 for fid in seed_ids}
        frontier = list(seed_ids)

        for depth in range(max_depth):
            next_frontier: list[str] = []
            for fid in frontier:
                current_activation = activations.get(fid, 0.0)
                if current_activation < 0.01:
                    continue

                edges = self._db.get_edges_for_node(
                    fid,
                    profile_id,
                    include_global=include_global,
                    include_shared=include_shared,
                )
                for edge in edges:
                    other_id = (
                        edge.target_id
                        if edge.source_id == fid
                        else edge.source_id
                    )
                    mu = _TYPED_MU.get(edge.edge_type.value, 0.8)
                    propagated = current_activation * _DECAY * mu

                    if propagated > activations.get(other_id, 0.0):
                        activations[other_id] = propagated
                        if other_id not in seed_ids:
                            next_frontier.append(other_id)

            frontier = next_frontier
            if not frontier:
                break

        # Return non-seed nodes with activation
        results = [
            (fid, score)
            for fid, score in activations.items()
            if fid not in set(seed_ids) and score > 0.01
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return filter_authorized_results(
            self._db,
            results,
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )[:budget]
