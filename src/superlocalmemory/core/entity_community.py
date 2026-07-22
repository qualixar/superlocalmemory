# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Entity-community backbone (Wave Q) — the single principled clustering spine.

Best-in-market memory graphs (GraphRAG, Graphiti/Zep) cluster the ENTITY
graph, not the raw fact/chunk graph. SLM's fact graph is capped-sparse
(5 edges/entity), so a fact-level clustering fragments. This module builds an
entity co-occurrence graph (two entities are linked when they appear together
in a fact, weighted by co-occurrence count) and runs Louvain community
detection over it — the correct target and, since entities are far fewer than
facts, a cheaper computation.

The resulting entity communities are the shared backbone for:
  - Q2 community summaries (one synthesized report per community), and
  - Q3 progressive abstraction (scenario/persona tiers + drill-down).

Runs in the background (consolidation lane), never on the hot recall path.
Fail-open and idempotent: a recompute fully replaces a profile's rows.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class EntityCommunityBuilder:
    """Build + persist entity communities via Louvain over co-occurrence."""

    def __init__(
        self,
        db: Any,
        min_community_size: int = 2,
        resolution: float = 1.0,
        seed: int = 42,
    ) -> None:
        self._db = db
        self._min_size = max(2, int(min_community_size))
        self._resolution = float(resolution)
        self._seed = int(seed)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _cooccurrence(self, profile_id: str) -> Counter:
        """Count entity pairs that co-occur within a fact (per profile)."""
        rows = self._db.execute(
            "SELECT canonical_entities_json FROM atomic_facts "
            "WHERE profile_id = ?",
            (profile_id,),
        )
        weights: Counter = Counter()
        for row in rows:
            raw = dict(row).get("canonical_entities_json")
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                continue
            if not isinstance(parsed, list):
                continue
            ents = sorted({str(e).strip() for e in parsed if str(e).strip()})
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    weights[(ents[i], ents[j])] += 1
        return weights

    def detect(self, profile_id: str) -> dict[str, int]:
        """Return {entity_id -> community_id}; communities below min size drop."""
        weights = self._cooccurrence(profile_id)
        if not weights:
            return {}

        import networkx as nx

        g = nx.Graph()
        for (a, b), w in weights.items():
            g.add_edge(a, b, weight=w)

        try:
            from networkx.algorithms.community import louvain_communities

            communities = louvain_communities(
                g, weight="weight", resolution=self._resolution, seed=self._seed,
            )
        except Exception as exc:  # pragma: no cover - fallback path
            logger.debug(
                "Louvain unavailable/failed (%s); using connected components",
                exc,
            )
            communities = nx.connected_components(g)

        result: dict[str, int] = {}
        cid = 0
        for comm in communities:
            members = list(comm)
            if len(members) < self._min_size:
                continue
            for node in members:
                result[node] = cid
            cid += 1
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def compute_and_store(self, profile_id: str) -> dict[str, int]:
        """Detect communities and replace this profile's stored rows."""
        mapping = self.detect(profile_id)
        try:
            self._db.execute(
                "DELETE FROM entity_communities WHERE profile_id = ?",
                (profile_id,),
            )
            for entity_id, community_id in mapping.items():
                self._db.execute(
                    "INSERT OR REPLACE INTO entity_communities "
                    "(profile_id, entity_id, community_id, computed_at) "
                    "VALUES (?, ?, ?, datetime('now'))",
                    (profile_id, entity_id, community_id),
                )
        except Exception as exc:
            logger.debug("entity_communities persist failed: %s", exc)

        return {
            "entity_count": len(mapping),
            "community_count": len(set(mapping.values())),
        }

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_communities(self, profile_id: str) -> dict[int, list[str]]:
        """Return {community_id -> [entity_id, ...]} for a profile."""
        try:
            rows = self._db.execute(
                "SELECT entity_id, community_id FROM entity_communities "
                "WHERE profile_id = ? ORDER BY community_id",
                (profile_id,),
            )
        except Exception as exc:
            logger.debug("get_communities failed: %s", exc)
            return {}
        out: dict[int, list[str]] = defaultdict(list)
        for row in rows:
            d = dict(row)
            out[int(d["community_id"])].append(str(d["entity_id"]))
        return dict(out)

    def get_community_for_entity(
        self, entity_id: str, profile_id: str,
    ) -> int | None:
        """Return the community id for one entity, or None."""
        try:
            rows = self._db.execute(
                "SELECT community_id FROM entity_communities "
                "WHERE profile_id = ? AND entity_id = ?",
                (profile_id, entity_id),
            )
        except Exception as exc:
            logger.debug("get_community_for_entity failed: %s", exc)
            return None
        for row in rows:
            return int(dict(row)["community_id"])
        return None
