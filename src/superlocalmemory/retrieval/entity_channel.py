# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Entity Graph Channel with Spreading Activation.

SA-RAG pattern: entities from query -> canonical lookup -> graph traversal
with decay. Handles BOTH uppercase and lowercase entity mentions.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: MIT
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superlocalmemory.encoding.entity_resolver import EntityResolver
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)

_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{1,}\b")

_ENTITY_STOP: frozenset[str] = frozenset({
    # Expanded stop list for query entity extraction
    "what", "when", "where", "who", "which", "how", "does", "did",
    "the", "that", "this", "there", "then", "than", "they", "them",
    "have", "has", "had", "been", "being", "about", "after", "before",
    "from", "into", "with", "some", "other", "would", "could", "should",
    "will", "because", "also", "just", "like", "know", "think",
    "feel", "want", "need", "make", "take", "give", "tell", "said",
    "wow", "gonna", "got", "by", "thanks", "thank", "hey", "hi",
    "hello", "bye", "good", "great", "nice", "cool", "right",
    "let", "can", "might", "much", "many", "more", "most",
    "something", "anything", "everything", "nothing", "someone",
    "it", "my", "your", "our", "their", "me", "you", "we", "us",
    "do", "if", "or", "no", "to", "at", "on", "in", "so",
    "go", "come", "see", "look", "say", "ask", "try", "keep",
    "yes", "yeah", "sure", "okay", "ok", "really", "actually",
    "maybe", "well", "still", "even", "very",
})


def extract_query_entities(query: str) -> list[str]:
    """Extract entity candidates from query (handles both cases).

    Strategy: find proper nouns in original + title-cased text,
    plus quoted phrases. Deduplicates case-insensitively.
    """
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        lo = name.lower()
        if lo not in seen and lo not in _ENTITY_STOP and len(name) >= 2:
            seen.add(lo)
            candidates.append(name)

    for m in _PROPER_NOUN_RE.finditer(query):
        _add(m.group(0))
    for m in _PROPER_NOUN_RE.finditer(query.title()):
        _add(m.group(0))
    for m in re.finditer(r'"([^"]+)"', query):
        _add(m.group(1).strip())

    return candidates


class EntityGraphChannel:
    """Entity-based retrieval with spreading activation (SA-RAG).

    V3.3.9: In-memory adjacency cache for O(1) edge lookup.
    Replaces per-node SQLite queries (23ms each) with dict lookup (<0.001ms).
    The cache is loaded once per profile and invalidated on store/edge changes.
    Memory cost: ~18 MB for 232K edges. Zero quality change — same algorithm.
    """

    def __init__(
        self, db: DatabaseManager,
        entity_resolver: EntityResolver | None = None,
        decay: float = 0.7, activation_threshold: float = 0.1,
        max_hops: int = 3,
    ) -> None:
        self._db = db
        self._resolver = entity_resolver
        self._decay = decay
        self._threshold = activation_threshold
        self._max_hops = max_hops
        # In-memory adjacency: {node_id -> [(neighbor_id, weight), ...]}
        self._adj: dict[str, list[tuple[str, float]]] = {}
        self._adj_profile: str = ""  # Track which profile is loaded
        self._adj_edge_count: int = 0  # Track edge count for staleness detection

    def _ensure_adjacency(self, profile_id: str) -> None:
        """Load graph adjacency into memory for fast spreading activation.

        Loads ALL edges for a profile into a bidirectional dict.
        Called once per profile switch or when edge count changes (new store).
        Cost: ~1s for 232K edges, ~18 MB RAM.
        """
        # Check staleness: profile changed or new edges added since last load
        current_count = self._get_edge_count(profile_id)
        if (self._adj_profile == profile_id
                and self._adj
                and self._adj_edge_count == current_count):
            return
        adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
        try:
            rows = self._db.execute(
                "SELECT source_id, target_id, weight FROM graph_edges WHERE profile_id = ?",
                (profile_id,),
            )
        except Exception:
            rows = []
        for r in rows:
            d = dict(r)
            s, t, w = d["source_id"], d["target_id"], float(d["weight"])
            adj[s].append((t, w))
            adj[t].append((s, w))
        self._adj = dict(adj)  # Convert defaultdict to regular dict (no accidental growth)
        self._adj_profile = profile_id
        self._adj_edge_count = current_count
        # Also load entity maps (same staleness lifecycle)
        self._load_entity_maps(profile_id)

        logger.info(
            "Loaded adjacency cache: %d nodes, %d edges, %d entity mappings for profile %s",
            len(self._adj), sum(len(v) for v in self._adj.values()) // 2,
            len(self._entity_to_facts), profile_id,
        )

    def _get_edge_count(self, profile_id: str) -> int:
        """Fast edge count for staleness check (~1ms)."""
        try:
            rows = self._db.execute(
                "SELECT COUNT(*) as cnt FROM graph_edges WHERE profile_id = ?",
                (profile_id,),
            )
            if rows:
                return int(dict(rows[0]).get("cnt", 0))
        except Exception:
            pass
        return 0

    def _load_entity_maps(self, profile_id: str) -> None:
        """Pre-load entity→fact and fact→entity maps into memory.

        Eliminates per-entity and per-fact SQL in the spreading activation loop.
        Same data, same algorithm — zero quality change.
        """
        # entity_id -> [fact_id, ...]
        self._entity_to_facts: dict[str, list[str]] = defaultdict(list)
        # fact_id -> [entity_id, ...]
        self._fact_to_entities: dict[str, list[str]] = defaultdict(list)

        try:
            rows = self._db.execute(
                "SELECT fact_id, canonical_entities_json FROM atomic_facts "
                "WHERE profile_id = ? AND canonical_entities_json IS NOT NULL "
                "AND canonical_entities_json != ''",
                (profile_id,),
            )
        except Exception:
            rows = []
        for r in rows:
            d = dict(r)
            fid = d["fact_id"]
            raw = d.get("canonical_entities_json")
            if not raw:
                continue
            try:
                eids = json.loads(raw)
                for eid in eids:
                    self._entity_to_facts[eid].append(fid)
                    self._fact_to_entities[fid].append(eid)
            except (ValueError, TypeError):
                continue

        logger.info(
            "Loaded entity maps: %d entities, %d facts with entities",
            len(self._entity_to_facts), len(self._fact_to_entities),
        )

    def invalidate_cache(self) -> None:
        """Clear all caches. Call after adding/removing edges or facts."""
        self._adj.clear()
        self._adj_profile = ""
        self._adj_edge_count = 0
        self._entity_to_facts = defaultdict(list)
        self._fact_to_entities = defaultdict(list)

    def search(self, query: str, profile_id: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Search via entity graph with spreading activation.

        V3.3.9: Uses in-memory adjacency for O(1) edge lookups.
        Same algorithm as before — zero quality change.
        """
        raw_entities = extract_query_entities(query)
        if not raw_entities:
            return []

        canonical_ids = self._resolve_entities(raw_entities, profile_id)
        if not canonical_ids:
            return []

        # Load adjacency cache (no-op if already loaded for this profile)
        self._ensure_adjacency(profile_id)

        # Seed activation from direct entity-linked facts
        # Use in-memory map when available, fall back to SQL for mock/test DBs
        activation: dict[str, float] = defaultdict(float)
        visited_entities: set[str] = set(canonical_ids)

        use_cache = bool(self._entity_to_facts)
        for eid in canonical_ids:
            if use_cache:
                for fid in self._entity_to_facts.get(eid, ()):
                    activation[fid] = max(activation[fid], 1.0)
            else:
                for fact in self._db.get_facts_by_entity(eid, profile_id):
                    activation[fact.fact_id] = max(activation[fact.fact_id], 1.0)

        # Spreading activation through graph edges (all in-memory O(1) lookups)
        frontier = set(activation.keys())
        for hop in range(1, self._max_hops):
            hop_decay = self._decay ** hop
            if hop_decay < self._threshold:
                break
            next_frontier: set[str] = set()

            for fid in frontier:
                if use_cache:
                    neighbors = self._adj.get(fid, ())
                    for neighbor, _weight in neighbors:
                        propagated = activation[fid] * self._decay
                        if propagated >= self._threshold and propagated > activation.get(neighbor, 0.0):
                            activation[neighbor] = propagated
                            next_frontier.add(neighbor)
                else:
                    for edge in self._db.get_edges_for_node(fid, profile_id):
                        neighbor = edge.target_id if edge.source_id == fid else edge.source_id
                        propagated = activation[fid] * self._decay
                        if propagated >= self._threshold and propagated > activation.get(neighbor, 0.0):
                            activation[neighbor] = propagated
                            next_frontier.add(neighbor)

            # Discover new entities from activated facts
            if use_cache:
                new_eids: list[str] = []
                for fid in frontier:
                    for eid in self._fact_to_entities.get(fid, ()):
                        if eid not in visited_entities:
                            visited_entities.add(eid)
                            new_eids.append(eid)
                for eid in new_eids:
                    for fid in self._entity_to_facts.get(eid, ()):
                        if hop_decay > activation.get(fid, 0.0):
                            activation[fid] = hop_decay
                            next_frontier.add(fid)
            else:
                # SQL fallback (mock/test DBs)
                new_eids_sql = self._discover_entities(frontier, profile_id, visited_entities)
                for eid in new_eids_sql:
                    visited_entities.add(eid)
                    for fact in self._db.get_facts_by_entity(eid, profile_id):
                        if hop_decay > activation.get(fact.fact_id, 0.0):
                            activation[fact.fact_id] = hop_decay
                            next_frontier.add(fact.fact_id)

            frontier = next_frontier
            if not frontier:
                break

        results = [(fid, sc) for fid, sc in activation.items() if sc >= self._threshold]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _resolve_entities(self, raw: list[str], profile_id: str) -> list[str]:
        """Resolve raw names to canonical entity IDs."""
        ids: list[str] = []
        seen: set[str] = set()
        if self._resolver is not None:
            for eid in self._resolver.resolve(raw, profile_id).values():
                if eid not in seen:
                    seen.add(eid)
                    ids.append(eid)
        else:
            for name in raw:
                ent = self._db.get_entity_by_name(name, profile_id)
                if ent and ent.entity_id not in seen:
                    seen.add(ent.entity_id)
                    ids.append(ent.entity_id)
        return ids

    def _discover_entities(
        self, fact_ids: set[str], profile_id: str, visited: set[str],
    ) -> list[str]:
        """Find new canonical entity IDs referenced by a set of facts."""
        new: list[str] = []
        seen = set(visited)
        for fid in fact_ids:
            rows = self._db.execute(
                "SELECT canonical_entities_json FROM atomic_facts WHERE fact_id = ?", (fid,),
            )
            if not rows:
                continue
            raw = dict(rows[0]).get("canonical_entities_json")
            if not raw:
                continue
            try:
                for eid in json.loads(raw):
                    if eid not in seen:
                        seen.add(eid)
                        new.append(eid)
            except (ValueError, TypeError):
                continue
        return new
