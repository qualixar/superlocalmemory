# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Entity Graph Channel with Spreading Activation.

SA-RAG pattern: entities from query -> canonical lookup -> graph traversal
with decay. Handles BOTH uppercase and lowercase entity mentions.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from superlocalmemory.retrieval import spreading
from superlocalmemory.retrieval.scope_policy import (
    authorized_fact_ids,
    filter_authorized_results,
)
from superlocalmemory.storage.database import (
    _scope_where,
    _unbounded_facts_ceiling,
)

if TYPE_CHECKING:
    from superlocalmemory.encoding.entity_resolver import EntityResolver
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


def _adj_ttl_seconds() -> float:
    """In-memory adjacency-cache TTL (seconds), env-overridable.

    v3.8.5: raised from a hard-coded 300s to 3600s. The TTL only exists to
    catch edge-WEIGHT mutations (pruning / MAX-merge) that leave the edge COUNT
    unchanged — new memories already force a reload via the count check. At 300s
    a 208K-edge graph rebuilt on the recall hot path every 5 idle minutes,
    causing a recurring multi-second latency spike. Weight drift is a minor
    ranking refinement, so a longer TTL trades negligible staleness for a big
    latency win. Set SLM_ENTITY_ADJ_TTL_S to tune (0 disables time-based reload;
    the count-based correctness reload always remains).
    """
    try:
        return max(0.0, float(os.environ.get("SLM_ENTITY_ADJ_TTL_S", "3600")))
    except (TypeError, ValueError):
        return 3600.0


def _adj_cache_profiles() -> int:
    """How many profile scopes the adjacency LRU keeps warm (>= 1).

    The cache used to hold a single slot, so interleaved per-request profiles
    rebuilt the whole graph (edges + entity maps + metrics + snapshot) on every
    switch. The LRU keeps the last N (profile, include_global, include_shared)
    scopes instead. Set SLM_ADJ_CACHE_PROFILES to tune; each slot costs roughly
    the old single cache (~18 MB on a 232K-edge store), so the default of 3
    bounds the worst case at ~3x that. Read at call time, same convention as
    SLM_ENTITY_ADJ_TTL_S above, so a test or operator can retune without a
    re-import.
    """
    try:
        return max(1, int(os.environ.get("SLM_ADJ_CACHE_PROFILES", "3")))
    except (TypeError, ValueError):
        return 3


@dataclass(frozen=True)
class _AdjSlot:
    """One cached adjacency load for one (profile, include_global, include_shared).

    Groups the instance attributes that used to form the single-slot cache, so
    the LRU can hand them back to readers exactly as a fresh load left them.
    """

    adj: dict[str, list[tuple[str, float]]]
    entity_to_facts: dict[str, list[str]]
    fact_to_entities: dict[str, list[str]]
    visible_fact_ids: set[str]
    edge_count: int
    fact_count: int
    loaded_at: float
    graph_metrics: dict[str, dict]
    graph_metrics_profile: str
    adjacency_source_name: str
    snapshot: Any  # graph_adjacency view; typed loosely to avoid an import cycle


_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{1,}\b")

_ENTITY_STOP: frozenset[str] = frozenset(
    {
        # Expanded stop list for query entity extraction
        "what",
        "when",
        "where",
        "who",
        "which",
        "how",
        "does",
        "did",
        "the",
        "that",
        "this",
        "there",
        "then",
        "than",
        "they",
        "them",
        "have",
        "has",
        "had",
        "been",
        "being",
        "about",
        "after",
        "before",
        "from",
        "into",
        "with",
        "some",
        "other",
        "would",
        "could",
        "should",
        "will",
        "because",
        "also",
        "just",
        "like",
        "know",
        "think",
        "feel",
        "want",
        "need",
        "make",
        "take",
        "give",
        "tell",
        "said",
        "wow",
        "gonna",
        "got",
        "by",
        "thanks",
        "thank",
        "hey",
        "hi",
        "hello",
        "bye",
        "good",
        "great",
        "nice",
        "cool",
        "right",
        "let",
        "can",
        "might",
        "much",
        "many",
        "more",
        "most",
        "something",
        "anything",
        "everything",
        "nothing",
        "someone",
        "it",
        "my",
        "your",
        "our",
        "their",
        "me",
        "you",
        "we",
        "us",
        "do",
        "if",
        "or",
        "no",
        "to",
        "at",
        "on",
        "in",
        "so",
        "go",
        "come",
        "see",
        "look",
        "say",
        "ask",
        "try",
        "keep",
        "yes",
        "yeah",
        "sure",
        "okay",
        "ok",
        "really",
        "actually",
        "maybe",
        "well",
        "still",
        "even",
        "very",
    }
)


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
    # Also extract multi-word capitalized sequences (e.g. "New York", "San Francisco")
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", query):
        _add(m.group(1))
    # Extract all-caps abbreviations (e.g. NYU, MIT, UCLA) — min 2 chars
    for m in re.finditer(r"\b([A-Z]{2,})\b", query):
        _add(m.group(1))

    return candidates


class EntityGraphChannel:
    """Entity-based retrieval with spreading activation (SA-RAG).

    V3.3.9: In-memory adjacency cache for O(1) edge lookup.
    Replaces per-node SQLite queries (23ms each) with dict lookup (<0.001ms).
    The cache is loaded once per profile and invalidated on store/edge changes.
    Memory cost: ~18 MB for 232K edges. Zero quality change — same algorithm.

    Per-request profiles: the cache holds the last SLM_ADJ_CACHE_PROFILES
    (default 3) profile scopes in an LRU, so alternating requests do not
    thrash a single slot into a full graph rebuild each time.
    """

    def __init__(
        self,
        db: DatabaseManager,
        entity_resolver: EntityResolver | None = None,
        decay: float = 0.7,
        activation_threshold: float = 0.05,
        max_hops: int = 4,
        graph_metrics: dict[str, dict] | None = None,
        cozo_backend: Any = None,  # v3.4.5: optional CozoDB backend
    ) -> None:
        self._db = db
        self._resolver = entity_resolver
        self._decay = decay
        self._threshold = activation_threshold
        self._max_hops = max_hops
        # v3.4.5: Optional CozoDB graph backend (Sprint 2)
        self._cozo = cozo_backend
        self._cache_lock = threading.RLock()
        # Profile-keyed LRU of adjacency loads, oldest first. Keyed by the full
        # scope (profile, include_global, include_shared) because each scope is
        # a different graph. The legacy single-slot attributes below always
        # mirror the scope served by the most recent _ensure_adjacency call.
        self._adj_slots: OrderedDict[tuple[str, bool, bool], _AdjSlot] = OrderedDict()
        # In-memory adjacency: {node_id -> [(neighbor_id, weight), ...]}
        self._adj: dict[str, list[tuple[str, float]]] = {}
        self._adj_profile: str = ""  # Track which profile is loaded
        self._adj_scope_key: tuple[str, bool, bool] | None = None
        self._adj_edge_count: int = 0  # Track edge count for staleness detection
        self._adj_fact_count: int = 0
        self._adj_loaded_at: float = 0.0  # TTL reference for the current slot
        self._entity_to_facts: dict[str, list[str]] = defaultdict(list)
        self._fact_to_entities: dict[str, list[str]] = defaultdict(list)
        self._visible_fact_ids: set[str] = set()
        # v3.4.1: Graph intelligence metrics (loaded from fact_importance)
        self._graph_metrics: dict[str, dict] = graph_metrics or {}
        self._graph_metrics_profile: str = ""

    def _ensure_adjacency(
        self,
        profile_id: str,
        *,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> None:
        """Load graph adjacency into memory for fast spreading activation.

        Loads ALL edges for a profile into a bidirectional dict. Cost: ~1s for
        232K edges, ~18 MB RAM. The last ``SLM_ADJ_CACHE_PROFILES`` scopes
        (default 3) stay warm in an LRU, so interleaved per-request profiles do
        not rebuild the graph on every switch.
        """
        # Check staleness: profile changed or new edges added since last load
        scope_key = (profile_id, bool(include_global), bool(include_shared))
        current_count = self._get_edge_count(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        try:
            current_fact_count = self._db.get_fact_count(
                profile_id,
                include_global=include_global,
                include_shared=include_shared,
            )
        except Exception:
            current_fact_count = -1
        import time as _t_ec

        _now_ec = _t_ec.monotonic()
        # memory-bounding-01: also reload if the cache is older than the TTL,
        # even when the edge COUNT is unchanged. Edge weights/pruning can mutate
        # the graph without changing the count (e.g. store_edge MAX-merge), and a
        # count-stable window would otherwise serve a stale adjacency map.
        # TTL=0 disables the time-based reload entirely (count-based correctness
        # reload still applies); otherwise the slot is fresh within the TTL.
        _ttl = _adj_ttl_seconds()
        slot = self._adj_slots.get(scope_key)
        if slot is not None:
            _fresh = _ttl <= 0.0 or ((_now_ec - slot.loaded_at) < _ttl)
            if (
                (slot.adj or slot.visible_fact_ids)
                and slot.edge_count == current_count
                and slot.fact_count == current_fact_count
                and _fresh
            ):
                self._adj_slots.move_to_end(scope_key)
                self._restore_slot(scope_key, slot)
                return
        slot = self._load_adjacency_from_db(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
            current_count=current_count,
            current_fact_count=current_fact_count,
            now=_now_ec,
        )
        # Replacing an existing key keeps its old position, so the reload also
        # counts as a use for LRU ordering.
        self._adj_slots[scope_key] = slot
        self._adj_slots.move_to_end(scope_key)
        while len(self._adj_slots) > _adj_cache_profiles():
            self._adj_slots.popitem(last=False)

    def _restore_slot(
        self,
        scope_key: tuple[str, bool, bool],
        slot: _AdjSlot,
    ) -> None:
        """Point the legacy single-slot attributes at a cached LRU slot.

        Everything downstream of _ensure_adjacency (search, score_candidates,
        _resolve_entities) reads these attributes; refreshing them per call is
        what keeps those readers untouched by the multi-slot cache.
        """
        self._adj = slot.adj
        self._adj_profile = scope_key[0]
        self._adj_scope_key = scope_key
        self._adj_edge_count = slot.edge_count
        self._adj_fact_count = slot.fact_count
        self._adj_loaded_at = slot.loaded_at
        self._entity_to_facts = slot.entity_to_facts
        self._fact_to_entities = slot.fact_to_entities
        self._visible_fact_ids = slot.visible_fact_ids
        self._graph_metrics = slot.graph_metrics
        self._graph_metrics_profile = slot.graph_metrics_profile
        self._adjacency_source_name = slot.adjacency_source_name
        self._snapshot = slot.snapshot

    def _load_adjacency_from_db(
        self,
        profile_id: str,
        *,
        include_global: bool = False,
        include_shared: bool = False,
        current_count: int = 0,
        current_fact_count: int = -1,
        now: float = 0.0,
    ) -> _AdjSlot:
        """Fetch one scope's graph from the stores and build its cache slot.

        The DB-loading half of the former _ensure_adjacency; the LRU and
        staleness bookkeeping live in the caller. Leaves the legacy instance
        attributes describing this scope (exactly as the single-slot cache did)
        and returns them grouped as a slot for the LRU to hold.
        """
        scope_key = (profile_id, bool(include_global), bool(include_shared))
        adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
        # The graph projection, when there is one, answers this in 395 ms where
        # SQLite takes 2,477 ms on the same 208k-edge store (hand-measured, see
        # graph/cozo_adjacency for the run; no test reproduces it) -- and this rebuild
        # sits on the recall path, triggered by any edge-count change or a TTL
        # expiry. It declines global and shared scope, because it stores one
        # profile per edge and a short answer here would silently shrink the
        # graph around a candidate. That decline is correct, not a degradation.
        #
        # This is the projection's only reader. Before it, the projection was
        # maintained by the outbox, held at parity, purged on erasure, and
        # queried by nothing.
        triples: list[tuple[str, str, float]] | None = None
        source_name = "sqlite"
        try:
            from superlocalmemory.graph.cozo_adjacency import adjacency_source

            projection = (
                adjacency_source() if self._projection_is_caught_up(profile_id) else None
            )
            if projection is not None:
                triples = projection.edges(
                    profile_id,
                    include_global=include_global,
                    include_shared=include_shared,
                )
                if triples is not None:
                    source_name = projection.name
        except Exception:  # noqa: BLE001 -- SQLite answers this unconditionally
            triples = None
        if triples is None:
            try:
                where, params = _scope_where(
                    profile_id,
                    include_global=include_global,
                    include_shared=include_shared,
                )
                rows = self._db.execute(
                    f"SELECT source_id, target_id, weight FROM graph_edges WHERE {where}",
                    (*params,),
                )
            except Exception:
                rows = []
            triples = []
            for r in rows:
                d = dict(r)
                triples.append(
                    (d["source_id"], d["target_id"], float(d["weight"])),
                )
        self._adjacency_source_name = source_name
        for edge_source, edge_target, edge_weight in triples:
            adj[edge_source].append((edge_target, edge_weight))
            adj[edge_target].append((edge_source, edge_weight))
        # Also load entity maps (same staleness lifecycle)
        self._load_entity_maps(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        # Edge scope alone cannot authorize an endpoint.  Prune both endpoints
        # against the visible fact corpus so denied facts cannot influence an
        # allowed candidate indirectly through propagation.
        self._adj = {
            node_id: [
                (neighbor_id, weight)
                for neighbor_id, weight in neighbors
                if neighbor_id in self._visible_fact_ids
            ]
            for node_id, neighbors in adj.items()
            if node_id in self._visible_fact_ids
        }
        self._adj_profile = profile_id
        self._adj_scope_key = scope_key
        self._adj_edge_count = current_count
        self._adj_fact_count = current_fact_count
        self._adj_loaded_at = now  # memory-bounding-01: TTL reference
        # v3.4.1: Load graph intelligence metrics (P0)
        self._load_graph_metrics(profile_id)

        # One array-shaped view of the same graph, for the walk to run over.
        # Built here rather than per query because it is derived entirely from
        # the maps above and shares their staleness lifecycle exactly.
        from superlocalmemory.retrieval.graph_adjacency import snapshot_from_maps

        self._snapshot = snapshot_from_maps(
            self._adj,
            self._entity_to_facts,
            self._fact_to_entities,
            self._graph_metrics,
            source=getattr(self, "_adjacency_source_name", "sqlite"),
            profile_id=profile_id,
            # Every visible fact is a node, including the ones with no edges
            # yet. Ingestion is queryable-first, so a memory stored a moment ago
            # has entities and no edges, and it must still be reachable.
            nodes=self._visible_fact_ids,
            # Only a real count. The staleness check above compares this value
            # with ``==``, which a MagicMock tolerates; an ordering comparison
            # does not, and the mock DBs in the test suite reach here.
            fact_count=(
                current_fact_count
                if isinstance(current_fact_count, int) and current_fact_count >= 0
                else 0
            ),
        )

        logger.info(
            "Loaded adjacency cache: %d nodes, %d edges, %d entity mappings for profile %s",
            len(self._adj),
            sum(len(v) for v in self._adj.values()) // 2,
            len(self._entity_to_facts),
            profile_id,
        )
        return _AdjSlot(
            adj=self._adj,
            entity_to_facts=self._entity_to_facts,
            fact_to_entities=self._fact_to_entities,
            visible_fact_ids=self._visible_fact_ids,
            edge_count=current_count,
            fact_count=current_fact_count,
            loaded_at=now,
            graph_metrics=self._graph_metrics,
            graph_metrics_profile=self._graph_metrics_profile,
            adjacency_source_name=self._adjacency_source_name,
            snapshot=self._snapshot,
        )

    def _projection_is_caught_up(self, profile_id: str) -> bool:
        """Whether the second graph store has seen every change SQLite has.

        The graph lives in two stores and no transaction spans them, so the
        durable record of "this fact still needs projecting" is a queue row
        written in the same transaction as the change. A row outstanding for
        this profile is that store telling us, in its own words, that it is
        behind -- and reading a graph that is behind means walking a link the
        store has already removed.

        This is a primary-key count on a table whose steady state is empty and
        whose size is bounded by the fact count, so it is microseconds. The
        alternative -- comparing the two edge sets -- costs 1.9 s on the
        author's store and 7.7 s on the larger one, which is the whole recall
        budget spent proving a cache is warm.
        """
        try:
            from superlocalmemory.storage import projection_outbox

            if not projection_outbox.is_available(self._db):
                return True
            rows = self._db.execute(
                "SELECT COUNT(*) AS cnt FROM projection_outbox WHERE profile_id = ?",
                (profile_id,),
            )
            pending = int(dict(rows[0]).get("cnt", 0)) if rows else 0
        except Exception:  # noqa: BLE001 -- an unreadable queue means read SQLite
            return False
        if pending:
            logger.debug(
                "adjacency: %d change(s) not yet in the graph projection for "
                "profile %s; reading the store directly", pending, profile_id,
            )
            return False
        return True

    def _get_edge_count(
        self,
        profile_id: str,
        *,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> int:
        """Fast edge count for staleness check (~1ms)."""
        try:
            where, params = _scope_where(
                profile_id,
                include_global=include_global,
                include_shared=include_shared,
            )
            rows = self._db.execute(
                f"SELECT COUNT(*) as cnt FROM graph_edges WHERE {where}",
                (*params,),
            )
            if rows:
                return int(dict(rows[0]).get("cnt", 0))
        except Exception:
            pass
        return 0

    def _load_entity_maps(
        self,
        profile_id: str,
        *,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> None:
        """Pre-load entity→fact and fact→entity maps into memory.

        Eliminates per-entity and per-fact SQL in the spreading activation loop.
        Fetch only the two columns this index consumes. Loading full AtomicFact
        objects also deserializes every 768-d embedding and Fisher vector; on a
        mature database that turned one new fact into a 5-second recall stall.
        The scope predicate and configurable 50k safety ceiling are identical
        to ``get_all_facts``; only heavyweight unused columns are omitted.
        """
        # entity_id -> [fact_id, ...]
        self._entity_to_facts: dict[str, list[str]] = defaultdict(list)
        # fact_id -> [entity_id, ...]
        self._fact_to_entities: dict[str, list[str]] = defaultdict(list)
        self._visible_fact_ids = set()

        try:
            where, params = _scope_where(
                profile_id,
                include_global=include_global,
                include_shared=include_shared,
            )
            # Withheld rows must not enter the entity map at all. They carry
            # their whole cluster's pooled entity list, which is exactly why
            # they out-ranked real memories here in the first place — leaving
            # them in the map would keep spending this channel's budget on
            # candidates that hydration then discards.
            rows = self._db.execute(
                "SELECT fact_id, canonical_entities_json "
                f"FROM atomic_facts WHERE {where}"
                f"{self._db.visible_fact_clause()} "
                "ORDER BY created_at DESC LIMIT ?",
                (*params, _unbounded_facts_ceiling()),
            )
        except Exception:
            rows = []
        for row in rows:
            data = dict(row)
            fact_id = str(data.get("fact_id") or "")
            if not fact_id:
                continue
            self._visible_fact_ids.add(fact_id)
            try:
                entity_ids = json.loads(
                    data.get("canonical_entities_json") or "[]",
                )
            except (TypeError, ValueError):
                entity_ids = []
            for entity_id in entity_ids:
                if not isinstance(entity_id, str) or not entity_id:
                    continue
                self._entity_to_facts[entity_id].append(fact_id)
                self._fact_to_entities[fact_id].append(entity_id)

        logger.info(
            "Loaded entity maps: %d entities, %d facts with entities",
            len(self._entity_to_facts),
            len(self._fact_to_entities),
        )

    def _load_graph_metrics(self, profile_id: str) -> None:
        """Load PageRank, community_id, degree_centrality from fact_importance.

        v3.4.1: Enables graph-enhanced retrieval (P0).
        Called alongside adjacency loading. Same staleness lifecycle.

        4.1.14 audit: always reloads — the old early-return on
        profile-match served stale PageRank/community after edge changes
        whenever a slot rebuild re-hit the previously live profile.
        """
        self._graph_metrics = {}
        self._graph_metrics_profile = profile_id
        try:
            rows = self._db.execute(
                "SELECT fact_id, pagerank_score, community_id, degree_centrality "
                "FROM fact_importance WHERE profile_id = ?",
                (profile_id,),
            )
            for r in rows:
                d = dict(r)
                self._graph_metrics[d["fact_id"]] = {
                    "pagerank_score": float(d.get("pagerank_score", 0) or 0),
                    "community_id": d.get("community_id"),
                    "degree_centrality": float(d.get("degree_centrality", 0) or 0),
                }
            logger.info(
                "Loaded graph metrics: %d facts for profile %s",
                len(self._graph_metrics),
                profile_id,
            )
        except Exception as exc:
            logger.debug("Graph metrics load failed (graceful degradation): %s", exc)
            self._graph_metrics = {}

    def invalidate_cache(self) -> None:
        """Clear all caches. Call after adding/removing edges or facts.

        4.1.14 audit: takes the cache lock like every other cache
        accessor — clearing aliased slot dicts in place while a routed
        recall walks them is a teardown race (RuntimeError or an empty
        graph served for the wrong reason).
        """
        with self._cache_lock:
            self._adj_slots.clear()
            # The in-place clears below now operate on the evicted slots'
            # dicts; dropping the slots above is what actually invalidates
            # them.
            self._adj.clear()
            self._adj_profile = ""
            self._adj_scope_key = None
            self._adj_edge_count = 0
            self._adj_fact_count = 0
            self._entity_to_facts = defaultdict(list)
            self._fact_to_entities = defaultdict(list)
            self._visible_fact_ids.clear()
            self._graph_metrics.clear()
            self._graph_metrics_profile = ""

    def search(self, query: str, profile_id: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Serialize access to the scope-keyed graph/entity cache."""
        with self._cache_lock:
            return self._search_locked(query, profile_id, top_k)

    def _search_locked(
        self,
        query: str,
        profile_id: str,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        """Search via entity graph with spreading activation.

        V3.3.9: Uses in-memory adjacency for O(1) edge lookups.
        V3.4.5: Routes to CozoDB if backend is active (Sprint 2).
        """
        include_global = bool(getattr(self, "include_global", False))
        include_shared = bool(getattr(self, "include_shared", False))
        raw_entities = extract_query_entities(query)

        if not raw_entities:
            return []

        # Load the visible fact/entity map before resolution. Canonical entity
        # IDs are profile-local UUIDs, so the same name may have a different ID
        # in an opted-in global/shared owner's partition.
        self._ensure_adjacency(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )

        canonical_ids = self._resolve_entities(raw_entities, profile_id)
        if not canonical_ids:
            return []

        # One walk, over the array-shaped snapshot when there is one. The
        # dict form below is kept for the mock/lightweight DBs that never build
        # an adjacency cache, and it is the only path that still pays a Python
        # loop per edge.
        snapshot = getattr(self, "_snapshot", None)
        if snapshot is not None and snapshot.node_count:
            activation_result = spreading.activate(
                snapshot,
                canonical_ids,
                decay=self._decay,
                threshold=self._threshold,
                max_hops=self._max_hops,
            )
            spreading.apply_community_bias(
                activation_result.scores, snapshot, canonical_ids,
            )
            activation = activation_result.as_mapping(snapshot, threshold=-1.0)
            if activation:
                self._suppress_contradictions(activation, profile_id)
            results = [
                (fid, sc) for fid, sc in activation.items() if sc >= self._threshold
            ]
            if not results:
                return []
            max_score = max(sc for _, sc in results)
            if max_score > 0:
                results = [(fid, sc / max_score) for fid, sc in results]
            results.sort(key=lambda x: (-x[1], x[0]))
            return filter_authorized_results(
                self._db,
                results,
                profile_id,
                include_global=include_global,
                include_shared=include_shared,
            )[:top_k]

        # Seed activation from direct entity-linked facts (no adjacency cache:
        # mock and lightweight DBs only). Graph intelligence is unavailable on
        # this path by design -- see Phase 7 LLD H-01.
        activation: dict[str, float] = defaultdict(float)
        visited_entities: set[str] = set(canonical_ids)
        use_cache = False
        for eid in canonical_ids:
            for fact in self._db.get_facts_by_entity(
                eid,
                profile_id,
                include_global=include_global,
                include_shared=include_shared,
            ):
                activation[fact.fact_id] = max(activation[fact.fact_id], 1.0)

        frontier = set(activation.keys())
        for hop in range(1, self._max_hops):
            hop_decay = self._decay**hop
            if hop_decay < self._threshold:
                break
            next_frontier: set[str] = set()
            for fid in frontier:
                for edge in self._db.get_edges_for_node(
                    fid,
                    profile_id,
                    include_global=include_global,
                    include_shared=include_shared,
                ):
                    neighbor = edge.target_id if edge.source_id == fid else edge.source_id
                    propagated = activation[fid] * self._decay
                    if propagated >= self._threshold and propagated > activation.get(
                        neighbor, 0.0
                    ):
                        activation[neighbor] = propagated
                        next_frontier.add(neighbor)
            new_eids_sql = self._discover_entities(frontier, profile_id, visited_entities)
            for eid in new_eids_sql:
                visited_entities.add(eid)
                for fact in self._db.get_facts_by_entity(
                    eid,
                    profile_id,
                    include_global=include_global,
                    include_shared=include_shared,
                ):
                    if hop_decay > activation.get(fact.fact_id, 0.0):
                        activation[fact.fact_id] = hop_decay
                        next_frontier.add(fact.fact_id)
            frontier = next_frontier
            if not frontier:
                break

        # v3.4.1 P2: Community-aware boosting
        if self._graph_metrics and use_cache:
            from collections import Counter as _Counter

            seed_communities: _Counter = _Counter()
            for eid in canonical_ids:
                for fid in self._entity_to_facts.get(eid, ()):
                    m = self._graph_metrics.get(fid, {})
                    comm = m.get("community_id")
                    if comm is not None:
                        seed_communities[comm] += 1
            if seed_communities:
                total_seeds = sum(seed_communities.values())
                for fid in list(activation.keys()):
                    m = self._graph_metrics.get(fid, {})
                    fact_comm = m.get("community_id")
                    if fact_comm is not None and fact_comm in seed_communities:
                        boost = min(1.0 + 0.15 * (seed_communities[fact_comm] / total_seeds), 1.3)
                        activation[fid] *= boost
                    elif fact_comm is not None and fact_comm not in seed_communities:
                        activation[fid] *= 0.9  # Mild penalty for unrelated communities

        # v3.4.1 P3: Contradiction suppression via graph_edges
        if use_cache and activation:
            self._suppress_contradictions(activation, profile_id)

        # v3.4.1: Score normalization to [0, 1]
        results = [(fid, sc) for fid, sc in activation.items() if sc >= self._threshold]
        if not results:
            return []
        max_score = max(sc for _, sc in results)
        if max_score > 0:
            results = [(fid, sc / max_score) for fid, sc in results]
        results.sort(key=lambda x: (-x[1], x[0]))
        return filter_authorized_results(
            self._db,
            results,
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )[:top_k]

    def score_candidates(
        self,
        query: str,
        candidate_fact_ids: list[str],
        profile_id: str,
        *,
        include_global: bool | None = None,
        include_shared: bool | None = None,
    ) -> dict[str, float]:
        """Serialize access to the scope-keyed graph/entity cache."""
        with self._cache_lock:
            return self._score_candidates_locked(
                query,
                candidate_fact_ids,
                profile_id,
                include_global=include_global,
                include_shared=include_shared,
            )

    def _score_candidates_locked(
        self,
        query: str,
        candidate_fact_ids: list[str],
        profile_id: str,
        *,
        include_global: bool | None = None,
        include_shared: bool | None = None,
    ) -> dict[str, float]:
        """Score candidate facts by their entity-graph proximity to query entities.

        V3.4.11 "Signal Enhancer" architecture: instead of returning its own
        independent set of fact_ids (which get outranked by multi-channel facts
        in RRF), this method scores EXISTING candidates from semantic/BM25
        by their graph connectivity to query entities.

        Research basis: Microsoft GraphRAG DRIFT Search, HippoRAG, Pistis-RAG
        cascaded architecture. Graph signals act as post-retrieval boosters,
        not independent retrievers. Avoids the "weakest link" phenomenon where
        non-overlapping result sets cause rank collapse in RRF fusion.

        Args:
            query: The user's query string.
            candidate_fact_ids: Fact IDs from semantic/BM25/other channels.
            profile_id: User profile.

        Returns:
            Dict mapping fact_id → entity_graph score [0, 1].
            Facts with no entity connection return 0.
            Facts directly linked to query entities score ~1.0.
            Facts 1-hop away score ~0.7 (decay factor).
        """
        if not candidate_fact_ids:
            return {}

        if include_global is None:
            include_global = bool(getattr(self, "include_global", False))
        if include_shared is None:
            include_shared = bool(getattr(self, "include_shared", False))
        allowed_candidates = authorized_fact_ids(
            self._db,
            candidate_fact_ids,
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        if not allowed_candidates:
            return {}

        raw_entities = extract_query_entities(query)
        if not raw_entities:
            return {}

        self._ensure_adjacency(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        canonical_ids = self._resolve_entities(raw_entities, profile_id)
        if not canonical_ids:
            return {}

        # The same walk as search(), over the same snapshot. This method used
        # to carry its own copy of the loop, which is how the two drifted: the
        # community bias here has never applied search()'s outsider penalty, and
        # the only record of that was the absence of six lines.
        snapshot = getattr(self, "_snapshot", None)
        if snapshot is None or not snapshot.node_count:
            return {}
        activation_result = spreading.activate(
            snapshot,
            canonical_ids,
            decay=self._decay,
            threshold=self._threshold,
            max_hops=self._max_hops,
        )
        spreading.apply_community_bias(
            activation_result.scores,
            snapshot,
            canonical_ids,
            # Re-scoring another channel's candidates, so a fact outside every
            # seed community is not damped -- see apply_community_bias.
            penalise_outsiders=False,
        )
        activation = {
            fid: float(activation_result.scores[idx])
            for fid, idx in snapshot.node_index.items()
        }

        # Extract scores ONLY for the candidate set, normalize to [0, 1]
        candidate_set = allowed_candidates
        scored = {fid: activation.get(fid, 0.0) for fid in candidate_set}

        max_score = max(scored.values()) if scored else 0
        if max_score > 0:
            scored = {fid: sc / max_score for fid, sc in scored.items()}

        return scored

    def _suppress_contradictions(
        self,
        activation: dict[str, float],
        profile_id: str,
    ) -> None:
        """P3: Penalize older fact in contradiction pairs, heavy-penalize superseded.

        Uses graph_edges (edge_type CHECK includes 'contradiction', 'supersedes').
        """
        candidate_ids = list(activation.keys())
        if not candidate_ids:
            return
        try:
            placeholders = ",".join("?" * len(candidate_ids))
            sql = (
                "SELECT source_id, target_id, edge_type FROM graph_edges "
                "WHERE profile_id = ? AND edge_type IN ('contradiction', 'supersedes') "
                "AND (source_id IN (" + placeholders + ") "
                "OR target_id IN (" + placeholders + "))"
            )
            rows = self._db.execute(sql, (profile_id, *candidate_ids, *candidate_ids))
            edges = [dict(r) for r in rows]
            if not edges:
                return

            # Batch load created_at for involved facts
            involved = set()
            for e in edges:
                involved.add(e["source_id"])
                involved.add(e["target_id"])
            involved = involved & set(candidate_ids)
            if not involved:
                return
            ph2 = ",".join("?" * len(involved))
            ts_rows = self._db.execute(
                "SELECT fact_id, created_at FROM atomic_facts "
                "WHERE fact_id IN (" + ph2 + ") AND profile_id = ?",
                (*involved, profile_id),
            )
            ts_map = {dict(r)["fact_id"]: dict(r).get("created_at", "") for r in ts_rows}

            for e in edges:
                src, tgt, etype = e["source_id"], e["target_id"], e["edge_type"]
                if etype == "supersedes" and src in activation:
                    activation[src] *= 0.3  # Heavy penalty: this fact was replaced
                elif etype == "contradiction":
                    src_ts = ts_map.get(src, "")
                    tgt_ts = ts_map.get(tgt, "")
                    if src_ts and tgt_ts:
                        older = src if src_ts < tgt_ts else tgt
                        if older in activation:
                            activation[older] *= 0.5
        except Exception as exc:
            logger.debug("Contradiction suppression failed: %s", exc)

    def _resolve_entities(self, raw: list[str], profile_id: str) -> list[str]:
        """Resolve local and visible cross-profile canonical entity IDs."""
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

        # Entity UUIDs are profile-local. Supplement local resolution with
        # same-name/alias IDs that are actually referenced by visible facts.
        names = [name.strip().lower() for name in raw if name.strip()]
        if names and self._visible_fact_ids and self._entity_to_facts:
            placeholders = ",".join("?" for _ in names)
            try:
                rows = self._db.execute(
                    "SELECT entity_id FROM canonical_entities "
                    f"WHERE LOWER(canonical_name) IN ({placeholders}) "
                    "UNION SELECT entity_id FROM entity_aliases "
                    f"WHERE LOWER(alias) IN ({placeholders})",
                    (*names, *names),
                )
            except Exception:
                rows = []
            visible_entity_ids = set(self._entity_to_facts)
            for row in rows:
                entity_id = str(dict(row)["entity_id"])
                if entity_id in visible_entity_ids and entity_id not in seen:
                    seen.add(entity_id)
                    ids.append(entity_id)
        return ids

    def _discover_entities(
        self,
        fact_ids: set[str],
        profile_id: str,
        visited: set[str],
    ) -> list[str]:
        """Find new canonical entity IDs referenced by a set of facts."""
        new: list[str] = []
        seen = set(visited)
        allowed_fact_ids = authorized_fact_ids(
            self._db,
            fact_ids,
            profile_id,
            include_global=bool(getattr(self, "include_global", False)),
            include_shared=bool(getattr(self, "include_shared", False)),
        )
        for fid in allowed_fact_ids:
            rows = self._db.execute(
                "SELECT canonical_entities_json FROM atomic_facts WHERE fact_id = ?",
                (fid,),
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

    # v3.4.5: CozoDB-backed search (Sprint 2)
