# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Loading one scope's entity graph from the stores (the full build).

Moved out of retrieval/entity_channel unchanged in behaviour; the channel keeps
thin methods of the same names, which tests and the RCU builder wrap. Since
4.1.25 a full load runs only for a first build, an evicted scope, a slot whose
currency the change log cannot prove, or a background compaction
(retrieval/graph_catchup) -- never because a memory was written.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from superlocalmemory.retrieval.adjacency_slot import AdjSlot as _AdjSlot
from superlocalmemory.storage.database import (
    _scope_where,
    _unbounded_facts_ceiling,
)
from superlocalmemory.storage.json_columns import load_json_column

logger = logging.getLogger("superlocalmemory.retrieval.entity_channel")


def load_adjacency_from_db(
    channel: Any,
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
    # graph/cozo_adjacency for the run; no test reproduces it). Since 4.1.25 this
    # load runs for a first build or a background compaction only
    # (retrieval/graph_catchup), never per write. It declines global and
    # shared scope, because it stores one profile per edge and a short answer
    # here would silently shrink the graph around a candidate. That decline is
    # correct, not a degradation.
    #
    # This is the projection's only reader. Before it, the projection was
    # maintained by the outbox, held at parity, purged on erasure, and
    # queried by nothing.
    triples: list[tuple[str, str, float]] | None = None
    source_name = "sqlite"
    try:
        from superlocalmemory.graph.cozo_adjacency import adjacency_source

        projection = (
            adjacency_source() if channel._projection_is_caught_up(profile_id) else None
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
            rows = channel._db.execute(
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
    channel._adjacency_source_name = source_name
    for edge_source, edge_target, edge_weight in triples:
        adj[edge_source].append((edge_target, edge_weight))
        adj[edge_target].append((edge_source, edge_weight))
    # Also load entity maps (same staleness lifecycle)
    channel._load_entity_maps(
        profile_id,
        include_global=include_global,
        include_shared=include_shared,
    )
    # Edge scope alone cannot authorize an endpoint.  Prune both endpoints
    # against the visible fact corpus so denied facts cannot influence an
    # allowed candidate indirectly through propagation.
    channel._adj = {
        node_id: [
            (neighbor_id, weight)
            for neighbor_id, weight in neighbors
            if neighbor_id in channel._visible_fact_ids
        ]
        for node_id, neighbors in adj.items()
        if node_id in channel._visible_fact_ids
    }
    channel._adj_profile = profile_id
    channel._adj_scope_key = scope_key
    channel._adj_edge_count = current_count
    channel._adj_fact_count = current_fact_count
    channel._adj_loaded_at = now  # memory-bounding-01: TTL reference
    # v3.4.1: Load graph intelligence metrics (P0)
    channel._load_graph_metrics(profile_id)

    # One array-shaped view of the same graph, for the walk to run over.
    # Built here rather than per query because it is derived entirely from
    # the maps above and shares their staleness lifecycle exactly.
    from superlocalmemory.retrieval.graph_adjacency import snapshot_from_maps

    channel._snapshot = snapshot_from_maps(
        channel._adj,
        channel._entity_to_facts,
        channel._fact_to_entities,
        channel._graph_metrics,
        source=getattr(channel, "_adjacency_source_name", "sqlite"),
        profile_id=profile_id,
        # Every visible fact is a node, including the ones with no edges
        # yet. Ingestion is queryable-first, so a memory stored a moment ago
        # has entities and no edges, and it must still be reachable.
        nodes=channel._visible_fact_ids,
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
        len(channel._adj),
        sum(len(v) for v in channel._adj.values()) // 2,
        len(channel._entity_to_facts),
        profile_id,
    )
    return _AdjSlot(
        adj=channel._adj,
        entity_to_facts=channel._entity_to_facts,
        fact_to_entities=channel._fact_to_entities,
        visible_fact_ids=channel._visible_fact_ids,
        edge_count=current_count,
        fact_count=current_fact_count,
        loaded_at=now,
        graph_metrics=channel._graph_metrics,
        graph_metrics_profile=channel._graph_metrics_profile,
        adjacency_source_name=channel._adjacency_source_name,
        snapshot=channel._snapshot,
    )

def projection_is_caught_up(channel: Any, profile_id: str) -> bool:
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

        if not projection_outbox.is_available(channel._db):
            return True
        rows = channel._db.execute(
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

def get_edge_count(
    channel: Any,
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
        rows = channel._db.execute(
            f"SELECT COUNT(*) as cnt FROM graph_edges WHERE {where}",
            (*params,),
        )
        if rows:
            return int(dict(rows[0]).get("cnt", 0))
    except Exception:
        pass
    return 0

def load_entity_maps(
    channel: Any,
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
    channel._entity_to_facts: dict[str, list[str]] = defaultdict(list)
    # fact_id -> [entity_id, ...]
    channel._fact_to_entities: dict[str, list[str]] = defaultdict(list)
    channel._visible_fact_ids = set()

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
        rows = channel._db.execute(
            "SELECT fact_id, canonical_entities_json "
            f"FROM atomic_facts WHERE {where}"
            f"{channel._db.visible_fact_clause()} "
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
        channel._visible_fact_ids.add(fact_id)
        try:
            entity_ids = load_json_column(
                data.get("canonical_entities_json"), [],
                column="atomic_facts.canonical_entities_json", row=fact_id,
            )
        except (TypeError, ValueError):
            entity_ids = []
        for entity_id in entity_ids:
            if not isinstance(entity_id, str) or not entity_id:
                continue
            channel._entity_to_facts[entity_id].append(fact_id)
            channel._fact_to_entities[fact_id].append(entity_id)

    logger.info(
        "Loaded entity maps: %d entities, %d facts with entities",
        len(channel._entity_to_facts),
        len(channel._fact_to_entities),
    )

def load_graph_metrics(channel: Any, profile_id: str) -> None:
    """Load PageRank, community_id, degree_centrality from fact_importance.

    v3.4.1: Enables graph-enhanced retrieval (P0).
    Called alongside adjacency loading. Same staleness lifecycle.

    4.1.14 audit: always reloads — the old early-return on
    profile-match served stale PageRank/community after edge changes
    whenever a slot rebuild re-hit the previously live profile.
    """
    channel._graph_metrics = {}
    channel._graph_metrics_profile = profile_id
    try:
        rows = channel._db.execute(
            "SELECT fact_id, pagerank_score, community_id, degree_centrality "
            "FROM fact_importance WHERE profile_id = ?",
            (profile_id,),
        )
        for r in rows:
            d = dict(r)
            channel._graph_metrics[d["fact_id"]] = {
                "pagerank_score": float(d.get("pagerank_score", 0) or 0),
                "community_id": d.get("community_id"),
                "degree_centrality": float(d.get("degree_centrality", 0) or 0),
            }
        logger.info(
            "Loaded graph metrics: %d facts for profile %s",
            len(channel._graph_metrics),
            profile_id,
        )
    except Exception as exc:
        logger.debug("Graph metrics load failed (graceful degradation): %s", exc)
        channel._graph_metrics = {}


__all__ = ["get_edge_count", "load_adjacency_from_db", "load_entity_maps",
           "load_graph_metrics", "projection_is_caught_up"]
