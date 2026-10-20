# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory v3.4.5 — CozoDB Graph Backend.

Embedded graph database backend powered by CozoDB (MPL-2.0).

It stores the graph. It does not traverse it, and the line that used to claim it
replaced NetworkX "for storage and traversal" outlived the traversal by a
release: the walk lives once in ``retrieval/spreading``, as a pure function of an
adjacency snapshot, and is numpy rather than NetworkX either way. What this
module supplies is the snapshot's edges, read by
``graph/cozo_adjacency.CozoAdjacencySource`` -- measured at 395 ms against
SQLite's 2,477 ms on a 208,151-edge store. Both figures come from a hand run on a
copy of that store, recorded in ``cozo_adjacency``; nothing in the test suite
reproduces them.

All Datalog queries are private to this module.
External code calls Python methods only — never raw Datalog strings.

Verified API: pycozo v0.7.6, Client('rocksdb', path), db.run(datalog), db.put(name, dicts)

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from superlocalmemory.storage.logical_edges import iter_logical_edges

logger = logging.getLogger(__name__)

# Availability via find_spec only — never import pycozo at module load.
# Native bindings belong to the same risk class as lancedb (background
# runtimes / GC races). Real import is deferred to CozoDBGraphBackend.__init__.
def _pycozo_spec_present() -> bool:
    try:
        return importlib.util.find_spec("pycozo") is not None
    except Exception:
        return False


_COZO_AVAILABLE: bool = _pycozo_spec_present()


class CozoDBError(Exception):
    """Base exception for CozoDB backend failures."""


class CozoDBNotAvailable(CozoDBError):
    """CozoDB not installed. Install with: pip install superlocalmemory[cozo]"""


class CozoDBConnectionError(CozoDBError):
    """CozoDB file not found or corrupted."""


class CozoDBQueryError(CozoDBError):
    """Datalog query execution failed."""


class _CozoRows:
    """Small pandas-values compatible view for PyCozo's dict results."""

    def __init__(self, rows: list[list[Any]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[Any]]:
        return self._rows


class _CozoResult:
    """Normalize old PyCozo dict responses to the dataframe surface we use."""

    def __init__(self, result: Any) -> None:
        self._result = result
        self.values = _CozoRows(list(result.get("rows", []))) if isinstance(result, dict) else result.values

    def __len__(self) -> int:
        return len(self.values.tolist())


class _CozoClientAdapter:
    """Bridge legacy PyCozo responses and the store-compatible client surface.

    SLM graph stores were established with PyCozo 0.3.0. That client returns
    dictionaries and exposes ``import_relations`` rather than ``put``; later
    clients return dataframe-like values and add ``put``. SLM only needs
    relation upserts and row results, so normalize both forms here.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def run(self, script: str, params: dict[str, Any] | None = None) -> Any:
        result = self._client.run(script) if params is None else self._client.run(script, params)
        return _CozoResult(result) if isinstance(result, dict) else result

    def put(self, relation: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        put = getattr(self._client, "put", None)
        if callable(put):
            put(relation, rows)
            return
        headers = list(rows[0])
        self._client.import_relations({
            relation: {
                "headers": headers,
                "rows": [[row.get(header) for header in headers] for row in rows],
            },
        })

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# CozoDBGraphBackend
# ---------------------------------------------------------------------------

class CozoDBGraphBackend:
    """Embedded graph backend powered by CozoDB.

    Wraps pycozo for graph storage, traversal, and algorithms.
    All Datalog queries are private. External code calls Python methods.
    """

    def __init__(self, db_path: str) -> None:
        if not _pycozo_spec_present():
            raise CozoDBNotAvailable(
                "CozoDB not installed. Run: pip install superlocalmemory[cozo]"
            )
        # Lazy import: only construct-time loads the native binding.
        from pycozo.client import Client as _CozoClient  # noqa: PLC0415

        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(path)
        client = _CozoClient("rocksdb", self._db_path, dataframe=False)
        self._db = _CozoClientAdapter(client)
        self._shadow_checks = 0
        self._shadow_mismatches = 0
        self._shadow_errors = 0
        self._ensure_schema()

    def close(self) -> None:
        """Close the CozoDB connection."""
        if hasattr(self, "_db") and self._db is not None:
            self._db.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create relations if they don't exist. Idempotent."""
        try:
            self._db.run("""
                :create entity {
                    id: String => name: String, entity_type: String,
                    tier: String default 'hot',
                    properties: String default '{}',
                    profile_id: String default 'default',
                    created_at: String, updated_at: String
                }
            """)
        except Exception:
            pass  # Already exists

        try:
            self._db.run("""
                :create edge {
                    from_id: String, to_id: String, edge_type: String =>
                    weight: Float default 1.0,
                    metadata: String default '{}',
                    profile_id: String default 'default',
                    created_at: String
                }
            """)
        except Exception:
            pass

        # ``edge`` is keyed by ``from_id`` first, so a node's incoming edges
        # cannot be found without scanning the relation. ``edge_rev`` mirrors
        # every edge keyed by ``to_id`` (#296: removing one fact cost 3.98 s on
        # a 4.3 M-edge store). ``::index create`` fails to parse on
        # cozo_embedded 0.3.0, so the mirror is maintained by hand.
        for script in (
            ":create edge_rev {to_id: String, from_id: String, edge_type: String}",
            ":create graph_meta {key: String => value: String}",
        ):
            try:
                self._db.run(script)
            except Exception:
                pass
        self._backfill_reverse_edges()

        # The entity recall channel resolves a query to canonical entity IDs,
        # while graph_edges links fact IDs.  Keeping those relations separate
        # is essential: treating fact IDs as entities produces a healthy but
        # semantically incompatible graph.  ``fact_entity`` is the bridge
        # that lets Cozo traverse the same two spaces as the SQLite channel.
        try:
            self._db.run("""
                :create fact_entity {
                    fact_id: String, entity_id: String =>
                    profile_id: String default 'default'
                }
            """)
        except Exception:
            pass

    def _backfill_reverse_edges(self) -> None:
        """Fill ``edge_rev`` once for a store created before it existed.

        The marker is written after the copy, so a copy cut short is simply
        redone on the next open (``:put`` is idempotent).
        """
        done = self._db.run(
            "?[value] := *graph_meta{key: 'edge_rev', value}"
        ).values.tolist()
        if done:
            return
        self._db.run("""
            ?[to_id, from_id, edge_type] := *edge{from_id, to_id, edge_type}
            :put edge_rev {to_id, from_id, edge_type}
        """)
        self._db.run("?[key, value] <- [['edge_rev', 'v1']] :put graph_meta {key => value}")

    def _remove_node_edges(self, node_id: str) -> None:
        """Remove every edge from or to ``node_id`` by key seeks, in one transaction."""
        self._db.run("""
            {
                ?[to_id, from_id, edge_type] :=
                    from_id = $id, *edge{from_id, to_id, edge_type}
                :rm edge_rev {to_id, from_id, edge_type}
            }
            {
                ?[from_id, to_id, edge_type, weight, metadata, profile_id, created_at] :=
                    to_id = $id, *edge_rev{to_id, from_id, edge_type},
                    *edge{from_id, to_id, edge_type, weight, metadata, profile_id, created_at}
                :rm edge {from_id, to_id, edge_type => weight, metadata, profile_id, created_at}
            }
            {
                ?[to_id, from_id, edge_type] :=
                    to_id = $id, *edge_rev{to_id, from_id, edge_type}
                :rm edge_rev {to_id, from_id, edge_type}
            }
            {
                ?[from_id, to_id, edge_type, weight, metadata, profile_id, created_at] :=
                    from_id = $id,
                    *edge{from_id, to_id, edge_type, weight, metadata, profile_id, created_at}
                :rm edge {from_id, to_id, edge_type => weight, metadata, profile_id, created_at}
            }
        """, {"id": node_id})

    # ------------------------------------------------------------------
    # Write Path
    # ------------------------------------------------------------------

    def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        properties: dict | None = None,
        profile_id: str = "default",
    ) -> None:
        """Insert or update a canonical entity node."""
        now = datetime.now().isoformat()
        props = json.dumps(properties or {})
        self._db.put("entity", [{
            "id": entity_id,
            "name": name,
            "entity_type": entity_type,
            "properties": props,
            "profile_id": profile_id,
            "tier": "hot",
            "created_at": now,
            "updated_at": now,
        }])

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: str,
        weight: float = 1.0,
        metadata: dict | None = None,
        profile_id: str = "default",
    ) -> None:
        """Insert a relationship edge between two entities."""
        now = datetime.now().isoformat()
        meta = json.dumps(metadata or {})
        self._put_edges([[from_id, to_id, edge_type, float(weight), meta, profile_id, now]])

    def _put_edges(self, rows: list[list[Any]]) -> None:
        """Upsert edges and their ``edge_rev`` mirror in one transaction.

        Each row is ``[from_id, to_id, edge_type, weight, metadata, profile_id, created_at]``.
        """
        if not rows:
            return
        self._db.run("""
            {
                ?[from_id, to_id, edge_type, weight, metadata, profile_id, created_at] <- $rows
                :put edge {from_id, to_id, edge_type => weight, metadata, profile_id, created_at}
            }
            {
                ?[to_id, from_id, edge_type] <- $rev
                :put edge_rev {to_id, from_id, edge_type}
            }
        """, {"rows": rows, "rev": [[r[1], r[0], r[2]] for r in rows]})

    def add_fact_entities(
        self,
        fact_id: str,
        entity_ids: list[str],
        profile_id: str = "default",
    ) -> None:
        """Upsert the canonical entities attached to one fact.

        This is deliberately a separate relation from ``edge``.  Fact edges
        and canonical entity IDs are different namespaces in SLM.
        """
        self._db.put("fact_entity", [
            {"fact_id": fact_id, "entity_id": entity_id, "profile_id": profile_id}
            for entity_id in dict.fromkeys(entity_ids)
            if entity_id
        ])

    def remove_fact(self, fact_id: str) -> None:
        """Remove a fact's derived graph records using bound query values."""
        # Cozo :rm needs every non-key column in the output relation.  Binding
        # ``fact_id`` keeps apostrophes and Datalog syntax in external IDs from
        # becoming executable query text.  Each id is bound *before* its stored
        # atom so Cozo seeks on the key instead of scanning the relation (#296).
        self._db.run("""
            ?[fact_id, entity_id, profile_id] :=
                fact_id = $fact_id, *fact_entity{fact_id, entity_id, profile_id}
            :rm fact_entity {fact_id, entity_id => profile_id}
        """, {"fact_id": fact_id})
        self._remove_node_edges(fact_id)

    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity node and every bridge that reaches it.

        ``remove_fact`` clears a fact's bridges and edges and leaves the entity
        nodes alone, which is right when a fact is deleted -- the entity is
        still real and other facts still reference it. It is wrong when the
        entity itself is erased: the node stayed behind with its name in it,
        so a graph query could still name somebody after their record had been
        removed from the store.

        The id is bound as a parameter, so an entity name containing a quote
        cannot become query text.
        """
        self._db.run("""
            ?[fact_id, entity_id, profile_id] :=
                *fact_entity{fact_id, entity_id, profile_id}, entity_id = $entity_id
            :rm fact_entity {fact_id, entity_id => profile_id}
        """, {"entity_id": entity_id})
        self._remove_node_edges(entity_id)
        self._db.run("""
            ?[id, name, entity_type, tier, properties, profile_id,
              created_at, updated_at] :=
                *entity{id, name, entity_type, tier, properties, profile_id,
                        created_at, updated_at},
                id = $entity_id
            :rm entity {id => name, entity_type, tier, properties, profile_id,
                        created_at, updated_at}
        """, {"entity_id": entity_id})

    def remove_profile(self, profile_id: str) -> None:
        """Remove every node, edge and fact bridge filed under ``profile_id``.

        Erasing a profile's facts one at a time leaves the entity nodes behind,
        and a node carries the name it was created with. This removes the
        profile's whole share of the graph; the profile id is a bound parameter.
        """
        self._db.run("""
            ?[fact_id, entity_id, profile_id] :=
                *fact_entity{fact_id, entity_id, profile_id}, profile_id = $pid
            :rm fact_entity {fact_id, entity_id => profile_id}
        """, {"pid": profile_id})
        self._db.run("""
            {
                ?[to_id, from_id, edge_type] :=
                    *edge{from_id, to_id, edge_type, profile_id}, profile_id = $pid
                :rm edge_rev {to_id, from_id, edge_type}
            }
            {
                ?[from_id, to_id, edge_type, weight, metadata, profile_id, created_at] :=
                    *edge{from_id, to_id, edge_type, weight, metadata, profile_id, created_at},
                    profile_id = $pid
                :rm edge {from_id, to_id, edge_type => weight, metadata, profile_id, created_at}
            }
        """, {"pid": profile_id})
        self._db.run("""
            ?[id, name, entity_type, tier, properties, profile_id, created_at, updated_at] :=
                *entity{id, name, entity_type, tier, properties, profile_id,
                        created_at, updated_at},
                profile_id = $pid
            :rm entity {id => name, entity_type, tier, properties, profile_id,
                        created_at, updated_at}
        """, {"pid": profile_id})

    def remove_fact_candidacy(self, fact_id: str) -> None:
        """Stop a fact being *offered* as a candidate, without touching the graph.

        Removes only the ``fact_entity`` bridge. The fact's edges stay.

        The two are not the same thing and the SQLite channel already treats them
        differently. Its entity map filters on ``visible_fact_clause()`` — a
        withheld row must never enter it, because it carries its whole cluster's
        pooled entity list and out-ranks real memories. Its edge walk filters on
        scope only, with no visibility predicate at all, so it traverses edges to
        withheld and archived facts and lets hydration drop them at the end.

        So a projection that deleted a hidden fact's edges would hold a smaller
        adjacency than the walk it is meant to replace, and give different
        answers for every fact that happened to neighbour a withheld one. Use
        this for a fact that has become invisible; use ``remove_fact`` only for
        one that is genuinely gone.
        """
        self._db.run("""
            ?[fact_id, entity_id, profile_id] :=
                *fact_entity{fact_id, entity_id, profile_id}, fact_id = $fact_id
            :rm fact_entity {fact_id, entity_id => profile_id}
        """, {"fact_id": fact_id})

    def record_shadow_comparison(
        self,
        *,
        matches: bool,
        projected: list[tuple[str, float]],
        canonical: list[tuple[str, float]],
    ) -> None:
        """Retain aggregate parity telemetry without persisting recalled text."""
        self._shadow_checks += 1
        if not matches:
            self._shadow_mismatches += 1
            logger.warning(
                "Cozo entity recall diverged from canonical SQLite; using SQLite "
                "(projected=%d canonical=%d)", len(projected), len(canonical),
            )

    def record_shadow_error(self, error: str) -> None:
        self._shadow_errors += 1
        logger.warning("Cozo entity recall failed closed to SQLite: %s", error)

    # ------------------------------------------------------------------
    # Bulk Import (SQLite → CozoDB)
    # ------------------------------------------------------------------

    def bulk_import_from_sqlite(
        self,
        conn: sqlite3.Connection,
        profile_id: str = "default",
        tier_filter: list[str] | None = None,
    ) -> int:
        """Export entities + edges from SQLite to CozoDB.

        Only imports facts + edges in tier_filter (default: hot+warm).
        Uses parameterized Datalog — no string injection.

        Returns number of edges imported.
        """
        if tier_filter is None:
            tier_filter = ["active", "warm"]

        # Step 1: Export canonical entity records.  Do *not* synthesize
        # entities from graph_edges: those are fact IDs and belong to the
        # separate fact graph relation below.
        entities_sql = """
            SELECT entity_id, canonical_name, entity_type, first_seen, last_seen, fact_count
            FROM canonical_entities WHERE profile_id = ?
        """
        rows = conn.execute(entities_sql, (profile_id,)).fetchall()

        entity_dicts = []
        now = datetime.now().isoformat()
        for entity_id, name, entity_type, first_seen, last_seen, fact_count in rows:
            entity_dicts.append({
                "id": entity_id,
                "name": name,
                "entity_type": entity_type or "concept",
                "tier": "active",
                "properties": json.dumps({"fact_count": int(fact_count or 0)}),
                "profile_id": profile_id,
                "created_at": first_seen or now,
                "updated_at": last_seen or now,
            })

        if entity_dicts:
            self._db.put("entity", entity_dicts)
        logger.info("CozoDB: imported %d entities", len(entity_dicts))

        # Step 2: Export fact-to-canonical-entity mappings.  This relation is
        # what allows a canonical query seed to enter the fact graph.
        #
        # Withheld and archived facts are excluded, with the same predicate the
        # SQLite entity map uses and for the reason it records: such a row
        # carries its whole cluster's pooled entity list, so it out-ranks real
        # memories, takes the top-k budget, and is then discarded at hydration.
        # This export predated that fix. Measured on a copy of the author's
        # store, it had put 1,257 unreturnable facts into the bridge, and the
        # Cozo graph search diverged from SQLite on every query tried — one
        # returned 9 results against SQLite's 20. The projection failed closed
        # each time, so recall stayed correct and the projection was useless.
        from superlocalmemory.storage.database import (
            visible_fact_clause_for_connection,
        )

        facts_sql = f"""
            SELECT fact_id, canonical_entities_json
            FROM atomic_facts WHERE profile_id = ?
            {visible_fact_clause_for_connection(conn)}
        """
        fact_entity_dicts: list[dict[str, str]] = []
        for fact_id, raw_entities in conn.execute(facts_sql, (profile_id,)).fetchall():
            try:
                entity_ids = json.loads(raw_entities or "[]")
            except (TypeError, ValueError, json.JSONDecodeError):
                entity_ids = []
            for entity_id in dict.fromkeys(entity_ids):
                if entity_id:
                    fact_entity_dicts.append({
                        "fact_id": fact_id,
                        "entity_id": str(entity_id),
                        "profile_id": profile_id,
                    })
        if fact_entity_dicts:
            self._db.put("fact_entity", fact_entity_dicts)

        # Step 3: Export fact graph edges directly.  Fact graph traversal is
        # intentionally kept in its native fact-ID namespace.
        edge_dicts = []
        for ea, eb, etype, weight, edge_profile in iter_logical_edges(conn, profile_id):
            edge_dicts.append({
                "from_id": ea,
                "to_id": eb,
                "edge_type": etype,
                "weight": float(weight),
                "metadata": "{}",
                "profile_id": edge_profile,
                "created_at": now,
            })

        columns = ("from_id", "to_id", "edge_type", "weight", "metadata", "profile_id",
                   "created_at")
        for start in range(0, len(edge_dicts), 50_000):
            self._put_edges([[e[c] for c in columns] for e in edge_dicts[start:start + 50_000]])
        logger.info("CozoDB: imported %d edges", len(edge_dicts))

        return len(edge_dicts)

    # recall_facts() and spreading_activation() were here, and they were a
    # second implementation of the walk in retrieval/entity_channel.py. They did
    # not compute the same function: the channel multiplies activation by a
    # PageRank factor every hop and these did not, so on a real store 3,567 of
    # 3,667 shared facts came out with different scores, the result sets
    # differed, and the projected path failed its shadow comparison on every
    # query and fell back to SQLite. The projection was correct data that
    # nothing could use.
    #
    # spreading_activation() also queried Cozo once per frontier node: 200
    # single-node queries measured 18,125 ms against 303 ms for one batched
    # query returning the identical 6,046 rows. With a mean degree of 228 over
    # four hops that is minutes per recall.
    #
    # The walk now lives in retrieval/spreading.py, once, as a pure function of
    # an AdjacencySnapshot. A storage backend supplies adjacency, not answers —
    # which is also why there is no longer anything to shadow-compare. Do not
    # reintroduce a traversal here; add an AdjacencySource adapter instead
    # (retrieval/graph_adjacency.py).

    def pagerank(
        self, damping: float = 0.85, max_iter: int = 100
    ) -> dict[str, float]:
        """Iterative PageRank on the current graph.

        Uses CozoDB for edge queries, Python for iteration.
        """
        try:
            # Get all entities
            entities_df = self._db.run("?[id] := *entity{id}")
            if entities_df is None or len(entities_df) == 0:
                return {}
            entities = [str(r[0]) for r in entities_df.values.tolist()]
            n = len(entities)
            if n == 0:
                return {}

            entity_index = {eid: i for i, eid in enumerate(entities)}
            scores = [1.0 / n] * n

            # Get all edges as adjacency list
            try:
                edges_df = self._db.run("?[from_id, to_id, weight] := *edge{from_id, to_id, weight}")
                if edges_df is not None and len(edges_df) > 0:
                    edges = edges_df.values.tolist()
                else:
                    edges = []
            except Exception:
                edges = []

            # Build outgoing edge map
            outgoing: dict[str, list[tuple[str, float]]] = {e: [] for e in entities}
            for from_id, to_id, weight in edges:
                outgoing[str(from_id)].append((str(to_id), float(weight)))

            # Iterative PageRank
            for _ in range(max_iter):
                new_scores = [(1.0 - damping) / n] * n
                for i, eid in enumerate(entities):
                    neighbors = outgoing.get(eid, [])
                    if neighbors:
                        total_weight = sum(w for _, w in neighbors)
                        if total_weight > 0:
                            for to_id, weight in neighbors:
                                j = entity_index.get(to_id)
                                if j is not None:
                                    new_scores[j] += damping * scores[i] * weight / total_weight
                scores = new_scores

            return {entities[i]: scores[i] for i in range(n)}

        except Exception as exc:
            logger.warning("CozoDB PageRank failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Community Detection (simplified label propagation)
    # ------------------------------------------------------------------

    def community_detect(self, method: str = "louvain") -> dict[str, int]:
        """Simplified community detection via label propagation.

        Uses CozoDB for edge queries, Python for iteration.
        Falls back to connected components if Louvain fails.
        """
        try:
            entities_df = self._db.run("?[id] := *entity{id}")
            if entities_df is None or len(entities_df) == 0:
                return {}
            entities = [str(r[0]) for r in entities_df.values.tolist()]

            # Get edges
            try:
                edges_df = self._db.run("?[from_id, to_id] := *edge{from_id, to_id}")
                if edges_df is not None and len(edges_df) > 0:
                    edges = edges_df.values.tolist()
                else:
                    edges = []
            except Exception:
                edges = []

            # Build adjacency for connected components
            adj: dict[str, set[str]] = {e: set() for e in entities}
            for from_id, to_id in edges:
                f, t = str(from_id), str(to_id)
                adj.setdefault(f, set()).add(t)
                adj.setdefault(t, set()).add(f)

            # Connected components via BFS
            community: dict[str, int] = {}
            visited: set[str] = set()
            comm_id = 0

            for entity in entities:
                if entity in visited:
                    continue
                # BFS from this entity
                queue = [entity]
                visited.add(entity)
                while queue:
                    current = queue.pop(0)
                    community[current] = comm_id
                    for neighbor in adj.get(current, set()):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                comm_id += 1

            return community

        except Exception as exc:
            logger.warning("CozoDB community detection failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Shortest Path (BFS)
    # ------------------------------------------------------------------

    def shortest_path(self, from_id: str, to_id: str) -> list[str]:
        """BFS shortest path between two entities."""
        try:
            if from_id == to_id:
                return [from_id]

            edges_df = self._db.run("?[from_id, to_id] := *edge{from_id, to_id}")
            if edges_df is None or len(edges_df) == 0:
                return []

            # Build adjacency
            adj: dict[str, list[str]] = {}
            for f, t in edges_df.values.tolist():
                adj.setdefault(str(f), []).append(str(t))
                adj.setdefault(str(t), []).append(str(f))

            # BFS
            from collections import deque
            queue = deque([(from_id, [from_id])])
            visited = {from_id}

            while queue:
                current, path = queue.popleft()
                for neighbor in adj.get(current, []):
                    if neighbor == to_id:
                        return path + [neighbor]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))

            return []
        except Exception as exc:
            logger.warning("CozoDB shortest path failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Tier Sync
    # ------------------------------------------------------------------

    # Rebind the whole entity row (Cozo has no partial ``:update`` op — that
    # token does not parse) while binding the id/tier/timestamp as query
    # parameters so an entity id containing a quote can never become Datalog.
    # The ``*entity{...}`` match means non-existent ids are a safe no-op rather
    # than creating a stub row.
    _TIER_SYNC_QUERY = (
        "?[id, name, entity_type, tier, properties, profile_id, created_at, updated_at] := "
        "*entity{id, name, entity_type, properties, profile_id, created_at}, "
        "id = $id, tier = $tier, updated_at = $now "
        ":put entity {id => name, entity_type, tier, properties, profile_id, created_at, updated_at}"
    )

    def sync_tier_changes(
        self, added: list[str], removed: list[str]
    ) -> None:
        """Sync tier changes: promote added entities to active, demote removed to cold."""
        now = datetime.now().isoformat()
        for entity_ids, tier in ((added, "active"), (removed, "cold")):
            for entity_id in entity_ids:
                try:
                    self._db.run(
                        self._TIER_SYNC_QUERY,
                        {"id": entity_id, "tier": tier, "now": now},
                    )
                except Exception:
                    logger.debug(
                        "cozo tier sync skipped for entity %s", entity_id, exc_info=True
                    )

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Return health status of the CozoDB backend."""
        try:
            entity_count = self._db.run(
                "?[count(id)] := *entity{id}"
            )
            edge_count = self._db.run(
                "?[count(from_id)] := *edge{from_id}"
            )
            # fact_entity bridge rows (fact_id, entity_id) — counted so parity
            # verification can catch a silent bridge-import failure that would
            # otherwise leave Cozo entity recall returning empty results.
            fe_count = self._db.run(
                "?[count(fact_id)] := *fact_entity{fact_id, entity_id}"
            )
            ec = entity_count.values.tolist()[0][0] if len(entity_count) > 0 else 0
            edc = edge_count.values.tolist()[0][0] if len(edge_count) > 0 else 0
            fec = fe_count.values.tolist()[0][0] if len(fe_count) > 0 else 0
            return {
                "status": "active",
                "entities": int(ec),
                "edges": int(edc),
                "fact_entity": int(fec),
                "shadow_checks": self._shadow_checks,
                "shadow_mismatches": self._shadow_mismatches,
                "shadow_errors": self._shadow_errors,
                "db_path": self._db_path,
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "db_path": self._db_path,
            }

    def entity_ids(self, limit: int = 1000) -> list[str]:
        """Return up to ``limit`` entity IDs, for scale-engine content parity.

        Count parity alone cannot prove the import reproduced the right rows;
        comparing this ID set against canonical SQLite catches an import that
        landed the correct *number* of entities with wrong identities.
        """
        try:
            res = self._db.run("?[id] := *entity{id} :limit " + str(int(limit)))
            return [row[0] for row in res.values.tolist()]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Rebuild (from SQLite canonical)
    # ------------------------------------------------------------------

    def rebuild_from_sqlite(
        self, conn: sqlite3.Connection, profile_id: str = "default"
    ) -> int:
        """Drop all CozoDB data, re-import from SQLite."""
        try:
            self._db.run("::remove entity")
        except Exception:
            pass
        try:
            self._db.run("::remove edge")
        except Exception:
            pass
        for relation in ("edge_rev", "graph_meta"):
            try:
                self._db.run(f"::remove {relation}")
            except Exception:
                pass
        try:
            self._db.run("::remove fact_entity")
        except Exception:
            pass

        self._ensure_schema()
        return self.bulk_import_from_sqlite(conn, profile_id)
