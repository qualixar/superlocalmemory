# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory v3.4.5 — CozoDB Graph Backend.

Embedded graph database backend powered by CozoDB (MPL-2.0).
Replaces NetworkX for entity graph storage and traversal.

All Datalog queries are private to this module.
External code calls Python methods only — never raw Datalog strings.

Verified API: pycozo v0.7.6, Client('rocksdb', path), db.run(datalog), db.put(name, dicts)

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from superlocalmemory.storage.logical_edges import iter_logical_edges

logger = logging.getLogger(__name__)

# Optional import — CozoDB is an optional dependency
try:
    from pycozo.client import Client as _CozoClient
    _COZO_AVAILABLE = True
except ImportError:
    _CozoClient = None  # type: ignore[assignment]
    _COZO_AVAILABLE = False


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
    """Bridge PyCozo 0.3 embedded bindings and later client conveniences.

    PyCozo 0.3 is the last client compatible with the published macOS native
    binding. It returns dictionaries and exposes ``import_relations`` rather
    than ``put``; later clients return dataframe-like values and add ``put``.
    SLM only needs relation upserts and row results, so normalize those here.
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
        if not _COZO_AVAILABLE:
            raise CozoDBNotAvailable(
                "CozoDB not installed. Run: pip install superlocalmemory[cozo]"
            )
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(path)
        client = _CozoClient("rocksdb", self._db_path, dataframe=False)  # type: ignore[misc]
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
        self._db.put("edge", [{
            "from_id": from_id,
            "to_id": to_id,
            "edge_type": edge_type,
            "weight": weight,
            "metadata": meta,
            "profile_id": profile_id,
            "created_at": now,
        }])

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
        # becoming executable query text.
        self._db.run("""
            ?[fact_id, entity_id, profile_id] :=
                *fact_entity{fact_id, entity_id, profile_id}, fact_id = $fact_id
            :rm fact_entity {fact_id, entity_id => profile_id}
        """, {"fact_id": fact_id})
        self._db.run("""
            ?[from_id, to_id, edge_type, weight, metadata, profile_id, created_at] :=
                *edge{from_id, to_id, edge_type, weight, metadata, profile_id, created_at},
                (from_id = $fact_id or to_id = $fact_id)
            :rm edge {from_id, to_id, edge_type => weight, metadata, profile_id, created_at}
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
        facts_sql = """
            SELECT fact_id, canonical_entities_json
            FROM atomic_facts WHERE profile_id = ?
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

        if edge_dicts:
            self._db.put("edge", edge_dicts)
        logger.info("CozoDB: imported %d edges", len(edge_dicts))

        return len(edge_dicts)

    def recall_facts(
        self,
        seed_entity_ids: list[str],
        *,
        profile_id: str = "default",
        depth: int = 4,
        decay: float = 0.7,
        threshold: float = 0.05,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        """Mirror SLM's entity-to-fact/fact-graph activation in Cozo storage.

        Query values never enter Datalog source.  Cozo is used as the durable
        projection; activation runs in Python so the algorithm stays aligned
        with the SQLite in-memory channel and can be shadow-compared exactly.
        """
        if not seed_entity_ids:
            return []
        entity_rows = self._db.run(
            "?[fact_id, entity_id] := *fact_entity{fact_id, entity_id, profile_id}, profile_id = $profile_id",
            {"profile_id": profile_id},
        )
        edge_rows = self._db.run(
            "?[from_id, to_id, weight] := *edge{from_id, to_id, weight, profile_id}, profile_id = $profile_id",
            {"profile_id": profile_id},
        )
        entity_to_facts: dict[str, list[str]] = {}
        fact_to_entities: dict[str, list[str]] = {}
        for fact_id, entity_id in entity_rows.values.tolist() if len(entity_rows) else []:
            entity_to_facts.setdefault(str(entity_id), []).append(str(fact_id))
            fact_to_entities.setdefault(str(fact_id), []).append(str(entity_id))
        adjacency: dict[str, list[tuple[str, float]]] = {}
        for source_id, target_id, weight in edge_rows.values.tolist() if len(edge_rows) else []:
            source, target = str(source_id), str(target_id)
            # Match EntityGraphChannel: graph edges are bidirectional during
            # activation even when stored as directed rows.
            adjacency.setdefault(source, []).append((target, float(weight)))
            adjacency.setdefault(target, []).append((source, float(weight)))

        activation: dict[str, float] = {}
        visited_entities = set(seed_entity_ids)
        for entity_id in seed_entity_ids:
            for fact_id in entity_to_facts.get(entity_id, ()):
                activation[fact_id] = max(activation.get(fact_id, 0.0), 1.0)
        frontier = set(activation)
        for hop in range(1, depth):
            hop_decay = decay ** hop
            if hop_decay < threshold:
                break
            next_frontier: set[str] = set()
            for fact_id in frontier:
                for neighbor_id, _weight in adjacency.get(fact_id, ()):
                    # SQLite intentionally ignores edge weights when graph
                    # metrics are unavailable; use that same baseline here.
                    score = activation[fact_id] * decay
                    if score >= threshold and score > activation.get(neighbor_id, 0.0):
                        activation[neighbor_id] = score
                        next_frontier.add(neighbor_id)
            for fact_id in frontier:
                for entity_id in fact_to_entities.get(fact_id, ()):
                    if entity_id in visited_entities:
                        continue
                    visited_entities.add(entity_id)
                    for related_fact_id in entity_to_facts.get(entity_id, ()):
                        if hop_decay > activation.get(related_fact_id, 0.0):
                            activation[related_fact_id] = hop_decay
                            next_frontier.add(related_fact_id)
            frontier = next_frontier
            if not frontier:
                break
        results = [(fact_id, score) for fact_id, score in activation.items() if score >= threshold]
        if not results:
            return []
        maximum = max(score for _, score in results)
        results = [(fact_id, score / maximum) for fact_id, score in results]
        return sorted(results, key=lambda item: item[1], reverse=True)[:top_k]

    # ------------------------------------------------------------------
    # Spreading Activation (Python BFS over CozoDB edges)
    # ------------------------------------------------------------------

    def spreading_activation(
        self,
        seed_entities: list[str],
        depth: int = 3,
        decay: float = 0.5,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        """BFS from seed nodes with weight decay per hop.

        Uses CozoDB as fast edge store, Python for BFS logic.
        Returns [(entity_id, activation_score), ...] sorted by score desc.
        """
        if not seed_entities:
            return []

        scores: dict[str, float] = {}
        current_frontier: set[str] = set(seed_entities)
        for s in seed_entities:
            scores[s] = 1.0

        for d in range(depth):
            if not current_frontier:
                break
            next_frontier: set[str] = set()
            hop_multiplier = decay ** (d + 1)

            for entity_id in current_frontier:
                # Query all outgoing edges from this entity
                try:
                    result = self._db.run("""
                        ?[to_id, weight] :=
                            *edge{from_id, to_id, weight}, from_id = $entity_id
                    """, {"entity_id": entity_id})
                    df = result if hasattr(result, "values") else result
                    if df is None or len(df) == 0:
                        continue
                    rows = df.values.tolist() if hasattr(df, "values") else []
                    for to_id, weight in rows:
                        to_id_str = str(to_id)
                        score = hop_multiplier * float(weight)
                        if to_id_str not in scores or score > scores[to_id_str]:
                            scores[to_id_str] = score
                        next_frontier.add(to_id_str)
                except Exception:
                    continue

            current_frontier = next_frontier

        # Sort by score desc, return top_k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # PageRank (Python iterative over CozoDB edges)
    # ------------------------------------------------------------------

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
        try:
            self._db.run("::remove fact_entity")
        except Exception:
            pass

        self._ensure_schema()
        return self.bulk_import_from_sqlite(conn, profile_id)
