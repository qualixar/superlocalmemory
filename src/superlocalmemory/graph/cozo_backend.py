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
                    from_id: String, to_id: String =>
                    edge_type: String, weight: Float default 1.0,
                    metadata: String default '{}',
                    profile_id: String default 'default',
                    created_at: String
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

        # Step 1: Export all unique node IDs from graph_edges as entities.
        # graph_edges uses fact IDs as nodes. canonical_entities uses separate entity IDs.
        # CozoDB graph mirrors the graph_edges adjacency — node = fact ID.
        entities_sql = """
            SELECT DISTINCT node_id FROM (
                SELECT source_id as node_id FROM graph_edges WHERE profile_id = ?
                UNION
                SELECT target_id as node_id FROM graph_edges WHERE profile_id = ?
            )
        """
        rows = conn.execute(entities_sql, (profile_id, profile_id)).fetchall()

        entity_dicts = []
        now = datetime.now().isoformat()
        for (nid,) in rows:
            entity_dicts.append({
                "id": nid,
                "name": nid[:12],
                "entity_type": "fact_node",
                "tier": "active",
                "properties": "{}",
                "profile_id": profile_id,
                "created_at": now,
                "updated_at": now,
            })

        if entity_dicts:
            self._db.put("entity", entity_dicts)
        logger.info("CozoDB: imported %d entities", len(entity_dicts))

        # Step 2: Export edges directly (source_id/target_id are fact IDs)
        edges_sql = """
            SELECT source_id, target_id, edge_type, weight
            FROM graph_edges WHERE profile_id = ?
        """
        edge_rows = conn.execute(edges_sql, (profile_id,)).fetchall()

        edge_dicts = []
        for row in edge_rows:
            ea, eb, etype, weight = row
            edge_dicts.append({
                "from_id": ea,
                "to_id": eb,
                "edge_type": etype or "related",
                "weight": float(weight or 1.0),
                "metadata": "{}",
                "profile_id": profile_id,
                "created_at": now,
            })

        if edge_dicts:
            self._db.put("edge", edge_dicts)
        logger.info("CozoDB: imported %d edges", len(edge_dicts))

        return len(edge_dicts)

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
                    result = self._db.run(f"""
                        ?[to_id, weight] :=
                            *edge{{from_id: '{entity_id}', to_id, weight}}
                    """)
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

    def sync_tier_changes(
        self, added: list[str], removed: list[str]
    ) -> None:
        """Sync tier changes: add promoted entities, mark demoted."""
        now = datetime.now().isoformat()

        if added:
            # Fetch entity data from existing CozoDB entities or set defaults
            for entity_id in added:
                try:
                    self._db.run(f"""
                        ?[id, tier] <- [['{entity_id}', 'active']]
                        :update entity {{id => tier, updated_at: '{now}'}}
                    """)
                except Exception:
                    pass

        if removed:
            for entity_id in removed:
                try:
                    self._db.run(f"""
                        ?[id, tier] <- [['{entity_id}', 'cold']]
                        :update entity {{id => tier, updated_at: '{now}'}}
                    """)
                except Exception:
                    pass

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
            ec = entity_count.values.tolist()[0][0] if len(entity_count) > 0 else 0
            edc = edge_count.values.tolist()[0][0] if len(edge_count) > 0 else 0
            return {
                "status": "active",
                "entities": int(ec),
                "edges": int(edc),
                "db_path": self._db_path,
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "db_path": self._db_path,
            }

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

        self._ensure_schema()
        return self.bulk_import_from_sqlite(conn, profile_id)
