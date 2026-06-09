# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Graph-integrity regression suite (v3.6.4, graph-integrity-07).

Locks the invariants fixed in this release:
  - store_edge dedups on logical identity, merging MAX weight (gi-02)
  - graph pruning caps IN-degree, not just OUT-degree (gi-03)
  - prune_graph removes orphaned association_edges (gi-04)
"""

from __future__ import annotations

import pytest

import sqlite3

from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    GraphEdge, EdgeType, AtomicFact, FactType, MemoryRecord,
)


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "graph.db")
    mgr.initialize(real_schema)
    return mgr


# --- gi-02: store_edge dedup + MAX-weight merge --------------------------------

class TestStoreEdgeDedup:
    def test_dedups_logical_edge_keeping_max_weight(self, db: DatabaseManager) -> None:
        for eid, w in (("e1", 0.3), ("e2", 0.7), ("e3", 0.5)):
            db.store_edge(GraphEdge(
                edge_id=eid, source_id="a", target_id="b",
                edge_type=EdgeType.SEMANTIC, weight=w,
            ))
        edges = db.get_edges_for_node("a", "default")
        assert len(edges) == 1, "duplicate logical edges were not merged"
        assert edges[0].weight == pytest.approx(0.7), "merge must keep MAX weight"

    def test_distinct_edges_kept_separate(self, db: DatabaseManager) -> None:
        db.store_edge(GraphEdge(edge_id="e1", source_id="a", target_id="b",
                                edge_type=EdgeType.SEMANTIC, weight=0.5))
        db.store_edge(GraphEdge(edge_id="e2", source_id="a", target_id="c",
                                edge_type=EdgeType.SEMANTIC, weight=0.5))
        db.store_edge(GraphEdge(edge_id="e3", source_id="a", target_id="b",
                                edge_type=EdgeType.ENTITY, weight=0.5))
        assert len(db.get_edges_for_node("a", "default")) == 3

    def test_weight_not_downgraded_by_weaker_duplicate(self, db: DatabaseManager) -> None:
        db.store_edge(GraphEdge(edge_id="e1", source_id="a", target_id="b",
                                edge_type=EdgeType.SEMANTIC, weight=0.9))
        db.store_edge(GraphEdge(edge_id="e2", source_id="a", target_id="b",
                                edge_type=EdgeType.SEMANTIC, weight=0.2))
        edges = db.get_edges_for_node("a", "default")
        assert len(edges) == 1
        assert edges[0].weight == pytest.approx(0.9)


# --- gi-03: cap IN-degree, not just OUT-degree ---------------------------------

def _raw_insert_edge(c, eid, src, tgt, w):
    c.execute(
        "INSERT INTO graph_edges (edge_id, profile_id, source_id, target_id, "
        "edge_type, weight, created_at) VALUES (?, 'default', ?, ?, 'semantic', ?, '2026-01-01')",
        (eid, src, tgt, w),
    )


class TestDegreeCap:
    def test_caps_in_degree_to_max(self, db: DatabaseManager) -> None:
        from superlocalmemory.core.graph_pruner import _cap_node_degree
        conn = sqlite3.connect(str(db.db_path)); conn.row_factory = sqlite3.Row
        c = conn.cursor()
        # 150 distinct sources → one hub target (each source out-degree 1).
        for i in range(150):
            _raw_insert_edge(c, f"e{i}", f"s{i}", "hub", i / 150.0)
        conn.commit()
        removed = _cap_node_degree(c, "default", 100, dry_run=False)
        conn.commit()
        in_deg = c.execute(
            "SELECT COUNT(*) AS x FROM graph_edges WHERE target_id='hub'"
        ).fetchone()["x"]
        conn.close()
        assert in_deg == 100, "in-degree was not capped to max"
        assert removed == 50

    def test_still_caps_out_degree(self, db: DatabaseManager) -> None:
        from superlocalmemory.core.graph_pruner import _cap_node_degree
        conn = sqlite3.connect(str(db.db_path)); conn.row_factory = sqlite3.Row
        c = conn.cursor()
        for i in range(150):
            _raw_insert_edge(c, f"e{i}", "hubsrc", f"t{i}", i / 150.0)
        conn.commit()
        _cap_node_degree(c, "default", 100, dry_run=False)
        conn.commit()
        out_deg = c.execute(
            "SELECT COUNT(*) AS x FROM graph_edges WHERE source_id='hubsrc'"
        ).fetchone()["x"]
        conn.close()
        assert out_deg == 100, "out-degree cap regressed"


# --- gi-04: prune_graph removes orphaned association_edges ----------------------

class TestAssociationOrphanPrune:
    def test_prune_removes_orphan_association_edges(self, db: DatabaseManager) -> None:
        from superlocalmemory.core.graph_pruner import prune_graph
        db.store_memory(MemoryRecord(memory_id="m0", profile_id="default", content="p"))
        db.store_fact(AtomicFact(fact_id="real", memory_id="m0", content="r",
                                 fact_type=FactType.SEMANTIC))
        # Valid edge (both endpoints exist).
        db.execute(
            "INSERT INTO association_edges (edge_id, profile_id, source_fact_id, "
            "target_fact_id, association_type, weight) "
            "VALUES ('a1','default','real','real','hebbian',0.5)"
        )
        # Orphan edge (target fact never existed) — inserted via FK-off connection.
        raw = sqlite3.connect(str(db.db_path))
        raw.execute(
            "INSERT INTO association_edges (edge_id, profile_id, source_fact_id, "
            "target_fact_id, association_type, weight) "
            "VALUES ('a2','default','real','ghost','hebbian',0.5)"
        )
        raw.commit(); raw.close()

        stats = prune_graph(db.db_path, profile_id="default")
        assert stats.get("association_orphans_removed") == 1
        rows = db.execute("SELECT edge_id FROM association_edges ORDER BY edge_id")
        assert [dict(r)["edge_id"] for r in rows] == ["a1"]
