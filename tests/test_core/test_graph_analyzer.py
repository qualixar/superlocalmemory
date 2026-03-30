# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for GraphAnalyzer -- PageRank, community detection, centrality."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.graph_analyzer import GraphAnalyzer


def _seed_phase3_db(conn):
    """Create V32 tables + seed data for Phase 3 tests."""
    from superlocalmemory.storage.schema_v32 import V32_DDL
    for ddl_block in V32_DDL:
        for stmt in ddl_block.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
    conn.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) "
        "VALUES ('default', 'Test')",
    )
    conn.execute(
        "INSERT OR IGNORE INTO memories "
        "(memory_id, profile_id, content, session_id, speaker, role, created_at) "
        "VALUES ('mem_1', 'default', 'test', 's1', '', 'user', datetime('now'))",
    )
    for fid in ("A", "B", "C", "D"):
        conn.execute(
            "INSERT OR IGNORE INTO atomic_facts "
            "(fact_id, memory_id, profile_id, content, fact_type, "
            "confidence, importance, evidence_count, access_count, "
            "lifecycle, signal_type, created_at) "
            "VALUES (?, 'mem_1', 'default', ?, 'episodic', "
            "0.8, 0.5, 1, 0, 'active', 'factual', datetime('now'))",
            (fid, f"Content of {fid}"),
        )
    conn.commit()


@pytest.fixture
def ga_db(in_memory_db):
    """In-memory DB with Phase 3 schema + seed data."""
    _seed_phase3_db(in_memory_db)
    return in_memory_db


@pytest.fixture
def mock_db(ga_db):
    """Wrap in-memory db to match DatabaseManager interface."""
    db = MagicMock()
    db.execute.side_effect = lambda sql, params=(): (
        ga_db.execute(sql, params).fetchall()
    )
    return db


def _insert_graph_edge(conn, eid, pid, src, tgt, etype, weight):
    """Insert a graph_edge."""
    conn.execute(
        "INSERT INTO graph_edges (edge_id, profile_id, source_id, target_id, "
        "edge_type, weight, created_at) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
        (eid, pid, src, tgt, etype, weight),
    )


def _insert_assoc_edge(conn, eid, pid, src, tgt, atype, weight):
    """Insert an association_edge."""
    conn.execute(
        "INSERT INTO association_edges (edge_id, profile_id, source_fact_id, "
        "target_fact_id, association_type, weight, co_access_count, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 0, datetime('now'))",
        (eid, pid, src, tgt, atype, weight),
    )


def test_empty_graph_returns_empty(mock_db):
    """No nodes -> empty dict."""
    ga = GraphAnalyzer(mock_db)
    result = ga.compute_and_store("default")
    assert result["node_count"] == 0
    assert result["community_count"] == 0
    assert result["top_5_nodes"] == []


def test_pagerank_computes_scores(ga_db, mock_db):
    """All nodes get a PageRank score."""
    _insert_graph_edge(ga_db, "e1", "default", "A", "B", "semantic", 0.8)
    _insert_graph_edge(ga_db, "e2", "default", "B", "C", "semantic", 0.7)
    ga_db.commit()

    ga = GraphAnalyzer(mock_db)
    pr = ga.compute_pagerank(profile_id="default")
    assert "A" in pr
    assert "B" in pr
    assert "C" in pr
    assert all(v > 0 for v in pr.values())


def test_pagerank_hub_gets_highest(ga_db, mock_db):
    """Most-connected node ranks highest."""
    # Hub node B: A->B, C->B, D->B
    _insert_graph_edge(ga_db, "e1", "default", "A", "B", "semantic", 0.8)
    _insert_graph_edge(ga_db, "e2", "default", "C", "B", "semantic", 0.8)
    _insert_graph_edge(ga_db, "e3", "default", "D", "B", "semantic", 0.8)
    ga_db.commit()

    ga = GraphAnalyzer(mock_db)
    pr = ga.compute_pagerank(profile_id="default")
    assert pr["B"] == max(pr.values())


def test_communities_detected(ga_db, mock_db):
    """Label Propagation finds groups."""
    # Two clusters: {A,B} and {C,D}
    _insert_graph_edge(ga_db, "e1", "default", "A", "B", "semantic", 0.9)
    _insert_graph_edge(ga_db, "e2", "default", "B", "A", "semantic", 0.9)
    _insert_graph_edge(ga_db, "e3", "default", "C", "D", "semantic", 0.9)
    _insert_graph_edge(ga_db, "e4", "default", "D", "C", "semantic", 0.9)
    ga_db.commit()

    ga = GraphAnalyzer(mock_db)
    communities = ga.detect_communities(profile_id="default")
    assert len(communities) == 4
    # A and B should be in the same community
    assert communities["A"] == communities["B"]
    # C and D should be in the same community
    assert communities["C"] == communities["D"]


def test_compute_and_store_persists(ga_db, mock_db):
    """fact_importance table populated after compute_and_store."""
    _insert_graph_edge(ga_db, "e1", "default", "A", "B", "semantic", 0.8)
    ga_db.commit()

    ga = GraphAnalyzer(mock_db)
    result = ga.compute_and_store("default")

    assert result["node_count"] == 2
    assert result["edge_count"] == 1

    # Verify persistence
    rows = ga_db.execute(
        "SELECT * FROM fact_importance WHERE profile_id = 'default'",
    ).fetchall()
    assert len(rows) == 2


def test_reads_both_tables(ga_db, mock_db):
    """Graph includes edges from both graph_edges and association_edges."""
    _insert_graph_edge(ga_db, "e1", "default", "A", "B", "semantic", 0.8)
    _insert_assoc_edge(ga_db, "ae1", "default", "B", "C", "auto_link", 0.7)
    ga_db.commit()

    ga = GraphAnalyzer(mock_db)
    graph = ga._build_networkx_graph("default")
    assert graph.has_node("A")
    assert graph.has_node("B")
    assert graph.has_node("C")
    assert graph.has_edge("A", "B")
    assert graph.has_edge("B", "C")


def test_degree_centrality_computed(ga_db, mock_db):
    """Degree centrality is computed for all nodes."""
    _insert_graph_edge(ga_db, "e1", "default", "A", "B", "semantic", 0.8)
    _insert_graph_edge(ga_db, "e2", "default", "A", "C", "semantic", 0.7)
    ga_db.commit()

    ga = GraphAnalyzer(mock_db)
    result = ga.compute_and_store("default")

    rows = ga_db.execute(
        "SELECT fact_id, degree_centrality FROM fact_importance "
        "WHERE profile_id = 'default' ORDER BY degree_centrality DESC",
    ).fetchall()
    # A connects to both B and C, so has highest centrality
    assert dict(rows[0])["fact_id"] == "A"
