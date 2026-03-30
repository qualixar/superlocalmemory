# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for SpreadingActivation -- SYNAPSE 5-step algorithm.

Tests each step of the algorithm independently (Implementation Rule 11).
"""

from __future__ import annotations

import math
import sqlite3
from unittest.mock import MagicMock

import numpy as np
import pytest

from superlocalmemory.retrieval.spreading_activation import (
    SpreadingActivation,
    SpreadingActivationConfig,
)


def _seed_phase3_db(conn):
    """Create V32 tables + seed data (profile + facts) for Phase 3 tests."""
    from superlocalmemory.storage.schema_v32 import V32_DDL
    for ddl_block in V32_DDL:
        for stmt in ddl_block.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
    # Seed a profile
    conn.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) "
        "VALUES ('default', 'Test')",
    )
    # Seed memory record (FK parent for facts)
    conn.execute(
        "INSERT OR IGNORE INTO memories "
        "(memory_id, profile_id, content, session_id, speaker, role, created_at) "
        "VALUES ('mem_1', 'default', 'test', 's1', '', 'user', datetime('now'))",
    )
    # Seed atomic facts
    for fid in ("fact_A", "fact_B", "fact_C", "fact_D", "fact_E",
                "existing_1", "existing_2", "new_fact", "low_sim"):
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
def sa_db(in_memory_db):
    """In-memory DB with Phase 3 schema tables + seed data."""
    _seed_phase3_db(in_memory_db)
    return in_memory_db


@pytest.fixture
def mock_db(sa_db):
    """Wrap in-memory db to match DatabaseManager interface."""
    db = MagicMock()
    db.execute.side_effect = lambda sql, params=(): (
        sa_db.execute(sql, params).fetchall()
    )
    return db


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore returning deterministic results."""
    vs = MagicMock()
    vs.available = True
    vs.search.return_value = [
        ("fact_A", 0.9),
        ("fact_B", 0.85),
        ("fact_C", 0.8),
    ]
    return vs


def test_search_returns_empty_when_disabled(mock_db, mock_vector_store):
    """Config.enabled=False -> empty results."""
    config = SpreadingActivationConfig(enabled=False)
    sa = SpreadingActivation(mock_db, mock_vector_store, config)
    result = sa.search(np.zeros(768), profile_id="default", top_k=7)
    assert result == []


def test_search_returns_empty_when_no_seeds(mock_db):
    """No VectorStore results -> empty."""
    vs = MagicMock()
    vs.available = True
    vs.search.return_value = []
    config = SpreadingActivationConfig(enabled=True)
    sa = SpreadingActivation(mock_db, vs, config)
    result = sa.search(np.zeros(768), profile_id="default", top_k=7)
    assert result == []


def test_initialization_scales_by_alpha(mock_db, mock_vector_store):
    """Seeds get alpha * similarity as initial activation."""
    config = SpreadingActivationConfig(
        enabled=True, alpha=2.0, max_iterations=0,
        tau_gate=0.0,
    )
    sa = SpreadingActivation(mock_db, mock_vector_store, config)
    # With max_iterations=0, _propagate returns raw initial activations
    seeds = [("fact_A", 0.9), ("fact_B", 0.5)]
    activations = sa._propagate(seeds, "default")
    assert activations["fact_A"] == pytest.approx(2.0 * 0.9)
    assert activations["fact_B"] == pytest.approx(2.0 * 0.5)


def test_propagation_spreads_to_neighbors(sa_db, mock_vector_store):
    """1-hop neighbor gets activated via spreading."""
    # Insert a graph edge: fact_A -> fact_D
    sa_db.execute(
        "INSERT INTO graph_edges (edge_id, profile_id, source_id, target_id, "
        "edge_type, weight, created_at) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
        ("e1", "default", "fact_A", "fact_D", "semantic", 0.8),
    )
    sa_db.commit()

    db = MagicMock()
    db.execute.side_effect = lambda sql, params=(): (
        sa_db.execute(sql, params).fetchall()
    )

    vs = MagicMock()
    vs.available = True
    vs.search.return_value = [("fact_A", 0.9)]

    config = SpreadingActivationConfig(
        enabled=True, max_iterations=1, tau_gate=0.0, top_m=10,
    )
    sa = SpreadingActivation(db, vs, config)
    result = sa.search(np.zeros(768), profile_id="default", top_k=10)

    result_ids = [r[0] for r in result]
    assert "fact_D" in result_ids, "1-hop neighbor should be activated"


def test_lateral_inhibition_prunes_to_top_m(mock_db, mock_vector_store):
    """More than top_m nodes get pruned to top_m."""
    config = SpreadingActivationConfig(
        enabled=True, top_m=3, max_iterations=1, tau_gate=0.0,
    )
    # Seeds have 3 items; with self-retention all 3 survive
    sa = SpreadingActivation(mock_db, mock_vector_store, config)
    seeds = [("f1", 0.9), ("f2", 0.8), ("f3", 0.7), ("f4", 0.6), ("f5", 0.5)]
    activations = sa._propagate(seeds, "default")
    assert len(activations) <= 3


def test_sigmoid_gating_applies_threshold():
    """Sigmoid(u - theta) transforms raw activation."""
    theta = 0.5
    raw = 1.0
    expected = 1.0 / (1.0 + math.exp(-(raw - theta)))
    assert expected == pytest.approx(0.6224593, rel=1e-4)

    raw_low = 0.0
    expected_low = 1.0 / (1.0 + math.exp(-(raw_low - theta)))
    assert expected_low < 0.5  # Below threshold


def test_fok_check_rejects_low_activation(mock_db, mock_vector_store):
    """All activations < tau_gate -> FOK rejects."""
    config = SpreadingActivationConfig(enabled=True, tau_gate=0.99)
    sa = SpreadingActivation(mock_db, mock_vector_store, config)
    assert sa._fok_check({"a": 0.1, "b": 0.2}) is False


def test_fok_check_accepts_high_activation(mock_db, mock_vector_store):
    """Max activation >= tau_gate -> FOK accepts."""
    config = SpreadingActivationConfig(enabled=True, tau_gate=0.12)
    sa = SpreadingActivation(mock_db, mock_vector_store, config)
    assert sa._fok_check({"a": 0.5, "b": 0.2}) is True


def test_fok_check_empty_activations(mock_db, mock_vector_store):
    """Empty activations -> FOK rejects."""
    config = SpreadingActivationConfig(enabled=True)
    sa = SpreadingActivation(mock_db, mock_vector_store, config)
    assert sa._fok_check({}) is False


def test_compute_query_hash_deterministic(mock_db, mock_vector_store):
    """Same input produces same hash."""
    sa = SpreadingActivation(mock_db, mock_vector_store)
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    h1 = sa._compute_query_hash(vec, "default")
    h2 = sa._compute_query_hash(vec, "default")
    assert h1 == h2
    assert len(h1) == 16


def test_compute_query_hash_differs_by_profile(mock_db, mock_vector_store):
    """Different profile produces different hash."""
    sa = SpreadingActivation(mock_db, mock_vector_store)
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    h1 = sa._compute_query_hash(vec, "profile_a")
    h2 = sa._compute_query_hash(vec, "profile_b")
    assert h1 != h2


def test_unified_neighbors_reads_both_tables(sa_db):
    """UNION query returns neighbors from both graph_edges and association_edges."""
    # Insert graph_edge: A -> B
    sa_db.execute(
        "INSERT INTO graph_edges (edge_id, profile_id, source_id, target_id, "
        "edge_type, weight, created_at) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
        ("e1", "default", "fact_A", "fact_B", "semantic", 0.8),
    )
    # Insert association_edge: A -> C
    sa_db.execute(
        "INSERT INTO association_edges (edge_id, profile_id, source_fact_id, "
        "target_fact_id, association_type, weight, co_access_count, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 0, datetime('now'))",
        ("ae1", "default", "fact_A", "fact_C", "auto_link", 0.7),
    )
    sa_db.commit()

    db = MagicMock()
    db.execute.side_effect = lambda sql, params=(): (
        sa_db.execute(sql, params).fetchall()
    )

    sa = SpreadingActivation(db, MagicMock())
    neighbors = sa._get_unified_neighbors("fact_A", "default")
    neighbor_ids = [n[0] for n in neighbors]
    assert "fact_B" in neighbor_ids
    assert "fact_C" in neighbor_ids
