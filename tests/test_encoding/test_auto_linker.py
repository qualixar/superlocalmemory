# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for AutoLinker -- A-MEM-inspired auto-linking + Hebbian strengthening."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from superlocalmemory.encoding.auto_linker import AutoLinker
from superlocalmemory.storage.models import AtomicFact


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
    for fid in ("fact_A", "fact_B", "fact_C", "new_fact", "existing_1",
                "existing_2", "low_sim", "only_one"):
        conn.execute(
            "INSERT OR IGNORE INTO atomic_facts "
            "(fact_id, memory_id, profile_id, content, fact_type, "
            "confidence, importance, evidence_count, access_count, "
            "lifecycle, signal_type, created_at) "
            "VALUES (?, 'mem_1', 'default', ?, 'episodic', "
            "0.8, 0.5, 1, 0, 'active', 'factual', datetime('now'))",
            (fid, f"Content of {fid}"),
        )
    # Extra facts for max_links test
    for i in range(15):
        conn.execute(
            "INSERT OR IGNORE INTO atomic_facts "
            "(fact_id, memory_id, profile_id, content, fact_type, "
            "confidence, importance, evidence_count, access_count, "
            "lifecycle, signal_type, created_at) "
            "VALUES (?, 'mem_1', 'default', ?, 'episodic', "
            "0.8, 0.5, 1, 0, 'active', 'factual', datetime('now'))",
            (f"fact_{i}", f"Content of fact_{i}"),
        )
    conn.commit()


@pytest.fixture
def linker_db(in_memory_db):
    """In-memory DB with Phase 3 schema + seed data."""
    _seed_phase3_db(in_memory_db)
    return in_memory_db


@pytest.fixture
def mock_db(linker_db):
    """Wrap in-memory db to match DatabaseManager interface."""
    db = MagicMock()
    db.execute.side_effect = lambda sql, params=(): (
        linker_db.execute(sql, params).fetchall()
    )
    db.get_fact.return_value = None
    return db


@pytest.fixture
def mock_vs():
    """Mock VectorStore."""
    vs = MagicMock()
    vs.available = True
    return vs


@pytest.fixture
def mock_config():
    """Mock config with mode attribute."""
    cfg = MagicMock()
    cfg.mode.value = "a"
    return cfg


def _make_fact(fact_id: str, embedding: list[float] | None = None) -> AtomicFact:
    """Helper to create a test fact."""
    return AtomicFact(
        fact_id=fact_id,
        content=f"Content of {fact_id}",
        embedding=embedding or [0.1] * 768,
        canonical_entities=["entity_1"],
    )


def test_link_new_fact_creates_edges(linker_db, mock_vs, mock_config):
    """Store fact, auto-link finds similar facts, creates edges."""
    mock_vs.search.return_value = [
        ("existing_1", 0.85),
        ("existing_2", 0.75),
    ]

    db = MagicMock()
    db.execute.side_effect = lambda sql, params=(): (
        linker_db.execute(sql, params).fetchall()
    )
    db.get_fact.return_value = None

    linker = AutoLinker(db, mock_vs, config=mock_config)
    new_fact = _make_fact("new_fact")
    linked = linker.link_new_fact(new_fact, "default")

    assert len(linked) == 2
    assert "existing_1" in linked
    assert "existing_2" in linked

    # Verify edges exist in DB
    edges = linker_db.execute(
        "SELECT * FROM association_edges WHERE profile_id = 'default'",
    ).fetchall()
    assert len(edges) >= 2


def test_link_threshold_enforced(linker_db, mock_vs, mock_config):
    """Similarity < 0.7 does not create edge."""
    mock_vs.search.return_value = [
        ("low_sim", 0.5),
    ]

    db = MagicMock()
    db.execute.side_effect = lambda sql, params=(): (
        linker_db.execute(sql, params).fetchall()
    )
    db.get_fact.return_value = None

    linker = AutoLinker(db, mock_vs, config=mock_config)
    new_fact = _make_fact("new_fact")
    linked = linker.link_new_fact(new_fact, "default")

    assert len(linked) == 0


def test_max_links_cap(linker_db, mock_vs, mock_config):
    """At most MAX_LINKS_PER_FACT edges created."""
    # Return 15 candidates all above threshold
    mock_vs.search.return_value = [
        (f"fact_{i}", 0.95 - i * 0.01) for i in range(15)
    ]

    db = MagicMock()
    db.execute.side_effect = lambda sql, params=(): (
        linker_db.execute(sql, params).fetchall()
    )
    db.get_fact.return_value = None

    linker = AutoLinker(db, mock_vs, config=mock_config)
    new_fact = _make_fact("new_fact")
    linked = linker.link_new_fact(new_fact, "default")

    assert len(linked) <= AutoLinker.MAX_LINKS_PER_FACT


def test_disabled_without_embeddings(mock_db, mock_vs, mock_config):
    """Returns [] when embedding is None."""
    linker = AutoLinker(mock_db, mock_vs, config=mock_config)
    fact = AtomicFact(fact_id="f1", content="test", embedding=None)
    assert linker.link_new_fact(fact, "default") == []


def test_link_writes_association_edges_only(linker_db, mock_vs, mock_config):
    """Auto-linking writes to association_edges, never graph_edges (Rule 13)."""
    mock_vs.search.return_value = [("existing_1", 0.85)]

    db = MagicMock()
    db.execute.side_effect = lambda sql, params=(): (
        linker_db.execute(sql, params).fetchall()
    )
    db.get_fact.return_value = None

    graph_edges_before = linker_db.execute(
        "SELECT COUNT(*) AS c FROM graph_edges",
    ).fetchone()["c"]

    linker = AutoLinker(db, mock_vs, config=mock_config)
    new_fact = _make_fact("new_fact")
    linker.link_new_fact(new_fact, "default")

    graph_edges_after = linker_db.execute(
        "SELECT COUNT(*) AS c FROM graph_edges",
    ).fetchone()["c"]
    assert graph_edges_after == graph_edges_before


def test_strengthen_increments_weight(linker_db, mock_config):
    """Hebbian +0.05 per co-access."""
    # Insert an association_edge
    linker_db.execute(
        "INSERT INTO association_edges (edge_id, profile_id, source_fact_id, "
        "target_fact_id, association_type, weight, co_access_count, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 0, datetime('now'))",
        ("e1", "default", "fact_A", "fact_B", "auto_link", 0.5),
    )
    linker_db.commit()

    db = MagicMock()
    db.execute.side_effect = lambda sql, params=(): (
        linker_db.execute(sql, params).fetchall()
    )

    linker = AutoLinker(db, MagicMock(), config=mock_config)
    count = linker.strengthen_co_access(["fact_A", "fact_B"], "default")

    assert count >= 1
    row = linker_db.execute(
        "SELECT weight FROM association_edges WHERE edge_id = 'e1'",
    ).fetchone()
    assert row["weight"] == pytest.approx(0.55)


def test_strengthen_caps_at_1(linker_db, mock_config):
    """Weight never exceeds 1.0."""
    linker_db.execute(
        "INSERT INTO association_edges (edge_id, profile_id, source_fact_id, "
        "target_fact_id, association_type, weight, co_access_count, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 0, datetime('now'))",
        ("e1", "default", "fact_A", "fact_B", "auto_link", 0.98),
    )
    linker_db.commit()

    db = MagicMock()
    db.execute.side_effect = lambda sql, params=(): (
        linker_db.execute(sql, params).fetchall()
    )

    linker = AutoLinker(db, MagicMock(), config=mock_config)
    linker.strengthen_co_access(["fact_A", "fact_B"], "default")

    row = linker_db.execute(
        "SELECT weight FROM association_edges WHERE edge_id = 'e1'",
    ).fetchone()
    assert row["weight"] <= 1.0


def test_strengthen_skips_single_fact(mock_db, mock_config):
    """len(fact_ids) < 2 -> 0 strengthened."""
    linker = AutoLinker(mock_db, MagicMock(), config=mock_config)
    assert linker.strengthen_co_access(["only_one"], "default") == 0


def test_strengthen_increments_co_access_count(linker_db, mock_config):
    """Co-access counter incremented."""
    linker_db.execute(
        "INSERT INTO association_edges (edge_id, profile_id, source_fact_id, "
        "target_fact_id, association_type, weight, co_access_count, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 5, datetime('now'))",
        ("e1", "default", "fact_A", "fact_B", "auto_link", 0.5),
    )
    linker_db.commit()

    db = MagicMock()
    db.execute.side_effect = lambda sql, params=(): (
        linker_db.execute(sql, params).fetchall()
    )

    linker = AutoLinker(db, MagicMock(), config=mock_config)
    linker.strengthen_co_access(["fact_A", "fact_B"], "default")

    row = linker_db.execute(
        "SELECT co_access_count FROM association_edges WHERE edge_id = 'e1'",
    ).fetchone()
    assert row["co_access_count"] == 6


def test_mode_b_evolution_makes_no_llm_calls_and_appends_keywords(linker_db):
    """4.2.9: Mode B regenerated every linked old fact through the LLM (up to
    10 calls per stored memory). The prompt held only the old fact, so the
    answer could not reflect the new link, and the ContextResult it returned
    was bound straight into SQL, which SQLite rejects - nothing was ever saved
    (since v3.2.2). Evolution now uses the keyword path in every mode."""
    for fid in ("existing_1", "existing_2"):
        linker_db.execute(
            "INSERT INTO fact_context (fact_id, profile_id, "
            "contextual_description, keywords) VALUES (?, 'default', 'old', 'kw')",
            (fid,),
        )
    linker_db.commit()
    db = MagicMock()
    db.execute.side_effect = lambda sql, params=(): (
        linker_db.execute(sql, params).fetchall()
    )
    new_fact = AtomicFact(fact_id="new_fact", content="Palette rule for videos",
                          canonical_entities=["Palette"])
    db.get_fact.side_effect = lambda fid: (
        new_fact if fid == "new_fact" else AtomicFact(fact_id=fid, content="old")
    )
    context_gen = MagicMock()
    cfg = MagicMock()
    cfg.mode.value = "b"

    linker = AutoLinker(db=db, vector_store=MagicMock(), context_generator=context_gen,
                        config=cfg)
    linker.evolve_linked_facts("new_fact", ["existing_1", "existing_2"], "default")

    context_gen.generate.assert_not_called()
    rows = linker_db.execute(
        "SELECT fact_id, contextual_description, keywords FROM fact_context "
        "ORDER BY fact_id").fetchall()
    assert [tuple(r) for r in rows] == [
        ("existing_1", "old", "kw, Palette"),
        ("existing_2", "old", "kw, Palette"),
    ]
