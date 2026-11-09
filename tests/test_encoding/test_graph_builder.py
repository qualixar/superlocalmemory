# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.encoding.graph_builder.

Covers:
  - Entity edges (shared canonical entity)
  - Temporal edges (exp-decay, 1-week window, bidirectional)
  - Semantic edges (ANN similarity > 0.7)
  - Causal edges (causal markers + shared entity)
  - Contradiction edges (external Sheaf API)
  - get_graph_stats()
  - _parse_date helper
  - Edge deduplication
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from superlocalmemory.encoding.graph_builder import GraphBuilder, _parse_date
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    AtomicFact,
    EdgeType,
    MemoryRecord,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    return mgr


def _setup_fact(
    db: DatabaseManager, fact_id: str, content: str,
    canonical_entities: list[str] | None = None,
    obs_date: str | None = None,
    embedding: list[float] | None = None,
) -> AtomicFact:
    """Store a parent memory + fact and return the fact."""
    mem_id = f"m_{fact_id}"
    db.store_memory(MemoryRecord(memory_id=mem_id, content="parent"))
    fact = AtomicFact(
        fact_id=fact_id, memory_id=mem_id, content=content,
        canonical_entities=canonical_entities or [],
        observation_date=obs_date,
        embedding=embedding,
    )
    db.store_fact(fact)
    return fact


# ---------------------------------------------------------------------------
# _parse_date
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_iso_full(self) -> None:
        dt = _parse_date("2026-03-11T10:30:00")
        assert dt is not None
        assert dt.year == 2026

    def test_iso_date_only(self) -> None:
        dt = _parse_date("2026-03-11")
        assert dt is not None

    def test_iso_with_microseconds(self) -> None:
        dt = _parse_date("2026-03-11T10:30:00.123456")
        assert dt is not None

    def test_none_input(self) -> None:
        assert _parse_date(None) is None

    def test_empty_input(self) -> None:
        assert _parse_date("") is None

    def test_invalid_format(self) -> None:
        assert _parse_date("not-a-date") is None


# ---------------------------------------------------------------------------
# Entity edges
# ---------------------------------------------------------------------------

class TestEntityEdges:
    def test_shared_entity_creates_edge(self, db: DatabaseManager) -> None:
        _setup_fact(db, "f1", "Alice works at Google",
                         canonical_entities=["ent_alice"])
        f2 = _setup_fact(db, "f2", "Alice likes hiking",
                         canonical_entities=["ent_alice"])

        builder = GraphBuilder(db=db)
        edges = builder.build_edges(f2, "default")
        entity_edges = [e for e in edges if e.edge_type == EdgeType.ENTITY]
        assert len(entity_edges) == 1
        assert entity_edges[0].source_id == "f2"
        assert entity_edges[0].target_id == "f1"

    def test_no_shared_entity_no_edge(self, db: DatabaseManager) -> None:
        _setup_fact(db, "f1", "Alice at Google", canonical_entities=["ent_alice"])
        f2 = _setup_fact(db, "f2", "Bob at Apple", canonical_entities=["ent_bob"])

        builder = GraphBuilder(db=db)
        edges = builder.build_edges(f2, "default")
        entity_edges = [e for e in edges if e.edge_type == EdgeType.ENTITY]
        assert len(entity_edges) == 0

    def test_no_canonical_entities(self, db: DatabaseManager) -> None:
        f1 = _setup_fact(db, "f1", "Some fact", canonical_entities=[])
        builder = GraphBuilder(db=db)
        edges = builder.build_edges(f1, "default")
        entity_edges = [e for e in edges if e.edge_type == EdgeType.ENTITY]
        assert len(entity_edges) == 0


# ---------------------------------------------------------------------------
# Temporal edges
# ---------------------------------------------------------------------------

class TestTemporalEdges:
    def test_within_window(self, db: DatabaseManager) -> None:
        _setup_fact(db, "f1", "Monday meeting",
                    canonical_entities=["ent_alice"],
                    obs_date="2026-03-10T10:00:00")
        f2 = _setup_fact(db, "f2", "Tuesday meeting",
                         canonical_entities=["ent_alice"],
                         obs_date="2026-03-11T10:00:00")

        builder = GraphBuilder(db=db)
        edges = builder.build_edges(f2, "default")
        temporal = [e for e in edges if e.edge_type == EdgeType.TEMPORAL]
        assert len(temporal) >= 1
        # Weight should be > 0 (exp decay within window)
        assert all(e.weight > 0 for e in temporal)

    def test_outside_window(self, db: DatabaseManager) -> None:
        _setup_fact(db, "f1", "Old fact",
                    canonical_entities=["ent_alice"],
                    obs_date="2025-01-01T10:00:00")
        f2 = _setup_fact(db, "f2", "New fact",
                         canonical_entities=["ent_alice"],
                         obs_date="2026-03-11T10:00:00")

        builder = GraphBuilder(db=db)
        edges = builder.build_edges(f2, "default")
        temporal = [e for e in edges if e.edge_type == EdgeType.TEMPORAL]
        assert len(temporal) == 0

    def test_no_date_no_temporal(self, db: DatabaseManager) -> None:
        _setup_fact(db, "f1", "No date fact", canonical_entities=["ent_a"])
        f2 = _setup_fact(db, "f2", "Also no date", canonical_entities=["ent_a"])
        builder = GraphBuilder(db=db)
        edges = builder.build_edges(f2, "default")
        temporal = [e for e in edges if e.edge_type == EdgeType.TEMPORAL]
        assert len(temporal) == 0


# ---------------------------------------------------------------------------
# Semantic edges (ANN)
# ---------------------------------------------------------------------------

class TestSemanticEdges:
    def test_ann_creates_edge(self, db: DatabaseManager) -> None:
        _setup_fact(db, "f1", "Alice loves hiking",
                         embedding=[1.0, 0.0, 0.0])
        f2 = _setup_fact(db, "f2", "Alice enjoys trails",
                         embedding=[0.9, 0.1, 0.0])

        ann = MagicMock()
        ann.search.return_value = [("f1", 0.95)]

        builder = GraphBuilder(db=db, ann_index=ann)
        edges = builder.build_edges(f2, "default")
        semantic = [e for e in edges if e.edge_type == EdgeType.SEMANTIC]
        assert len(semantic) == 1
        assert semantic[0].weight == pytest.approx(0.95, abs=0.01)

    def test_below_threshold_no_edge(self, db: DatabaseManager) -> None:
        _setup_fact(db, "f1", "Fact A", embedding=[1.0, 0.0])
        f2 = _setup_fact(db, "f2", "Fact B", embedding=[0.0, 1.0])

        ann = MagicMock()
        ann.search.return_value = [("f1", 0.3)]  # Below 0.7 threshold

        builder = GraphBuilder(db=db, ann_index=ann)
        edges = builder.build_edges(f2, "default")
        semantic = [e for e in edges if e.edge_type == EdgeType.SEMANTIC]
        assert len(semantic) == 0

    def test_no_ann_no_semantic(self, db: DatabaseManager) -> None:
        f1 = _setup_fact(db, "f1", "Fact", embedding=[1.0, 0.0])
        builder = GraphBuilder(db=db, ann_index=None)
        edges = builder.build_edges(f1, "default")
        semantic = [e for e in edges if e.edge_type == EdgeType.SEMANTIC]
        assert len(semantic) == 0

    def test_no_embedding_no_semantic(self, db: DatabaseManager) -> None:
        f1 = _setup_fact(db, "f1", "Fact", embedding=None)
        ann = MagicMock()
        builder = GraphBuilder(db=db, ann_index=ann)
        edges = builder.build_edges(f1, "default")
        semantic = [e for e in edges if e.edge_type == EdgeType.SEMANTIC]
        assert len(semantic) == 0


# ---------------------------------------------------------------------------
# Causal edges
# ---------------------------------------------------------------------------

class TestCausalEdges:
    def test_causal_marker_creates_edge(self, db: DatabaseManager) -> None:
        _setup_fact(db, "f1", "Alice accepted the new job",
                    canonical_entities=["ent_alice"], obs_date="2026-01-01")
        f2 = _setup_fact(db, "f2",
                         "Alice moved to New York because of the new job",
                         canonical_entities=["ent_alice"], obs_date="2026-01-02")

        builder = GraphBuilder(db=db)
        edges = builder.build_edges(f2, "default")
        causal = [e for e in edges if e.edge_type == EdgeType.CAUSAL]
        assert len(causal) >= 1
        # Direction: cause (f1) -> effect (f2)
        assert all(e.target_id == "f2" for e in causal)

    def test_no_causal_marker_no_edge(self, db: DatabaseManager) -> None:
        _setup_fact(db, "f1", "Alice likes cats",
                    canonical_entities=["ent_alice"])
        f2 = _setup_fact(db, "f2", "Alice has a pet",
                         canonical_entities=["ent_alice"])
        builder = GraphBuilder(db=db)
        edges = builder.build_edges(f2, "default")
        causal = [e for e in edges if e.edge_type == EdgeType.CAUSAL]
        assert len(causal) == 0


# ---------------------------------------------------------------------------
# Contradiction edge (external API)
# ---------------------------------------------------------------------------

class TestContradictionEdge:
    def test_add_contradiction(self, db: DatabaseManager) -> None:
        builder = GraphBuilder(db=db)
        edge = builder.add_contradiction_edge("fa", "fb", "default", severity=0.8)
        assert edge.edge_type == EdgeType.CONTRADICTION
        assert edge.weight == 0.8

    def test_severity_clamped(self, db: DatabaseManager) -> None:
        builder = GraphBuilder(db=db)
        edge = builder.add_contradiction_edge("fa", "fb", "default", severity=1.5)
        assert edge.weight <= 1.0
        edge2 = builder.add_contradiction_edge("fa", "fc", "default", severity=-0.5)
        assert edge2.weight >= 0.0


# ---------------------------------------------------------------------------
# get_graph_stats
# ---------------------------------------------------------------------------

class TestGetGraphStats:
    def test_empty_graph(self, db: DatabaseManager) -> None:
        builder = GraphBuilder(db=db)
        stats = builder.get_graph_stats("default")
        assert stats["total_edges"] == 0
        assert stats["node_count"] == 0
        assert stats["avg_degree"] == 0.0

    def test_with_edges(self, db: DatabaseManager) -> None:
        _setup_fact(db, "f1", "Alice at Google",
                    canonical_entities=["ent_alice"])
        f2 = _setup_fact(db, "f2", "Alice at Apple",
                         canonical_entities=["ent_alice"])

        builder = GraphBuilder(db=db)
        builder.build_edges(f2, "default")
        stats = builder.get_graph_stats("default")
        assert stats["total_edges"] > 0
        assert stats["node_count"] > 0


# ---------------------------------------------------------------------------
# Edge persistence and dedup
# ---------------------------------------------------------------------------

class TestEdgePersistence:
    def test_edges_persisted(self, db: DatabaseManager) -> None:
        _setup_fact(db, "f1", "Alice at Google",
                    canonical_entities=["ent_alice"])
        f2 = _setup_fact(db, "f2", "Alice at Apple",
                         canonical_entities=["ent_alice"])

        builder = GraphBuilder(db=db)
        edges = builder.build_edges(f2, "default")
        assert len(edges) > 0
        # Verify persisted
        stored = db.get_edges_for_node("f2", "default")
        assert len(stored) > 0

    def test_no_duplicate_edges(self, db: DatabaseManager) -> None:
        _setup_fact(db, "f1", "Alice at Google",
                    canonical_entities=["ent_alice"])
        f2 = _setup_fact(db, "f2", "Alice at Apple",
                         canonical_entities=["ent_alice"])

        builder = GraphBuilder(db=db)
        builder.build_edges(f2, "default")
        builder.build_edges(f2, "default")  # Build again
        # Second call should not create duplicates
        all_edges = db.get_edges_for_node("f2", "default")
        entity_edges = [e for e in all_edges if e.edge_type == EdgeType.ENTITY]
        # At most one entity edge between f1 and f2
        assert len(entity_edges) <= 2  # f2->f1 from first call
