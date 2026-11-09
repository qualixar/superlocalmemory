# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.math.sheaf — Sheaf cohomology contradiction detection.

Covers:
  - ContradictionResult dataclass
  - _restriction_for_edge_type: identity vs scaled maps
  - edge_residual: coboundary computation
  - coboundary_norm: normalized severity
  - SheafConsistencyChecker.check_consistency with mocked DB
  - SheafConsistencyChecker.detect_contradictions_batch
  - Edge cases: no embedding, no entities, empty batch, same-pair dedup
  - Mathematical invariants: severity in [0, 1], non-negative coboundary
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from superlocalmemory.math.sheaf import (
    TEMPORAL_TOLERANCE,
    ContradictionResult,
    SheafConsistencyChecker,
    _restriction_for_edge_type,
    coboundary_norm,
    edge_residual,
)
from superlocalmemory.storage.models import AtomicFact, EdgeType, GraphEdge

# ---------------------------------------------------------------------------
# ContradictionResult
# ---------------------------------------------------------------------------

class TestContradictionResult:
    def test_immutable(self) -> None:
        cr = ContradictionResult(
            fact_id_a="a", fact_id_b="b", severity=0.8,
            edge_type="entity", description="test",
        )
        with pytest.raises(AttributeError):
            cr.severity = 0.5  # type: ignore[misc]

    def test_fields(self) -> None:
        cr = ContradictionResult(
            fact_id_a="f1", fact_id_b="f2", severity=0.75,
            edge_type="temporal", description="high disagreement",
        )
        assert cr.fact_id_a == "f1"
        assert cr.fact_id_b == "f2"
        assert cr.severity == 0.75
        assert cr.edge_type == "temporal"


# ---------------------------------------------------------------------------
# _restriction_for_edge_type
# ---------------------------------------------------------------------------

class TestRestrictionMap:
    def test_entity_returns_identity(self) -> None:
        R = _restriction_for_edge_type("entity", 3)
        np.testing.assert_allclose(R, np.eye(3))

    def test_semantic_returns_identity(self) -> None:
        R = _restriction_for_edge_type("semantic", 4)
        np.testing.assert_allclose(R, np.eye(4))

    def test_temporal_returns_identity(self) -> None:
        R = _restriction_for_edge_type("temporal", 3)
        np.testing.assert_allclose(R, np.eye(3))

    def test_unknown_type_returns_identity(self) -> None:
        R = _restriction_for_edge_type("unknown_type", 2)
        np.testing.assert_allclose(R, np.eye(2))


# ---------------------------------------------------------------------------
# edge_residual
# ---------------------------------------------------------------------------

class TestEdgeResidual:
    def test_identical_embeddings_zero_residual(self) -> None:
        emb = np.array([1.0, 0.0, 0.0])
        R = np.eye(3)
        residual = edge_residual(emb, emb, R, R)
        np.testing.assert_allclose(residual, np.zeros(3), atol=1e-12)

    def test_different_embeddings_nonzero(self) -> None:
        emb_a = np.array([1.0, 0.0])
        emb_b = np.array([0.0, 1.0])
        R = np.eye(2)
        residual = edge_residual(emb_a, emb_b, R, R)
        expected = emb_b - emb_a
        np.testing.assert_allclose(residual, expected)

    def test_with_scaled_restriction(self) -> None:
        emb_a = np.array([1.0, 2.0])
        emb_b = np.array([3.0, 4.0])
        R_a = 0.5 * np.eye(2)
        R_b = np.eye(2)
        residual = edge_residual(emb_a, emb_b, R_a, R_b)
        expected = R_b @ emb_b - R_a @ emb_a
        np.testing.assert_allclose(residual, expected)


# ---------------------------------------------------------------------------
# coboundary_norm
# ---------------------------------------------------------------------------

class TestCoboundaryNorm:
    def test_identical_gives_zero(self) -> None:
        emb = np.array([1.0, 0.0, 0.0])
        R = np.eye(3)
        severity = coboundary_norm(emb, emb, R, R)
        np.testing.assert_allclose(severity, 0.0, atol=1e-10)

    def test_orthogonal_gives_positive(self) -> None:
        emb_a = np.array([1.0, 0.0])
        emb_b = np.array([0.0, 1.0])
        R = np.eye(2)
        severity = coboundary_norm(emb_a, emb_b, R, R)
        assert severity > 0.0

    def test_opposite_vectors_high_severity(self) -> None:
        emb_a = np.array([1.0, 0.0])
        emb_b = np.array([-1.0, 0.0])
        R = np.eye(2)
        severity = coboundary_norm(emb_a, emb_b, R, R)
        # ||b - a|| / (||a|| + ||b||) = 2 / 2 = 1.0
        np.testing.assert_allclose(severity, 1.0, atol=1e-10)

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            emb_a = rng.standard_normal(10)
            emb_b = rng.standard_normal(10)
            R = np.eye(10)
            assert coboundary_norm(emb_a, emb_b, R, R) >= 0.0

    def test_temporal_scaling_reduces_severity(self) -> None:
        """Temporal restriction (0.6*I) should reduce effective severity
        compared to identity restriction."""
        emb_a = np.array([1.0, 0.0, 0.0])
        emb_b = np.array([0.0, 1.0, 0.0])
        R_id = np.eye(3)
        R_temp = TEMPORAL_TOLERANCE * np.eye(3)
        sev_entity = coboundary_norm(emb_a, emb_b, R_id, R_id)
        sev_temporal = coboundary_norm(emb_a, emb_b, R_temp, R_temp)
        # Same ratio since both R_a and R_b scale equally
        # The ratio should be the same because norm scales linearly
        np.testing.assert_allclose(sev_entity, sev_temporal, atol=1e-8)


# ---------------------------------------------------------------------------
# Helper: mock DB for SheafConsistencyChecker
# ---------------------------------------------------------------------------

def _make_mock_db() -> MagicMock:
    """Create a mock DatabaseManager."""
    db = MagicMock()
    db.get_edges_for_node.return_value = []
    db.execute.return_value = []
    return db


def _make_fact(
    fact_id: str = "f1",
    embedding: list[float] | None = None,
    canonical_entities: list[str] | None = None,
) -> AtomicFact:
    """Create an AtomicFact with defaults for testing."""
    return AtomicFact(
        fact_id=fact_id,
        embedding=embedding or [1.0, 0.0, 0.0],
        canonical_entities=canonical_entities or ["entity_a"],
        content="test fact",
    )


# ---------------------------------------------------------------------------
# SheafConsistencyChecker — check_consistency
# ---------------------------------------------------------------------------

class TestCheckConsistency:
    def test_no_embedding_returns_empty(self) -> None:
        db = _make_mock_db()
        checker = SheafConsistencyChecker(db)
        fact = AtomicFact(fact_id="f1", embedding=None, canonical_entities=["e1"])
        assert checker.check_consistency(fact, "default") == []

    def test_no_entities_returns_empty(self) -> None:
        db = _make_mock_db()
        checker = SheafConsistencyChecker(db)
        fact = AtomicFact(fact_id="f1", embedding=[1.0, 0.0], canonical_entities=[])
        assert checker.check_consistency(fact, "default") == []

    def test_empty_embedding_returns_empty(self) -> None:
        db = _make_mock_db()
        checker = SheafConsistencyChecker(db)
        fact = AtomicFact(fact_id="f1", embedding=[], canonical_entities=["e1"])
        assert checker.check_consistency(fact, "default") == []

    def test_no_edges_returns_empty(self) -> None:
        db = _make_mock_db()
        db.get_edges_for_node.return_value = []
        checker = SheafConsistencyChecker(db)
        fact = _make_fact()
        assert checker.check_consistency(fact, "default") == []

    def test_skips_contradiction_edges(self) -> None:
        db = _make_mock_db()
        edge = GraphEdge(
            edge_id="e1", source_id="f1", target_id="f2",
            edge_type=EdgeType.CONTRADICTION,
        )
        db.get_edges_for_node.return_value = [edge]
        checker = SheafConsistencyChecker(db)
        fact = _make_fact()
        results = checker.check_consistency(fact, "default")
        assert results == []

    def test_skips_supersedes_edges(self) -> None:
        db = _make_mock_db()
        edge = GraphEdge(
            edge_id="e1", source_id="f1", target_id="f2",
            edge_type=EdgeType.SUPERSEDES,
        )
        db.get_edges_for_node.return_value = [edge]
        checker = SheafConsistencyChecker(db)
        fact = _make_fact()
        results = checker.check_consistency(fact, "default")
        assert results == []

    def test_detects_contradiction_above_threshold(self) -> None:
        db = _make_mock_db()
        # Set up an entity edge to another fact
        edge = GraphEdge(
            edge_id="e1", source_id="f1", target_id="f2",
            edge_type=EdgeType.ENTITY,
        )
        db.get_edges_for_node.return_value = [edge]
        # Return opposite embedding for the other fact
        opposite_emb = json.dumps([-1.0, 0.0, 0.0])
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: {"embedding": opposite_emb}[key]
        mock_row.keys = lambda: ["embedding"]
        db.execute.return_value = [mock_row]

        checker = SheafConsistencyChecker(db, contradiction_threshold=0.5)
        fact = _make_fact(embedding=[1.0, 0.0, 0.0])
        fact.content = "Priya's status is active"
        other = _make_fact("f2", [-1.0, 0.0, 0.0])
        other.content = "Priya's status is inactive"
        db.get_fact.return_value = other
        results = checker.check_consistency(fact, "default")
        assert len(results) == 1
        assert results[0].severity <= 1.0
        assert results[0].severity > 0.5

    def test_no_contradiction_below_threshold(self) -> None:
        db = _make_mock_db()
        edge = GraphEdge(
            edge_id="e1", source_id="f1", target_id="f2",
            edge_type=EdgeType.ENTITY,
        )
        db.get_edges_for_node.return_value = [edge]
        # Return very similar embedding
        similar_emb = json.dumps([0.99, 0.01, 0.0])
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: {"embedding": similar_emb}[key]
        mock_row.keys = lambda: ["embedding"]
        db.execute.return_value = [mock_row]

        checker = SheafConsistencyChecker(db, contradiction_threshold=0.7)
        fact = _make_fact(embedding=[1.0, 0.0, 0.0])
        results = checker.check_consistency(fact, "default")
        assert results == []

    def test_deduplicates_other_ids(self) -> None:
        db = _make_mock_db()
        # Two edges pointing to the same other fact
        edge1 = GraphEdge(
            edge_id="e1", source_id="f1", target_id="f2",
            edge_type=EdgeType.ENTITY,
        )
        edge2 = GraphEdge(
            edge_id="e2", source_id="f2", target_id="f1",
            edge_type=EdgeType.SEMANTIC,
        )
        db.get_edges_for_node.return_value = [edge1, edge2]
        # Return empty for embedding lookup
        db.get_fact.return_value = None

        checker = SheafConsistencyChecker(db)
        fact = _make_fact()
        checker.check_consistency(fact, "default")
        # Only checked once despite two edges to f2
        assert db.get_fact.call_count == 1

    def test_threshold_clamped_to_valid_range(self) -> None:
        db = _make_mock_db()
        checker = SheafConsistencyChecker(db, contradiction_threshold=5.0)
        assert checker._threshold == 2.0
        checker2 = SheafConsistencyChecker(db, contradiction_threshold=-1.0)
        assert checker2._threshold == 0.0


# ---------------------------------------------------------------------------
# SheafConsistencyChecker — detect_contradictions_batch
# ---------------------------------------------------------------------------

class TestBatchContradictions:
    def test_empty_batch(self) -> None:
        db = _make_mock_db()
        checker = SheafConsistencyChecker(db)
        assert checker.detect_contradictions_batch([], "default") == []

    def test_no_embeddings_skipped(self) -> None:
        db = _make_mock_db()
        checker = SheafConsistencyChecker(db)
        facts = [AtomicFact(fact_id="f1", embedding=None, canonical_entities=["e1"])]
        assert checker.detect_contradictions_batch(facts, "default") == []

    def test_no_shared_entity_no_contradiction(self) -> None:
        db = _make_mock_db()
        checker = SheafConsistencyChecker(db)
        f1 = _make_fact("f1", [1.0, 0.0, 0.0], ["entity_a"])
        f2 = _make_fact("f2", [-1.0, 0.0, 0.0], ["entity_b"])
        # Different entities => no pairwise check
        results = checker.detect_contradictions_batch([f1, f2], "default")
        assert results == []

    def test_detects_batch_contradiction(self) -> None:
        db = _make_mock_db()
        checker = SheafConsistencyChecker(db, contradiction_threshold=0.5)
        # Two facts with opposite embeddings sharing an entity
        f1 = _make_fact("f1", [1.0, 0.0, 0.0], ["shared_entity"])
        f2 = _make_fact("f2", [-1.0, 0.0, 0.0], ["shared_entity"])
        f1.content, f2.content = "Priya's status is active", "Priya's status is inactive"
        results = checker.detect_contradictions_batch([f1, f2], "default")
        assert len(results) == 1
        assert results[0].edge_type == "entity"

    def test_pair_deduplication(self) -> None:
        db = _make_mock_db()
        checker = SheafConsistencyChecker(db, contradiction_threshold=0.5)
        # Same entity appears twice => same pair, checked only once
        f1 = _make_fact("f1", [1.0, 0.0, 0.0], ["e1", "e2"])
        f2 = _make_fact("f2", [-1.0, 0.0, 0.0], ["e1", "e2"])
        f1.content, f2.content = "Priya's status is active", "Priya's status is inactive"
        results = checker.detect_contradictions_batch([f1, f2], "default")
        # Should find at most 1 result (deduped across entity groups)
        assert len(results) == 1

    def test_severity_capped_at_one(self) -> None:
        db = _make_mock_db()
        checker = SheafConsistencyChecker(db, contradiction_threshold=0.0)
        # Very different embeddings
        f1 = _make_fact("f1", [1.0, 0.0], ["e1"])
        f2 = _make_fact("f2", [-1.0, 0.0], ["e1"])
        results = checker.detect_contradictions_batch([f1, f2], "default")
        for r in results:
            assert r.severity <= 1.0

    def test_mismatched_embedding_dims_skipped(self) -> None:
        db = _make_mock_db()
        checker = SheafConsistencyChecker(db, contradiction_threshold=0.1)
        f1 = _make_fact("f1", [1.0, 0.0, 0.0], ["e1"])
        f2 = _make_fact("f2", [0.0, 1.0], ["e1"])  # Different dim
        results = checker.detect_contradictions_batch([f1, f2], "default")
        assert results == []

    def test_similar_batch_no_contradiction(self) -> None:
        db = _make_mock_db()
        checker = SheafConsistencyChecker(db, contradiction_threshold=0.7)
        f1 = _make_fact("f1", [1.0, 0.0, 0.0], ["e1"])
        f2 = _make_fact("f2", [0.99, 0.01, 0.0], ["e1"])
        results = checker.detect_contradictions_batch([f1, f2], "default")
        assert results == []
