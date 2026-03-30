# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for SemanticChannel with VectorStore integration.

Covers:
  - Constructor accepts vector_store=None (backward compat)
  - Fast path via VectorStore when available
  - Fallback to full scan when VectorStore is empty
  - Fallback to full scan when VectorStore is None
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from superlocalmemory.retrieval.semantic_channel import SemanticChannel
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, MemoryRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 8


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    db_path = tmp_path / "test_semantic_vec.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    return mgr


def _make_embedding(seed: int, dim: int = DIM) -> list[float]:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v.tolist()


def _seed_fact(
    db: DatabaseManager, profile_id: str, content: str, seed: int,
) -> AtomicFact:
    record = MemoryRecord(
        profile_id=profile_id, content=content, session_id="s1",
    )
    db.store_memory(record)
    fact = AtomicFact(
        profile_id=profile_id,
        memory_id=record.memory_id,
        content=content,
        embedding=_make_embedding(seed),
    )
    db.store_fact(fact)
    return fact


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """SemanticChannel works with vector_store=None (pre-Phase-1 behavior)."""

    def test_no_vector_store_uses_full_scan(self, db: DatabaseManager) -> None:
        _seed_fact(db, "default", "Alice went to Paris", seed=1)
        _seed_fact(db, "default", "Bob stayed in London", seed=2)

        ch = SemanticChannel(db, vector_store=None)
        query = _make_embedding(1)  # Similar to fact 1
        results = ch.search(query, "default", top_k=5)
        assert len(results) >= 1
        # Should find the seeded facts via full scan
        fact_ids = [fid for fid, _ in results]
        assert len(fact_ids) > 0


class TestVectorStoreFastPath:
    """SemanticChannel uses VectorStore when available."""

    def test_uses_vector_store_results(self, db: DatabaseManager) -> None:
        f1 = _seed_fact(db, "default", "Alice went to Paris", seed=1)
        f2 = _seed_fact(db, "default", "Bob stayed in London", seed=2)

        # Mock VectorStore
        mock_vs = MagicMock()
        mock_vs.available = True
        mock_vs.search.return_value = [
            (f1.fact_id, 0.95),
            (f2.fact_id, 0.70),
        ]

        ch = SemanticChannel(db, vector_store=mock_vs)
        query = _make_embedding(1)
        results = ch.search(query, "default", top_k=5)

        # Should have called VectorStore.search
        mock_vs.search.assert_called_once()
        assert len(results) >= 1


class TestFallbackOnEmptyVecStore:
    """SemanticChannel falls back to full scan if VectorStore is empty."""

    def test_empty_vec_store_falls_to_full_scan(
        self, db: DatabaseManager,
    ) -> None:
        _seed_fact(db, "default", "Alice went to Paris", seed=1)

        mock_vs = MagicMock()
        mock_vs.available = True
        mock_vs.search.return_value = []  # Empty vec0

        ch = SemanticChannel(db, vector_store=mock_vs)
        query = _make_embedding(1)
        results = ch.search(query, "default", top_k=5)

        # VectorStore.search was called but returned empty
        mock_vs.search.assert_called_once()
        # Full scan should still find the fact
        assert len(results) >= 1


class TestFallbackOnUnavailableVecStore:
    """SemanticChannel falls back when VectorStore.available is False."""

    def test_unavailable_vec_store_uses_full_scan(
        self, db: DatabaseManager,
    ) -> None:
        _seed_fact(db, "default", "Alice went to Paris", seed=1)

        mock_vs = MagicMock()
        mock_vs.available = False

        ch = SemanticChannel(db, vector_store=mock_vs)
        query = _make_embedding(1)
        results = ch.search(query, "default", top_k=5)

        # Should NOT have called VectorStore.search (unavailable)
        mock_vs.search.assert_not_called()
        # Full scan should still work
        assert len(results) >= 1
