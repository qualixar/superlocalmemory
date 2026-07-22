# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
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
    *, scope: str = "personal", shared_with: list[str] | None = None,
    access_count: int = 0, fisher_variance: list[float] | None = None,
) -> AtomicFact:
    record = MemoryRecord(
        profile_id=profile_id, content=content, session_id="s1",
        scope=scope, shared_with=shared_with,
    )
    db.store_memory(record)
    fact = AtomicFact(
        profile_id=profile_id,
        memory_id=record.memory_id,
        content=content,
        embedding=_make_embedding(seed),
        scope=scope,
        shared_with=shared_with,
        access_count=access_count,
        fisher_variance=fisher_variance,
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

    def test_fast_and_fallback_fisher_scores_are_equivalent(
        self, db: DatabaseManager,
    ) -> None:
        """Candidate-source choice must not change Fisher contribution."""
        facts = [
            _seed_fact(
                db, "default", "fresh fact", seed=1,
                access_count=0, fisher_variance=[0.2] * DIM,
            ),
            _seed_fact(
                db, "default", "used fact", seed=2,
                access_count=8, fisher_variance=[0.4] * DIM,
            ),
        ]
        query = np.asarray(_make_embedding(3), dtype=np.float32)
        knn = []
        for fact in facts:
            vector = np.asarray(fact.embedding, dtype=np.float32)
            cosine = float(np.dot(query, vector) / (
                np.linalg.norm(query) * np.linalg.norm(vector)
            ))
            # Model the real VectorStore.search contract: max(0.0, 1.0 - distance),
            # i.e. max(0, cosine) for the cosine metric — NOT the canonical
            # (cosine+1)/2 normalization (that is applied inside the channel).
            knn.append((fact.fact_id, max(0.0, cosine)))

        mock_vs = MagicMock(available=True)
        mock_vs.search.return_value = sorted(
            knn, key=lambda item: item[1], reverse=True,
        )
        fast = SemanticChannel(db, vector_store=mock_vs).search(
            query.tolist(), "default", top_k=10,
        )
        fallback = SemanticChannel(db, vector_store=None).search(
            query.tolist(), "default", top_k=10,
        )

        assert [fact_id for fact_id, _ in fast] == [
            fact_id for fact_id, _ in fallback
        ]
        assert dict(fast) == pytest.approx(dict(fallback), abs=1e-6)

    def test_fast_path_merges_global_and_authorized_shared_candidates(
        self, db: DatabaseManager,
    ) -> None:
        for profile_id in ("requester", "publisher"):
            db.execute(
                "INSERT OR IGNORE INTO profiles (profile_id, name, description) "
                "VALUES (?, ?, '')",
                (profile_id, profile_id),
            )
        local = _seed_fact(db, "requester", "local", seed=2)
        global_fact = _seed_fact(
            db, "publisher", "global", seed=1, scope="global",
        )
        shared_fact = _seed_fact(
            db, "publisher", "shared", seed=1, scope="shared",
            shared_with=["requester"],
        )
        private_fact = _seed_fact(db, "publisher", "private", seed=1)
        project_fact = _seed_fact(
            db, "publisher", "project", seed=1, scope="project",
        )
        denied_fact = _seed_fact(
            db, "publisher", "denied", seed=1, scope="shared",
            shared_with=["someone-else"],
        )

        mock_vs = MagicMock()
        mock_vs.available = True
        mock_vs.search.return_value = [(local.fact_id, 0.5)]
        channel = SemanticChannel(db, vector_store=mock_vs)
        channel.include_global = True
        channel.include_shared = True

        result_ids = {
            fid for fid, _ in channel.search(_make_embedding(1), "requester", top_k=10)
        }
        assert {local.fact_id, global_fact.fact_id, shared_fact.fact_id} <= result_ids
        assert private_fact.fact_id not in result_ids
        assert project_fact.fact_id not in result_ids
        assert denied_fact.fact_id not in result_ids


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

    def test_full_scan_scope_contract_matches_fast_path_policy(
        self, db: DatabaseManager,
    ) -> None:
        for profile_id in ("requester", "publisher"):
            db.execute(
                "INSERT OR IGNORE INTO profiles (profile_id, name, description) "
                "VALUES (?, ?, '')",
                (profile_id, profile_id),
            )
        global_fact = _seed_fact(
            db, "publisher", "global", seed=1, scope="global",
        )
        private_fact = _seed_fact(db, "publisher", "private", seed=1)

        channel = SemanticChannel(db, vector_store=None)
        channel.include_global = True
        result_ids = {
            fid for fid, _ in channel.search(_make_embedding(1), "requester", top_k=10)
        }
        assert global_fact.fact_id in result_ids
        assert private_fact.fact_id not in result_ids
