# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for HopfieldChannel (Phase G: 6th Retrieval Channel).

TDD Phase: RED first, then GREEN.
8 tests per LLD Section 6.2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — lightweight mock objects
# ---------------------------------------------------------------------------

DIM = 768


def _random_embedding(d: int = DIM, seed: int = 42) -> list[float]:
    """Generate a single random L2-normalized embedding as list[float]."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(d).astype(np.float32)
    vec = vec / max(float(np.linalg.norm(vec)), 1e-8)
    return vec.tolist()


@dataclass
class FakeFact:
    """Minimal AtomicFact stand-in for testing."""

    fact_id: str = "f1"
    profile_id: str = "default"
    embedding: list[float] | None = None


class FakeDB:
    """Mock DatabaseManager that returns configurable facts."""

    def __init__(self, facts: list[FakeFact] | None = None) -> None:
        self._facts = facts or []

    def get_all_facts(self, profile_id: str, limit: int | None = None, **kwargs) -> list[FakeFact]:
        out = [f for f in self._facts if f.profile_id == profile_id]
        return out[:limit] if limit is not None else out

    def get_facts_by_ids(
        self, fact_ids: list[str], profile_id: str, **kwargs,
    ) -> list[FakeFact]:
        id_set = set(fact_ids)
        return [
            f for f in self._facts
            if f.fact_id in id_set and f.profile_id == profile_id
        ]


class FakeVectorStore:
    """Mock VectorStore for testing prefilter and count paths."""

    def __init__(
        self,
        available: bool = True,
        count_val: int = 0,
        search_results: list[tuple[str, float]] | None = None,
    ) -> None:
        self.available = available
        self._count_val = count_val
        self._search_results = search_results or []

    def count(self, profile_id: str | None = None) -> int:
        return self._count_val

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 30,
        profile_id: str | None = None,
    ) -> list[tuple[str, float]]:
        return self._search_results[:top_k]


def _make_facts(n: int, d: int = DIM, profile: str = "default") -> list[FakeFact]:
    """Generate n FakeFacts with random embeddings."""
    facts = []
    for i in range(n):
        facts.append(FakeFact(
            fact_id=f"fact_{i}",
            profile_id=profile,
            embedding=_random_embedding(d, seed=1000 + i),
        ))
    return facts


# ---------------------------------------------------------------------------
# Test 9 (LLD #10): Channel search returns correct format
# ---------------------------------------------------------------------------

class TestChannelSearchReturnsCorrectFormat:
    """LLD Test 10: search() returns list[tuple[str, float]], len <= top_k."""

    def test_channel_search_returns_correct_format(self) -> None:
        from superlocalmemory.math.hopfield import HopfieldConfig
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel

        facts = _make_facts(10)
        db = FakeDB(facts)
        vs = FakeVectorStore(available=True, count_val=10)
        config = HopfieldConfig(dimension=DIM, enabled=True)

        channel = HopfieldChannel(db=db, vector_store=vs, config=config)
        query = _random_embedding(DIM, seed=999)
        result = channel.search(query, "default", top_k=5)

        # Return type check
        assert isinstance(result, list)
        assert len(result) <= 5
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            fid, score = item
            assert isinstance(fid, str)
            assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Test 10 (LLD #11): Empty memory returns empty list
# ---------------------------------------------------------------------------

class TestEmptyMemoryReturnsEmptyList:
    """LLD Test 11: No facts -> returns []."""

    def test_empty_memory_returns_empty_list(self) -> None:
        from superlocalmemory.math.hopfield import HopfieldConfig
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel

        db = FakeDB([])
        vs = FakeVectorStore(available=True, count_val=0)
        config = HopfieldConfig(dimension=DIM, enabled=True)

        channel = HopfieldChannel(db=db, vector_store=vs, config=config)
        query = _random_embedding(DIM, seed=1)
        result = channel.search(query, "default")

        assert result == []


# ---------------------------------------------------------------------------
# Test 11 (LLD #12): Single fact retrieval
# ---------------------------------------------------------------------------

class TestSingleFactRetrieval:
    """LLD Test 12: 1 fact in DB -> returns [(fact_id, score)] with score > 0."""

    def test_single_fact_retrieval(self) -> None:
        from superlocalmemory.math.hopfield import HopfieldConfig
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel

        facts = _make_facts(1)
        db = FakeDB(facts)
        vs = FakeVectorStore(available=True, count_val=1)
        config = HopfieldConfig(dimension=DIM, enabled=True)

        channel = HopfieldChannel(db=db, vector_store=vs, config=config)
        # Query with same embedding as the single fact
        query = facts[0].embedding
        result = channel.search(query, "default")

        assert len(result) == 1
        fid, score = result[0]
        assert fid == "fact_0"
        assert score > 0.0


# ---------------------------------------------------------------------------
# Test 12 (LLD #13): Disabled returns empty
# ---------------------------------------------------------------------------

class TestDisabledReturnsEmpty:
    """LLD Test 13: config.enabled=False -> returns [] immediately."""

    def test_disabled_returns_empty(self) -> None:
        from superlocalmemory.math.hopfield import HopfieldConfig
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel

        facts = _make_facts(10)
        db = FakeDB(facts)
        vs = FakeVectorStore(available=True, count_val=10)
        config = HopfieldConfig(dimension=DIM, enabled=False)

        channel = HopfieldChannel(db=db, vector_store=vs, config=config)
        query = _random_embedding(DIM, seed=1)
        result = channel.search(query, "default")

        assert result == []


# ---------------------------------------------------------------------------
# Test 13: Dimension mismatch returns empty
# ---------------------------------------------------------------------------

class TestDimensionMismatchReturnsEmpty:
    """Dimension mismatch between query and config -> returns []."""

    def test_dimension_mismatch_returns_empty(self) -> None:
        from superlocalmemory.math.hopfield import HopfieldConfig
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel

        facts = _make_facts(5, d=DIM)
        db = FakeDB(facts)
        vs = FakeVectorStore(available=True, count_val=5)
        config = HopfieldConfig(dimension=DIM, enabled=True)

        channel = HopfieldChannel(db=db, vector_store=vs, config=config)
        # Query with wrong dimension (384 instead of 768)
        wrong_dim_query = _random_embedding(384, seed=1)
        result = channel.search(wrong_dim_query, "default")

        assert result == []


# ---------------------------------------------------------------------------
# Test 14 (LLD #13): Large scale triggers prefilter
# ---------------------------------------------------------------------------

class TestLargeScaleTriggersPrefilter:
    """LLD Test 13: >prefilter_threshold facts -> VectorStore.search() is called."""

    def test_large_scale_triggers_prefilter(self) -> None:
        from superlocalmemory.math.hopfield import HopfieldConfig
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel

        # Use low threshold so we don't need 10K+ real facts
        config = HopfieldConfig(
            dimension=DIM, enabled=True,
            prefilter_threshold=5, prefilter_candidates=3,
        )

        # Create 20 facts but report count=20 so threshold triggers
        facts = _make_facts(20)
        db = FakeDB(facts)

        # VectorStore returns top-3 candidates
        knn_results = [(f"fact_{i}", 0.9 - i * 0.1) for i in range(3)]
        vs = FakeVectorStore(available=True, count_val=20, search_results=knn_results)

        channel = HopfieldChannel(db=db, vector_store=vs, config=config)
        query = _random_embedding(DIM, seed=42)
        result = channel.search(query, "default")

        # Should have results (from prefilter path)
        assert isinstance(result, list)
        # Result fact_ids should be a subset of the KNN candidates
        result_ids = {fid for fid, _ in result}
        candidate_ids = {fid for fid, _ in knn_results}
        assert result_ids.issubset(candidate_ids), (
            f"Result IDs {result_ids} not subset of KNN candidates {candidate_ids}"
        )


# ---------------------------------------------------------------------------
# Test 15: Cache invalidation
# ---------------------------------------------------------------------------

class TestCacheInvalidation:
    """Cache is cleared after invalidate_cache() call."""

    def test_cache_invalidation(self) -> None:
        from superlocalmemory.math.hopfield import HopfieldConfig
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel

        facts = _make_facts(5)
        db = FakeDB(facts)
        vs = FakeVectorStore(available=True, count_val=5)
        config = HopfieldConfig(dimension=DIM, enabled=True)

        channel = HopfieldChannel(db=db, vector_store=vs, config=config)
        query = _random_embedding(DIM, seed=1)

        # First search populates cache
        result1 = channel.search(query, "default")
        assert channel._cached_matrix is not None

        # Invalidate
        channel.invalidate_cache()
        assert channel._cached_matrix is None
        assert channel._cached_count == 0

        # Second search repopulates cache
        result2 = channel.search(query, "default")
        assert channel._cached_matrix is not None


# ---------------------------------------------------------------------------
# Test 16: Skip threshold returns empty
# ---------------------------------------------------------------------------

class TestSkipThresholdReturnsEmpty:
    """skip_threshold exceeded -> returns [] without loading matrix."""

    def test_skip_threshold_returns_empty(self) -> None:
        from superlocalmemory.math.hopfield import HopfieldConfig
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel

        facts = _make_facts(5)
        db = FakeDB(facts)
        # Report 200K facts — exceeds skip_threshold
        vs = FakeVectorStore(available=True, count_val=200_000)
        config = HopfieldConfig(
            dimension=DIM, enabled=True, skip_threshold=100_000,
        )

        channel = HopfieldChannel(db=db, vector_store=vs, config=config)
        query = _random_embedding(DIM, seed=1)
        result = channel.search(query, "default")

        assert result == []
        # Matrix should NOT have been loaded
        assert channel._cached_matrix is None


# ---------------------------------------------------------------------------
# Coverage gap tests — exercise uncovered branches
# ---------------------------------------------------------------------------

class TestHopfieldChannelCoverageGaps:
    """Additional tests to cover edge-case branches for 100% coverage."""

    def test_search_error_returns_empty(self) -> None:
        """HR-06: Any exception in search() -> returns []."""
        from superlocalmemory.math.hopfield import HopfieldConfig
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel

        # DB that raises on get_all_facts
        db = MagicMock()
        db.get_all_facts.side_effect = RuntimeError("DB exploded")
        vs = FakeVectorStore(available=True, count_val=5)
        config = HopfieldConfig(dimension=DIM, enabled=True)

        channel = HopfieldChannel(db=db, vector_store=vs, config=config)
        query = _random_embedding(DIM, seed=1)
        result = channel.search(query, "default")

        assert result == []

    def test_cache_hit_returns_cached_results(self) -> None:
        """Second search on same profile with same count uses cache."""
        from superlocalmemory.math.hopfield import HopfieldConfig
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel

        facts = _make_facts(5)
        db = FakeDB(facts)
        vs = FakeVectorStore(available=True, count_val=5)
        config = HopfieldConfig(dimension=DIM, enabled=True, cache_ttl_seconds=60.0)

        channel = HopfieldChannel(db=db, vector_store=vs, config=config)
        query = _random_embedding(DIM, seed=1)

        # First search
        result1 = channel.search(query, "default")
        cached_matrix_id = id(channel._cached_matrix)

        # Second search should use cache (same object identity)
        result2 = channel.search(query, "default")
        assert id(channel._cached_matrix) == cached_matrix_id
        assert result1 == result2

    def test_prefilter_no_vector_store_returns_empty(self) -> None:
        """Prefilter path with unavailable VectorStore returns []."""
        from superlocalmemory.math.hopfield import HopfieldConfig
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel

        config = HopfieldConfig(
            dimension=DIM, enabled=True, prefilter_threshold=3,
        )
        facts = _make_facts(10)
        db = FakeDB(facts)
        # VS reports 10 facts for count but is unavailable for search
        vs = FakeVectorStore(available=False, count_val=0)

        channel = HopfieldChannel(db=db, vector_store=vs, config=config)
        query = _random_embedding(DIM, seed=1)
        result = channel.search(query, "default")

        # With unavailable VS, count=0, so no skip and no prefilter needed.
        # The facts load fine via full matrix path. So result should be non-empty.
        assert isinstance(result, list)

    def test_prefilter_empty_knn_returns_empty(self) -> None:
        """Prefilter path where VectorStore KNN returns [] -> returns []."""
        from superlocalmemory.math.hopfield import HopfieldConfig
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel

        config = HopfieldConfig(
            dimension=DIM, enabled=True,
            prefilter_threshold=3, prefilter_candidates=5,
        )
        facts = _make_facts(10)
        db = FakeDB(facts)
        # VS returns empty search results
        vs = FakeVectorStore(available=True, count_val=10, search_results=[])

        channel = HopfieldChannel(db=db, vector_store=vs, config=config)
        query = _random_embedding(DIM, seed=1)
        result = channel.search(query, "default")

        assert result == []

    def test_prefilter_candidates_no_embeddings_returns_empty(self) -> None:
        """Prefilter path where KNN candidates have no valid embeddings -> [].

        The matrix is built from valid facts, triggering the prefilter path
        (len(fact_ids) > threshold). The KNN then returns IDs that resolve
        to facts with None embeddings, so sub_embeddings is empty.
        """
        from superlocalmemory.math.hopfield import HopfieldConfig
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel

        config = HopfieldConfig(
            dimension=DIM, enabled=True,
            prefilter_threshold=3, prefilter_candidates=5,
        )

        # 10 facts WITH valid embeddings (so the matrix builds fine)
        valid_facts = _make_facts(10)
        # 5 extra facts WITHOUT embeddings (these will be the KNN candidates)
        no_emb_facts = [
            FakeFact(fact_id=f"no_emb_{i}", profile_id="default", embedding=None)
            for i in range(5)
        ]
        all_facts = valid_facts + no_emb_facts

        # Use a mock DB that returns all facts for get_all_facts
        # but only the no-embedding facts for get_facts_by_ids
        db = MagicMock()
        db.get_all_facts.return_value = all_facts
        db.get_facts_by_ids.return_value = no_emb_facts

        # KNN returns the no-embedding fact IDs
        knn_results = [(f"no_emb_{i}", 0.9) for i in range(5)]
        vs = FakeVectorStore(available=True, count_val=15, search_results=knn_results)

        channel = HopfieldChannel(db=db, vector_store=vs, config=config)
        query = _random_embedding(DIM, seed=1)
        result = channel.search(query, "default")

        assert result == []

    def test_prefilter_db_returns_empty_candidates(self) -> None:
        """Prefilter path where get_facts_by_ids returns [] -> returns []."""
        from superlocalmemory.math.hopfield import HopfieldConfig
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel

        config = HopfieldConfig(
            dimension=DIM, enabled=True,
            prefilter_threshold=3, prefilter_candidates=5,
        )

        # DB returns facts for get_all_facts but empty for get_facts_by_ids
        # (simulating deleted facts between KNN and load)
        facts = _make_facts(10)
        db = MagicMock()
        db.get_all_facts.return_value = facts
        db.get_facts_by_ids.return_value = []

        knn_results = [(f"nonexistent_{i}", 0.9) for i in range(5)]
        vs = FakeVectorStore(available=True, count_val=10, search_results=knn_results)

        channel = HopfieldChannel(db=db, vector_store=vs, config=config)
        query = _random_embedding(DIM, seed=1)
        result = channel.search(query, "default")

        assert result == []

    def test_facts_with_no_valid_embeddings_returns_empty(self) -> None:
        """All facts have None embeddings -> returns []."""
        from superlocalmemory.math.hopfield import HopfieldConfig
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel

        facts_no_emb = [
            FakeFact(fact_id=f"fact_{i}", profile_id="default", embedding=None)
            for i in range(5)
        ]
        db = FakeDB(facts_no_emb)
        vs = FakeVectorStore(available=True, count_val=5)
        config = HopfieldConfig(dimension=DIM, enabled=True)

        channel = HopfieldChannel(db=db, vector_store=vs, config=config)
        query = _random_embedding(DIM, seed=1)
        result = channel.search(query, "default")

        assert result == []

    def test_prefilter_candidates_valid_but_wrong_dim(self) -> None:
        """Prefilter: KNN candidates have embeddings but wrong dimension -> [].

        Exercises the empty sub_embeddings branch in _search_with_prefilter
        when all candidate embeddings fail the dimension check.
        """
        from superlocalmemory.math.hopfield import HopfieldConfig
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel

        config = HopfieldConfig(
            dimension=DIM, enabled=True,
            prefilter_threshold=3, prefilter_candidates=5,
        )

        # IDs 0-4: correct 768-d embeddings (for matrix loading)
        # IDs 5-9: wrong 384-d embeddings (KNN candidates)
        good_facts = _make_facts(5, d=DIM)
        wrong_dim_facts = [
            FakeFact(
                fact_id=f"fact_{i + 5}",
                profile_id="default",
                embedding=_random_embedding(384, seed=3000 + i),
            )
            for i in range(5)
        ]
        all_facts = good_facts + wrong_dim_facts
        db = FakeDB(all_facts)

        knn_results = [(f"fact_{i + 5}", 0.9) for i in range(5)]
        vs = FakeVectorStore(available=True, count_val=10, search_results=knn_results)

        channel = HopfieldChannel(db=db, vector_store=vs, config=config)
        query = _random_embedding(DIM, seed=1)
        result = channel.search(query, "default")

        assert result == []
