# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for LanceDBVectorBackend — Sprint 3."""

from __future__ import annotations

import importlib.util
import numpy as np
import pytest
import shutil

# LanceDB is an optional backend (pip install superlocalmemory[lancedb]). When
# it is absent these tests cannot run — skip cleanly instead of erroring at
# fixture setup, so the suite stays honestly green on installs without it.
pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(
        importlib.util.find_spec("lancedb") is None,
        reason="LanceDB optional dependency not installed (pip install superlocalmemory[lancedb])",
    ),
]


@pytest.fixture
def backend():
    """Create temporary LanceDB backend."""
    from superlocalmemory.vector.lancedb_backend import LanceDBVectorBackend

    path = "/tmp/test_lancedb_backend"
    be = LanceDBVectorBackend(path)
    yield be
    be.close()
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def populated(backend):
    """Backend with 10 vectors across tiers."""
    np.random.seed(42)
    ids = [f"f{i}" for i in range(10)]
    vecs = [np.random.rand(768).tolist() for _ in range(10)]
    tiers = ["active"] * 4 + ["warm"] * 3 + ["cold"] * 2 + ["archived"] * 1
    backend.add_vectors(ids, vecs, tiers)
    return backend


def _make_vec(seed: float = 0.5) -> list[float]:
    return np.full(768, seed, dtype=np.float32).tolist()


class TestLifecycle:
    def test_open_and_close(self):
        from superlocalmemory.vector.lancedb_backend import LanceDBVectorBackend

        be = LanceDBVectorBackend("/tmp/test_lance_lifecycle")
        health = be.health_check()
        assert health["status"] == "active"
        be.close()
        assert be._table is None
        assert be._db is None
        shutil.rmtree("/tmp/test_lance_lifecycle", ignore_errors=True)

    def test_empty_backend_returns_zero(self, backend):
        results = backend.similarity_search(_make_vec(), top_k=10)
        assert results == []


class TestSimilaritySearch:
    def test_basic_search(self, populated):
        results = populated.similarity_search(_make_vec(0.5), top_k=5)
        assert len(results) == 5

    def test_cosine_scores_in_range(self, populated):
        results = populated.similarity_search(_make_vec(0.5), top_k=10)
        for _, score in results:
            assert 0.0 <= score <= 1.01, f"Score {score} out of [0, 1]"

    def test_tier_filter_excludes_cold(self, populated):
        results = populated.similarity_search(
            _make_vec(0.5), top_k=10, tier_filter=["active", "warm"]
        )
        fact_ids = {r[0] for r in results}
        # cold facts: f7, f8. archive: f9
        assert "f7" not in fact_ids
        assert "f8" not in fact_ids
        assert "f9" not in fact_ids

    def test_deep_recall_includes_all(self, populated):
        results = populated.similarity_search(
            _make_vec(0.5), top_k=20,
            tier_filter=["active", "warm", "cold", "archived"],
        )
        assert len(results) == 10

    def test_invalid_tier_raises(self, populated):
        with pytest.raises(AssertionError):
            populated.similarity_search(
                _make_vec(), tier_filter=["hot", "warm"]  # "hot" not valid
            )


class TestWrite:
    def test_add_vectors_returns_count(self, backend):
        count = backend.add_vectors(
            ["a1"], [_make_vec()], ["active"]
        )
        assert count == 1

    def test_update_tier(self, populated):
        populated.update_tier("f0", "cold")
        results = populated.similarity_search(
            _make_vec(0.5), top_k=10, tier_filter=["active", "warm"]
        )
        fact_ids = {r[0] for r in results}
        assert "f0" not in fact_ids  # Now cold, excluded


class TestHealth:
    def test_health_counts_vectors(self, populated):
        health = populated.health_check()
        assert health["vectors"] == 10
        assert health["status"] == "active"
