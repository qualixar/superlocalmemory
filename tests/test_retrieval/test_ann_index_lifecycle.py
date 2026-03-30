# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for ANNIndex lifecycle — cold start, add, search, remove baselines.

Phase 0 Safety Net: captures current ANNIndex behavior with real numpy math
(no mocks) before Phase 1 replaces the index with VectorStore.

Covers:
  - Cold start: empty index search, empty index size
  - Add-then-search round trip
  - Multiple vector operations
  - top_k enforcement
  - Score ordering (descending)
  - Remove reduces size and excludes removed vectors
  - Dimension property reflects init value
"""

from __future__ import annotations

import numpy as np
import pytest

from superlocalmemory.retrieval.ann_index import ANNIndex


# ---------------------------------------------------------------------------
# Helpers — small dimension (4) for speed, real numpy math
# ---------------------------------------------------------------------------

DIM = 4


def _unit_vec(axis: int) -> list[float]:
    """One-hot unit vector along given axis in DIM-space."""
    v = [0.0] * DIM
    v[axis % DIM] = 1.0
    return v


def _random_vec(seed: int = 0) -> list[float]:
    """Deterministic random vector in DIM-space."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return v.tolist()


# ---------------------------------------------------------------------------
# Cold start — fresh index before any add()
# ---------------------------------------------------------------------------

class TestANNIndexColdStart:
    """Verify index behavior when completely empty (cold boot scenario)."""

    def test_empty_index_search_returns_empty(self) -> None:
        """Search on a fresh index with zero vectors must return []."""
        idx = ANNIndex(dimension=DIM)
        results = idx.search(_unit_vec(0), top_k=10)
        assert results == []

    def test_empty_index_size_zero(self) -> None:
        """A freshly constructed index reports size == 0."""
        idx = ANNIndex(dimension=DIM)
        assert idx.size == 0

    def test_add_then_search_finds_vector(self) -> None:
        """After adding one vector, search with matching query finds it."""
        idx = ANNIndex(dimension=DIM)
        idx.add("f1", _unit_vec(0))
        results = idx.search(_unit_vec(0), top_k=5)
        assert len(results) == 1
        assert results[0][0] == "f1"
        # Cosine similarity of identical normalized vectors is ~1.0
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Add / search / remove operations
# ---------------------------------------------------------------------------

class TestANNIndexOperations:
    """Verify add-multiple, top_k, ordering, and remove behavior."""

    def test_add_multiple_vectors(self) -> None:
        """Adding N distinct vectors results in size == N."""
        idx = ANNIndex(dimension=DIM)
        n = 8
        for i in range(n):
            idx.add(f"f{i}", _random_vec(seed=i))
        assert idx.size == n

    def test_search_top_k_respected(self) -> None:
        """Search must return at most top_k results, even if more vectors exist."""
        idx = ANNIndex(dimension=DIM)
        for i in range(10):
            idx.add(f"f{i}", _random_vec(seed=i))
        results = idx.search(_random_vec(seed=99), top_k=3)
        assert len(results) <= 3

    def test_search_scores_descending(self) -> None:
        """Results are ordered by cosine similarity, highest first."""
        idx = ANNIndex(dimension=DIM)
        idx.add("f_x", _unit_vec(0))
        idx.add("f_y", _unit_vec(1))
        idx.add("f_z", _unit_vec(2))
        idx.add("f_w", _unit_vec(3))
        results = idx.search(_unit_vec(0), top_k=4)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_remove_reduces_size(self) -> None:
        """Removing a vector decreases size by 1."""
        idx = ANNIndex(dimension=DIM)
        idx.add("f1", _unit_vec(0))
        idx.add("f2", _unit_vec(1))
        assert idx.size == 2
        idx.remove("f1")
        assert idx.size == 1

    def test_search_after_remove_excludes_removed(self) -> None:
        """A removed vector must not appear in subsequent search results."""
        idx = ANNIndex(dimension=DIM)
        idx.add("f_keep", _unit_vec(0))
        idx.add("f_gone", _unit_vec(1))
        idx.add("f_also_keep", _unit_vec(2))
        idx.remove("f_gone")
        results = idx.search(_unit_vec(1), top_k=10)
        result_ids = {fid for fid, _ in results}
        assert "f_gone" not in result_ids
        assert "f_keep" in result_ids
        assert "f_also_keep" in result_ids


# ---------------------------------------------------------------------------
# Dimension property
# ---------------------------------------------------------------------------

class TestANNIndexDimension:
    """Verify dimension property reflects the value passed at construction."""

    def test_dimension_property(self) -> None:
        """dimension getter must return the value passed to __init__."""
        for d in (4, 128, 768):
            idx = ANNIndex(dimension=d)
            assert idx.dimension == d
