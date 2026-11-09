# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.retrieval.ann_index — Approximate Nearest Neighbor.

Covers:
  - add / remove / clear
  - search: top-k, cosine ordering, zero vector, dimension mismatch
  - rebuild from bulk data
  - update existing vector
  - swap-and-pop removal correctness
  - thread-safety properties (size)
  - empty index search
"""

from __future__ import annotations

import numpy as np
import pytest

from superlocalmemory.retrieval.ann_index import ANNIndex

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(dim: int, axis: int = 0) -> list[float]:
    """One-hot unit vector along given axis."""
    v = [0.0] * dim
    v[axis] = 1.0
    return v


def _random_vec(dim: int, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v.tolist()


# ---------------------------------------------------------------------------
# Basic add / size / clear
# ---------------------------------------------------------------------------

class TestANNBasicOps:
    def test_empty_index(self) -> None:
        idx = ANNIndex(dimension=4)
        assert idx.size == 0
        assert idx.dimension == 4

    def test_add_single(self) -> None:
        idx = ANNIndex(dimension=4)
        idx.add("f1", [1.0, 0.0, 0.0, 0.0])
        assert idx.size == 1

    def test_add_multiple(self) -> None:
        idx = ANNIndex(dimension=4)
        idx.add("f1", _unit_vec(4, 0))
        idx.add("f2", _unit_vec(4, 1))
        idx.add("f3", _unit_vec(4, 2))
        assert idx.size == 3

    def test_add_duplicate_updates(self) -> None:
        idx = ANNIndex(dimension=4)
        idx.add("f1", _unit_vec(4, 0))
        idx.add("f1", _unit_vec(4, 1))  # Update f1 with new vector
        assert idx.size == 1
        # Search should match axis 1 now
        results = idx.search(_unit_vec(4, 1), top_k=1)
        assert results[0][0] == "f1"
        assert results[0][1] > 0.9

    def test_add_wrong_dimension_skipped(self) -> None:
        idx = ANNIndex(dimension=4)
        idx.add("f_bad", [1.0, 2.0])  # Wrong dim
        assert idx.size == 0

    def test_clear(self) -> None:
        idx = ANNIndex(dimension=4)
        idx.add("f1", _unit_vec(4, 0))
        idx.add("f2", _unit_vec(4, 1))
        idx.clear()
        assert idx.size == 0


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------

class TestANNRemove:
    def test_remove_single(self) -> None:
        idx = ANNIndex(dimension=4)
        idx.add("f1", _unit_vec(4, 0))
        idx.remove("f1")
        assert idx.size == 0

    def test_remove_nonexistent_noop(self) -> None:
        idx = ANNIndex(dimension=4)
        idx.add("f1", _unit_vec(4, 0))
        idx.remove("f_not_there")
        assert idx.size == 1

    def test_remove_middle_swap_and_pop(self) -> None:
        idx = ANNIndex(dimension=4)
        idx.add("f1", _unit_vec(4, 0))
        idx.add("f2", _unit_vec(4, 1))
        idx.add("f3", _unit_vec(4, 2))
        idx.remove("f1")  # Should swap f3 into f1's slot
        assert idx.size == 2
        # Both f2 and f3 should still be searchable
        results = idx.search(_unit_vec(4, 1), top_k=5)
        ids = [r[0] for r in results]
        assert "f2" in ids
        assert "f3" in ids
        assert "f1" not in ids

    def test_remove_last_element(self) -> None:
        idx = ANNIndex(dimension=4)
        idx.add("f1", _unit_vec(4, 0))
        idx.add("f2", _unit_vec(4, 1))
        idx.remove("f2")  # Remove last — no swap needed
        assert idx.size == 1
        results = idx.search(_unit_vec(4, 0), top_k=5)
        assert results[0][0] == "f1"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestANNSearch:
    def test_search_empty_index(self) -> None:
        idx = ANNIndex(dimension=4)
        results = idx.search(_unit_vec(4, 0), top_k=5)
        assert results == []

    def test_search_returns_correct_order(self) -> None:
        idx = ANNIndex(dimension=4)
        idx.add("f_x", _unit_vec(4, 0))
        idx.add("f_y", _unit_vec(4, 1))
        idx.add("f_z", _unit_vec(4, 2))
        # Search along axis 0 -> f_x should be first
        results = idx.search(_unit_vec(4, 0), top_k=3)
        assert results[0][0] == "f_x"
        assert results[0][1] > 0.9

    def test_search_top_k_limit(self) -> None:
        idx = ANNIndex(dimension=4)
        for i in range(10):
            idx.add(f"f{i}", _random_vec(4, seed=i))
        results = idx.search(_random_vec(4, seed=42), top_k=3)
        assert len(results) <= 3

    def test_search_wrong_dimension_returns_empty(self) -> None:
        idx = ANNIndex(dimension=4)
        idx.add("f1", _unit_vec(4, 0))
        results = idx.search([1.0, 2.0], top_k=5)  # Wrong dim
        assert results == []

    def test_search_zero_vector_returns_empty(self) -> None:
        idx = ANNIndex(dimension=4)
        idx.add("f1", _unit_vec(4, 0))
        results = idx.search([0.0, 0.0, 0.0, 0.0], top_k=5)
        assert results == []

    def test_search_scores_are_cosine(self) -> None:
        idx = ANNIndex(dimension=4)
        idx.add("f1", [1.0, 1.0, 0.0, 0.0])
        # Query identical direction
        results = idx.search([1.0, 1.0, 0.0, 0.0], top_k=1)
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_search_opposite_direction(self) -> None:
        idx = ANNIndex(dimension=4)
        idx.add("f1", [1.0, 0.0, 0.0, 0.0])
        results = idx.search([-1.0, 0.0, 0.0, 0.0], top_k=1)
        assert results[0][1] == pytest.approx(-1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------

class TestANNRebuild:
    def test_rebuild_populates_index(self) -> None:
        idx = ANNIndex(dimension=4)
        ids = ["a", "b", "c"]
        embs = [_unit_vec(4, 0), _unit_vec(4, 1), _unit_vec(4, 2)]
        count = idx.rebuild(ids, embs)
        assert count == 3
        assert idx.size == 3

    def test_rebuild_replaces_existing(self) -> None:
        idx = ANNIndex(dimension=4)
        idx.add("old", _unit_vec(4, 0))
        assert idx.size == 1
        idx.rebuild(["new1", "new2"], [_unit_vec(4, 1), _unit_vec(4, 2)])
        assert idx.size == 2
        results = idx.search(_unit_vec(4, 0), top_k=5)
        ids = [r[0] for r in results]
        assert "old" not in ids

    def test_rebuild_mismatched_lengths_returns_zero(self) -> None:
        idx = ANNIndex(dimension=4)
        count = idx.rebuild(["a", "b"], [_unit_vec(4, 0)])
        assert count == 0
        assert idx.size == 0

    def test_rebuild_empty(self) -> None:
        idx = ANNIndex(dimension=4)
        idx.add("pre", _unit_vec(4, 0))
        count = idx.rebuild([], [])
        assert count == 0
        assert idx.size == 0
