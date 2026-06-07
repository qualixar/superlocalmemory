"""TDD tests for centroid_store.py — SAFE-CACHE adversarial defense.

LLD-03 §4.3.
"""

from __future__ import annotations

import numpy as np
import pytest

from superlocalmemory.optimize.cache.centroid_store import (
    CentroidStore,
    _cosine_similarity,
)


def test_cosine_similarity_identical_vectors():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert abs(_cosine_similarity(a, a) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert abs(_cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_zero_vector_safe():
    """Zero vector → returns 0.0 (no division by zero)."""
    a = np.zeros(3, dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert _cosine_similarity(a, b) == 0.0


def test_centroid_store_empty_returns_no_adversarial():
    """No centroid established → not adversarial (fail-open skip)."""
    cs = CentroidStore()
    q = np.random.default_rng(0).standard_normal(768).astype(np.float32)
    assert cs.is_adversarial("t", q, distance_floor=0.15) is False


def test_centroid_store_few_entries_skip_defense():
    """< 5 entries → not enough data → skip defense."""
    cs = CentroidStore()
    for i in range(3):
        v = np.zeros(768, dtype=np.float32)
        v[0] = 1.0
        cs.update("t", v)
    q = np.zeros(768, dtype=np.float32)
    q[0] = 1.0
    assert cs.is_adversarial("t", q, distance_floor=0.15) is False


def test_centroid_store_detects_far_query_as_adversarial():
    """Query far from centroid (low sim) → adversarial=True."""
    cs = CentroidStore()
    # Cluster: all vectors point along axis 0.
    for i in range(10):
        v = np.zeros(768, dtype=np.float32)
        v[0] = 1.0
        cs.update("t", v)
    # Query points along axis 1 (orthogonal → low sim).
    q = np.zeros(768, dtype=np.float32)
    q[1] = 1.0
    assert cs.is_adversarial("t", q, distance_floor=0.15) is True


def test_centroid_store_accepts_close_query():
    """Query close to centroid → not adversarial."""
    cs = CentroidStore()
    for i in range(10):
        v = np.zeros(768, dtype=np.float32)
        v[0] = 1.0
        v[1] = 0.1
        cs.update("t", v)
    q = np.zeros(768, dtype=np.float32)
    q[0] = 1.0
    q[1] = 0.1
    assert cs.is_adversarial("t", q, distance_floor=0.15) is False


def test_centroid_store_welford_updates_incremental():
    """Welford running mean: after 1 update, centroid equals that vector."""
    cs = CentroidStore()
    v = np.array([0.0] * 767 + [1.0], dtype=np.float32)
    cs.update("t", v)
    centroid = cs.get_centroid("t")
    assert centroid is not None
    assert np.allclose(centroid, v, atol=1e-6)
    assert cs.count("t") == 1


def test_centroid_store_count_zero_for_unknown_tenant():
    cs = CentroidStore()
    assert cs.count("unknown") == 0
    assert cs.get_centroid("unknown") is None


def test_centroid_store_thread_safety():
    """Concurrent updates do not corrupt the centroid count."""
    import threading
    cs = CentroidStore()
    errors: list[Exception] = []

    def updater(seed: int) -> None:
        try:
            rng = np.random.default_rng(seed)
            for _ in range(50):
                v = rng.standard_normal(768).astype(np.float32)
                cs.update("t", v)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=updater, args=(i,)) for i in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert errors == []
    assert cs.count("t") == 200
