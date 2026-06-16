# tests/test_optimize/test_centroid_rebuild.py
# Regression test for WP-C: CentroidStore.rebuild_from_db() 3-tuple unpack bug.
#
# BUG: rebuild_from_db unpacked `for _entry_id, blob in rows:` but
# CacheDB.get_all_vectors() returns 3-tuples (entry_id, vector_blob, context_fp).
# This raised ValueError inside the fail-open except, leaving centroid=None on
# every warm-start, silently disabling SAFE-CACHE adversarial defense.
#
# FIX: change unpack to `for _entry_id, blob, _ctx_fp in rows:`.

from __future__ import annotations

import logging
import struct
import unittest.mock as mock

import numpy as np
import pytest

from superlocalmemory.optimize.cache.centroid_store import CentroidStore

_TENANT = "test-tenant-rebuild"
_EMBED_DIM = 768


def _make_blob(seed: int) -> bytes:
    """Return a 768-float32 blob whose values are deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(_EMBED_DIM).astype(np.float32)
    # normalise so it's a proper unit vector
    vec /= np.linalg.norm(vec) + 1e-9
    return vec.tobytes()


def _stub_db(num_vectors: int = 10) -> mock.MagicMock:
    """Return a mock CacheDB whose get_all_vectors returns real 3-tuples."""
    db = mock.MagicMock()
    rows = [
        (f"entry-{i}", _make_blob(i), f"ctx-fp-{i}")
        for i in range(num_vectors)
    ]
    db.get_all_vectors.return_value = rows
    return db


class TestCentroidRebuildFromDb:
    """Regression suite for the 3-tuple unpack fix in rebuild_from_db."""

    def test_rebuild_sets_centroid_not_none(self):
        """After rebuild with 3-tuple rows, get_centroid must NOT return None."""
        cs = CentroidStore()
        db = _stub_db(num_vectors=10)

        cs.rebuild_from_db(db, _TENANT)

        centroid = cs.get_centroid(_TENANT)
        assert centroid is not None, (
            "centroid is None — rebuild_from_db failed to unpack 3-tuples "
            "(ValueError swallowed by fail-open except)"
        )

    def test_rebuild_count_matches_vector_count(self):
        """count() must equal the number of valid vectors fed to rebuild."""
        num = 7
        cs = CentroidStore()
        db = _stub_db(num_vectors=num)

        cs.rebuild_from_db(db, _TENANT)

        assert cs.count(_TENANT) == num, (
            f"Expected count={num}, got {cs.count(_TENANT)}"
        )

    def test_rebuild_no_warning_logged(self, caplog):
        """rebuild_from_db must NOT emit a WARNING (which indicates the fail-open
        path was triggered by the ValueError)."""
        cs = CentroidStore()
        db = _stub_db(num_vectors=5)

        with caplog.at_level(logging.WARNING, logger="superlocalmemory.optimize.cache.centroid_store"):
            cs.rebuild_from_db(db, _TENANT)

        warning_records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
        ]
        assert warning_records == [], (
            f"Unexpected warnings logged (fail-open triggered?): "
            f"{[r.message for r in warning_records]}"
        )

    def test_rebuild_centroid_shape_correct(self):
        """Rebuilt centroid must be a float32 array of shape (768,)."""
        cs = CentroidStore()
        db = _stub_db(num_vectors=6)

        cs.rebuild_from_db(db, _TENANT)

        centroid = cs.get_centroid(_TENANT)
        assert centroid is not None
        assert centroid.dtype == np.float32
        assert centroid.shape == (_EMBED_DIM,)

    def test_rebuild_adversarial_defense_active_after_rebuild(self):
        """is_adversarial must work (return True for outlier) after a warm rebuild.

        This confirms the defence is NOT silently disabled when centroid is valid
        and count >= 5.
        """
        # Build a tight cluster: all vectors point along axis 0.
        db = mock.MagicMock()
        cluster_vec = np.zeros(_EMBED_DIM, dtype=np.float32)
        cluster_vec[0] = 1.0
        rows = [
            (f"entry-{i}", cluster_vec.tobytes(), f"ctx-{i}")
            for i in range(10)
        ]
        db.get_all_vectors.return_value = rows

        cs = CentroidStore()
        cs.rebuild_from_db(db, _TENANT)

        # Query along orthogonal axis → should be flagged adversarial.
        outlier = np.zeros(_EMBED_DIM, dtype=np.float32)
        outlier[1] = 1.0

        assert cs.is_adversarial(_TENANT, outlier, distance_floor=0.15) is True, (
            "Adversarial defence is inactive — centroid likely None after rebuild"
        )

    def test_rebuild_empty_tenant_clears_centroid(self):
        """If DB returns no rows for a tenant, centroid is cleared (not left stale)."""
        cs = CentroidStore()
        # pre-populate via update so there IS an existing centroid
        cs.update(_TENANT, np.ones(_EMBED_DIM, dtype=np.float32))
        assert cs.get_centroid(_TENANT) is not None

        db = mock.MagicMock()
        db.get_all_vectors.return_value = []
        cs.rebuild_from_db(db, _TENANT)

        assert cs.get_centroid(_TENANT) is None

    def test_rebuild_get_all_vectors_called_with_correct_tenant(self):
        """rebuild_from_db must query the DB with the supplied tenant_id."""
        cs = CentroidStore()
        db = _stub_db(num_vectors=3)

        cs.rebuild_from_db(db, _TENANT)

        db.get_all_vectors.assert_called_once_with(tenant_id=_TENANT)
