# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for LanceDBVectorBackend — Sprint 3."""

from __future__ import annotations

import importlib.util
import shutil
import sqlite3

import numpy as np
import pytest
import sqlite_vec

from superlocalmemory.storage.sqlite_vectors import CanonicalVectorError

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


def _load_sqlite_vec_or_skip(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec where CPython exposes extension loading."""
    enable = getattr(conn, "enable_load_extension", None)
    if not callable(enable):
        pytest.skip("this CPython build does not support SQLite extensions")
    try:
        enable(True)
        sqlite_vec.load(conn)
    except (AttributeError, sqlite3.Error) as exc:
        pytest.skip(f"sqlite-vec extension loading unavailable: {exc}")
    finally:
        try:
            enable(False)
        except (AttributeError, sqlite3.Error):
            pass


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

    def test_replayed_vector_write_replaces_instead_of_duplicates(self, backend):
        backend.add_vectors(["a1"], [_make_vec(0.1)], ["active"])
        backend.add_vectors(["a1"], [_make_vec(0.2)], ["warm"])
        assert backend.health_check()["vectors"] == 1
        result = backend.similarity_search(_make_vec(0.2), top_k=1, tier_filter=["warm"])
        assert result and result[0][0] == "a1"

    def test_update_tier(self, populated):
        populated.update_tier("f0", "cold")
        results = populated.similarity_search(
            _make_vec(0.5), top_k=10, tier_filter=["active", "warm"]
        )
        fact_ids = {r[0] for r in results}
        assert "f0" not in fact_ids  # Now cold, excluded

    def test_profile_scope_and_delete_are_projection_safe(self, backend):
        backend.add_vectors(["same-name"], [_make_vec(0.1)], ["active"], "alpha")
        backend.add_vectors(["other"], [_make_vec(0.1)], ["active"], "beta")
        assert [fact_id for fact_id, _ in backend.similarity_search(
            _make_vec(0.1), profile_id="alpha"
        )] == ["same-name"]
        backend.remove_vector("same-name")
        assert backend.similarity_search(_make_vec(0.1), profile_id="alpha") == []

    def test_bulk_import_reads_supported_sqlite_vec_virtual_table(self, backend):
        conn = sqlite3.connect(":memory:")
        _load_sqlite_vec_or_skip(conn)
        conn.executescript(
            """
            CREATE TABLE atomic_facts (
                fact_id TEXT PRIMARY KEY,
                lifecycle TEXT NOT NULL,
                profile_id TEXT NOT NULL
            );
            CREATE TABLE embedding_metadata (
                vec_rowid INTEGER PRIMARY KEY,
                fact_id TEXT NOT NULL UNIQUE,
                profile_id TEXT NOT NULL,
                model_name TEXT NOT NULL DEFAULT '',
                dimension INTEGER NOT NULL DEFAULT 768
            );
            CREATE VIRTUAL TABLE fact_embeddings USING vec0(
                profile_id TEXT PARTITION KEY,
                embedding float[768] distance_metric=cosine
            );
            INSERT INTO atomic_facts VALUES
                ('default-a', 'active', 'default'),
                ('default-b', 'warm', 'default'),
                ('foreign', 'active', 'other');
            """
        )
        for fact_id, profile_id, seed in (
            ("default-a", "default", 0.1),
            ("default-b", "default", 0.2),
            ("foreign", "other", 0.3),
        ):
            cursor = conn.execute(
                "INSERT INTO fact_embeddings(profile_id, embedding) VALUES (?, ?)",
                (profile_id, np.full(768, seed, dtype=np.float32).tobytes()),
            )
            conn.execute(
                "INSERT INTO embedding_metadata "
                "(vec_rowid, fact_id, profile_id) VALUES (?, ?, ?)",
                (cursor.lastrowid, fact_id, profile_id),
            )
        conn.commit()

        assert backend.bulk_import_from_sqlite(conn, "default") == 2
        assert backend.health_check()["vectors"] == 2
        imported_ids = {
            fact_id
            for fact_id, _ in backend.similarity_search(
                _make_vec(0.1), top_k=10, profile_id="default"
            )
        }
        assert imported_ids == {"default-a", "default-b"}

    def test_bulk_import_fails_closed_when_metadata_vector_is_missing(self, backend):
        conn = sqlite3.connect(":memory:")
        _load_sqlite_vec_or_skip(conn)
        conn.executescript(
            """
            CREATE TABLE atomic_facts (
                fact_id TEXT PRIMARY KEY,
                lifecycle TEXT NOT NULL,
                profile_id TEXT NOT NULL
            );
            CREATE TABLE embedding_metadata (
                vec_rowid INTEGER PRIMARY KEY,
                fact_id TEXT NOT NULL UNIQUE,
                profile_id TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE fact_embeddings USING vec0(
                profile_id TEXT PARTITION KEY,
                embedding float[768] distance_metric=cosine
            );
            INSERT INTO atomic_facts VALUES ('missing', 'active', 'default');
            INSERT INTO embedding_metadata VALUES (999, 'missing', 'default');
            """
        )

        with pytest.raises(CanonicalVectorError, match="canonical vector"):
            backend.bulk_import_from_sqlite(conn, "default")


class TestHealth:
    def test_health_counts_vectors(self, populated):
        health = populated.health_check()
        assert health["vectors"] == 10
        assert health["status"] == "active"
