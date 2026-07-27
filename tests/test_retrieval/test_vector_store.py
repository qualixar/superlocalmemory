# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for superlocalmemory.retrieval.vector_store — VectorStore KNN.

Covers:
  - Config frozen dataclass
  - Feature flag (enabled=False → unavailable)
  - Extension loading fallback (sqlite_vec import fail)
  - Upsert + search round-trip
  - Search with profile isolation
  - Search returns sorted by similarity desc
  - Delete removes vector and metadata
  - Count (global + per-profile)
  - rebuild_from_facts migration
  - needs_binary_quantization threshold
  - Dimension mismatch rejection
  - Thread safety (no crashes)
  - Empty store returns empty results
  - Update existing vector via upsert
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from superlocalmemory.retrieval.vector_store import VectorStore, VectorStoreConfig
from superlocalmemory.storage import schema as real_schema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 4  # Small dimension for fast tests


def _vec(*vals: float) -> list[float]:
    """Create a normalized vector from values."""
    v = np.array(vals, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v.tolist()


def _make_db(tmp_path: Path) -> Path:
    """Create a DB with schema applied (for embedding_metadata table)."""
    import sqlite3

    db_path = tmp_path / "test_vec.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    real_schema.create_all_tables(conn)
    conn.commit()
    conn.close()
    return db_path


def _concurrent_process_upsert(
    db_path: str,
    fact_index: int,
    start_event,
    result_queue,
) -> None:
    """Spawn-safe worker for the cross-process row allocator contract."""
    store = VectorStore(
        Path(db_path),
        VectorStoreConfig(dimension=DIM, enabled=True),
    )
    start_event.wait(timeout=10)
    result_queue.put(
        store.upsert(
            f"process-fact-{fact_index}",
            "p1",
            _vec(1, float(fact_index + 1), 0, 0),
        )
    )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestVectorStoreConfig:
    """Test VectorStoreConfig frozen dataclass (Rule 10)."""

    def test_defaults(self) -> None:
        cfg = VectorStoreConfig()
        assert cfg.dimension == 768
        assert cfg.enabled is True
        assert cfg.binary_quantization_threshold == 100_000

    def test_frozen(self) -> None:
        cfg = VectorStoreConfig()
        with pytest.raises(AttributeError):
            cfg.dimension = 512  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Feature flag tests
# ---------------------------------------------------------------------------


class TestFeatureFlag:
    """Test that enabled=False disables VectorStore."""

    def test_disabled_by_default(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(dimension=DIM, enabled=False)
        vs = VectorStore(db_path, cfg)
        assert vs.available is False

    def test_disabled_store_returns_false(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(dimension=DIM, enabled=False)
        vs = VectorStore(db_path, cfg)
        assert vs.upsert("f1", "p1", [1.0] * DIM) is False

    def test_disabled_search_returns_empty(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(dimension=DIM, enabled=False)
        vs = VectorStore(db_path, cfg)
        assert vs.search([1.0] * DIM) == []


# ---------------------------------------------------------------------------
# Fallback tests (sqlite_vec import failure)
# ---------------------------------------------------------------------------


class TestFallback:
    """Test graceful fallback when sqlite_vec is unavailable."""

    def test_import_failure_makes_unavailable(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(dimension=DIM, enabled=True)
        with patch.dict("sys.modules", {"sqlite_vec": None}):
            vs = VectorStore(db_path, cfg)
            assert vs.available is False

    def test_unavailable_methods_are_noop(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(dimension=DIM, enabled=True)
        with patch.dict("sys.modules", {"sqlite_vec": None}):
            vs = VectorStore(db_path, cfg)
            assert vs.upsert("f1", "p1", [1.0] * DIM) is False
            assert vs.search([1.0] * DIM) == []
            assert vs.delete("f1") is False
            assert vs.count() == 0


# ---------------------------------------------------------------------------
# Core CRUD tests (requires sqlite-vec installed)
# ---------------------------------------------------------------------------


def _skip_if_no_sqlite_vec():
    """Skip test if sqlite-vec can't load at runtime (not just import)."""
    try:
        import sqlite3

        import sqlite_vec

        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.close()
        return False
    except Exception:
        return True


_needs_sqlite_vec = pytest.mark.skipif(
    _skip_if_no_sqlite_vec(),
    reason="sqlite-vec not installed",
)


@_needs_sqlite_vec
class TestUpsert:
    """Test vector insert and update."""

    def test_upsert_new_returns_true(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(dimension=DIM, enabled=True)
        vs = VectorStore(db_path, cfg)
        assert vs.available
        result = vs.upsert("f1", "p1", _vec(1, 0, 0, 0))
        assert result is True

    def test_upsert_updates_existing(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(dimension=DIM, enabled=True)
        vs = VectorStore(db_path, cfg)
        vs.upsert("f1", "p1", _vec(1, 0, 0, 0))
        vs.upsert("f1", "p1", _vec(0, 1, 0, 0))  # update
        # Should still have count=1 (updated, not duplicated)
        assert vs.count("p1") == 1

    def test_upsert_allocates_past_orphaned_metadata_rowid(
        self,
        tmp_path: Path,
    ) -> None:
        """Mature projection drift must not poison every later vector write."""
        import sqlite3

        db_path = _make_db(tmp_path)
        vs = VectorStore(
            db_path,
            VectorStoreConfig(dimension=DIM, enabled=True),
        )
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT INTO embedding_metadata "
                "(vec_rowid, fact_id, profile_id, model_name, dimension) "
                "VALUES (?, ?, ?, ?, ?)",
                (500, "orphaned-metadata", "p1", "legacy", DIM),
            )

        assert vs.upsert("f1", "p1", _vec(1, 0, 0, 0)) is True

        with sqlite3.connect(str(db_path)) as conn:
            rowid = conn.execute(
                "SELECT vec_rowid FROM embedding_metadata WHERE fact_id = ?",
                ("f1",),
            ).fetchone()[0]
        assert rowid == 501
        assert (
            vs.search(
                _vec(1, 0, 0, 0),
                top_k=1,
                profile_id="p1",
            )[0][0]
            == "f1"
        )

    def test_upsert_repairs_metadata_without_vector_payload(
        self,
        tmp_path: Path,
    ) -> None:
        import sqlite3

        db_path = _make_db(tmp_path)
        vs = VectorStore(
            db_path,
            VectorStoreConfig(dimension=DIM, enabled=True),
        )
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT INTO embedding_metadata "
                "(vec_rowid, fact_id, profile_id, model_name, dimension) "
                "VALUES (?, ?, ?, ?, ?)",
                (500, "f1", "p1", "legacy", DIM),
            )

        assert vs.upsert("f1", "p1", _vec(1, 0, 0, 0)) is True
        assert (
            vs.search(
                _vec(1, 0, 0, 0),
                top_k=1,
                profile_id="p1",
            )[0][0]
            == "f1"
        )

    def test_cross_profile_rowid_collision_is_not_a_complete_pair(
        self,
        tmp_path: Path,
    ) -> None:
        """Legacy row-id drift cannot map one profile's vector to another."""
        db_path = _make_db(tmp_path)
        vs = VectorStore(
            db_path,
            VectorStoreConfig(dimension=DIM, enabled=True),
        )
        with vs._managed_connection() as conn:
            conn.execute(
                "INSERT INTO fact_embeddings(rowid, profile_id, embedding) VALUES (?, ?, ?)",
                (500, "p1", vs._serialize_f32(_vec(1, 0, 0, 0))),
            )
            conn.execute(
                "INSERT INTO embedding_metadata "
                "(vec_rowid, fact_id, profile_id, model_name, dimension) "
                "VALUES (?, ?, ?, ?, ?)",
                (500, "f2", "p2", "legacy", DIM),
            )
            conn.commit()

        assert vs.count("p1") == 0
        assert vs.count("p2") == 0
        assert vs.indexed_fact_ids("p2") == set()
        assert (
            vs.search(
                _vec(1, 0, 0, 0),
                top_k=1,
                profile_id="p1",
            )
            == []
        )

        assert vs.upsert("f2", "p2", _vec(0, 1, 0, 0)) is True
        assert vs.count("p2") == 1
        assert vs.indexed_fact_ids("p2") == {"f2"}
        assert (
            vs.search(
                _vec(0, 1, 0, 0),
                top_k=1,
                profile_id="p2",
            )[0][0]
            == "f2"
        )
        with vs._managed_connection() as conn:
            repaired = conn.execute(
                "SELECT em.vec_rowid, fe.profile_id "
                "FROM embedding_metadata em "
                "JOIN fact_embeddings fe ON fe.rowid = em.vec_rowid "
                "WHERE em.fact_id = ?",
                ("f2",),
            ).fetchone()
        assert repaired["vec_rowid"] != 500
        assert repaired["profile_id"] == "p2"

    def test_count_and_indexed_ids_exclude_orphaned_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        import sqlite3

        db_path = _make_db(tmp_path)
        vs = VectorStore(
            db_path,
            VectorStoreConfig(dimension=DIM, enabled=True),
        )
        assert vs.upsert("paired", "p1", _vec(1, 0, 0, 0))
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT INTO embedding_metadata "
                "(vec_rowid, fact_id, profile_id, model_name, dimension) "
                "VALUES (?, ?, ?, ?, ?)",
                (500, "orphaned", "p1", "legacy", DIM),
            )

        assert vs.count("p1") == 1
        assert vs.indexed_fact_ids("p1") == {"paired"}

    def test_upsert_dimension_mismatch_returns_false(
        self,
        tmp_path: Path,
    ) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(dimension=DIM, enabled=True)
        vs = VectorStore(db_path, cfg)
        result = vs.upsert("f1", "p1", [1.0, 0.0])  # wrong dim
        assert result is False

    def test_failed_commit_rolls_back_closes_and_releases_writer(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """A fail-soft vector projection may not strand SQLite's writer lock."""
        import sqlite3

        db_path = _make_db(tmp_path)
        vs = VectorStore(
            db_path,
            VectorStoreConfig(dimension=DIM, enabled=True),
        )
        real_conn = vs._connect()

        class CommitFailure:
            rolled_back = False
            closed = False

            def __getattr__(self, name):
                return getattr(real_conn, name)

            @property
            def in_transaction(self):
                return real_conn.in_transaction

            def commit(self):
                raise sqlite3.OperationalError("forced sqlite-vec commit failure")

            def rollback(self):
                self.rolled_back = True
                return real_conn.rollback()

            def close(self):
                self.closed = True
                return real_conn.close()

        failing = CommitFailure()
        monkeypatch.setattr(vs, "_connect", lambda: failing)

        assert vs.upsert("f1", "p1", _vec(1, 0, 0, 0)) is False
        assert failing.rolled_back is True
        assert failing.closed is True

        contender = sqlite3.connect(str(db_path), timeout=0.1)
        try:
            contender.execute("BEGIN IMMEDIATE")
            contender.rollback()
        finally:
            contender.close()

    def test_upsert_rowid_allocation_is_cross_process_atomic(
        self,
        tmp_path: Path,
    ) -> None:
        """Separate MCP/agent processes cannot race MAX(rowid)+1."""
        db_path = _make_db(tmp_path)
        VectorStore(
            db_path,
            VectorStoreConfig(dimension=DIM, enabled=True),
        )
        ctx = mp.get_context("spawn")
        start_event = ctx.Event()
        result_queue = ctx.Queue()
        workers = [
            ctx.Process(
                target=_concurrent_process_upsert,
                args=(str(db_path), index, start_event, result_queue),
            )
            for index in range(8)
        ]
        for worker in workers:
            worker.start()
        start_event.set()
        results = [result_queue.get(timeout=20) for _ in workers]
        for worker in workers:
            worker.join(timeout=20)
            assert worker.exitcode == 0

        store = VectorStore(
            db_path,
            VectorStoreConfig(dimension=DIM, enabled=True),
        )
        assert results == [True] * len(workers)
        assert store.count("p1") == len(workers)


@_needs_sqlite_vec
class TestSearch:
    """Test KNN search."""

    def test_search_returns_results(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(dimension=DIM, enabled=True)
        vs = VectorStore(db_path, cfg)
        vs.upsert("f1", "p1", _vec(1, 0, 0, 0))
        vs.upsert("f2", "p1", _vec(0, 1, 0, 0))
        results = vs.search(_vec(1, 0, 0, 0), top_k=5, profile_id="p1")
        assert len(results) == 2
        # f1 should be most similar to the query
        assert results[0][0] == "f1"
        assert results[0][1] > results[1][1]

    def test_search_profile_isolation(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(dimension=DIM, enabled=True)
        vs = VectorStore(db_path, cfg)
        vs.upsert("f1", "p1", _vec(1, 0, 0, 0))
        vs.upsert("f2", "p2", _vec(0, 1, 0, 0))
        results_p1 = vs.search(
            _vec(1, 0, 0, 0),
            top_k=5,
            profile_id="p1",
        )
        assert len(results_p1) == 1
        assert results_p1[0][0] == "f1"

    def test_search_all_profiles(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(dimension=DIM, enabled=True)
        vs = VectorStore(db_path, cfg)
        vs.upsert("f1", "p1", _vec(1, 0, 0, 0))
        vs.upsert("f2", "p2", _vec(0, 1, 0, 0))
        results = vs.search(_vec(1, 0, 0, 0), top_k=5, profile_id=None)
        assert len(results) == 2

    def test_search_empty_store_returns_empty(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(dimension=DIM, enabled=True)
        vs = VectorStore(db_path, cfg)
        results = vs.search(_vec(1, 0, 0, 0), top_k=5, profile_id="p1")
        assert results == []

    def test_search_dimension_mismatch_returns_empty(
        self,
        tmp_path: Path,
    ) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(dimension=DIM, enabled=True)
        vs = VectorStore(db_path, cfg)
        vs.upsert("f1", "p1", _vec(1, 0, 0, 0))
        results = vs.search([1.0, 0.0], top_k=5)  # wrong dim
        assert results == []

    def test_search_similarity_scores_valid(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(dimension=DIM, enabled=True)
        vs = VectorStore(db_path, cfg)
        vs.upsert("f1", "p1", _vec(1, 0, 0, 0))
        results = vs.search(_vec(1, 0, 0, 0), top_k=1, profile_id="p1")
        assert len(results) == 1
        fid, score = results[0]
        assert fid == "f1"
        assert 0.0 <= score <= 1.0
        assert score > 0.9  # Near-identical vector


@_needs_sqlite_vec
class TestDelete:
    """Test vector deletion."""

    def test_delete_existing(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(dimension=DIM, enabled=True)
        vs = VectorStore(db_path, cfg)
        vs.upsert("f1", "p1", _vec(1, 0, 0, 0))
        assert vs.count() == 1
        result = vs.delete("f1")
        assert result is True
        assert vs.count() == 0

    def test_delete_nonexistent_returns_false(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(dimension=DIM, enabled=True)
        vs = VectorStore(db_path, cfg)
        result = vs.delete("nonexistent")
        assert result is False


@_needs_sqlite_vec
class TestCount:
    """Test count method."""

    def test_count_global(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(dimension=DIM, enabled=True)
        vs = VectorStore(db_path, cfg)
        vs.upsert("f1", "p1", _vec(1, 0, 0, 0))
        vs.upsert("f2", "p2", _vec(0, 1, 0, 0))
        assert vs.count() == 2

    def test_count_per_profile(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(dimension=DIM, enabled=True)
        vs = VectorStore(db_path, cfg)
        vs.upsert("f1", "p1", _vec(1, 0, 0, 0))
        vs.upsert("f2", "p1", _vec(0, 1, 0, 0))
        vs.upsert("f3", "p2", _vec(0, 0, 1, 0))
        assert vs.count("p1") == 2
        assert vs.count("p2") == 1


@_needs_sqlite_vec
class TestRebuild:
    """Test rebuild_from_facts migration."""

    def test_rebuild_migrates_facts(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(dimension=DIM, enabled=True)
        vs = VectorStore(db_path, cfg)
        facts = [
            ("f1", "p1", _vec(1, 0, 0, 0)),
            ("f2", "p1", _vec(0, 1, 0, 0)),
            ("f3", "p1", _vec(0, 0, 1, 0)),
        ]
        migrated = vs.rebuild_from_facts(facts)
        assert migrated == 3
        assert vs.count("p1") == 3


@_needs_sqlite_vec
class TestBinaryQuantization:
    """Test needs_binary_quantization threshold check."""

    def test_below_threshold_returns_false(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(
            dimension=DIM,
            enabled=True,
            binary_quantization_threshold=10,
        )
        vs = VectorStore(db_path, cfg)
        vs.upsert("f1", "p1", _vec(1, 0, 0, 0))
        assert vs.needs_binary_quantization("p1") is False

    def test_at_threshold_returns_true(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        cfg = VectorStoreConfig(
            dimension=DIM,
            enabled=True,
            binary_quantization_threshold=2,
        )
        vs = VectorStore(db_path, cfg)
        vs.upsert("f1", "p1", _vec(1, 0, 0, 0))
        vs.upsert("f2", "p1", _vec(0, 1, 0, 0))
        assert vs.needs_binary_quantization("p1") is True
