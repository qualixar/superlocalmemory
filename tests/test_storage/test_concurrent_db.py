# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for concurrent database operations.

Verifies that multiple processes/threads can safely read and write
to the same SQLite database without deadlocks or data corruption.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    AtomicFact,
    FactType,
    MemoryRecord,
)


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    """Fresh database for each test."""
    mgr = DatabaseManager(tmp_path / "test.db")
    mgr.initialize(schema)
    return mgr


def _make_record(profile_id: str = "default", n: int = 0) -> MemoryRecord:
    return MemoryRecord(profile_id=profile_id, content=f"Test memory #{n}")


def _store_fact_with_parent(db: DatabaseManager, profile_id: str = "default", n: int = 0) -> AtomicFact:
    """Store a fact with its required parent memory record."""
    rec = _make_record(profile_id=profile_id, n=n)
    db.store_memory(rec)
    fact = AtomicFact(
        profile_id=profile_id,
        memory_id=rec.memory_id,
        content=f"Test fact #{n}",
        fact_type=FactType.SEMANTIC,
    )
    db.store_fact(fact)
    return fact


# ---------------------------------------------------------------------------
# WAL mode verification
# ---------------------------------------------------------------------------

class TestWALMode:
    """Verify WAL mode is enabled."""

    def test_wal_enabled(self, db: DatabaseManager) -> None:
        rows = db.execute("PRAGMA journal_mode")
        mode = dict(rows[0])["journal_mode"]
        assert mode == "wal"

    def test_busy_timeout_set(self, db: DatabaseManager) -> None:
        rows = db.execute("PRAGMA busy_timeout")
        # PRAGMA returns a single unnamed column
        timeout = list(dict(rows[0]).values())[0]
        assert timeout >= 5000


# ---------------------------------------------------------------------------
# Concurrent writes from threads
# ---------------------------------------------------------------------------

class TestConcurrentThreadWrites:
    """Multiple threads writing simultaneously."""

    def test_10_threads_store_memories(self, db: DatabaseManager) -> None:
        """10 threads each storing a memory — all should succeed."""
        errors: list[Exception] = []

        def _store(n: int) -> None:
            try:
                db.store_memory(_make_record(n=n))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_store, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread errors: {errors}"
        rows = db.execute(
            "SELECT COUNT(*) AS c FROM memories WHERE profile_id = 'default'"
        )
        assert int(rows[0]["c"]) == 10

    def test_10_threads_store_facts(self, db: DatabaseManager) -> None:
        """10 threads each storing a fact — all should succeed."""
        errors: list[Exception] = []

        def _store(n: int) -> None:
            try:
                _store_fact_with_parent(db, n=n)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_store, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread errors: {errors}"
        count = db.get_fact_count("default")
        assert count == 10

    def test_concurrent_read_during_write(self, db: DatabaseManager) -> None:
        """Reads should not block during writes (WAL mode)."""
        # Pre-populate
        for i in range(5):
            _store_fact_with_parent(db, n=i)

        read_results: list[int] = []
        write_done = threading.Event()
        read_done = threading.Event()

        def _writer() -> None:
            for i in range(5, 15):
                _store_fact_with_parent(db, n=i)
                time.sleep(0.01)
            write_done.set()

        def _reader() -> None:
            while not write_done.is_set():
                facts = db.get_all_facts("default")
                read_results.append(len(facts))
                time.sleep(0.005)
            read_done.set()

        w = threading.Thread(target=_writer)
        r = threading.Thread(target=_reader)
        r.start()
        w.start()
        w.join(timeout=30)
        r.join(timeout=5)

        # Reader should have gotten results while writer was active
        assert len(read_results) > 0
        # Final count should be 15
        assert db.get_fact_count("default") == 15


# ---------------------------------------------------------------------------
# Concurrent writes from ThreadPoolExecutor (simulates multi-process)
# ---------------------------------------------------------------------------

class TestConcurrentPoolWrites:
    """ThreadPoolExecutor simulating multi-client access."""

    def test_20_concurrent_stores(self, db: DatabaseManager) -> None:
        """20 concurrent store operations via thread pool."""
        def _store_fact(n: int) -> str:
            fact = _store_fact_with_parent(db, n=n)
            return fact.fact_id

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_store_fact, i) for i in range(20)]
            results = [f.result(timeout=30) for f in as_completed(futures)]

        assert len(results) == 20
        assert db.get_fact_count("default") == 20

    def test_mixed_reads_and_writes(self, db: DatabaseManager) -> None:
        """Mix of reads and writes — no deadlocks."""
        # Pre-populate
        for i in range(10):
            _store_fact_with_parent(db, n=i)

        results: dict[str, list] = {"reads": [], "writes": []}

        def _read(n: int) -> str:
            facts = db.get_all_facts("default")
            results["reads"].append(len(facts))
            return f"read-{n}"

        def _write(n: int) -> str:
            _store_fact_with_parent(db, n=100 + n)
            results["writes"].append(n)
            return f"write-{n}"

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = []
            for i in range(10):
                futures.append(pool.submit(_read, i))
                futures.append(pool.submit(_write, i))
            all_results = [f.result(timeout=30) for f in as_completed(futures)]

        assert len(all_results) == 20
        assert db.get_fact_count("default") == 20  # 10 pre-populated + 10 new


# ---------------------------------------------------------------------------
# Multi-process simulation (separate DatabaseManager instances)
# ---------------------------------------------------------------------------

class TestMultiProcessSimulation:
    """Simulate multiple processes by creating separate DatabaseManager instances."""

    def test_two_managers_concurrent_writes(self, tmp_path: Path) -> None:
        """Two DatabaseManager instances (simulating two processes) writing."""
        db_path = tmp_path / "shared.db"
        db1 = DatabaseManager(db_path)
        db1.initialize(schema)
        db2 = DatabaseManager(db_path)

        errors: list[Exception] = []

        def _write_from_db(db: DatabaseManager, start: int) -> None:
            try:
                for i in range(5):
                    _store_fact_with_parent(db, n=start + i)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=_write_from_db, args=(db1, 0))
        t2 = threading.Thread(target=_write_from_db, args=(db2, 100))
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert not errors, f"Errors: {errors}"
        assert db1.get_fact_count("default") == 10

    def test_three_managers_mixed_operations(self, tmp_path: Path) -> None:
        """Three managers: one writes, two read — no deadlocks."""
        db_path = tmp_path / "shared.db"
        db1 = DatabaseManager(db_path)
        db1.initialize(schema)
        db2 = DatabaseManager(db_path)
        db3 = DatabaseManager(db_path)

        # Pre-populate via db1
        for i in range(5):
            _store_fact_with_parent(db1, n=i)

        errors: list[Exception] = []
        read_counts: list[int] = []

        def _writer() -> None:
            try:
                for i in range(5, 15):
                    _store_fact_with_parent(db1, n=i)
                    time.sleep(0.01)
            except Exception as exc:
                errors.append(exc)

        def _reader(db: DatabaseManager) -> None:
            try:
                for _ in range(10):
                    facts = db.get_all_facts("default")
                    read_counts.append(len(facts))
                    time.sleep(0.01)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_writer),
            threading.Thread(target=_reader, args=(db2,)),
            threading.Thread(target=_reader, args=(db3,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors: {errors}"
        assert db1.get_fact_count("default") == 15
        assert len(read_counts) > 0
