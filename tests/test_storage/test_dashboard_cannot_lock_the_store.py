# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""The dashboard must never be able to block recall or remember.

WHY THIS EXISTS
---------------
The dashboard reads memory.db directly, and it reads a lot: panes poll, users
refresh, and several cards each issue their own query. Recall and remember run
from the CLI, the desktop app and MCP against the SAME file. If a dashboard read
could take a lock, a refreshing browser tab would slow down every agent on the
machine — and the symptom would appear as "SLM got slow", nowhere near the
dashboard.

Four independent properties make that impossible, and each is asserted here
rather than trusted:

1. **``mode=ro``** — the connection is opened read-only at the URI level, so
   SQLite physically refuses to grant it a write lock. Not a convention: an
   attempted write raises.
2. **``PRAGMA query_only=ON``** — a second, independent barrier, so a future
   refactor that loses the URI flag still cannot write.
3. **WAL journal mode** — readers never block the writer and the writer never
   blocks readers. Without WAL, properties 1 and 2 would still leave a reader
   holding a shared lock that stalls a commit.
4. **A capped ``busy_timeout``** — a dashboard read waits milliseconds, not
   seconds, so even a pathological case degrades the dashboard rather than the
   agent.

These are cheap assertions guarding an expensive, hard-to-diagnose failure.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from superlocalmemory.storage.read_connection import ReadConnectionFactory


@pytest.fixture()
def db(tmp_path):
    """A WAL database with one table, mirroring the real store's setup."""
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE facts (id INTEGER PRIMARY KEY, content TEXT)")
    conn.executemany(
        "INSERT INTO facts (content) VALUES (?)", [(f"fact {i}",) for i in range(200)]
    )
    conn.commit()
    conn.close()
    return path


class TestDashboardConnectionIsPhysicallyReadOnly:
    def test_insert_is_refused(self, db):
        with ReadConnectionFactory(db).snapshot() as conn:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("INSERT INTO facts (content) VALUES ('written')")

    def test_update_is_refused(self, db):
        with ReadConnectionFactory(db).snapshot() as conn:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("UPDATE facts SET content='x' WHERE id=1")

    def test_delete_is_refused(self, db):
        with ReadConnectionFactory(db).snapshot() as conn:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("DELETE FROM facts WHERE id=1")

    def test_query_only_pragma_is_set(self, db):
        """The second barrier. Losing the URI flag alone must not allow writes."""
        with ReadConnectionFactory(db).snapshot() as conn:
            assert conn.execute("PRAGMA query_only").fetchone()[0] == 1

    def test_reads_still_work(self, db):
        with ReadConnectionFactory(db).snapshot() as conn:
            assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 200

    def test_busy_timeout_is_capped(self, db):
        """A dashboard read must wait milliseconds, never seconds."""
        with ReadConnectionFactory(db, timeout_ms=250).snapshot() as conn:
            timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert 0 < timeout <= 1000, (
            f"busy_timeout={timeout} ms — a dashboard read that waits this long "
            f"is holding up the request thread, not just itself"
        )


class TestWritesProceedDuringDashboardReads:
    """The property that actually matters to the owner: an agent writing while
    the dashboard reads must not be delayed."""

    def test_writer_is_not_blocked_by_open_dashboard_reads(self, db):
        errors: list[Exception] = []
        write_ms: list[float] = []
        stop = threading.Event()

        def reader():
            # Long-lived read connections, exactly like a dashboard pane holding
            # one across several queries.
            while not stop.is_set():
                try:
                    with ReadConnectionFactory(db).snapshot() as conn:
                        for _ in range(20):
                            conn.execute("SELECT COUNT(*) FROM facts").fetchone()
                            if stop.is_set():
                                break
                except Exception as exc:  # pragma: no cover — would be the bug
                    errors.append(exc)


        threads = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
        for t in threads:
            t.start()
        time.sleep(0.05)

        writer = sqlite3.connect(db, timeout=5)
        writer.execute("PRAGMA busy_timeout=5000")
        try:
            for i in range(40):
                t0 = time.perf_counter()
                writer.execute("INSERT INTO facts (content) VALUES (?)", (f"w{i}",))
                writer.commit()
                write_ms.append((time.perf_counter() - t0) * 1000)
        finally:
            writer.close()
            stop.set()
            for t in threads:
                t.join(timeout=2)

        assert not errors, f"dashboard reads errored during writes: {errors[:2]}"
        worst = max(write_ms)
        assert worst < 1000, (
            f"a write waited {worst:.0f} ms while the dashboard was reading — "
            f"dashboard traffic is blocking the agent write path"
        )

    def test_reader_sees_a_consistent_snapshot(self, db):
        """WAL gives the reader a stable view; it must not error or see torn
        state while the writer commits underneath it."""
        stop = threading.Event()

        def writer():
            conn = sqlite3.connect(db, timeout=5)
            try:
                i = 0
                while not stop.is_set():
                    conn.execute("INSERT INTO facts (content) VALUES (?)", (f"x{i}",))
                    conn.commit()
                    i += 1
            finally:
                conn.close()

        t = threading.Thread(target=writer, daemon=True)
        t.start()
        try:
            with ReadConnectionFactory(db).snapshot() as conn:
                for _ in range(50):
                    n = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
                    assert n >= 200
        finally:
            stop.set()
            t.join(timeout=2)


class TestRealStoreIsInWalMode:
    """Properties 1-2 are not enough without WAL. If the shipped store ever
    reverts to the default rollback journal, dashboard reads WOULD stall
    commits — so assert the mode the code depends on."""

    def test_new_databases_are_created_in_wal(self, tmp_path):
        from superlocalmemory.storage import schema as _schema
        from superlocalmemory.storage.database import DatabaseManager

        db = DatabaseManager(tmp_path / "memory.db")
        db.initialize(_schema)
        conn = sqlite3.connect(tmp_path / "memory.db")
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            conn.close()
        assert str(mode).lower() == "wal", (
            f"store created in {mode!r} journal mode; dashboard reads would "
            f"block agent writes"
        )
