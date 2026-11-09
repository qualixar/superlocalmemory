# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com
"""Tests for write-lock + busy_timeout hardening (3.8.4-dev).

Covers:
  1. Two threads calling memory_write() concurrently on the same temp DB
     serialise without SQLITE_BUSY.
  2. A route-write smoke test: ensure_profile_in_db() and
     delete_profile_from_db() succeed against a real temp DB and preserve
     correct state.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_db(path: Path) -> None:
    """Provision minimal schema: profiles + tool_events tables."""
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS profiles (
            profile_id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            last_used TEXT
        );
        CREATE TABLE IF NOT EXISTS tool_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            profile_id TEXT,
            project_path TEXT,
            tool_name TEXT,
            event_type TEXT,
            input_summary TEXT,
            output_summary TEXT,
            duration_ms INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            created_at TEXT
        );
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Test 1: concurrent memory_write() calls serialise without SQLITE_BUSY
# ---------------------------------------------------------------------------

def test_concurrent_memory_write_no_busy(tmp_path):
    """Two threads both calling memory_write() on the same DB never raise
    SQLITE_BUSY.  Each inserts a row; both rows must be present after join.
    """
    from superlocalmemory.storage.memory_write import memory_write

    db = tmp_path / "test_concurrent.db"
    _create_test_db(db)

    errors: list[Exception] = []
    insert_counts: list[int] = []

    def writer(profile_id: str) -> None:
        try:
            with memory_write(db) as conn:
                # Small sleep INSIDE the lock to stress the serialisation.
                time.sleep(0.02)
                conn.execute(
                    "INSERT INTO profiles (profile_id, name, description) VALUES (?, ?, ?)",
                    (profile_id, profile_id, f"test {profile_id}"),
                )
            insert_counts.append(1)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=writer, args=("profile_alpha",))
    t2 = threading.Thread(target=writer, args=("profile_beta",))
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)

    assert not errors, f"Unexpected errors in writer threads: {errors}"
    assert sum(insert_counts) == 2, f"Expected 2 successful inserts, got {insert_counts}"

    # Verify both rows landed.
    conn = sqlite3.connect(str(db))
    rows = conn.execute("SELECT profile_id FROM profiles ORDER BY profile_id").fetchall()
    conn.close()
    profile_ids = {r[0] for r in rows}
    assert "profile_alpha" in profile_ids
    assert "profile_beta" in profile_ids


# ---------------------------------------------------------------------------
# Test 2: ensure_profile_in_db / delete_profile_from_db smoke test
# ---------------------------------------------------------------------------

def test_ensure_and_delete_profile_in_db(tmp_path, monkeypatch):
    """ensure_profile_in_db inserts idempotently; delete_profile_from_db
    removes the row.  Verifies the two helpers produce correct DB state.
    """
    db = tmp_path / "test_helpers.db"
    _create_test_db(db)

    # Patch DB_PATH used by helpers to point at our temp DB.
    import superlocalmemory.server.routes.helpers as helpers_mod

    # We need DB_PATH.exists() to return True and str(DB_PATH) to be our path.
    class _FakeDBPath:
        def exists(self):
            return True
        def __str__(self):
            return str(db)
        def __fspath__(self):
            return str(db)

    monkeypatch.setattr(helpers_mod, "DB_PATH", _FakeDBPath())

    # Also patch memory_write's db_path resolution (it receives the DB_PATH object).
    # memory_write calls str(db_path) and Path(db_path) — both work via __str__/__fspath__.

    from superlocalmemory.server.routes.helpers import (
        delete_profile_from_db,
        ensure_profile_in_db,
    )

    # Insert (idempotent x2).
    ensure_profile_in_db("smoke_test_profile", "Smoke test profile")
    ensure_profile_in_db("smoke_test_profile", "Smoke test profile")  # no-op

    conn = sqlite3.connect(str(db))
    rows = conn.execute(
        "SELECT profile_id FROM profiles WHERE profile_id = ?", ("smoke_test_profile",)
    ).fetchall()
    conn.close()
    assert len(rows) == 1, f"Expected exactly 1 row, got {len(rows)}"

    # Delete.
    delete_profile_from_db("smoke_test_profile")

    conn = sqlite3.connect(str(db))
    rows = conn.execute(
        "SELECT profile_id FROM profiles WHERE profile_id = ?", ("smoke_test_profile",)
    ).fetchall()
    conn.close()
    assert len(rows) == 0, "Row should be gone after delete"


# ---------------------------------------------------------------------------
# Test 3: memory_write rollback on exception leaves DB clean
# ---------------------------------------------------------------------------

def test_memory_write_rollback_on_exception(tmp_path):
    """If an exception is raised inside the memory_write block the transaction
    is rolled back and no partial write remains.
    """
    from superlocalmemory.storage.memory_write import memory_write

    db = tmp_path / "test_rollback.db"
    _create_test_db(db)

    class _Boom(RuntimeError):
        pass

    with pytest.raises(_Boom):
        with memory_write(db) as conn:
            conn.execute(
                "INSERT INTO profiles (profile_id, name, description) VALUES (?, ?, ?)",
                ("should_not_persist", "x", "x"),
            )
            raise _Boom("intentional error")

    conn = sqlite3.connect(str(db))
    rows = conn.execute(
        "SELECT profile_id FROM profiles WHERE profile_id = ?", ("should_not_persist",)
    ).fetchall()
    conn.close()
    assert len(rows) == 0, "Rollback should have removed the partial insert"
