# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for migration v3.4.25 -> v3.4.26."""

from __future__ import annotations

from pathlib import Path


def _imports():
    from superlocalmemory.migrations import v3_4_25_to_v3_4_26 as mig
    return mig


def test_fresh_migration_creates_queue_and_marker(tmp_path: Path) -> None:
    mig = _imports()
    result = mig.migrate(tmp_path)
    assert any("recall_queue.db" in c for c in result["created"])
    assert any(".slm-v3.4.26-ready" in c for c in result["created"])
    assert mig.is_ready(tmp_path)


def test_migration_is_idempotent(tmp_path: Path) -> None:
    mig = _imports()
    mig.migrate(tmp_path)
    r2 = mig.migrate(tmp_path)
    # Second run creates nothing new
    assert r2["created"] == []
    assert mig.is_ready(tmp_path)


def test_migration_does_not_touch_memory_db(tmp_path: Path) -> None:
    # Backward-compat guard: existing memory.db must be left alone.
    mig = _imports()
    memory_db = tmp_path / "memory.db"
    memory_db.write_bytes(b"existing_user_data")
    original = memory_db.read_bytes()
    mig.migrate(tmp_path)
    assert memory_db.read_bytes() == original


def test_migration_preserves_existing_queue_db(tmp_path: Path) -> None:
    mig = _imports()
    mig.migrate(tmp_path)
    # Insert something into the queue
    from superlocalmemory.core.recall_queue import RecallQueue
    q = RecallQueue(db_path=tmp_path / "recall_queue.db")
    rid = q.enqueue(
        query="preserved", limit_n=10, mode="B",
        agent_id="a", session_id="s",
    )
    q.close()
    # Re-run migration — the queue row must survive
    mig.migrate(tmp_path)
    q2 = RecallQueue(db_path=tmp_path / "recall_queue.db")
    assert q2._get_row(rid) is not None
    q2.close()
