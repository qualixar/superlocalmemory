# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-03 §3.6 + §7.2 (B8)

"""Retention sweep tests for ``bandit_plays``.

Covers hard rule B8: sweep deletes only settled rows older than cutoff.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from superlocalmemory.learning.bandit import retention_sweep
from superlocalmemory.storage.migration_runner import apply_all


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    learning = tmp_path / "learning.db"
    memory = tmp_path / "memory.db"
    stats = apply_all(learning, memory)
    assert "M005_bandit_tables" in stats["applied"], stats
    return learning


def _seed(db: Path, rows: list[tuple[str, str | None]]) -> None:
    """Seed bandit_plays with (played_at, settled_at) tuples."""
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        for i, (played, settled) in enumerate(rows):
            conn.execute(
                "INSERT INTO bandit_plays "
                "(profile_id, query_id, stratum, arm_id, played_at, "
                " reward, settled_at, settlement_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "p", f"q{i}", "s", "fallback_default", played,
                    0.5 if settled else None,
                    settled,
                    "proxy_position" if settled else None,
                ),
            )
    finally:
        conn.close()


def _count(db: Path) -> int:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM bandit_plays"
        ).fetchone()[0]
    finally:
        conn.close()


def _count_unsettled(db: Path) -> int:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM bandit_plays WHERE settled_at IS NULL"
        ).fetchone()[0]
    finally:
        conn.close()


def test_bandit_plays_retention_sweep(db: Path):
    """B8: delete 500 settled+old, keep 100 settled+new + 50 unsettled."""
    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    old_settled = (now - timedelta(days=10)).isoformat(timespec="seconds")
    new_settled = (now - timedelta(days=1)).isoformat(timespec="seconds")
    played = (now - timedelta(hours=1)).isoformat(timespec="seconds")

    rows: list[tuple[str, str | None]] = []
    # (a) 500 settled rows older than 7 days → SHOULD be deleted.
    rows += [(played, old_settled)] * 500
    # (b) 100 settled rows younger than 7 days → SHOULD be retained.
    rows += [(played, new_settled)] * 100
    # (c) 50 unsettled rows → SHOULD be retained (never touched).
    rows += [(played, None)] * 50
    _seed(db, rows)

    assert _count(db) == 650
    deleted = retention_sweep(db, retention_days=7, now=now)
    assert deleted == 500
    assert _count(db) == 150
    # Unsettled rows untouched.
    assert _count_unsettled(db) == 50


def test_retention_sweep_does_not_delete_unsettled(db: Path):
    """Unsettled rows survive even if played_at is ancient."""
    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    ancient_played = (now - timedelta(days=365)).isoformat(timespec="seconds")
    _seed(db, [(ancient_played, None)] * 20)
    deleted = retention_sweep(db, retention_days=7, now=now)
    assert deleted == 0
    assert _count_unsettled(db) == 20


def test_retention_sweep_zero_deletion_on_all_recent(db: Path):
    """All rows within retention window → nothing deleted."""
    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    fresh = (now - timedelta(hours=1)).isoformat(timespec="seconds")
    _seed(db, [(fresh, fresh)] * 40)
    deleted = retention_sweep(db, retention_days=7, now=now)
    assert deleted == 0
    assert _count(db) == 40


def test_retention_sweep_chunk_boundary_drains_all(db: Path):
    """Chunk size smaller than total → loop drains remaining."""
    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    old = (now - timedelta(days=30)).isoformat(timespec="seconds")
    _seed(db, [(old, old)] * 2500)
    deleted = retention_sweep(db, retention_days=7, now=now, chunk_size=500)
    assert deleted == 2500
    assert _count(db) == 0


def test_retention_sweep_rejects_negative_days(db: Path):
    with pytest.raises(ValueError):
        retention_sweep(db, retention_days=-1)


def test_retention_sweep_on_missing_table(tmp_path: Path):
    """Missing bandit_plays → sweep returns 0, no exception."""
    db = tmp_path / "missing.db"
    # Create empty DB without bandit tables.
    sqlite3.connect(str(db)).close()
    deleted = retention_sweep(db, retention_days=7)
    assert deleted == 0
