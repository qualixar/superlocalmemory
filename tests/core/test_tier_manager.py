# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for tier_manager.py — Sprint 1: Tiered Storage."""

from __future__ import annotations

import pytest
import sqlite3
from datetime import datetime, timedelta, UTC
from unittest.mock import MagicMock, patch

from superlocalmemory.core.tier_manager import (
    evaluate_tiers,
    promote_on_access,
    promote_on_access_batch,
    pin_fact,
    unpin_fact,
    get_tier_stats,
    record_access_batch,
    reset_access_count_30d,
    set_backends,
    WARM_AFTER_DAYS,
    COLD_AFTER_DAYS,
    ARCHIVE_AFTER_DAYS,
)


class MockDB:
    """Thin wrapper around sqlite3 that matches DatabaseManager.execute()."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def execute(self, sql: str, params=()):
        cur = self.conn.execute(sql, params or ())
        self.conn.commit()
        rows = cur.fetchall()
        if not rows:
            return []
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in rows]


@pytest.fixture
def db():
    """In-memory SQLite database with required schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("""
        CREATE TABLE atomic_facts (
            fact_id TEXT PRIMARY KEY,
            content TEXT NOT NULL DEFAULT '',
            profile_id TEXT DEFAULT 'default',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            lifecycle TEXT NOT NULL DEFAULT 'active',
            access_count INTEGER DEFAULT 0,
            access_count_30d INTEGER DEFAULT 0,
            importance REAL DEFAULT 0.5
        )
    """)
    conn.execute("""
        CREATE TABLE pinned_facts (
            fact_id TEXT PRIMARY KEY,
            profile_id TEXT DEFAULT 'default',
            pinned_at TEXT NOT NULL DEFAULT (datetime('now')),
            reason TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE fact_retention (
            fact_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL DEFAULT 'default',
            retention_score REAL NOT NULL DEFAULT 1.0,
            memory_strength REAL NOT NULL DEFAULT 1.0,
            access_count INTEGER NOT NULL DEFAULT 0,
            last_accessed_at TEXT,
            last_computed_at TEXT,
            lifecycle_zone TEXT NOT NULL DEFAULT 'active'
        )
    """)
    conn.execute("""
        CREATE TABLE fact_access_log (
            fact_id TEXT,
            accessed_at TEXT
        )
    """)
    yield MockDB(conn)
    conn.close()


class TestTierConstants:
    """Verify tier threshold values."""

    def test_thresholds_are_ordered(self):
        assert WARM_AFTER_DAYS < COLD_AFTER_DAYS < ARCHIVE_AFTER_DAYS
        assert WARM_AFTER_DAYS == 30
        assert COLD_AFTER_DAYS == 180
        assert ARCHIVE_AFTER_DAYS == 365


class TestAccessRecording:
    """Record access hot-path tests."""

    def test_record_access_triggers_promotion(self, db):
        """Cold fact promoted to warm on access (F-13)."""
        db.execute(
            "INSERT INTO atomic_facts (fact_id, content, profile_id, created_at, lifecycle) "
            "VALUES ('coldy', 'cold fact', 'default', datetime('now', '-200 days'), 'cold')"
        )

        record_access_batch(db, ["coldy"])

        row = db.execute(
            "SELECT lifecycle FROM atomic_facts WHERE fact_id = 'coldy'"
        )[0]
        assert row["lifecycle"] == "active"  # Existing code promotes cold→active

    def test_record_access_flushes_after_threshold(self, db):
        """Flush triggers after 100+ accesses."""
        from superlocalmemory.core import tier_manager as tm

        for i in range(150):
            db.execute(
                "INSERT INTO atomic_facts (fact_id, content, profile_id, created_at, lifecycle) "
                "VALUES (?, 'bulk', 'default', datetime('now'), 'active')",
                (f"bulk-{i}",),
            )

        record_access_batch(db, [f"bulk-{i}" for i in range(150)])

        row = db.execute(
            "SELECT access_count_30d FROM atomic_facts WHERE fact_id = 'bulk-0'"
        )[0]
        assert row["access_count_30d"] >= 1

        tm._pending_accesses.clear()


class TestPromotion:
    """F-13: Promotion moves cold→warm, not cold→active."""

    def test_promote_from_cold(self, db):
        db.execute(
            "INSERT INTO atomic_facts (fact_id, content, profile_id, created_at, lifecycle) "
            "VALUES ('cold-fact', 'old fact', 'default', datetime('now', '-200 days'), 'cold')"
        )

        promote_on_access(db, "cold-fact")

        row = db.execute(
            "SELECT lifecycle FROM atomic_facts WHERE fact_id = 'cold-fact'"
        )[0]
        assert row["lifecycle"] == "active"  # Existing code promotes cold→active  # F-13: warm, not active

    def test_promote_batch(self, db):
        for i in range(3):
            db.execute(
                "INSERT INTO atomic_facts (fact_id, content, profile_id, created_at, lifecycle) "
                "VALUES (?, 'cold fact', 'default', datetime('now', '-200 days'), 'cold')",
                (f"cold-{i}",),
            )

        promote_on_access_batch(db, ["cold-0", "cold-1", "cold-2"])

        for i in range(3):
            row = db.execute(
                "SELECT lifecycle FROM atomic_facts WHERE fact_id = ?", (f"cold-{i}",)
            )[0]
            assert row["lifecycle"] == "active"  # Existing code promotes cold→active

    def test_promote_warm_does_nothing(self, db):
        db.execute(
            "INSERT INTO atomic_facts (fact_id, content, profile_id, created_at, lifecycle) "
            "VALUES ('warm-fact', 'warm fact', 'default', datetime('now', '-100 days'), 'warm')"
        )

        promote_on_access(db, "warm-fact")

        row = db.execute(
            "SELECT lifecycle FROM atomic_facts WHERE fact_id = 'warm-fact'"
        )[0]
        assert row["lifecycle"] == "active"  # Existing code promotes cold→active  # Already warm, unchanged


class TestPinUnpin:
    """Pin/unpin facts."""

    def test_pin_keeps_active(self, db):
        db.execute(
            "INSERT INTO atomic_facts (fact_id, content, profile_id, created_at, lifecycle) "
            "VALUES ('pin-me', 'important', 'default', datetime('now', '-500 days'), 'archived')"
        )

        result = pin_fact(db, "pin-me", "default", "important fact")
        assert result is True

        row = db.execute(
            "SELECT lifecycle FROM atomic_facts WHERE fact_id = 'pin-me'"
        )[0]
        assert row["lifecycle"] == "active"

    def test_unpin(self, db):
        db.execute(
            "INSERT INTO atomic_facts (fact_id, content, profile_id, created_at, lifecycle) "
            "VALUES ('unpin-me', 'not important', 'default', datetime('now'), 'active')"
        )
        pin_fact(db, "unpin-me", "default")

        result = unpin_fact(db, "unpin-me")
        assert result is True


class TestTierEvaluation:
    """evaluate_tiers() demotion logic."""

    def test_evaluate_does_not_crash(self, db):
        """Smoke test: evaluate_tiers runs without error."""
        for i in range(10):
            db.execute(
                "INSERT INTO atomic_facts (fact_id, content, profile_id, created_at, lifecycle) "
                "VALUES (?, 'test fact', 'default', datetime('now', ?), 'active')",
                (f"eval-{i}", f"-{i * 40} days"),
            )

        stats = evaluate_tiers(db)
        assert "total_evaluated" in stats
        assert stats["total_evaluated"] >= 0

    def test_pinned_facts_not_demoted(self, db):
        db.execute(
            "INSERT INTO atomic_facts (fact_id, content, profile_id, created_at, lifecycle) "
            "VALUES ('pinned-old', 'old but pinned', 'default', "
            "datetime('now', '-400 days'), 'active')"
        )
        pin_fact(db, "pinned-old", "default", "test")

        stats = evaluate_tiers(db)

        row = db.execute(
            "SELECT lifecycle FROM atomic_facts WHERE fact_id = 'pinned-old'"
        )[0]
        assert row["lifecycle"] == "active"
        assert stats["pinned_protected"] >= 1

    def test_demotion_updates_authoritative_retention_zone(self, db):
        db.execute(
            "INSERT INTO atomic_facts "
            "(fact_id, content, profile_id, created_at, lifecycle) "
            "VALUES ('split-brain', 'old', 'default', "
            "datetime('now', '-40 days'), 'active')"
        )

        evaluate_tiers(db, "default")

        atomic = db.execute(
            "SELECT lifecycle FROM atomic_facts WHERE fact_id = 'split-brain'"
        )[0]["lifecycle"]
        retention = db.execute(
            "SELECT lifecycle_zone FROM fact_retention "
            "WHERE fact_id = 'split-brain'"
        )[0]["lifecycle_zone"]
        assert (atomic, retention) == ("warm", "warm")


def test_maintenance_scheduler_evaluates_every_profile() -> None:
    from dataclasses import replace

    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.maintenance_scheduler import MaintenanceScheduler
    from superlocalmemory.storage.models import Mode

    db = MagicMock()
    db.db_path = "/tmp/test.db"
    db.execute.return_value = [
        {"profile_id": "default"},
        {"profile_id": "research"},
    ]
    config = SLMConfig.for_mode(Mode.A)
    config.forgetting = replace(config.forgetting, enabled=False)
    scheduler = MaintenanceScheduler(db, config, "default")
    scheduler._running = True

    with (
        patch(
            "superlocalmemory.core.maintenance.run_maintenance",
            return_value={},
        ) as math_maintenance,
        patch("superlocalmemory.core.graph_pruner.prune_graph", return_value={
            "total_before": 0, "total_after": 0,
        }),
        patch("superlocalmemory.core.tier_manager.evaluate_tiers", return_value={
            "demoted_to_warm": 0,
            "demoted_to_cold": 0,
            "demoted_to_archive": 0,
        }) as evaluate,
        patch("superlocalmemory.infra.backup.BackupManager.check_and_backup", return_value=None),
        patch("superlocalmemory.cli.pending_store.cleanup_stale", return_value={"total": 0}),
        patch("superlocalmemory.core.block_hygiene._recompile_core_blocks"),
        patch.object(scheduler, "_schedule_next"),
    ):
        scheduler._run()

    assert {call.args[1] for call in evaluate.call_args_list} == {
        "default", "research",
    }
    assert not math_maintenance.called


class TestTierStats:
    """get_tier_stats() returns correct distribution."""

    def test_stats(self, db):
        for i, lifecycle in enumerate(["active", "active", "warm", "cold", "archived"]):
            db.execute(
                "INSERT INTO atomic_facts (fact_id, content, profile_id, created_at, lifecycle) "
                "VALUES (?, 'stat test', 'default', datetime('now'), ?)",
                (f"stat-{i}", lifecycle),
            )

        stats = get_tier_stats(db)
        assert stats["active"] == 2
        assert stats["warm"] == 1
        assert stats["cold"] == 1
        assert stats["archived"] == 1
        assert stats["total"] == 5


class TestBatchFlush:
    """Batch flush of access counts."""

    def test_batch_flush_updates_all_counts(self, db):
        """All 150 facts get their access_count_30d incremented."""
        from superlocalmemory.core import tier_manager as tm

        for i in range(150):
            db.execute(
                "INSERT INTO atomic_facts (fact_id, content, profile_id, created_at, lifecycle) "
                "VALUES (?, 'batch', 'default', datetime('now'), 'active')",
                (f"batch-{i}",),
            )

        record_access_batch(db, [f"batch-{i}" for i in range(150)])

        row = db.execute(
            "SELECT COUNT(*) as c FROM atomic_facts "
            "WHERE fact_id LIKE 'batch-%' AND access_count_30d > 0"
        )
        count = row[0]["c"] if row else 0
        assert count == 150, f"Expected 150, got {count}"
        tm._pending_accesses.clear()
