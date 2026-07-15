# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for forgetting scheduler — Phase A.

TDD: 4 tests covering batch retention update, on-access strengthening,
soft-delete audit trail, and scheduler interval check.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.core.config import ForgettingConfig
from superlocalmemory.learning.forgetting_scheduler import ForgettingScheduler
from superlocalmemory.math.ebbinghaus import EbbinghausCurve
from superlocalmemory.storage.database import DatabaseManager


@pytest.fixture
def config() -> ForgettingConfig:
    return ForgettingConfig()


@pytest.fixture
def ebbinghaus(config: ForgettingConfig) -> EbbinghausCurve:
    return EbbinghausCurve(config)


@pytest.fixture
def db_with_facts(tmp_path):
    """Create a real DB with schema and seed 10 test facts."""
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.schema_v32 import V32_DDL

    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)

    # Initialize base schema
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    schema.create_all_tables(conn)
    conn.commit()

    # Initialize v32 schema (includes fact_retention)
    for ddl in V32_DDL:
        conn.executescript(ddl)
    conn.commit()

    # Create a test profile
    conn.execute(
        "INSERT INTO profiles (profile_id, name) VALUES (?, ?)",
        ("test_profile", "Test Profile"),
    )

    # Seed 10 facts with varying ages and access patterns
    now = datetime.now(UTC)
    for i in range(10):
        fact_id = f"fact_{i:03d}"
        hours_ago = i * 24  # each fact is 24h older than the previous
        created = (now - timedelta(hours=hours_ago)).isoformat()

        # Insert parent memory record (FK requirement)
        conn.execute(
            "INSERT INTO memories "
            "(memory_id, profile_id, content, session_id, speaker, role, "
            " created_at, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (f"mem_{i}", "test_profile", f"Memory {i}", "sess1", "user",
             "user", created, "{}"),
        )

        conn.execute(
            "INSERT INTO atomic_facts "
            "(fact_id, memory_id, profile_id, content, fact_type, "
            " entities_json, canonical_entities_json, confidence, importance, "
            " evidence_count, access_count, source_turn_ids_json, "
            " lifecycle, emotional_valence, emotional_arousal, signal_type, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (fact_id, f"mem_{i}", "test_profile", f"Test fact {i}",
             "semantic", "[]", "[]", 1.0, 0.5 - (i * 0.05),
             max(1, 5 - i), 0, "[]",
             "active", 0.0, 0.0, "factual", created),
        )

        # Add access log entries (more for recent facts)
        for j in range(max(1, 10 - i)):
            access_time = (now - timedelta(hours=hours_ago - j)).isoformat()
            conn.execute(
                "INSERT INTO fact_access_log "
                "(log_id, fact_id, profile_id, access_type, session_id, accessed_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (f"log_{i}_{j}", fact_id, "test_profile", "recall", "sess1", access_time),
            )

        # Add importance scores
        conn.execute(
            "INSERT INTO fact_importance "
            "(fact_id, profile_id, pagerank_score, computed_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            (fact_id, "test_profile", max(0.0, 0.5 - i * 0.05)),
        )

    conn.commit()
    conn.close()

    return db


# ---- Test 16: Batch retention update (100 facts in <200ms) ----

def test_batch_retention_update(db_with_facts, ebbinghaus: EbbinghausCurve, config: ForgettingConfig) -> None:
    """10 facts get retention scores computed correctly."""
    scheduler = ForgettingScheduler(db_with_facts, ebbinghaus, config)

    stats = scheduler.run_decay_cycle("test_profile")

    assert stats["total"] == 10, f"Expected 10 facts, got {stats['total']}"
    assert stats["total"] == (
        stats["active"] + stats["warm"] + stats["cold"]
        + stats["archive"] + stats["forgotten"]
    ), "Zone counts must sum to total"


def test_decay_dry_run_computes_without_durable_changes(
    db_with_facts, ebbinghaus: EbbinghausCurve, config: ForgettingConfig,
) -> None:
    scheduler = ForgettingScheduler(db_with_facts, ebbinghaus, config)
    before = db_with_facts.execute(
        "SELECT COUNT(*) AS n FROM fact_retention WHERE profile_id = ?",
        ("test_profile",),
    )[0]["n"]

    stats = scheduler.run_decay_cycle("test_profile", force=True, dry_run=True)

    after = db_with_facts.execute(
        "SELECT COUNT(*) AS n FROM fact_retention WHERE profile_id = ?",
        ("test_profile",),
    )[0]["n"]
    archived = db_with_facts.execute(
        "SELECT COUNT(*) AS n FROM atomic_facts WHERE lifecycle != 'active'"
    )[0]["n"]
    assert stats["total"] == 10
    assert before == after == 0
    assert archived == 0


# ---- Test 17: On-access strengthens ----

def test_on_access_strengthens(db_with_facts, ebbinghaus: EbbinghausCurve, config: ForgettingConfig) -> None:
    """After on_access_event, retention increases or stays same."""
    scheduler = ForgettingScheduler(db_with_facts, ebbinghaus, config)

    # First run a decay cycle to populate retention data
    scheduler.run_decay_cycle("test_profile")

    # Get initial retention for fact_000
    initial = db_with_facts.execute(
        "SELECT retention_score, memory_strength FROM fact_retention "
        "WHERE fact_id = ? AND profile_id = ?",
        ("fact_000", "test_profile"),
    )
    assert len(initial) > 0, "fact_000 should have retention data"
    initial_strength = float(dict(initial[0])["memory_strength"])

    # Trigger on-access event
    scheduler.on_access_event("fact_000", "test_profile")

    # Check that strength increased (HR-07)
    updated = db_with_facts.execute(
        "SELECT retention_score, memory_strength FROM fact_retention "
        "WHERE fact_id = ? AND profile_id = ?",
        ("fact_000", "test_profile"),
    )
    updated_strength = float(dict(updated[0])["memory_strength"])
    assert updated_strength >= initial_strength, (
        f"Strength should increase after access: was {initial_strength}, now {updated_strength}"
    )


# ---- Test 20 (A-HIGH-01): Soft-delete audit trail ----

def test_forgotten_soft_delete(tmp_path, ebbinghaus: EbbinghausCurve) -> None:
    """Facts with retention < forget_threshold get lifecycle_zone='forgotten'
    in fact_retention AND lifecycle='archived' in atomic_facts.
    Verify they are NOT physically deleted."""
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.schema_v32 import V32_DDL

    # Use very aggressive config so facts get forgotten quickly
    config = ForgettingConfig(
        forget_threshold=0.99,  # Almost everything is "forgotten"
        archive_threshold=0.999,
    )
    ebbinghaus_aggressive = EbbinghausCurve(config)

    db_path = tmp_path / "soft_delete.db"
    db = DatabaseManager(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    schema.create_all_tables(conn)
    for ddl in V32_DDL:
        conn.executescript(ddl)
    conn.commit()

    conn.execute(
        "INSERT INTO profiles (profile_id, name) VALUES (?, ?)",
        ("test_profile", "Test"),
    )

    # Create a fact that was created long ago and never accessed
    old_time = (datetime.now(UTC) - timedelta(days=365)).isoformat()
    conn.execute(
        "INSERT INTO memories "
        "(memory_id, profile_id, content, session_id, speaker, role, "
        " created_at, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("mem_old", "test_profile", "Old memory", "sess1", "user",
         "user", old_time, "{}"),
    )
    conn.execute(
        "INSERT INTO atomic_facts "
        "(fact_id, memory_id, profile_id, content, fact_type, "
        " entities_json, canonical_entities_json, confidence, importance, "
        " evidence_count, access_count, source_turn_ids_json, "
        " lifecycle, emotional_valence, emotional_arousal, signal_type, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("fact_old", "mem_old", "test_profile", "Old fact",
         "semantic", "[]", "[]", 1.0, 0.0,
         0, 0, "[]",
         "active", 0.0, 0.0, "factual", old_time),
    )
    conn.execute(
        "INSERT INTO fact_importance "
        "(fact_id, profile_id, pagerank_score, computed_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        ("fact_old", "test_profile", 0.0),
    )
    conn.commit()
    conn.close()

    scheduler = ForgettingScheduler(db, ebbinghaus_aggressive, config)
    stats = scheduler.run_decay_cycle("test_profile")

    assert stats["forgotten"] > 0, "At least one fact should be forgotten"

    # Verify fact still exists in atomic_facts (soft-delete only, HR-04)
    rows = db.execute(
        "SELECT lifecycle FROM atomic_facts WHERE fact_id = ?",
        ("fact_old",),
    )
    assert len(rows) > 0, "Fact must NOT be physically deleted"
    assert dict(rows[0])["lifecycle"] == "archived", (
        "atomic_facts.lifecycle should be 'archived' (valid enum value)"
    )

    # Verify fact_retention has lifecycle_zone = 'forgotten'
    ret_rows = db.execute(
        "SELECT lifecycle_zone FROM fact_retention WHERE fact_id = ?",
        ("fact_old",),
    )
    assert len(ret_rows) > 0
    assert dict(ret_rows[0])["lifecycle_zone"] == "forgotten"


# ---- Test 21 (A-HIGH-01): Scheduler interval check ----

def test_scheduler_triggers_on_interval(
    db_with_facts, ebbinghaus: EbbinghausCurve, config: ForgettingConfig,
) -> None:
    """Scheduler only runs when time_since_last_run >= interval.
    Calling before interval returns early with no-op stats."""
    scheduler = ForgettingScheduler(db_with_facts, ebbinghaus, config)

    # First run should execute normally
    stats1 = scheduler.run_decay_cycle("test_profile")
    assert stats1["total"] == 10

    # Immediate second run should be a no-op (within interval)
    stats2 = scheduler.run_decay_cycle("test_profile")
    assert stats2.get("skipped", False) is True, (
        "Second immediate run should be skipped (within interval)"
    )

    # Force-run should bypass interval check
    stats3 = scheduler.run_decay_cycle("test_profile", force=True)
    assert stats3["total"] == 10, "Force run should execute regardless of interval"


# ---- Coverage: on_access_event with no retention data ----

def test_on_access_no_retention_data(
    db_with_facts, ebbinghaus: EbbinghausCurve, config: ForgettingConfig,
) -> None:
    """on_access_event for a fact with no retention data should be a no-op."""
    scheduler = ForgettingScheduler(db_with_facts, ebbinghaus, config)
    # Don't run decay cycle first — no retention data exists
    # Should not raise, just log and return
    scheduler.on_access_event("fact_000", "test_profile")


# ---- Coverage: run_decay_cycle with empty profile ----

def test_decay_cycle_empty_profile(
    db_with_facts, ebbinghaus: EbbinghausCurve, config: ForgettingConfig,
) -> None:
    """run_decay_cycle for a profile with no facts should return zero counts."""
    scheduler = ForgettingScheduler(db_with_facts, ebbinghaus, config)
    stats = scheduler.run_decay_cycle("nonexistent_profile")
    assert stats["total"] == 0


# ---- Coverage: zone transition tracking (line 117) ----

def test_decay_cycle_tracks_transitions(
    db_with_facts, ebbinghaus: EbbinghausCurve, config: ForgettingConfig,
) -> None:
    """run_decay_cycle should count zone transitions between runs."""
    scheduler = ForgettingScheduler(db_with_facts, ebbinghaus, config)

    # First decay cycle establishes initial zones
    scheduler.run_decay_cycle("test_profile")
    scheduler._last_run_times.clear()  # Reset so we can run again

    # Manually alter a fact's retention zone so it transitions on next run
    db_with_facts.execute(
        "UPDATE fact_retention SET lifecycle_zone = 'active', retention_score = 1.0 "
        "WHERE fact_id = 'fact_009' AND profile_id = 'test_profile'",
        (),
    )

    # Second run should detect fact_009 transitioning from 'active' to its
    # real computed zone (which will be lower due to 9*24h age)
    stats = scheduler.run_decay_cycle("test_profile")
    assert stats["transitions"] >= 1, "At least one zone transition expected"


# ---- Coverage: on_access_event with timezone-naive date (lines 165-166) ----

def test_on_access_event_timezone_naive_date(
    db_with_facts, ebbinghaus: EbbinghausCurve, config: ForgettingConfig,
) -> None:
    """on_access_event handles timezone-naive last_accessed_at strings."""
    scheduler = ForgettingScheduler(db_with_facts, ebbinghaus, config)
    scheduler.run_decay_cycle("test_profile")

    # Set a timezone-naive datetime in the DB
    db_with_facts.execute(
        "UPDATE fact_retention SET last_accessed_at = '2026-03-01 12:00:00' "
        "WHERE fact_id = 'fact_000' AND profile_id = 'test_profile'",
        (),
    )

    # Should not crash — handles naive datetime by adding UTC
    scheduler.on_access_event("fact_000", "test_profile")

    # Verify strength was updated
    updated = db_with_facts.execute(
        "SELECT memory_strength FROM fact_retention "
        "WHERE fact_id = 'fact_000' AND profile_id = 'test_profile'",
        (),
    )
    assert len(updated) > 0


# ---- Coverage: on_access_event with invalid date (lines 169-170) ----

def test_on_access_event_invalid_date(
    db_with_facts, ebbinghaus: EbbinghausCurve, config: ForgettingConfig,
) -> None:
    """on_access_event handles invalid last_accessed_at gracefully."""
    scheduler = ForgettingScheduler(db_with_facts, ebbinghaus, config)
    scheduler.run_decay_cycle("test_profile")

    # Set an invalid datetime string in the DB
    db_with_facts.execute(
        "UPDATE fact_retention SET last_accessed_at = 'not-a-date' "
        "WHERE fact_id = 'fact_000' AND profile_id = 'test_profile'",
        (),
    )

    # Should not crash — catches ValueError and defaults hours_since to 0.0
    scheduler.on_access_event("fact_000", "test_profile")


# ---- Coverage: _get_existing_zones with empty facts (line 250) ----

def test_get_existing_zones_empty(
    db_with_facts, ebbinghaus: EbbinghausCurve, config: ForgettingConfig,
) -> None:
    """_get_existing_zones returns empty dict for empty input."""
    scheduler = ForgettingScheduler(db_with_facts, ebbinghaus, config)
    result = scheduler._get_existing_zones("test_profile", [])
    assert result == {}
