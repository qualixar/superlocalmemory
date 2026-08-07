# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4

"""W2: maintenance auto-closes orphaned application sessions.

Nothing auto-calls close_session except the MCP tool. Un-closed sessions
never get temporal summaries. Maintenance must close stale sessions
(idle > configurable window), bounded per pass, without double-summarising.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.core.maintenance import close_stale_sessions, run_maintenance
from superlocalmemory.core.store_pipeline import run_close_session
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, MemoryRecord


def _iso_hours_ago(hours: float) -> str:
    return (datetime.now(UTC) - timedelta(hours=hours)).isoformat()


def _seed_session(
    db: DatabaseManager,
    *,
    session_id: str,
    profile_id: str = "default",
    hours_ago: float = 48.0,
    entity_id: str = "ent-alice",
) -> None:
    db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
        (profile_id, profile_id),
    )
    db.execute(
        "INSERT OR IGNORE INTO canonical_entities "
        "(entity_id, profile_id, canonical_name) VALUES (?, ?, ?)",
        (entity_id, profile_id, "Alice"),
    )
    mid = f"m-{session_id}"
    db.store_memory(
        MemoryRecord(
            memory_id=mid,
            profile_id=profile_id,
            content=f"memory for {session_id}",
            session_id=session_id,
            created_at=_iso_hours_ago(hours_ago),
        )
    )
    fact = AtomicFact(
        fact_id=f"f-{session_id}",
        memory_id=mid,
        profile_id=profile_id,
        content=f"Alice decided X in {session_id}",
        session_id=session_id,
        canonical_entities=[entity_id],
        observation_date=_iso_hours_ago(hours_ago)[:10],
        created_at=_iso_hours_ago(hours_ago),
    )
    db.store_fact(fact)


def _summary_count(db: DatabaseManager, profile_id: str, session_id: str) -> int:
    rows = db.execute(
        "SELECT COUNT(*) AS c FROM temporal_events "
        "WHERE profile_id = ? AND description LIKE ?",
        (profile_id, f"Session {session_id}:%"),
    )
    return int(dict(rows[0])["c"])


@pytest.fixture
def db(tmp_path):
    manager = DatabaseManager(tmp_path / "stale-sessions.db")
    manager.initialize(schema)
    yield manager
    manager.close()


class TestCloseStaleSessions:
    def test_closes_idle_session(self, db: DatabaseManager) -> None:
        _seed_session(db, session_id="sess-old", hours_ago=48.0)
        n = close_stale_sessions(db, "default", idle_hours=24.0, max_per_pass=50)
        assert n == 1
        assert _summary_count(db, "default", "sess-old") >= 1

    def test_skips_fresh_session(self, db: DatabaseManager) -> None:
        _seed_session(db, session_id="sess-fresh", hours_ago=1.0)
        n = close_stale_sessions(db, "default", idle_hours=24.0, max_per_pass=50)
        assert n == 0
        assert _summary_count(db, "default", "sess-fresh") == 0

    def test_idempotent_no_double_summarise(self, db: DatabaseManager) -> None:
        _seed_session(db, session_id="sess-once", hours_ago=48.0)
        assert close_stale_sessions(db, "default", idle_hours=24.0) == 1
        first = _summary_count(db, "default", "sess-once")
        assert first >= 1
        # Second maintenance pass must not create more summaries.
        assert close_stale_sessions(db, "default", idle_hours=24.0) == 0
        assert _summary_count(db, "default", "sess-once") == first
        # Explicit close_session is also idempotent.
        assert run_close_session("sess-once", "default", db=db) == 0
        assert _summary_count(db, "default", "sess-once") == first

    def test_bounded_per_pass(self, db: DatabaseManager) -> None:
        for i in range(5):
            _seed_session(
                db,
                session_id=f"sess-b{i}",
                hours_ago=48.0 + i,
                entity_id=f"ent-{i}",
            )
        n = close_stale_sessions(db, "default", idle_hours=24.0, max_per_pass=2)
        assert n == 2
        # Remaining stale sessions still open.
        remaining = close_stale_sessions(db, "default", idle_hours=24.0, max_per_pass=50)
        assert remaining == 3

    def test_run_maintenance_includes_stale_close(self, db: DatabaseManager) -> None:
        _seed_session(db, session_id="sess-maint", hours_ago=72.0)
        cfg = SLMConfig()
        cfg.session_idle_close_hours = 24.0
        cfg.session_idle_close_max_per_pass = 50
        # Disable math-heavy paths that need more setup.
        cfg.math.langevin_persist_positions = False
        cfg.math.sheaf_at_encoding = False
        cfg.math.fisher_bayesian_update = False
        cfg.math.ebbinghaus_langevin_coupling_enabled = False
        counts = run_maintenance(db, cfg, profile_id="default")
        assert "stale_sessions_closed" in counts
        assert counts["stale_sessions_closed"] == 1
        assert _summary_count(db, "default", "sess-maint") >= 1
