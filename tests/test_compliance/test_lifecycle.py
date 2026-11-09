# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.compliance.lifecycle — Memory Lifecycle Management.

Covers:
  - get_lifecycle_state: time-based fallback (Active/Warm/Cold/Archived)
  - get_lifecycle_state: Langevin-based (when langevin is available)
  - update_lifecycle: recomputes and persists state
  - update_lifecycle: no-op when fact not found
  - run_maintenance: processes all facts, returns counts
  - run_maintenance: Langevin path vs time-based path
  - get_archived_facts: returns only archived facts
  - _time_based_state: high access_count keeps Active
  - _days_since: edge cases (empty string, invalid date)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from superlocalmemory.compliance.lifecycle import LifecycleManager
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    AtomicFact,
    MemoryLifecycle,
    MemoryRecord,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    db_path = tmp_path / "lifecycle_test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    db_path_str = str(db_path)
    mgr.store_memory(MemoryRecord(memory_id="m0", content="parent"))
    return mgr


@pytest.fixture()
def manager(db: DatabaseManager) -> LifecycleManager:
    return LifecycleManager(db)


def _make_fact(
    fact_id: str = "f_test",
    created_at: str | None = None,
    access_count: int = 0,
    lifecycle: MemoryLifecycle = MemoryLifecycle.ACTIVE,
    langevin_position: list[float] | None = None,
    importance: float = 0.5,
) -> AtomicFact:
    """Create an AtomicFact with controlled timestamps."""
    if created_at is None:
        created_at = datetime.now(UTC).isoformat()
    return AtomicFact(
        fact_id=fact_id,
        memory_id="m0",
        content="test fact",
        created_at=created_at,
        access_count=access_count,
        lifecycle=lifecycle,
        langevin_position=langevin_position,
        importance=importance,
    )


def _days_ago(days: int) -> str:
    """Return ISO date string N days ago."""
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


# ---------------------------------------------------------------------------
# _time_based_state
# ---------------------------------------------------------------------------

class TestTimeBasedState:
    def test_recent_fact_is_active(self, manager: LifecycleManager) -> None:
        fact = _make_fact(created_at=datetime.now(UTC).isoformat())
        state = manager._time_based_state(fact)
        assert state == MemoryLifecycle.ACTIVE

    def test_high_access_keeps_active(self, manager: LifecycleManager) -> None:
        """Access count > 10 keeps fact active regardless of age."""
        fact = _make_fact(created_at=_days_ago(50), access_count=15)
        state = manager._time_based_state(fact)
        assert state == MemoryLifecycle.ACTIVE

    def test_10_days_old_is_warm(self, manager: LifecycleManager) -> None:
        fact = _make_fact(created_at=_days_ago(10), access_count=0)
        state = manager._time_based_state(fact)
        assert state == MemoryLifecycle.WARM

    def test_40_days_old_is_cold(self, manager: LifecycleManager) -> None:
        fact = _make_fact(created_at=_days_ago(40), access_count=0)
        state = manager._time_based_state(fact)
        assert state == MemoryLifecycle.COLD

    def test_100_days_old_is_archived(self, manager: LifecycleManager) -> None:
        fact = _make_fact(created_at=_days_ago(100), access_count=0)
        state = manager._time_based_state(fact)
        assert state == MemoryLifecycle.ARCHIVED

    def test_boundary_7_days_is_warm(self, manager: LifecycleManager) -> None:
        """Exactly at 7 days boundary (> 7 days)."""
        fact = _make_fact(created_at=_days_ago(8), access_count=0)
        state = manager._time_based_state(fact)
        assert state == MemoryLifecycle.WARM


# ---------------------------------------------------------------------------
# _days_since edge cases
# ---------------------------------------------------------------------------

class TestDaysSince:
    def test_empty_string_returns_zero(self) -> None:
        assert LifecycleManager._days_since("") == 0.0

    def test_invalid_date_returns_zero(self) -> None:
        assert LifecycleManager._days_since("not-a-date") == 0.0

    def test_recent_date_returns_small_positive(self) -> None:
        recent = datetime.now(UTC).isoformat()
        days = LifecycleManager._days_since(recent)
        assert 0.0 <= days < 0.01  # Within a few seconds

    def test_old_date_returns_large_positive(self) -> None:
        old = (datetime.now(UTC) - timedelta(days=365)).isoformat()
        days = LifecycleManager._days_since(old)
        assert days >= 364.0


# ---------------------------------------------------------------------------
# get_lifecycle_state
# ---------------------------------------------------------------------------

class TestGetLifecycleState:
    def test_uses_time_based_when_no_langevin(
        self, manager: LifecycleManager
    ) -> None:
        fact = _make_fact(created_at=_days_ago(50), access_count=0)
        state = manager.get_lifecycle_state(fact)
        assert state == MemoryLifecycle.COLD

    def test_uses_langevin_when_available(self, db: DatabaseManager) -> None:
        mock_langevin = MagicMock()
        mock_langevin.compute_lifecycle_weight.return_value = 0.9
        mock_langevin.get_lifecycle_state.return_value = MemoryLifecycle.ACTIVE

        mgr = LifecycleManager(db, langevin=mock_langevin)
        fact = _make_fact(
            created_at=_days_ago(100),
            langevin_position=[0.1, 0.2, 0.3],
        )
        state = mgr.get_lifecycle_state(fact)
        assert state == MemoryLifecycle.ACTIVE
        mock_langevin.compute_lifecycle_weight.assert_called_once_with([0.1, 0.2, 0.3])

    def test_falls_back_when_no_langevin_position(
        self, db: DatabaseManager
    ) -> None:
        """Even with langevin available, no position => time-based fallback."""
        mock_langevin = MagicMock()
        mgr = LifecycleManager(db, langevin=mock_langevin)
        fact = _make_fact(created_at=_days_ago(50), access_count=0)
        state = mgr.get_lifecycle_state(fact)
        assert state == MemoryLifecycle.COLD
        mock_langevin.compute_lifecycle_weight.assert_not_called()


# ---------------------------------------------------------------------------
# update_lifecycle
# ---------------------------------------------------------------------------

class TestUpdateLifecycle:
    def test_returns_active_for_missing_fact(
        self, manager: LifecycleManager
    ) -> None:
        state = manager.update_lifecycle("nonexistent", "default")
        assert state == MemoryLifecycle.ACTIVE

    def test_updates_state_in_db(
        self, manager: LifecycleManager, db: DatabaseManager
    ) -> None:
        old_date = _days_ago(50)
        db.store_fact(AtomicFact(
            fact_id="f_update", memory_id="m0", content="old fact",
            created_at=old_date, access_count=0,
        ))
        state = manager.update_lifecycle("f_update", "default")
        assert state == MemoryLifecycle.COLD

    def test_high_access_stays_active(
        self, manager: LifecycleManager, db: DatabaseManager
    ) -> None:
        db.store_fact(AtomicFact(
            fact_id="f_active", memory_id="m0", content="active fact",
            created_at=_days_ago(50), access_count=15,
        ))
        state = manager.update_lifecycle("f_active", "default")
        assert state == MemoryLifecycle.ACTIVE


# ---------------------------------------------------------------------------
# run_maintenance
# ---------------------------------------------------------------------------

class TestRunMaintenance:
    def test_processes_all_facts(
        self, manager: LifecycleManager, db: DatabaseManager
    ) -> None:
        db.store_fact(AtomicFact(
            fact_id="f_new", memory_id="m0", content="new",
            created_at=datetime.now(UTC).isoformat(),
        ))
        db.store_fact(AtomicFact(
            fact_id="f_old", memory_id="m0", content="old",
            created_at=_days_ago(100), access_count=0,
        ))
        counts = manager.run_maintenance("default")
        assert counts["active"] >= 1
        assert counts["archived"] >= 1
        assert "transitions" in counts

    def test_empty_profile_returns_zero_counts(
        self, manager: LifecycleManager
    ) -> None:
        counts = manager.run_maintenance("empty_profile")
        assert counts["transitions"] == 0

    def test_langevin_path_called_when_available(
        self, db: DatabaseManager
    ) -> None:
        mock_langevin = MagicMock()
        mock_langevin.step.return_value = ([0.2, 0.3], 0.8)
        mock_langevin.get_lifecycle_state.return_value = MemoryLifecycle.ACTIVE

        mgr = LifecycleManager(db, langevin=mock_langevin)

        # Store a fact with langevin_position
        db.store_fact(AtomicFact(
            fact_id="f_lang", memory_id="m0", content="langevin fact",
            langevin_position=[0.1, 0.2],
            created_at=datetime.now(UTC).isoformat(),
        ))
        counts = mgr.run_maintenance("default")
        assert mock_langevin.step.called


# ---------------------------------------------------------------------------
# get_archived_facts
# ---------------------------------------------------------------------------

class TestGetArchivedFacts:
    def test_returns_only_archived(
        self, manager: LifecycleManager, db: DatabaseManager
    ) -> None:
        db.store_fact(AtomicFact(
            fact_id="f_active2", memory_id="m0", content="active",
            lifecycle=MemoryLifecycle.ACTIVE,
        ))
        db.store_fact(AtomicFact(
            fact_id="f_archived2", memory_id="m0", content="archived",
            lifecycle=MemoryLifecycle.ARCHIVED,
        ))
        archived = manager.get_archived_facts("default")
        assert len(archived) == 1
        assert archived[0].fact_id == "f_archived2"

    def test_empty_when_none_archived(
        self, manager: LifecycleManager, db: DatabaseManager
    ) -> None:
        db.store_fact(AtomicFact(
            fact_id="f_warm", memory_id="m0", content="warm",
            lifecycle=MemoryLifecycle.WARM,
        ))
        assert manager.get_archived_facts("default") == []
