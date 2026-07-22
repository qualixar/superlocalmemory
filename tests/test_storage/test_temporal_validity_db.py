# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""DB-level tests for bi-temporal validity queries (Phase 4, T1).

Proves the closed loop against real SQLite + schema:
    invalidate_fact_temporal -> get_invalidated_fact_ids -> excluded by
    get_valid_facts. Facts with no temporal record are treated as valid.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "test.db")
    mgr.initialize(real_schema)
    return mgr


def _seed_three_facts(db: DatabaseManager) -> None:
    db.store_memory(MemoryRecord(memory_id="m0", content="parent"))
    for fid, text in (("f1", "lives in Delhi"),
                      ("f2", "lives in Mumbai"),
                      ("f3", "works at Qualixar")):
        db.store_fact(AtomicFact(
            fact_id=fid, memory_id="m0", content=text,
            fact_type=FactType.SEMANTIC,
        ))


class TestGetInvalidatedFactIds:
    def test_empty_input_returns_empty(self, db: DatabaseManager) -> None:
        assert db.get_invalidated_fact_ids([], "default") == set()

    def test_facts_without_record_are_valid(self, db: DatabaseManager) -> None:
        _seed_three_facts(db)
        # No temporal records stored at all -> none are invalid.
        assert db.get_invalidated_fact_ids(["f1", "f2", "f3"], "default") == set()

    def test_open_validity_record_is_valid(self, db: DatabaseManager) -> None:
        _seed_three_facts(db)
        db.store_temporal_validity("f1", "default")  # open-ended, not expired
        assert db.get_invalidated_fact_ids(["f1"], "default") == set()

    def test_system_invalidated_fact_is_returned(self, db: DatabaseManager) -> None:
        _seed_three_facts(db)
        db.store_temporal_validity("f1", "default")
        db.store_temporal_validity("f2", "default")
        # f1 is superseded by f2.
        db.invalidate_fact_temporal("f1", invalidated_by="f2",
                                    invalidation_reason="contradicted")

        invalid = db.get_invalidated_fact_ids(["f1", "f2", "f3"], "default")
        assert invalid == {"f1"}

    def test_closed_loop_excluded_from_valid_facts(self, db: DatabaseManager) -> None:
        _seed_three_facts(db)
        db.store_temporal_validity("f1", "default")
        db.invalidate_fact_temporal("f1", invalidated_by="f2",
                                    invalidation_reason="contradicted")

        valid = set(db.get_valid_facts("default"))
        assert "f1" not in valid          # superseded fact drops out
        assert {"f2", "f3"} <= valid       # others remain
        # And the two views agree.
        assert db.get_invalidated_fact_ids(["f1", "f2", "f3"], "default") == {"f1"}

    def test_profile_scoped(self, db: DatabaseManager) -> None:
        _seed_three_facts(db)
        db.store_temporal_validity("f1", "default")
        db.invalidate_fact_temporal("f1", invalidated_by="f2",
                                    invalidation_reason="contradicted")
        # Querying under a different profile must not see this profile's row.
        assert db.get_invalidated_fact_ids(["f1"], "other") == set()


class TestGetFactEventTimes:
    def test_empty_input(self, db: DatabaseManager) -> None:
        assert db.get_fact_event_times([], "default") == {}

    def test_created_at_fallback(self, db: DatabaseManager) -> None:
        _seed_three_facts(db)  # no dates set -> event_time falls back to created_at
        et = db.get_fact_event_times(["f1", "f2", "f3"], "default")
        assert set(et) == {"f1", "f2", "f3"}
        assert all(v for v in et.values())  # every fact has a non-empty time

    def test_referenced_date_wins(self, db: DatabaseManager) -> None:
        db.store_memory(MemoryRecord(memory_id="m0", content="parent"))
        db.store_fact(AtomicFact(
            fact_id="fx", memory_id="m0", content="met on a specific date",
            fact_type=FactType.SEMANTIC,
            referenced_date="2026-03-15", observation_date="2026-07-01",
        ))
        et = db.get_fact_event_times(["fx"], "default")
        assert et["fx"].startswith("2026-03-15")

    def test_valid_from_used_when_no_fact_dates(self, db: DatabaseManager) -> None:
        _seed_three_facts(db)  # f1 has no referenced/observation date
        db.store_temporal_validity("f1", "default", valid_from="2025-01-01")
        et = db.get_fact_event_times(["f1"], "default")
        assert et["f1"].startswith("2025-01-01")

    def test_profile_scoped(self, db: DatabaseManager) -> None:
        _seed_three_facts(db)
        assert db.get_fact_event_times(["f1"], "other") == {}
