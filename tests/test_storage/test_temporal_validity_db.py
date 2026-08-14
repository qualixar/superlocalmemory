# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""DB-level tests for bi-temporal validity queries (Phase 4, T1).

Proves the closed loop against real SQLite + schema:
    invalidate_fact_temporal -> get_invalidated_fact_ids -> excluded by
    get_valid_facts. Facts with no temporal record are treated as valid.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
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

    def test_global_fact_uses_its_owner_temporal_record(self, db: DatabaseManager) -> None:
        """A global fact remains visible but must inherit its owner's correction."""
        db.execute(
            "INSERT INTO profiles (profile_id, name) VALUES (?, ?)", ("owner", "Owner"),
        )
        db.store_memory(
            MemoryRecord(memory_id="owner-memory", profile_id="owner", content="parent")
        )
        db.store_fact(AtomicFact(
            fact_id="old-global", memory_id="owner-memory", profile_id="owner",
            scope="global", content="old global fact", fact_type=FactType.SEMANTIC,
        ))
        db.store_temporal_validity("old-global", "owner")
        db.invalidate_fact_temporal(
            "old-global", invalidated_by="new-global", invalidation_reason="corrected",
        )

        invalid = db.get_invalidated_fact_ids(
            ["old-global"], "requester", include_global=True,
        )

        assert invalid == {"old-global"}
        assert db.get_invalidated_fact_ids(
            ["old-global"], "requester", as_of="2000-01-01T00:00:00+00:00",
            include_global=True,
        ) == set()

    def test_shared_fact_uses_its_owner_temporal_record(self, db: DatabaseManager) -> None:
        """Shared-scope admission also follows the fact owner's correction."""
        db.execute(
            "INSERT INTO profiles (profile_id, name) VALUES (?, ?)", ("owner", "Owner"),
        )
        db.execute(
            "INSERT INTO profiles (profile_id, name) VALUES (?, ?)", ("requester", "Requester"),
        )
        db.store_memory(
            MemoryRecord(memory_id="shared-memory", profile_id="owner", content="parent")
        )
        db.store_fact(AtomicFact(
            fact_id="old-shared", memory_id="shared-memory", profile_id="owner",
            scope="shared", shared_with=["requester"], content="old shared fact",
            fact_type=FactType.SEMANTIC,
        ))
        db.store_temporal_validity("old-shared", "owner")
        db.invalidate_fact_temporal(
            "old-shared", invalidated_by="new-shared", invalidation_reason="corrected",
        )

        invalid = db.get_invalidated_fact_ids(
            ["old-shared"], "requester", include_shared=True,
        )

        assert invalid == {"old-shared"}


class TestPinnedTemporalAdmission:
    def test_m015_is_safe_after_fresh_schema_bootstrap(self, db: DatabaseManager) -> None:
        """Fresh installs already carry the column; the upgrade migration is a no-op."""
        from superlocalmemory.storage.migrations import M015_add_pinned_column as m015

        conn = sqlite3.connect(str(db.db_path))
        try:
            m015.apply(conn)
            conn.commit()
            columns = {row[1] for row in conn.execute("PRAGMA table_info(atomic_facts)")}
        finally:
            conn.close()

        assert "pinned" in columns

    def test_current_pinned_recall_excludes_system_invalidated_fact(
        self, db: DatabaseManager,
    ) -> None:
        """A stale pin must not bypass correction-aware session injection."""
        _seed_three_facts(db)
        db.set_pinned("f1", True)
        db.set_pinned("f2", True)
        db.store_temporal_validity("f1", "default")
        db.invalidate_fact_temporal("f1", invalidated_by="f2", invalidation_reason="corrected")

        pinned = {fact.fact_id for fact in db.get_pinned("default")}

        assert pinned == {"f2"}


class TestStrictTwoClockAdmission:
    """S402-A: transaction knowledge and event validity are independent."""

    def test_known_as_of_excludes_fact_committed_one_second_later(
        self, db: DatabaseManager,
    ) -> None:
        _seed_three_facts(db)
        db.execute(
            "UPDATE fact_temporal_validity SET system_created_at = ? WHERE fact_id = ?",
            ("2026-01-01T00:00:01+00:00", "f1"),
        )
        assert db.get_strict_temporal_inadmissible_fact_ids(
            ["f1"], "default", known_as_of="2026-01-01T00:00:00+00:00",
        ) == {"f1"}

    def test_store_fact_anchor_is_canonical_and_excludes_same_day_future_knowledge(
        self, db: DatabaseManager,
    ) -> None:
        db.store_memory(MemoryRecord(memory_id="m0", content="parent"))
        db.store_fact(AtomicFact(
            fact_id="new", memory_id="m0", content="new fact", fact_type=FactType.SEMANTIC,
        ))
        anchor = db.get_temporal_validity("new", "default")["system_created_at"]
        assert "T" in anchor and anchor.endswith("+00:00")
        one_second_before = (datetime.fromisoformat(anchor) - timedelta(seconds=1)).isoformat()
        assert db.get_strict_temporal_inadmissible_fact_ids(
            ["new"], "default", known_as_of=one_second_before,
        ) == {"new"}

    def test_known_as_of_preserves_one_microsecond_boundary(
        self, db: DatabaseManager,
    ) -> None:
        _seed_three_facts(db)
        db.execute(
            "UPDATE fact_temporal_validity SET system_created_at = ? WHERE fact_id = ?",
            ("2026-01-01T00:00:00.000001+00:00", "f1"),
        )
        assert db.get_strict_temporal_inadmissible_fact_ids(
            ["f1"], "default", known_as_of="2026-01-01T00:00:00+00:00",
        ) == {"f1"}

    def test_anchor_failure_rolls_back_fact_and_retry_repairs_missing_anchor(
        self, db: DatabaseManager, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db.store_memory(MemoryRecord(memory_id="m0", content="parent"))
        fact = AtomicFact(
            fact_id="atomic", memory_id="m0", content="atomic fact", fact_type=FactType.SEMANTIC,
        )
        original = db.store_temporal_validity

        def _fail_anchor(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("anchor write interrupted")

        monkeypatch.setattr(db, "store_temporal_validity", _fail_anchor)
        with pytest.raises(RuntimeError, match="anchor write interrupted"):
            db.store_fact(fact)
        assert db.get_fact("atomic") is None

        monkeypatch.setattr(db, "store_temporal_validity", original)
        db.store_fact(fact)
        assert db.get_temporal_validity("atomic", "default") is not None

        # The recovery path also repairs a legacy-shaped missing anchor when a
        # repeated store resolves to the existing canonical fact.
        db.execute("DELETE FROM fact_temporal_validity WHERE fact_id = ?", ("atomic",))
        db.store_fact(AtomicFact(
            fact_id="retry", memory_id="m0", content="atomic fact", fact_type=FactType.SEMANTIC,
        ))
        assert db.get_temporal_validity("atomic", "default") is not None

    def test_known_as_of_does_not_use_current_supersession_to_rewrite_history(
        self, db: DatabaseManager,
    ) -> None:
        _seed_three_facts(db)
        db.execute(
            "UPDATE fact_temporal_validity SET system_created_at = ?, system_expired_at = ? "
            "WHERE fact_id = ?",
            ("2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00", "f1"),
        )
        assert db.get_strict_temporal_inadmissible_fact_ids(
            ["f1"], "default", known_as_of="2026-01-15T00:00:00+00:00",
        ) == set()
        assert db.get_strict_temporal_inadmissible_fact_ids(
            ["f1"], "default", known_as_of="2026-02-01T00:00:00+00:00",
        ) == {"f1"}

    def test_valid_at_and_known_as_of_are_independent_axes(
        self, db: DatabaseManager,
    ) -> None:
        _seed_three_facts(db)
        db.execute(
            "UPDATE fact_temporal_validity SET system_created_at = ?, valid_from = ? "
            "WHERE fact_id = ?",
            ("2020-01-01T00:00:00+00:00", "2030-01-01T00:00:00+00:00", "f1"),
        )
        assert db.get_strict_temporal_inadmissible_fact_ids(
            ["f1"], "default", known_as_of="2021-01-01T00:00:00+00:00",
        ) == set()
        assert db.get_strict_temporal_inadmissible_fact_ids(
            ["f1"], "default", valid_at="2021-01-01T00:00:00+00:00",
        ) == {"f1"}

    def test_legacy_unknown_is_excluded_unless_explicitly_included(
        self, db: DatabaseManager,
    ) -> None:
        db.store_memory(MemoryRecord(memory_id="legacy_memory", content="parent"))
        db.store_fact(AtomicFact(
            fact_id="legacy", memory_id="legacy_memory", content="legacy fact",
            fact_type=FactType.SEMANTIC,
        ))
        # Simulate an existing pre-4.0.2 row: it has no knowledge-time anchor.
        db.execute(
            "DELETE FROM fact_temporal_validity WHERE fact_id = ?", ("legacy",),
        )
        assert db.get_strict_temporal_inadmissible_fact_ids(
            ["legacy"], "default", known_as_of="2021-01-01T00:00:00+00:00",
        ) == {"legacy"}
        assert db.get_strict_temporal_inadmissible_fact_ids(
            ["legacy"], "default", known_as_of="2021-01-01T00:00:00+00:00",
            include_unknown=True,
        ) == set()


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
