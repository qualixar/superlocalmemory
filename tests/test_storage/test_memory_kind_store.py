# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for superlocalmemory.storage.memory_kind_store.MemoryKindStore (WP-2).

M052 (WP-1) has not landed in this tree yet, so the fixture applies its DDL
by hand (LLD §4.2) — identical columns/tables/index a migrated store would
carry — so these tests exercise the real code path, not a mock of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.compliance.gdpr import GDPRCompliance
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.memory_kind_store import KindChange, MemoryKindStore
from superlocalmemory.storage.memory_kinds import KindAssignment, KindSource, MemoryKind
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord


def _apply_memory_kind_schema(db: DatabaseManager) -> None:
    """Upgrade the store with the real M052 migration (idempotent)."""
    from superlocalmemory.storage.migrations import M052_memory_kinds as m052

    with db.raw_connection() as conn:
        m052.apply(conn)
    db._kind_columns_present = True  # noqa: SLF001 - test shortcut


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "test.db")
    mgr.initialize(real_schema)
    _apply_memory_kind_schema(mgr)
    return mgr


@pytest.fixture()
def store(db: DatabaseManager) -> MemoryKindStore:
    return MemoryKindStore(db)


def _fact(db: DatabaseManager, profile_id: str, content: str, *, kind: str | None = None,
          source: str | None = None, quarantined: bool = False) -> str:
    mid = db.store_memory(MemoryRecord(profile_id=profile_id, content="session"))
    f = AtomicFact(profile_id=profile_id, memory_id=mid, content=content, fact_type=FactType.SEMANTIC)
    f.memory_kind = kind
    f.memory_kind_source = source
    fid = db.store_fact(f)
    if quarantined:
        db.execute("UPDATE atomic_facts SET quarantined = 1 WHERE fact_id = ?", (fid,))
    return fid


def _rowid(db: DatabaseManager, fact_id: str) -> int:
    return int(db.execute("SELECT rowid FROM atomic_facts WHERE fact_id = ?", (fact_id,))[0]["rowid"])


class TestSelectBatch:
    def test_select_batch_owner_profile_only(self, db: DatabaseManager, store: MemoryKindStore) -> None:
        db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('work', 'Work')")
        _fact(db, "default", "mine")
        _fact(db, "work", "not mine")
        batch = store.select_batch("default", after_rowid=0, limit=10, mode="untyped")
        assert {c.content for c in batch} == {"mine"}
        assert all(c.profile_id == "default" for c in batch)

    def test_select_batch_skips_quarantined(self, db: DatabaseManager, store: MemoryKindStore) -> None:
        _fact(db, "default", "visible")
        _fact(db, "default", "withheld", quarantined=True)
        batch = store.select_batch("default", after_rowid=0, limit=10, mode="untyped")
        assert {c.content for c in batch} == {"visible"}

    def test_refresh_mode_never_selects_confirmed(
        self, db: DatabaseManager, store: MemoryKindStore,
    ) -> None:
        _fact(db, "default", "untyped")
        _fact(db, "default", "rules suggestion", kind="rule", source="rules")
        _fact(db, "default", "llm suggestion", kind="decision", source="model:llm")
        _fact(db, "default", "caller confirmed", kind="rule", source="caller")
        _fact(db, "default", "user confirmed", kind="decision", source="user")

        batch = store.select_batch("default", after_rowid=0, limit=10, mode="refresh")
        contents = {c.content for c in batch}
        assert contents == {"untyped", "rules suggestion", "llm suggestion"}


class TestApplyBatch:
    def _running_run(self, db: DatabaseManager, store: MemoryKindStore, profile_id: str = "default") -> str:
        run = store.create_run(
            profile_id, backend="rules", recipe_id="kinds-rules-v1", mode="untyped",
            requested_by="test", total_estimate=1,
        )
        assert store.set_run_status(run["run_id"], "running", expected=["queued"])
        return run["run_id"]

    def test_apply_batch_guard_skips_row_changed_meanwhile(
        self, db: DatabaseManager, store: MemoryKindStore,
    ) -> None:
        fid = _fact(db, "default", "a fact")
        rowid = _rowid(db, fid)
        run_id = self._running_run(db, store)

        change = KindChange(
            rowid=rowid, fact_id=fid, old_kind=None, old_source=None, old_confidence=None,
            fact_type="semantic",
            new=KindAssignment(MemoryKind.RULE, KindSource.RULES, None, "kinds-rules-v1"),
        )
        # Simulate a concurrent writer re-typing the row between selection and apply.
        db.execute(
            "UPDATE atomic_facts SET memory_kind = 'decision', memory_kind_source = 'caller' "
            "WHERE fact_id = ?", (fid,),
        )
        result = store.apply_batch(run_id, "default", [change], new_cursor=rowid, actor="test")
        assert result.applied == 0
        assert result.skipped == 1
        row = db.get_all_facts("default")[0]
        assert row.memory_kind == "decision"  # untouched by the stale write
        assert row.memory_kind_source == "caller"

    def test_apply_batch_rolls_back_if_run_paused(
        self, db: DatabaseManager, store: MemoryKindStore,
    ) -> None:
        fid = _fact(db, "default", "a fact")
        rowid = _rowid(db, fid)
        run_id = self._running_run(db, store)
        # Another actor paused the run after the batch was selected.
        assert store.set_run_status(run_id, "paused", expected=["running"])

        change = KindChange(
            rowid=rowid, fact_id=fid, old_kind=None, old_source=None, old_confidence=None,
            fact_type="semantic",
            new=KindAssignment(MemoryKind.RULE, KindSource.RULES, None, "kinds-rules-v1"),
        )
        result = store.apply_batch(run_id, "default", [change], new_cursor=rowid, actor="test")
        assert result.run_still_active is False
        assert result.applied == 0
        row = db.get_all_facts("default")[0]
        assert row.memory_kind is None  # rolled back, not applied
        history = db.execute("SELECT * FROM memory_kind_history WHERE fact_id = ?", (fid,))
        assert history == []  # no partial history row survived the rollback

    def test_apply_batch_does_not_clobber_fact_type_changed_since_select(
        self, db: DatabaseManager, store: MemoryKindStore,
    ) -> None:
        """L1-07: the batch's guard compared only ``memory_kind``/
        ``memory_kind_source`` against the stale read, so a concurrent writer
        that changed ``fact_type`` alone (the kind unchanged) was invisible to
        it — the batch wrote its own stale ``fact_type`` back over the live
        value, a lost update."""
        fid = _fact(db, "default", "a fact")  # fact_type defaults to semantic
        rowid = _rowid(db, fid)
        run_id = self._running_run(db, store)

        change = KindChange(
            rowid=rowid, fact_id=fid, old_kind=None, old_source=None, old_confidence=None,
            fact_type="semantic",  # the batch's own (now stale) read
            new=KindAssignment(MemoryKind.RULE, KindSource.RULES, None, "kinds-rules-v1"),
        )
        # A concurrent writer changes fact_type only (e.g. a correction
        # successor) between select_batch and apply_batch.
        db.execute("UPDATE atomic_facts SET fact_type = 'episodic' WHERE fact_id = ?", (fid,))

        result = store.apply_batch(run_id, "default", [change], new_cursor=rowid, actor="test")
        assert result.applied == 0
        assert result.skipped == 1
        row = db.get_all_facts("default")[0]
        assert row.fact_type == FactType.EPISODIC  # untouched by the stale write
        assert row.memory_kind is None  # no suggestion applied either


class TestRevertBatch:
    def test_revert_restores_old_values_but_not_over_user_edit(
        self, db: DatabaseManager, store: MemoryKindStore,
    ) -> None:
        fid_a = _fact(db, "default", "fact A")
        fid_b = _fact(db, "default", "fact B")
        run = store.create_run(
            "default", backend="rules", recipe_id="kinds-rules-v1", mode="untyped",
            requested_by="test", total_estimate=2,
        )
        run_id = run["run_id"]
        store.set_run_status(run_id, "running", expected=["queued"])

        changes = [
            KindChange(
                rowid=_rowid(db, fid_a), fact_id=fid_a, old_kind=None, old_source=None,
                old_confidence=None, fact_type="semantic",
                new=KindAssignment(MemoryKind.RULE, KindSource.RULES, None, "kinds-rules-v1"),
            ),
            KindChange(
                rowid=_rowid(db, fid_b), fact_id=fid_b, old_kind=None, old_source=None,
                old_confidence=None, fact_type="semantic",
                new=KindAssignment(MemoryKind.DECISION, KindSource.RULES, None, "kinds-rules-v1"),
            ),
        ]
        result = store.apply_batch(run_id, "default", changes, new_cursor=_rowid(db, fid_b), actor="test")
        assert result.applied == 2

        # A user confirms fact B's kind by hand after the backfill ran.
        db.execute(
            "UPDATE atomic_facts SET memory_kind = 'procedure', memory_kind_source = 'user' "
            "WHERE fact_id = ?", (fid_b,),
        )

        store.set_run_status(run_id, "reverting", expected=["running"])
        revert_result = store.revert_batch(run_id, "default", limit=200)
        assert revert_result.applied == 1  # only fact A

        row_a = next(f for f in db.get_all_facts("default") if f.fact_id == fid_a)
        row_b = next(f for f in db.get_all_facts("default") if f.fact_id == fid_b)
        assert row_a.memory_kind is None  # reverted to its pre-backfill value
        assert row_b.memory_kind == "procedure"  # the user's edit is untouched
        assert row_b.memory_kind_source == "user"

    def test_revert_restores_fact_type_exactly(
        self, db: DatabaseManager, store: MemoryKindStore,
    ) -> None:
        """L1-07: undo must restore ``fact_type`` exactly, not only the kind
        columns. Uses a confirmed source so the apply side actually moves
        ``fact_type`` (a real backfill run never selects user/caller rows,
        per ``_REFRESH_ELIGIBLE_SOURCES``, but ``apply_batch`` must still
        guard and restore it correctly on this defensive path)."""
        fid = _fact(db, "default", "a fact")  # fact_type defaults to semantic
        rowid = _rowid(db, fid)
        run = store.create_run(
            "default", backend="rules", recipe_id="kinds-rules-v1", mode="untyped",
            requested_by="test", total_estimate=1,
        )
        run_id = run["run_id"]
        store.set_run_status(run_id, "running", expected=["queued"])

        change = KindChange(
            rowid=rowid, fact_id=fid, old_kind=None, old_source=None, old_confidence=None,
            fact_type="semantic",
            new=KindAssignment(MemoryKind.DECISION, KindSource.CALLER, None, "test-recipe"),
        )
        result = store.apply_batch(run_id, "default", [change], new_cursor=rowid, actor="test")
        assert result.applied == 1
        row = db.get_all_facts("default")[0]
        assert row.fact_type == FactType.EPISODIC  # COARSE[DECISION]

        store.set_run_status(run_id, "reverting", expected=["running"])
        revert_result = store.revert_batch(run_id, "default", limit=200)
        assert revert_result.applied == 1

        row = db.get_all_facts("default")[0]
        assert row.memory_kind is None
        assert row.fact_type == FactType.SEMANTIC  # restored exactly, not left at episodic


class TestReconcileConfirmed:
    def test_reconcile_confirmed_repairs_fact_type(
        self, db: DatabaseManager, store: MemoryKindStore,
    ) -> None:
        fid = _fact(db, "default", "decision kind, wrong fact_type", kind="decision", source="user")
        # Simulate the downgrade-window corruption LLD §6.5 repairs: a 4.1.18
        # write changed fact_type without touching the (unknown-to-it) kind.
        db.execute("UPDATE atomic_facts SET fact_type = 'semantic' WHERE fact_id = ?", (fid,))

        fixed = store.reconcile_confirmed("default")
        assert fixed == 1
        row = db.get_all_facts("default")[0]
        assert row.fact_type == FactType.EPISODIC  # COARSE[DECISION] == "episodic"
        history = db.execute(
            "SELECT * FROM memory_kind_history WHERE fact_id = ? AND origin = 'reconcile'", (fid,),
        )
        assert len(history) == 1

    def test_reconcile_confirmed_stops_at_its_time_bound(
        self, db: DatabaseManager, store: MemoryKindStore, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """L1-15: bounded by time as well as by ``limit``, so a large store
        never delays engine start — the remaining rows are picked up on a
        later boot, not processed unbounded in this one."""
        import time as time_mod

        fid_a = _fact(db, "default", "a", kind="decision", source="user")
        fid_b = _fact(db, "default", "b", kind="decision", source="user")
        db.execute(
            "UPDATE atomic_facts SET fact_type = 'semantic' WHERE fact_id IN (?, ?)",
            (fid_a, fid_b),
        )
        # started, pre-row-1 check (still within budget), pre-row-2 check
        # (over budget) — deterministic instead of racing a real clock.
        # Any later clock read (the lock-hold measurement) stays past the bound.
        import itertools
        ticks = itertools.chain([0.0, 0.0, 10.0], itertools.repeat(10.0))
        monkeypatch.setattr(time_mod, "perf_counter", lambda: next(ticks))

        fixed = store.reconcile_confirmed("default", max_seconds=1.0)
        assert fixed == 1  # stopped before the second row


class TestGDPRDiscovery:
    def test_history_rows_are_discoverable_by_gdpr(self, db: DatabaseManager) -> None:
        tables = GDPRCompliance(db)._profile_scoped_tables()  # noqa: SLF001 - exercising discovery
        assert "memory_kind_runs" in tables
        assert "memory_kind_history" in tables
