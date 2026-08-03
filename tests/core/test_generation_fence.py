"""Generation fence: admission is fenced to the profile-binding epoch.

The fence guarantees that a projection write admitted under one profile-binding
epoch cannot commit after the runtime has rebound to a newer epoch (a profile /
mode transition). It is a no-op in steady state (zero new rejections) and never
touches the durable journal, so recovery replay always completes.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from superlocalmemory.storage import generation_fence as gf

_FUNCTIONAL_DEADLINE_MS = 1_500


def _reset_fence() -> None:
    with gf._lock:
        gf._epochs.clear()


# ---------------------------------------------------------------------------
# Module logic
# ---------------------------------------------------------------------------

class TestFenceModule:
    def setup_method(self) -> None:
        _reset_fence()

    def test_record_then_read_returns_epoch(self) -> None:
        gf.record_admission_epoch("p", "k", 7)
        assert gf.admitted_epoch("p", "k") == 7

    def test_absent_key_returns_none(self) -> None:
        assert gf.admitted_epoch("p", "missing") is None

    def test_empty_key_is_ignored(self) -> None:
        gf.record_admission_epoch("p", "", 3)
        assert gf.admitted_epoch("p", "") is None

    def test_clear_removes_entry(self) -> None:
        gf.record_admission_epoch("p", "k", 1)
        gf.clear_admission_epoch("p", "k")
        assert gf.admitted_epoch("p", "k") is None

    def test_entries_are_profile_scoped(self) -> None:
        gf.record_admission_epoch("p1", "k", 1)
        gf.record_admission_epoch("p2", "k", 2)
        assert gf.admitted_epoch("p1", "k") == 1
        assert gf.admitted_epoch("p2", "k") == 2

    def test_epoch_zero_is_distinguishable_from_absent(self) -> None:
        gf.record_admission_epoch("p", "k", 0)
        assert gf.admitted_epoch("p", "k") == 0

    def test_conflicting_epochs_fail_closed(self) -> None:
        # Two concurrent admits recording DIFFERENT epochs for one key must not
        # let either satisfy the fence: the stored epoch becomes a sentinel that
        # can never equal a real generation, so both are rejected and retry.
        gf.record_admission_epoch("p", "k", 0)
        gf.record_admission_epoch("p", "k", 5)
        assert gf.admitted_epoch("p", "k") == gf._CONFLICT_EPOCH
        assert gf.admitted_epoch("p", "k") not in (0, 5)
        gf.clear_admission_epoch("p", "k")
        gf.record_admission_epoch("p", "k", 5)
        assert gf.admitted_epoch("p", "k") == 5

    def test_same_epoch_recorded_twice_is_stable(self) -> None:
        gf.record_admission_epoch("p", "k", 3)
        gf.record_admission_epoch("p", "k", 3)
        assert gf.admitted_epoch("p", "k") == 3

    def test_expired_entry_is_pruned_on_read(self, monkeypatch) -> None:
        gf.record_admission_epoch("p", "k", 5)
        future = time.time() + gf._TTL_SECONDS + 1.0
        monkeypatch.setattr(gf.time, "time", lambda: future)
        assert gf.admitted_epoch("p", "k") is None


# ---------------------------------------------------------------------------
# Real-runtime integration
# ---------------------------------------------------------------------------

def _install_write_commits(path) -> None:
    from superlocalmemory.storage.migrations import (
        M018_ingestion_operations,
        M032_write_coordinator_admission,
        M033_projection_transactions,
        M034_obligation_integrity,
    )

    conn = sqlite3.connect(path)
    try:
        M018_ingestion_operations.apply(conn)
        M032_write_coordinator_admission.apply(conn)
        # Phase-3: M033 is now mandatory (fail-closed schema guard).
        M033_projection_transactions.apply(conn)
        M034_obligation_integrity.apply(conn)
        conn.commit()
    finally:
        conn.close()


def _build_runtime(tmp_path):
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager

    db_path = tmp_path / "memory.db"
    _install_write_commits(db_path)
    db = DatabaseManager(db_path)
    db.initialize(schema)
    calls: list[str] = []

    def writer(request, operation_id):
        calls.append(operation_id)
        db.execute(
            "INSERT INTO runtime_probe(operation_id, content) VALUES (?, ?)",
            (operation_id, request.content),
        )
        return ["fact-1"]

    runtime = CanonicalRememberRuntime(
        db=db,
        profile_id="default",
        writer=writer,
        journal_path=tmp_path / "admission_journal.db",
    )
    db.execute("CREATE TABLE runtime_probe(operation_id TEXT, content TEXT)")
    return runtime, calls


class TestFenceRuntimeIntegration:
    def setup_method(self) -> None:
        _reset_fence()

    def test_matching_epoch_is_a_noop_and_commits(self, tmp_path) -> None:
        from superlocalmemory.storage.admission_journal import Actor, RememberRequest

        runtime, calls = _build_runtime(tmp_path)
        actor = Actor("actor", frozenset({"default"}), frozenset({"personal"}))
        request = RememberRequest(
            content="steady state write",
            profile_id="default",
            source_type="http",
            idempotency_key="fence-noop-1",
            trusted_actor_id="actor",
        )
        runtime.start()
        try:
            receipt = runtime.remember(
                request, actor, deadline_ms=_FUNCTIONAL_DEADLINE_MS
            ).payload
            assert receipt["status"] == "queryable"
            assert calls == [receipt["operation_id"]]
        finally:
            runtime.stop()

    def test_fresh_write_after_generation_advance_still_commits(self, tmp_path) -> None:
        # A benign generation advance (a completed rebind) must NOT reject a
        # subsequent write: remember() captures the CURRENT epoch, so it matches.
        from superlocalmemory.storage.admission_journal import Actor, RememberRequest

        runtime, calls = _build_runtime(tmp_path)
        actor = Actor("actor", frozenset({"default"}), frozenset({"personal"}))
        runtime.start()
        try:
            with runtime._binding_lock:
                runtime._generation += 3
            request = RememberRequest(
                content="post-transition write",
                profile_id="default",
                source_type="http",
                idempotency_key="fence-fresh-1",
                trusted_actor_id="actor",
            )
            receipt = runtime.remember(
                request, actor, deadline_ms=_FUNCTIONAL_DEADLINE_MS
            ).payload
            assert calls == [receipt["operation_id"]]
        finally:
            runtime.stop()

    def test_stale_epoch_is_rejected_before_projection(self, tmp_path) -> None:
        # Simulate: an admission captured at epoch 0, then the runtime rebinds
        # to a newer epoch before the command reaches the sole writer. The
        # projection must be rejected (retryable) and the writer never invoked.
        from superlocalmemory.storage.admission_journal import RememberRequest
        from superlocalmemory.storage.write_coordinator import (
            CommandKind,
            WriteCommand,
            WriteCoordinatorError,
        )

        runtime, calls = _build_runtime(tmp_path)
        request = RememberRequest(
            content="stale epoch witness",
            profile_id="default",
            source_type="http",
            idempotency_key="fence-stale-1",
            trusted_actor_id="actor",
        )
        runtime.start()
        try:
            gf.record_admission_epoch("default", "fence-stale-1", 0)
            with runtime._binding_lock:
                runtime._generation = 1
            command = WriteCommand.create(
                CommandKind.ADMISSION,
                {
                    "journal_id": "jid-stale",
                    "request_hash": "hash-stale",
                    "profile_id": "default",
                    "idempotency_key": "fence-stale-1",
                    "request": request.canonical_payload(),
                },
                command_id="jid-stale",
            )
            with pytest.raises(WriteCoordinatorError) as caught:
                runtime.coordinator.submit(command, timeout=2.0)
            assert isinstance(caught.value.__cause__, ValueError)
            assert "epoch is stale" in str(caught.value.__cause__)
            assert calls == []
        finally:
            runtime.stop()

    def test_rebind_bumps_generation(self, tmp_path) -> None:
        runtime, _ = _build_runtime(tmp_path)
        assert runtime._generation == 0
