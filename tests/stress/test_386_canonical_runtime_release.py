# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Release-stress contracts for the 3.8.6 canonical writer.

These tests deliberately use a fresh ``SLM_DATA_DIR`` and direct temporary
database paths.  They never invoke the CLI, MCP, a public daemon port, or a
developer's installed SuperLocalMemory state.  The aim is deterministic
ordering/integrity evidence, not a hardware-dependent throughput benchmark.
"""

from __future__ import annotations

import multiprocessing
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_REQUEST_COUNT = 128
_UNIQUE_REQUEST_COUNT = _REQUEST_COUNT // 2
_ACTOR_ID = "release-stress-daemon"


@dataclass
class _CanonicalHarness:
    """One fully isolated runtime backed by the real immediate projection."""

    db: Any
    runtime: Any
    db_path: Path
    journal_path: Path

    def stop(self) -> None:
        self.runtime.stop()


def _new_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _CanonicalHarness:
    """Install the real schemas and runtime in a fixture-owned namespace."""
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.migrations import (
        M018_ingestion_operations,
        M032_write_coordinator_admission,
        M033_projection_transactions,  # required: _record_projection_obligations fail-closed
        M034_obligation_integrity,      # required: obligation FK + index
    )

    data_dir = tmp_path / "slm-data"
    data_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))
    db_path = data_dir / "memory.db"
    journal_path = data_dir / "admission_journal.db"
    db = DatabaseManager(db_path)
    db.initialize(schema)
    with db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M032_write_coordinator_admission.apply(conn)
        M033_projection_transactions.apply(conn)
        M034_obligation_integrity.apply(conn)
    return _open_runtime(db_path, journal_path, owner_id="release-stress-runtime")


def _open_runtime(db_path: Path, journal_path: Path, *, owner_id: str) -> _CanonicalHarness:
    """Open an existing canonical database without replaying its setup DDL."""
    from superlocalmemory.core.engine_ingestion import build_immediate_admission_handler
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage.database import DatabaseManager

    db = DatabaseManager(db_path)
    runtime = CanonicalRememberRuntime(
        db=db,
        profile_id="default",
        writer=build_immediate_admission_handler(db, profile_id="default"),
        journal_path=journal_path,
        owner_id=owner_id,
    )
    return _CanonicalHarness(db, runtime, db_path, journal_path)


def _actor():
    from superlocalmemory.storage.admission_journal import Actor

    return Actor(_ACTOR_ID, frozenset({"default"}), frozenset({"personal"}))


def _request(sequence: int, *, key: str | None = None):
    from superlocalmemory.storage.admission_journal import RememberRequest

    return RememberRequest(
        content=(
            f"Release stress fact {sequence}: canonical admission remains durable, "
            "queryable, and independent of model enrichment."
        ),
        profile_id="default",
        source_type="release-stress",
        idempotency_key=key or f"release-stress:{sequence}",
        trusted_actor_id=_ACTOR_ID,
        session_id="release-stress-session",
        session_date="2026-07-27",
    )


def _crash_admission_process(
    db_path: str,
    journal_path: str,
    window: str,
) -> None:
    """Hard-exit at a named admission window without running cleanup."""
    from superlocalmemory.core.remember_admission import RememberAdmissionCommand
    from superlocalmemory.core.remember_runtime import _CoordinatorAdapter

    harness = _open_runtime(
        Path(db_path),
        Path(journal_path),
        owner_id=f"release-hard-crash-{window}",
    )
    harness.runtime.start()
    request = _request(386, key=f"release-hard-crash:{window}")
    prepared = harness.runtime.journal.prepare(request, _actor())
    if window == "after_dispatch":
        harness.runtime.journal.mark_dispatched(prepared.journal_id)
    elif window == "after_canonical_commit":
        _CoordinatorAdapter(harness.runtime.coordinator).submit(
            RememberAdmissionCommand.from_prepared(prepared, request),
            wait_ms=2_000,
        )
    else:  # pragma: no cover - parent parametrization owns the contract
        raise ValueError(window)
    os._exit(97)


def _assert_no_busy(errors: list[BaseException]) -> None:
    busy = [
        error
        for error in errors
        if "locked" in str(error).lower() or "busy" in str(error).lower()
    ]
    assert not busy, f"canonical path leaked SQLite contention: {busy!r}"


def _scalar_read(db_path: Path, sql: str, params: tuple[object, ...] = ()) -> int:
    """Read a scalar only through the strict read-only query boundary."""
    from superlocalmemory.storage.read_connection import ReadConnectionFactory

    with ReadConnectionFactory(db_path, timeout_ms=250).snapshot() as conn:
        row = conn.execute(sql, params).fetchone()
    assert row is not None
    return int(row[0])


def test_release_386_128_concurrent_remembers_are_exactly_once_and_queryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """128 daemon submissions create one fact and receipt per unique request.

    The production deadline is the bounded coordination contract.  This test
    does not assert elapsed wall-clock time: each caller is instead subject to
    the same two-second API budget that the daemon exposes to clients.
    """
    harness = _new_runtime(tmp_path, monkeypatch)
    harness.runtime.start()
    errors: list[BaseException] = []
    receipts: list[tuple[int, dict[str, object]]] = []
    receipt_lock = threading.Lock()
    try:
        def submit(sequence: int) -> None:
            try:
                unique_request = sequence % _UNIQUE_REQUEST_COUNT
                receipt = harness.runtime.remember(
                    _request(
                        unique_request,
                        key=f"release-stress:duplicate:{unique_request}",
                    ),
                    _actor(),
                    deadline_ms=2_000,
                )
                with receipt_lock:
                    receipts.append((unique_request, receipt.payload))
            except BaseException as exc:  # Assertion below retains every failure.
                with receipt_lock:
                    errors.append(exc)

        with ThreadPoolExecutor(max_workers=32) as pool:
            list(pool.map(submit, range(_REQUEST_COUNT)))

        assert not errors, f"concurrent remember failures: {errors!r}"
        _assert_no_busy(errors)
        assert len(receipts) == _REQUEST_COUNT
        assert all(receipt["status"] == "queryable" for _, receipt in receipts)
        fact_ids_by_request: dict[int, set[str]] = {}
        commits_by_request: dict[int, set[int]] = {}
        for unique_request, receipt in receipts:
            fact_ids_by_request.setdefault(unique_request, set()).add(
                str(receipt["fact_ids"][0])
            )
            commits_by_request.setdefault(unique_request, set()).add(
                int(receipt["commit_sequence"])
            )
        assert set(fact_ids_by_request) == set(range(_UNIQUE_REQUEST_COUNT))
        assert all(len(ids) == 1 for ids in fact_ids_by_request.values())
        assert all(len(commits) == 1 for commits in commits_by_request.values())
        assert (
            _scalar_read(harness.db_path, "SELECT COUNT(*) FROM atomic_facts")
            == _UNIQUE_REQUEST_COUNT
        )
        assert (
            _scalar_read(harness.db_path, "SELECT COUNT(*) FROM write_commits")
            == _UNIQUE_REQUEST_COUNT
        )
        assert harness.runtime.journal.count() == _UNIQUE_REQUEST_COUNT
        assert _scalar_read(
            harness.db_path,
            "SELECT COUNT(*) FROM atomic_facts_fts WHERE atomic_facts_fts MATCH ?",
            ("release OR canonical",),
        ) == _UNIQUE_REQUEST_COUNT
    finally:
        harness.stop()


def test_release_386_strict_read_snapshots_remain_available_while_remembers_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent FTS snapshots never mutate or contend with the writer."""
    from superlocalmemory.storage.read_connection import ReadConnectionFactory

    harness = _new_runtime(tmp_path, monkeypatch)
    harness.runtime.start()
    writer_errors: list[BaseException] = []
    reader_errors: list[BaseException] = []
    write_done = threading.Event()
    try:
        with ReadConnectionFactory(harness.db_path, timeout_ms=250).snapshot() as conn:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("CREATE TABLE forbidden_read_write(value TEXT)")

        def write_many() -> None:
            try:
                for sequence in range(_REQUEST_COUNT):
                    harness.runtime.remember(
                        _request(sequence), _actor(), deadline_ms=2_000,
                    )
            except BaseException as exc:
                writer_errors.append(exc)
            finally:
                write_done.set()

        def snapshot_until_done() -> int:
            snapshots = 0
            try:
                while not write_done.is_set():
                    with ReadConnectionFactory(harness.db_path, timeout_ms=250).snapshot() as conn:
                        conn.execute("SELECT COUNT(*) FROM atomic_facts").fetchone()
                        conn.execute(
                            "SELECT COUNT(*) FROM atomic_facts_fts "
                            "WHERE atomic_facts_fts MATCH ?",
                            ("release",),
                        ).fetchone()
                    snapshots += 1
            except BaseException as exc:
                reader_errors.append(exc)
            return snapshots

        writer = threading.Thread(target=write_many, name="release-stress-writer")
        writer.start()
        with ThreadPoolExecutor(max_workers=4) as readers:
            snapshot_counts = list(readers.map(lambda _: snapshot_until_done(), range(4)))
        writer.join(timeout=5)
        assert not writer.is_alive(), "canonical writer did not complete its bounded workload"

        assert not writer_errors, f"remember errors during snapshot contention: {writer_errors!r}"
        assert not reader_errors, (
            f"read-only snapshots failed during remember flow: {reader_errors!r}"
        )
        _assert_no_busy([*writer_errors, *reader_errors])
        assert sum(snapshot_counts) > 0
        assert _scalar_read(harness.db_path, "SELECT COUNT(*) FROM atomic_facts") == _REQUEST_COUNT
        assert _scalar_read(
            harness.db_path,
            "SELECT COUNT(*) FROM atomic_facts_fts WHERE atomic_facts_fts MATCH ?",
            ("release",),
        ) == _REQUEST_COUNT
    finally:
        harness.stop()


def test_release_386_restart_replays_128_prepared_requests_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process crash before dispatch recovers every journaled request once."""
    harness = _new_runtime(tmp_path, monkeypatch)
    actor = _actor()
    for sequence in range(_REQUEST_COUNT):
        harness.runtime.journal.prepare(_request(sequence), actor)

    harness.runtime.start()
    try:
        assert _scalar_read(harness.db_path, "SELECT COUNT(*) FROM atomic_facts") == _REQUEST_COUNT
        assert _scalar_read(harness.db_path, "SELECT COUNT(*) FROM write_commits") == _REQUEST_COUNT
        assert all(
            harness.runtime.journal.get_by_idempotency_key(
                "default", f"release-stress:{sequence}"
            ).state
            == "committed"
            for sequence in range(_REQUEST_COUNT)
        )
    finally:
        harness.stop()

    restarted = _open_runtime(
        harness.db_path,
        harness.journal_path,
        owner_id="release-stress-restarted-runtime",
    )
    restarted.runtime.start()
    try:
        assert restarted.runtime.replay_pending() == 0
        assert (
            _scalar_read(restarted.db_path, "SELECT COUNT(*) FROM atomic_facts")
            == _REQUEST_COUNT
        )
        assert (
            _scalar_read(restarted.db_path, "SELECT COUNT(*) FROM write_commits")
            == _REQUEST_COUNT
        )
    finally:
        restarted.stop()


@pytest.mark.parametrize("window", ("after_dispatch", "after_canonical_commit"))
def test_release_386_hard_process_crash_recovers_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    window: str,
) -> None:
    """SIGKILL-equivalent exit at both durable admission gaps replays once."""
    harness = _new_runtime(tmp_path, monkeypatch)
    process = multiprocessing.get_context("spawn").Process(
        target=_crash_admission_process,
        args=(str(harness.db_path), str(harness.journal_path), window),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 97

    request = _request(386, key=f"release-hard-crash:{window}")
    harness.runtime.start()
    try:
        receipt = harness.runtime.remember(
            request,
            _actor(),
            deadline_ms=2_000,
        ).payload
        assert receipt["status"] == "queryable"
        assert len(receipt["fact_ids"]) == 1
        assert _scalar_read(
            harness.db_path,
            "SELECT COUNT(*) FROM atomic_facts WHERE profile_id = ?",
            ("default",),
        ) == 1
        assert _scalar_read(
            harness.db_path,
            "SELECT COUNT(*) FROM write_commits WHERE profile_id = ?",
            ("default",),
        ) == 1
        entry = harness.runtime.journal.get_by_idempotency_key(
            "default",
            request.idempotency_key,
        )
        assert entry is not None
        assert entry.state == "committed"
        assert entry.original_receipt == receipt
        assert harness.runtime.replay_pending() == 0
    finally:
        harness.stop()


def test_release_386_foreground_work_preempts_queued_background_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One foreground admission is ordered before queued background writes.

    A test-only gate holds the first background execution after it is dequeued.
    This creates a deterministic load state without a slow SQL expression or a
    machine-dependent latency assertion.
    """
    from superlocalmemory.storage.write_coordinator import Lane, WriteCoordinator

    data_dir = tmp_path / "slm-data"
    data_dir.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))
    coordinator = WriteCoordinator(data_dir / "memory.db", owner_id="release-stress-priority")
    assert coordinator.claim_ownership()
    first_background_started = threading.Event()
    release_first_background = threading.Event()
    original_execute_item = coordinator._execute_item
    original_enqueue = coordinator._enqueue
    first_background = True
    execute_lock = threading.Lock()

    def gated_execute(conn, item) -> None:
        nonlocal first_background
        with execute_lock:
            should_gate = first_background and item.lane is Lane.BACKGROUND
            if should_gate:
                first_background = False
        if should_gate:
            first_background_started.set()
            assert release_first_background.wait(timeout=2), "test gate was not released"
        original_execute_item(conn, item)

    foreground_enqueued = threading.Event()

    def observed_enqueue(item) -> None:
        original_enqueue(item)
        if item.lane is Lane.FOREGROUND:
            foreground_enqueued.set()

    monkeypatch.setattr(coordinator, "_execute_item", gated_execute)
    monkeypatch.setattr(coordinator, "_enqueue", observed_enqueue)
    try:
        coordinator.execute(
            "CREATE TABLE events(ordinal INTEGER PRIMARY KEY AUTOINCREMENT, "
            "sequence INTEGER UNIQUE NOT NULL, lane TEXT NOT NULL)"
        )
        first = threading.Thread(
            target=lambda: coordinator.execute(
                "INSERT INTO events(sequence, lane) VALUES (0, 'background')",
                priority="background",
                timeout=2,
            ),
        )
        first.start()
        assert first_background_started.wait(timeout=2)

        queued: list[threading.Thread] = []
        for sequence in range(1, 17):
            thread = threading.Thread(
                target=lambda value=sequence: coordinator.execute(
                    "INSERT INTO events(sequence, lane) VALUES (?, 'background')",
                    (value,),
                    priority="background",
                    timeout=2,
                ),
            )
            thread.start()
            queued.append(thread)
        deadline = threading.Event()
        # Wait until every bounded background submission has joined the queue;
        # the private count is a test synchronisation probe, not an assertion
        # about the public API.
        for _ in range(200):
            if coordinator._queued_count >= len(queued):
                deadline.set()
                break
            threading.Event().wait(0.005)
        assert deadline.is_set(), "background load did not reach the coordinator queue"

        foreground = threading.Thread(
            target=lambda: coordinator.execute(
                "INSERT INTO events(sequence, lane) VALUES (999, 'foreground')",
                priority="foreground",
                timeout=2,
            ),
        )
        foreground.start()
        assert foreground_enqueued.wait(timeout=2), "foreground write was not queued"
        release_first_background.set()
        first.join(timeout=2)
        foreground.join(timeout=2)
        for thread in queued:
            thread.join(timeout=2)
        assert not first.is_alive() and not foreground.is_alive()
        assert not any(thread.is_alive() for thread in queued)

        rows = coordinator.execute("SELECT lane FROM events ORDER BY rowid")
        lanes = [str(row[0]) for row in rows]
        assert lanes[0] == "background"
        assert lanes[1] == "foreground"
        assert lanes.count("background") == 17
    finally:
        release_first_background.set()
        coordinator.release_ownership()


def test_release_386_stuck_embedding_child_is_reaped_during_bounded_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shutdown path kills a non-responsive model child without descendants."""
    import subprocess
    import sys

    from superlocalmemory.core.embeddings import EmbeddingService

    data_dir = tmp_path / "slm-data"
    data_dir.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    service = EmbeddingService.__new__(EmbeddingService)
    service._lock = threading.Lock()
    service._idle_timer = None
    service._worker_ready = True
    service._owns_worker_lock = False
    service._http_client = None
    service._worker_proc = child
    try:
        service.shutdown(timeout=0.05)
        assert child.wait(timeout=2) is not None
        assert service._worker_proc is None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=2)
