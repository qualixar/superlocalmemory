# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""3.8.6 contention contract for the daemon-owned canonical write coordinator."""

from __future__ import annotations

import multiprocessing
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


def _competing_process_claim(db_path: str, result_queue) -> None:
    """A CLI/MCP fallback may not become a second canonical writer."""
    from superlocalmemory.storage.write_coordinator import WriteCoordinator

    started = time.monotonic()
    coordinator = WriteCoordinator(Path(db_path), owner_id="mcp-fallback")
    claimed = coordinator.claim_ownership()
    try:
        result_queue.put((claimed, time.monotonic() - started))
    finally:
        if claimed:
            coordinator.release_ownership()


def test_386_daemon_owner_serializes_threads_and_rejects_second_process(tmp_path) -> None:
    """One owner, many callers: no SQLITE_BUSY and no unbounded wait.

    The daemon owns the canonical path.  Concurrent in-process work is sent to
    the same coordinator; a separate process representing a CLI/MCP fallback
    is rejected quickly instead of opening its own writable ``memory.db``
    connection.  This is the regression that process-local ``RLock`` tests
    cannot catch.
    """
    from superlocalmemory.storage.write_coordinator import WriteCoordinator

    db_path = tmp_path / "memory.db"
    coordinator = WriteCoordinator(db_path, owner_id="daemon-386")
    assert coordinator.claim_ownership() is True
    try:
        coordinator.execute(
            "CREATE TABLE admissions (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
            priority="foreground",
            timeout=0.5,
        )

        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        contender = context.Process(
            target=_competing_process_claim,
            args=(str(db_path), result_queue),
        )
        contender.start()
        contender.join(timeout=5)
        assert contender.exitcode == 0
        claimed, elapsed = result_queue.get(timeout=1)
        assert claimed is False
        assert elapsed < 0.5

        def remember(sequence: int) -> float:
            started = time.monotonic()
            coordinator.execute(
                "INSERT INTO admissions(value) VALUES (?)",
                (f"remember-{sequence}",),
                priority="foreground",
                timeout=0.5,
            )
            return time.monotonic() - started

        with ThreadPoolExecutor(max_workers=12) as pool:
            elapsed_times = list(pool.map(remember, range(120)))

        rows = coordinator.execute("SELECT COUNT(*) FROM admissions", timeout=0.5)
        assert rows[0][0] == 120
        assert max(elapsed_times) < 1.5
    finally:
        coordinator.release_ownership()


def test_386_owner_release_and_input_contract_are_explicit(tmp_path) -> None:
    """A stopped daemon releases ownership and never accepts ambiguous commands.

    This closes a gap in the original contention test: its ``finally`` block
    called ``release_ownership`` but never verified that a replacement daemon
    could claim the same path.  It also locks the bounded public API against
    empty SQL, invalid scheduling lanes, and writes attempted before ownership.
    """
    from superlocalmemory.storage.write_coordinator import (
        CommandKind,
        OwnershipRequiredError,
        WriteCommand,
        WriteCoordinator,
    )

    db_path = tmp_path / "memory.db"
    with pytest.raises(ValueError, match="at least one"):
        WriteCoordinator(db_path, max_queue_depth=0)

    coordinator = WriteCoordinator(db_path, owner_id="daemon-first")
    assert coordinator.db_path == db_path.resolve()
    assert coordinator.owner_id == "daemon-first"
    with pytest.raises(ValueError, match="command_id"):
        WriteCommand.create(CommandKind.ADMISSION, command_id="")
    with pytest.raises(OwnershipRequiredError):
        coordinator.execute("SELECT 1")

    assert coordinator.claim_ownership() is True
    assert coordinator.claim_ownership() is True
    try:
        with pytest.raises(ValueError, match="non-empty"):
            coordinator.execute("", timeout=0.5)
        with pytest.raises(ValueError, match="greater than zero"):
            coordinator.execute("SELECT 1", timeout=0)
        with pytest.raises(ValueError, match="unknown coordinator priority"):
            coordinator.execute("SELECT 1", priority="urgent", timeout=0.5)  # type: ignore[arg-type]
    finally:
        coordinator.release_ownership()

    replacement = WriteCoordinator(db_path, owner_id="daemon-replacement")
    assert replacement.claim_ownership() is True
    replacement.release_ownership()


def test_386_concurrent_start_constructs_exactly_one_worker(
    tmp_path,
    monkeypatch,
) -> None:
    """Concurrent first submissions cannot launch multiple writer threads."""
    import superlocalmemory.storage.write_coordinator as coordinator_module
    from superlocalmemory.storage.write_coordinator import WriteCoordinator

    coordinator = WriteCoordinator(tmp_path / "memory.db", owner_id="start-race")
    assert coordinator.claim_ownership()
    real_thread = threading.Thread
    start_gate = threading.Event()
    constructed_workers: list[threading.Thread] = []
    errors: list[BaseException] = []
    error_lock = threading.Lock()

    def call_start() -> None:
        start_gate.wait()
        try:
            coordinator.start()
        except BaseException as exc:  # pragma: no cover - regression witness
            with error_lock:
                errors.append(exc)

    # Build the callers before replacing the constructor used by the module.
    callers = [real_thread(target=call_start) for _ in range(8)]

    def delayed_thread(*args, **kwargs):
        # Before the fix, every caller could observe _worker=None during this
        # delay and construct its own connection-owning thread.
        time.sleep(0.02)
        worker = real_thread(*args, **kwargs)
        constructed_workers.append(worker)
        return worker

    monkeypatch.setattr(coordinator_module.threading, "Thread", delayed_thread)
    try:
        for caller in callers:
            caller.start()
        start_gate.set()
        for caller in callers:
            caller.join(timeout=2)

        assert all(not caller.is_alive() for caller in callers)
        assert errors == []
        assert len(constructed_workers) == 1
        assert coordinator._worker is constructed_workers[0]
    finally:
        coordinator.release_ownership()


def test_386_stalled_handler_keeps_lease_until_worker_exits_then_releases_it(tmp_path) -> None:
    """A timed-out shutdown cannot expose a live writer to a replacement daemon.

    The handler stays active for slightly longer than the public two-second
    shutdown budget.  ``release_ownership`` must report that bounded shutdown
    failed, reject a replacement while the handler is live, then release the
    lease automatically once the writer thread has actually exited.
    """
    from superlocalmemory.storage.migrations import M032_write_coordinator_admission
    from superlocalmemory.storage.write_coordinator import (
        CommandKind,
        WriteCommand,
        WriteCoordinator,
        WriteCoordinatorError,
        WriteResult,
    )

    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(db_path)
    try:
        M032_write_coordinator_admission.apply(conn)
        conn.commit()
    finally:
        conn.close()

    coordinator = WriteCoordinator(db_path, owner_id="stalled-handler-owner")
    replacement = WriteCoordinator(db_path, owner_id="replacement-owner")
    entered_handler = threading.Event()
    release_handler = threading.Event()
    outcome: list[object] = []
    assert coordinator.claim_ownership() is True

    def stalled_handler(_conn, _capability, command):
        entered_handler.set()
        assert release_handler.wait(timeout=5), "test handler was not released"
        return WriteResult.from_receipt(
            command,
            {"operation_id": "operation:stalled-handler"},
        )

    coordinator.register_handler(CommandKind.ADMISSION, stalled_handler)
    command = WriteCommand.create(
        CommandKind.ADMISSION,
        {
            "journal_id": "journal:stalled-handler",
            "request_hash": "hash:stalled-handler",
            "profile_id": "default",
            "idempotency_key": "idempotency:stalled-handler",
        },
    )

    def submit() -> None:
        try:
            outcome.append(coordinator.submit(command, timeout=5))
        except BaseException as exc:  # pragma: no cover - regression witness
            outcome.append(exc)

    submitter = threading.Thread(target=submit)
    submitter.start()
    assert entered_handler.wait(timeout=1)
    worker = coordinator._worker
    assert worker is not None
    try:
        started = time.monotonic()
        with pytest.raises(WriteCoordinatorError, match="did not stop"):
            coordinator.release_ownership()
        elapsed = time.monotonic() - started
        assert elapsed >= 1.9
        assert coordinator._ownership_context is not None
        assert worker.is_alive()
        assert replacement.claim_ownership() is False
        with pytest.raises(WriteCoordinatorError, match="shutting down"):
            coordinator.claim_ownership()
        with pytest.raises(WriteCoordinatorError, match="stopping"):
            coordinator.start()
        time.sleep(0.15)
        assert worker.is_alive(), "handler did not remain stalled past the shutdown budget"

        release_handler.set()
        submitter.join(timeout=2)
        assert not submitter.is_alive()
        assert len(outcome) == 1 and not isinstance(outcome[0], BaseException)
        worker.join(timeout=2)
        assert not worker.is_alive()

        deadline = time.monotonic() + 2
        while not replacement.claim_ownership() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert replacement._ownership_context is not None
    finally:
        release_handler.set()
        submitter.join(timeout=2)
        if coordinator._ownership_context is not None:
            coordinator.release_ownership()
        if replacement._ownership_context is not None:
            replacement.release_ownership()


def test_386_coordinator_serializes_with_transitional_memory_write_callers(
    tmp_path,
) -> None:
    """Typed admission and legacy in-process writers share one path lock.

    The 3.8.6 migration is deliberately incremental: background/control
    writers still using ``memory_write`` must not race the coordinator's
    connection while those paths move behind typed commands.
    """
    from superlocalmemory.storage.memory_write import memory_write
    from superlocalmemory.storage.write_coordinator import WriteCoordinator
    from superlocalmemory.storage.write_lock import get_write_lock

    db_path = tmp_path / "memory.db"
    coordinator = WriteCoordinator(db_path, owner_id="mixed-writers")
    assert coordinator.claim_ownership()
    errors: list[BaseException] = []
    errors_lock = threading.Lock()
    try:
        coordinator.execute(
            "CREATE TABLE mixed_writes (value TEXT NOT NULL)",
            timeout=0.5,
        )
        assert coordinator._process_write_lock is get_write_lock(db_path)

        def write(sequence: int) -> None:
            try:
                if sequence % 2:
                    coordinator.execute(
                        "INSERT INTO mixed_writes(value) VALUES (?)",
                        (f"coordinator:{sequence}",),
                        timeout=2.0,
                    )
                else:
                    with memory_write(db_path) as conn:
                        conn.execute(
                            "INSERT INTO mixed_writes(value) VALUES (?)",
                            (f"legacy:{sequence}",),
                        )
            except BaseException as exc:
                with errors_lock:
                    errors.append(exc)

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(write, range(200)))

        assert not errors
        assert (
            coordinator.execute(
                "SELECT COUNT(*) FROM mixed_writes",
                timeout=0.5,
            )[0][0]
            == 200
        )
    finally:
        coordinator.release_ownership()
