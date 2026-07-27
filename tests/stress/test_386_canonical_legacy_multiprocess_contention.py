# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Cross-process contention contract for the 3.8.6 canonical writer.

The canonical runtime owns foreground remember admission while two independent
spawned processes represent transitional legacy writers.  The workload uses a
pytest-owned database only: no daemon, CLI, MCP endpoint, or installed SLM
state is touched.
"""

from __future__ import annotations

import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

import pytest

_FOREGROUND_WRITES = 48
_LEGACY_WRITES_PER_PROCESS = 48
_READER_THREADS = 4
_REMEMBER_DEADLINE_MS = 2_000
_PROCESS_DEADLINE_SECONDS = 8.0
_ACTOR_ID = "multiprocess-contention-daemon"


@dataclass
class _Harness:
    db_path: Path
    journal_path: Path
    runtime: Any

    def stop(self) -> None:
        self.runtime.stop()


def _new_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Harness:
    """Create a canonical runtime in a namespace inherited safely by children."""
    from superlocalmemory.core.engine_ingestion import build_immediate_admission_handler
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.migrations import (
        M018_ingestion_operations,
        M032_write_coordinator_admission,
    )

    data_dir = tmp_path / "slm-multiprocess-contention"
    data_dir.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))
    db_path = data_dir / "memory.db"
    journal_path = data_dir / "admission_journal.db"
    db = DatabaseManager(db_path)
    db.initialize(schema)
    with db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M032_write_coordinator_admission.apply(conn)
        conn.execute(
            "CREATE TABLE legacy_memory_write_events("
            "sequence INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE legacy_manager_events("
            "sequence INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
        )
    runtime = CanonicalRememberRuntime(
        db=db,
        profile_id="default",
        writer=build_immediate_admission_handler(db, profile_id="default"),
        journal_path=journal_path,
        owner_id="multiprocess-contention-runtime",
    )
    return _Harness(db_path=db_path, journal_path=journal_path, runtime=runtime)


def _legacy_memory_write_worker(
    db_path: str,
    start: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
    count: int,
) -> None:
    """Use the legacy short-lived ``memory_write`` path in a spawned process."""
    from superlocalmemory.storage.memory_write import memory_write

    started = time.monotonic()
    max_operation_seconds = 0.0
    try:
        if not start.wait(timeout=_PROCESS_DEADLINE_SECONDS):
            raise TimeoutError("parent did not release the legacy writer")
        for sequence in range(count):
            operation_started = time.monotonic()
            with memory_write(db_path) as conn:
                conn.execute(
                    "INSERT INTO legacy_memory_write_events(sequence, payload) VALUES (?, ?)",
                    (sequence, f"memory-write:{sequence}"),
                )
            max_operation_seconds = max(
                max_operation_seconds,
                time.monotonic() - operation_started,
            )
        results.put(
            {
                "worker": "memory_write",
                "ok": True,
                "elapsed_seconds": time.monotonic() - started,
                "max_operation_seconds": max_operation_seconds,
            }
        )
    except BaseException as exc:
        results.put(
            {
                "worker": "memory_write",
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def _legacy_database_manager_worker(
    db_path: str,
    start: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
    count: int,
) -> None:
    """Use a separate legacy ``DatabaseManager`` in a spawned process."""
    from superlocalmemory.storage.database import DatabaseManager

    db = DatabaseManager(db_path)
    started = time.monotonic()
    max_operation_seconds = 0.0
    try:
        if not start.wait(timeout=_PROCESS_DEADLINE_SECONDS):
            raise TimeoutError("parent did not release the legacy manager")
        for sequence in range(count):
            operation_started = time.monotonic()
            db.execute(
                "INSERT INTO legacy_manager_events(sequence, payload) VALUES (?, ?)",
                (sequence, f"database-manager:{sequence}"),
            )
            max_operation_seconds = max(
                max_operation_seconds,
                time.monotonic() - operation_started,
            )
        results.put(
            {
                "worker": "database_manager",
                "ok": True,
                "elapsed_seconds": time.monotonic() - started,
                "max_operation_seconds": max_operation_seconds,
            }
        )
    except BaseException as exc:
        results.put(
            {
                "worker": "database_manager",
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        db.close()


def _actor():
    from superlocalmemory.storage.admission_journal import Actor

    return Actor(_ACTOR_ID, frozenset({"default"}), frozenset({"personal"}))


def _remember_request(sequence: int):
    from superlocalmemory.storage.admission_journal import RememberRequest

    return RememberRequest(
        content=(
            f"Multiprocess canonical fact {sequence}: foreground admission "
            "remains queryable while legacy writers contend."
        ),
        profile_id="default",
        source_type="stress-multiprocess",
        idempotency_key=f"multiprocess-contention:{sequence}",
        trusted_actor_id=_ACTOR_ID,
    )


def _strict_read_count(db_path: Path, sql: str, params: tuple[object, ...] = ()) -> int:
    """Read through the physical mode=ro/query_only path only."""
    from superlocalmemory.storage.read_connection import ReadConnectionFactory

    with ReadConnectionFactory(db_path, timeout_ms=250).snapshot() as conn:
        row = conn.execute(sql, params).fetchone()
    assert row is not None
    return int(row[0])


def _assert_no_busy(errors: list[BaseException | str]) -> None:
    leaked = [
        error for error in errors if "locked" in str(error).lower() or "busy" in str(error).lower()
    ]
    assert not leaked, f"SQLite contention leaked to a caller: {leaked!r}"


def test_386_canonical_remember_survives_legacy_multiprocess_contention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical writes, legacy DML, and strict RO FTS snapshots all finish.

    The individual 2s remember deadline is the foreground service contract;
    legacy child processes have an 8s total cap.  Counts prove that contention
    caused neither duplicated work nor silently lost writes.
    """
    harness = _new_harness(tmp_path, monkeypatch)
    harness.runtime.start()
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    children = [
        context.Process(
            target=_legacy_memory_write_worker,
            args=(
                str(harness.db_path),
                start,
                results,
                _LEGACY_WRITES_PER_PROCESS,
            ),
        ),
        context.Process(
            target=_legacy_database_manager_worker,
            args=(
                str(harness.db_path),
                start,
                results,
                _LEGACY_WRITES_PER_PROCESS,
            ),
        ),
    ]
    foreground_errors: list[BaseException] = []
    reader_errors: list[BaseException] = []
    foreground_elapsed: list[float] = []
    writes_complete = threading.Event()
    errors_lock = threading.Lock()
    try:
        for child in children:
            child.start()

        def foreground_remember(sequence: int) -> None:
            started = time.monotonic()
            try:
                receipt = harness.runtime.remember(
                    _remember_request(sequence),
                    _actor(),
                    deadline_ms=_REMEMBER_DEADLINE_MS,
                )
                assert receipt.payload["status"] == "queryable"
            except BaseException as exc:
                with errors_lock:
                    foreground_errors.append(exc)
            finally:
                with errors_lock:
                    foreground_elapsed.append(time.monotonic() - started)

        def strict_reader() -> int:
            snapshots = 0
            try:
                while not writes_complete.is_set():
                    _strict_read_count(
                        harness.db_path,
                        "SELECT COUNT(*) FROM atomic_facts",
                    )
                    _strict_read_count(
                        harness.db_path,
                        "SELECT COUNT(*) FROM atomic_facts_fts WHERE atomic_facts_fts MATCH ?",
                        ("multiprocess OR canonical",),
                    )
                    snapshots += 1
            except BaseException as exc:
                with errors_lock:
                    reader_errors.append(exc)
            return snapshots

        start.set()
        with ThreadPoolExecutor(max_workers=_READER_THREADS) as reader_pool:
            with ThreadPoolExecutor(max_workers=16) as writer_pool:
                foreground_futures = [
                    writer_pool.submit(foreground_remember, sequence)
                    for sequence in range(_FOREGROUND_WRITES)
                ]
                snapshot_futures = [
                    reader_pool.submit(strict_reader) for _ in range(_READER_THREADS)
                ]
                for future in foreground_futures:
                    future.result(timeout=_PROCESS_DEADLINE_SECONDS)
                writes_complete.set()
                snapshot_counts = [
                    future.result(timeout=_PROCESS_DEADLINE_SECONDS) for future in snapshot_futures
                ]

        for child in children:
            child.join(timeout=_PROCESS_DEADLINE_SECONDS)
            assert child.exitcode == 0, f"{child.name} failed with {child.exitcode}"
        child_results = []
        for _ in children:
            try:
                child_results.append(results.get(timeout=1.0))
            except Empty as exc:
                pytest.fail(f"legacy worker did not report a result: {exc}")

        child_errors = [str(result["error"]) for result in child_results if not result["ok"]]
        assert not foreground_errors, f"foreground remember failures: {foreground_errors!r}"
        assert not reader_errors, f"strict read failures: {reader_errors!r}"
        assert not child_errors, f"legacy child failures: {child_errors!r}"
        _assert_no_busy([*foreground_errors, *reader_errors, *child_errors])
        assert foreground_elapsed
        assert max(foreground_elapsed) < _REMEMBER_DEADLINE_MS / 1_000 + 0.5
        assert all(
            result["elapsed_seconds"] < _PROCESS_DEADLINE_SECONDS
            and result["max_operation_seconds"] < _PROCESS_DEADLINE_SECONDS
            for result in child_results
        )
        assert sum(snapshot_counts) > 0
        assert (
            _strict_read_count(
                harness.db_path,
                "SELECT COUNT(*) FROM atomic_facts",
            )
            == _FOREGROUND_WRITES
        )
        assert (
            _strict_read_count(
                harness.db_path,
                "SELECT COUNT(*) FROM write_commits",
            )
            == _FOREGROUND_WRITES
        )
        assert (
            _strict_read_count(
                harness.db_path,
                "SELECT COUNT(*) FROM atomic_facts_fts WHERE atomic_facts_fts MATCH ?",
                ("multiprocess OR canonical",),
            )
            == _FOREGROUND_WRITES
        )
        assert (
            _strict_read_count(
                harness.db_path,
                "SELECT COUNT(*) FROM legacy_memory_write_events",
            )
            == _LEGACY_WRITES_PER_PROCESS
        )
        assert (
            _strict_read_count(
                harness.db_path,
                "SELECT COUNT(*) FROM legacy_manager_events",
            )
            == _LEGACY_WRITES_PER_PROCESS
        )
    finally:
        writes_complete.set()
        for child in children:
            if child.is_alive():
                child.join(timeout=1.0)
            if child.is_alive():
                child.terminate()
                child.join(timeout=1.0)
        harness.stop()
