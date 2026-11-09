# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A write whose caller has given up stops holding memory.db's write lock.

The coordinator already rolls back a command that ran past its deadline, but
only after the command's statement finished. Live daemon, 13 Oct 2026: an
accepted remember retried every 5 s (WriteDeadlineExceededError), each attempt
re-running a slow statement to the end inside BEGIN IMMEDIATE while holding the
process write lock -- work that was always going to be rolled back -- and every
other writer (mesh heartbeats, events, the materializer) waited behind it. The
statement is now interrupted at the deadline, so the lock is held for at most
the caller's budget, not for the statement's whole run.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest
from tests.storage.test_write_coordinator import _admission_payload, _install_write_commits

# Tens of seconds of VM work on any machine this suite runs on.
_SLOW = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c "
         "WHERE x < 400000000) SELECT count(*) FROM c")


def test_an_expired_command_releases_the_write_lock_at_its_deadline(tmp_path) -> None:
    from superlocalmemory.storage.write_coordinator import (
        CommandKind,
        WriteCommand,
        WriteCoordinator,
        WriteDeadlineExceededError,
        WriteResult,
    )
    from superlocalmemory.storage.write_lock import get_write_lock

    db_path = tmp_path / "memory.db"
    _install_write_commits(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE admissions (value TEXT NOT NULL)")
    coordinator = WriteCoordinator(db_path)
    assert coordinator.claim_ownership()
    started = threading.Event()
    try:
        def slow_handler(conn, _capability, command):
            conn.execute("INSERT INTO admissions(value) VALUES ('must-rollback')")
            started.set()
            conn.execute(_SLOW).fetchall()
            return WriteResult.from_receipt(command, {"operation_id": "operation:slow"})

        coordinator.register_handler(CommandKind.ADMISSION, slow_handler)
        t0 = time.monotonic()
        with pytest.raises(WriteDeadlineExceededError):
            coordinator.submit(
                WriteCommand.create(CommandKind.ADMISSION, _admission_payload("slow")),
                timeout=2.0,  # time for the worker to reach the handler on a busy runner
            )
        assert started.is_set()
        lock = get_write_lock(db_path)
        assert lock.acquire(timeout=60.0), "the expired write kept the process write lock"
        lock.release()
        held_for = time.monotonic() - t0
        assert held_for < 15.0, held_for  # the statement alone runs for tens of seconds

        probe = sqlite3.connect(db_path, timeout=5.0)
        try:
            probe.execute("BEGIN IMMEDIATE")  # SQLite's write lock is free too
            probe.rollback()
        finally:
            probe.close()
        assert coordinator.execute("SELECT COUNT(*) FROM admissions")[0][0] == 0
        assert coordinator.execute("SELECT COUNT(*) FROM write_commits")[0][0] == 0
    finally:
        coordinator.release_ownership()


def test_a_command_inside_its_deadline_is_not_interrupted(tmp_path) -> None:
    from superlocalmemory.storage.write_coordinator import (
        CommandKind,
        WriteCommand,
        WriteCoordinator,
        WriteResult,
    )

    db_path = tmp_path / "memory.db"
    _install_write_commits(db_path)
    coordinator = WriteCoordinator(db_path)
    assert coordinator.claim_ownership()
    try:
        def handler(conn, _capability, command):
            total = conn.execute("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 "
                                 "FROM c WHERE x < 20000) SELECT count(*) FROM c").fetchone()[0]
            return WriteResult.from_receipt(command, {"operation_id": f"operation:{total}"})

        coordinator.register_handler(CommandKind.ADMISSION, handler)
        result = coordinator.submit(
            WriteCommand.create(CommandKind.ADMISSION, _admission_payload("ok")), timeout=30.0)
        assert result.receipt["operation_id"] == "operation:20000"
    finally:
        coordinator.release_ownership()
