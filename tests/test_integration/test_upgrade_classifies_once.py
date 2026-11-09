# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""An upgraded store's older memories are classified once, by the daemon itself.

Composed path: a REAL daemon subprocess on a private port and data folder.
Memories are saved, then made "older" (their kind removed, exactly what a
store written before 4.1.19 looks like), and the daemon restarted: it queues
one on-device run, finishes it, and says so on its status. Restarted again
with a memory untyped once more, it does not run a second time.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

from tests.test_integration.test_mcp_declared_kind_transport import _start_daemon
from tests.test_integration.test_per_request_profile_e2e import (
    PRODUCTION_PORTS,
    _foreign_daemon_pids,
    _reserve_private_port,
)

RUN = uuid.uuid4().hex[:6].upper()


def _sql(data_root: Path, sql: str, args: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(data_root / "memory.db", timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        rows = conn.execute(sql, args).fetchall()
        conn.commit()
        return rows
    finally:
        conn.close()


def _wait(predicate, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.25)
    raise AssertionError(f"timed out waiting for {what}")


def test_an_upgraded_store_is_classified_once_and_never_again(tmp_path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    port = _reserve_private_port()
    assert port not in PRODUCTION_PORTS
    foreign = _foreign_daemon_pids()
    tokens = [f"UG{i}{RUN}" for i in range(3)]

    daemon = _start_daemon(data_root, port, tmp_path)
    try:
        for i, token in enumerate(tokens):
            daemon.remember(f"Fixture entry {token}: the synthetic dial shows {i} marks.",
                            "", f"upgrade-{token}")
        _wait(lambda: len(_sql(data_root, "SELECT 1 FROM atomic_facts WHERE content LIKE ?",
                               (f"%{RUN}%",))) >= 3, 60, "the saves")
    finally:
        daemon.stop(foreign)
    # What a store written before memory kinds existed looks like.
    _sql(data_root, "UPDATE atomic_facts SET memory_kind=NULL, memory_kind_source=NULL, "
                    "memory_kind_confidence=NULL, memory_kind_recipe=NULL, memory_kind_at=NULL")
    _sql(data_root, "DELETE FROM memory_kind_runs")
    untyped = _sql(data_root, "SELECT COUNT(*) FROM atomic_facts WHERE memory_kind IS NULL")[0][0]
    assert untyped >= 3

    daemon = _start_daemon(data_root, port, tmp_path)
    try:
        _wait(lambda: _sql(data_root, "SELECT status FROM memory_kind_runs") == [("completed",)],
              120, "the automatic run to finish")
        code, status = daemon.request("GET", "/api/memory-kinds/status")
        assert code == 200, status
        assert status["automatic_classification"]["state"] == "queued"
        assert "explanation" in status
    finally:
        daemon.stop(foreign)
    runs = _sql(data_root, "SELECT run_id, requested_by, processed FROM memory_kind_runs")
    assert len(runs) == 1 and runs[0][1] == "automatic-upgrade" and runs[0][2] >= 3
    assert _sql(data_root, "SELECT COUNT(*) FROM atomic_facts WHERE memory_kind IS NULL "
                           "AND COALESCE(quarantined,0)=0")[0][0] == 0

    # Untyped again (as if an older version wrote one), then restarted: never twice.
    _sql(data_root, "UPDATE atomic_facts SET memory_kind=NULL, memory_kind_source=NULL "
                    "WHERE content LIKE ?", (f"%{tokens[0]}%",))
    daemon = _start_daemon(data_root, port, tmp_path)
    try:
        code, status = daemon.request("GET", "/api/memory-kinds/status")
        assert code == 200, status
        _wait(lambda: daemon.request("GET", "/api/memory-kinds/status")[1]
              .get("automatic_classification", {}).get("state") != "not_checked", 60,
              "the start-up check")
        state = daemon.request("GET", "/api/memory-kinds/status")[1]["automatic_classification"]
        assert state["state"] == "already_classified" and state["run_id"] == runs[0][0]
    finally:
        daemon.stop(foreign)
    assert len(_sql(data_root, "SELECT 1 FROM memory_kind_runs")) == 1
