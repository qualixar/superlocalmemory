# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The daemon builds the mesh broker off the event loop.

MeshBroker.for_store may migrate mesh.db out of memory.db, wait for another
process's migration lock (30 s) or wait out a busy file: none of that may
freeze the async startup, so it runs in a worker thread.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from superlocalmemory.mesh.broker import MeshBroker
from superlocalmemory.server import unified_daemon


class _FakeBroker:
    def __init__(self) -> None:
        self.cleanup_started_in: str | None = None

    def start_cleanup(self) -> None:
        self.cleanup_started_in = threading.current_thread().name


def _config(tmp_path: Path, **extra) -> SimpleNamespace:
    return SimpleNamespace(db_path=tmp_path / "memory.db", mesh_enabled=True,
                           mesh_retention_days=5, **extra)


def test_slow_mesh_open_does_not_block_the_event_loop(tmp_path: Path, monkeypatch) -> None:
    broker = _FakeBroker()
    seen: dict = {}

    def slow_for_store(db_path, *, retention_days):
        seen.update(thread=threading.current_thread(), db=db_path, days=retention_days)
        time.sleep(0.6)  # a migration or a wait on another process's lock
        return broker

    monkeypatch.setattr(MeshBroker, "for_store", staticmethod(slow_for_store))

    async def scenario() -> tuple[object, int]:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        task = asyncio.create_task(ticker())
        started = await unified_daemon._start_mesh_broker(_config(tmp_path))
        task.cancel()
        return started, ticks

    started, ticks = asyncio.run(scenario())
    assert started is broker
    assert seen["thread"] is not threading.main_thread()
    assert seen["db"] == tmp_path / "memory.db" and seen["days"] == 5
    assert ticks >= 10, f"event loop starved while mesh.db opened ({ticks} ticks)"
    assert broker.cleanup_started_in is not None  # only after the broker exists


def test_mesh_disabled_starts_nothing(tmp_path: Path, monkeypatch) -> None:
    def never(*_a, **_k):
        raise AssertionError("mesh disabled: no broker")

    monkeypatch.setattr(MeshBroker, "for_store", staticmethod(never))
    config = _config(tmp_path)
    config.mesh_enabled = False
    assert asyncio.run(unified_daemon._start_mesh_broker(config)) is None


def test_mesh_open_failure_is_logged_not_fatal(tmp_path: Path, monkeypatch, caplog) -> None:
    def broken(*_a, **_k):
        raise RuntimeError("mesh.db could not be opened (database is locked)")

    monkeypatch.setattr(MeshBroker, "for_store", staticmethod(broken))
    assert asyncio.run(unified_daemon._start_mesh_broker(_config(tmp_path))) is None
    assert any("Mesh broker init failed" in r.getMessage() for r in caplog.records)
