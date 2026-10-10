# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""The projection queue must not claim to be draining when nothing can drain it.

On the default store (Local Core) no graph or vector projection is open, so the
drain returns without touching a row, by design, while every write still queues
a row as the catch-up record for a future promotion.  /health nevertheless said
``draining: true, behind: true`` forever.  Nothing is deleted or marked here;
only the status is made truthful.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.core import backend_orchestrator as bo
from superlocalmemory.core.backend_orchestrator import BackendOrchestrator
from superlocalmemory.core.config import SLMConfig
from superlocalmemory.core.projection_drain import ProjectionDrain
from superlocalmemory.storage import projection_outbox
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.schema import create_all_tables


def _orchestrator(tmp_path: Path) -> BackendOrchestrator:
    db_path = tmp_path / "m.db"
    conn = sqlite3.connect(str(db_path))
    create_all_tables(conn)
    conn.commit()
    conn.close()
    config = SLMConfig()
    config.base_dir = str(tmp_path)
    config.data_dir = str(tmp_path)
    return BackendOrchestrator(config=config, db=DatabaseManager(str(db_path)))


@pytest.fixture
def orch(tmp_path, monkeypatch):
    o = _orchestrator(tmp_path)
    monkeypatch.setattr(
        projection_outbox, "health",
        lambda db: {"available": True, "depth": 5, "stalled": 0},
    )
    monkeypatch.setattr(ProjectionDrain, "running", property(lambda self: True))
    return o


def test_drain_knows_whether_a_projection_is_open():
    none = ProjectionDrain(None, lambda: None, lambda: None)
    assert none.has_target() is False
    assert ProjectionDrain(None, lambda: object(), lambda: None).has_target() is True
    assert ProjectionDrain(None, lambda: None, lambda: object()).has_target() is True


def test_no_projection_open_is_not_draining(orch):
    health = orch.outbox_health()
    assert health["projection_open"] is False
    assert health["draining"] is False
    assert health["depth"] == 5
    assert orch.health_check()["projection_queue"]["draining"] is False
    assert orch.health_check()["projection_queue"]["projection_open"] is False


def test_open_vector_backend_is_draining(orch):
    orch._lancedb = object()
    orch._lancedb_status = lambda: "active"
    health = orch.outbox_health()
    assert health["projection_open"] is True
    assert health["draining"] is True
    assert orch.health_check()["projection_queue"]["draining"] is True


def _daemon_report(monkeypatch, health):
    from superlocalmemory.server.unified_daemon import _projection_health

    class _O:
        @staticmethod
        def outbox_health():
            return dict(health)

    monkeypatch.setattr(bo, "get_orchestrator", lambda: _O())
    return _projection_health()


def test_daemon_health_waits_for_promotion_when_nothing_is_open(monkeypatch):
    r = _daemon_report(monkeypatch, {
        "depth": 7, "stalled": 0, "draining": False, "projection_open": False,
    })
    assert r["behind"] is False
    assert r["waiting_for_promotion"] is True


def test_daemon_health_is_behind_when_a_projection_is_open(monkeypatch):
    r = _daemon_report(monkeypatch, {
        "depth": 7, "stalled": 0, "draining": True, "projection_open": True,
    })
    assert r["behind"] is True
    assert r["waiting_for_promotion"] is False


def test_stalled_rows_are_behind_even_without_a_projection(monkeypatch):
    r = _daemon_report(monkeypatch, {
        "depth": 7, "stalled": 2, "draining": False, "projection_open": False,
    })
    assert r["behind"] is True


def test_empty_queue_is_neither(monkeypatch):
    r = _daemon_report(monkeypatch, {
        "depth": 0, "stalled": 0, "draining": False, "projection_open": False,
    })
    assert (r["behind"], r["waiting_for_promotion"]) == (False, False)
