"""Closing a session reads that session only and never holds the daemon's event loop.

Found live on a 32k-fact store: every Claude session end called POST /session/close,
whose async route ran close_session on the event loop. close_session loaded the whole
store (get_all_facts) and scanned all 87k temporal events with LIKE, so the daemon
stopped answering /health for as long as that took, many times an hour.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from superlocalmemory.core.store_pipeline import run_close_session
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from tests.test_core.test_stale_session_close import _seed_session, _summary_count


@pytest.fixture
def db(tmp_path):
    manager = DatabaseManager(tmp_path / "close.db")
    manager.initialize(schema)
    yield manager
    manager.close()


def test_an_older_session_beyond_the_fetch_ceiling_is_still_summarised(db, monkeypatch):
    _seed_session(db, session_id="sess-old", hours_ago=72.0, entity_id="ent-old")
    _seed_session(db, session_id="sess-new", hours_ago=1.0, entity_id="ent-new")
    monkeypatch.setenv("SLM_MAX_FACTS_UNBOUNDED", "1")  # only the newest fact fits

    assert run_close_session("sess-old", "default", db=db) == 1
    assert _summary_count(db, "default", "sess-old") == 1


def test_close_never_loads_the_whole_store(db, monkeypatch):
    _seed_session(db, session_id="sess-a")

    def whole_store(*_a, **_kw):
        raise AssertionError("close_session must not read every fact")

    monkeypatch.setattr(db, "get_all_facts", whole_store)
    assert run_close_session("sess-a", "default", db=db) == 1
    assert run_close_session("sess-a", "default", db=db) == 0  # still idempotent


def test_a_session_without_facts_never_scans_the_summaries(db, monkeypatch):
    statements: list[str] = []
    original = db.execute

    def spy(sql, *a, **kw):
        statements.append(sql)
        return original(sql, *a, **kw)

    monkeypatch.setattr(db, "execute", spy)
    assert run_close_session("sess-empty", "default", db=db) == 0
    assert not [s for s in statements if "temporal_events" in s]


def test_the_close_route_runs_off_the_event_loop(engine_with_mock_deps, monkeypatch):
    from superlocalmemory.server.unified_daemon import create_app

    engine = engine_with_mock_deps
    on_loop: list[bool] = []

    def close_session(session_id, profile_id=None):
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return 0

    monkeypatch.setattr(engine, "close_session", close_session)
    app = create_app()
    app.state.engine = engine
    client = TestClient(app)
    client.headers["X-SLM-Daemon-Capability"] = app.state.daemon_descriptor.capability
    client.headers["X-SLM-Target-Instance"] = app.state.daemon_descriptor.instance_id

    resp = client.post("/session/close", json={"session_id": "sess-x"})
    assert resp.status_code == 200, resp.text
    assert on_loop == [False]
