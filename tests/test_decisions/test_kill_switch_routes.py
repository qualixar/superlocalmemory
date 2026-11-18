# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-23 over HTTP: read the switch, set it, and bring retired memories back as a recorded decision."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from superlocalmemory.core.agent_attribution import CHANNEL_HEADER, CONNECTION_HEADER
from superlocalmemory.core.feature_flags import KILL_ENV, kill_switch_on
from superlocalmemory.server.routes import kill_switch as routes
from tests.test_server.test_per_request_profile import _daemon
from tests.test_truth.test_slot_pass import _fact, _reconcile, _retired, _settled_by_a_person

AGENT = {CONNECTION_HEADER: "claude-code", CHANNEL_HEADER: "mcp"}


@pytest.fixture
def world(engine_with_mock_deps, monkeypatch, tmp_path):
    home = tmp_path / "slm-home"
    home.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(home))
    monkeypatch.delenv(KILL_ENV, raising=False)
    with _daemon(engine_with_mock_deps, profiles=("work",)) as (client, app):
        yield client, app, app.state.engine._db, home


def _retire_one(db):
    old = _fact(db, "The release cadence of SLM is one release every week", "2026-03-01T00:00:00+00:00")
    new = _fact(db, "The release cadence of SLM is one big release", "2026-11-14T00:00:00+00:00")
    _settled_by_a_person(db, old, new)
    _reconcile(db)
    assert _retired(db, old)[2] == new
    return old


def test_the_router_is_registered_in_the_daemon() -> None:
    from pathlib import Path

    text = (Path(routes.__file__).parents[1] / "unified_daemon.py").read_text(encoding="utf-8")
    assert '"kill_switch"' in text


def test_a_fresh_store_reads_off_with_the_plain_words_and_nothing_to_bring_back(world) -> None:
    client, _app, _db, _home = world
    body = client.get("/api/flags/kill").json()
    assert body["success"] and body["kill_switch"]["on"] is False
    assert body["kill_switch"]["source"] == "off" and body["would_come_back"] == 0
    assert "does not bring back" in body["does_not"] and "retired" in body["does_not"]
    assert "before 4.2.30" in body["does"]


def test_post_turns_the_switch_on_and_off_in_the_file_for_a_local_caller(world) -> None:
    client, _app, _db, home = world
    on = client.post("/api/flags/kill", json={"on": True}, headers=AGENT).json()
    assert on["kill_switch"]["on"] is True and kill_switch_on(home, {})
    assert client.get("/api/flags/kill").json()["kill_switch"]["source"] == "file"
    off = client.post("/api/flags/kill", json={"on": False}).json()
    assert off["kill_switch"]["on"] is False and not kill_switch_on(home, {})


def test_the_environment_keeps_it_on_and_the_words_say_why(world, monkeypatch) -> None:
    client, _app, _db, _home = world
    monkeypatch.setenv(KILL_ENV, "1")
    body = client.post("/api/flags/kill", json={"on": False}).json()
    assert body["kill_switch"]["on"] is True and body["kill_switch"]["source"] == "environment"
    assert KILL_ENV in body["kill_switch"]["words"]


def test_the_body_must_say_on_or_off(world) -> None:
    client, *_ = world
    assert client.post("/api/flags/kill", json={}).status_code == 422


def test_the_get_counts_what_would_come_back_and_a_dry_run_changes_nothing(world) -> None:
    client, _app, db, _home = world
    old = _retire_one(db)
    assert client.get("/api/flags/kill").json()["would_come_back"] == 1
    r = client.post("/api/flags/rollback-retirements", json={"dry_run": True}, headers=AGENT)
    assert r.json()["would_come_back"] == 1 and r.json()["came_back"] == 0
    assert _retired(db, old)[2] is not None
    assert client.get("/api/decisions").json()["decisions"] == []


def test_bringing_them_back_is_recorded_with_who_and_how_and_can_be_undone(world) -> None:
    client, _app, db, _home = world
    old = _retire_one(db)
    r = client.post("/api/flags/rollback-retirements", json={}, headers=AGENT)
    assert r.status_code == 200 and r.json()["came_back"] == 1
    assert _retired(db, old) == (None, None, None)
    [row] = client.get("/api/decisions").json()["decisions"]
    assert (row["kind"], row["actor"], row["channel"]) == (
        "rollback_retirements", "agent:claude-code", "mcp")
    assert row["can_undo"] and "Brought back 1 memory" in row["summary"]
    assert client.post(f"/api/decisions/{row['decision_id']}/undo", json={}).status_code == 200
    assert _retired(db, old)[2] is not None  # retired again for the same newer value
    assert client.get("/api/flags/kill").json()["would_come_back"] == 1


def test_the_rollback_works_with_the_switch_on(world) -> None:
    client, _app, db, _home = world
    old = _retire_one(db)
    client.post("/api/flags/kill", json={"on": True})
    assert client.post("/api/flags/rollback-retirements", json={}).json()["came_back"] == 1
    assert _retired(db, old) == (None, None, None)


def test_a_viewer_from_another_machine_can_read_but_not_change(world, monkeypatch) -> None:
    from superlocalmemory.access.rbac import RbacEngine

    client, app, db, home = world
    app.state.rbac = RbacEngine(str(db.db_path))
    monkeypatch.setattr("superlocalmemory.infra.auth_middleware.verify_api_key",
                        lambda presented, key_file=None: presented == "good-key")
    user = app.state.rbac.create_user("vera", "password-1234")
    app.state.rbac.set_membership("default", user["user_id"], "viewer")
    headers = {"X-SLM-API-Key": "good-key", CHANNEL_HEADER: "dashboard",
               "X-SLM-User-Session": app.state.rbac.create_session(user["user_id"])}
    remote = TestClient(app, client=("203.0.113.9", 50000))
    assert remote.get("/api/flags/kill", headers=headers).status_code == 200
    assert remote.post("/api/flags/kill", json={"on": True}, headers=headers).status_code == 403
    assert remote.post("/api/flags/rollback-retirements", json={}, headers=headers).status_code == 403
    assert not kill_switch_on(home, {})


def _kill_rows(client) -> list[dict]:
    rows = client.get("/api/decisions").json()["decisions"]
    return [r for r in rows if r["kind"] == "kill_switch"]


def test_turning_the_switch_is_a_recorded_decision_with_who_and_channel(world) -> None:
    """D-49: the biggest decision a caller can make is listed like every other one."""
    client, _app, _db, _home = world
    client.post("/api/flags/kill", json={"on": True}, headers=AGENT)
    rows = _kill_rows(client)
    assert len(rows) == 1
    assert rows[0]["action"] == "on" and rows[0]["actor"] == "agent:claude-code"
    assert rows[0]["channel"] == "mcp" and rows[0]["can_undo"] is True
    assert "everything new in 4.2.30" in rows[0]["summary"]


def test_undo_sets_the_switch_back_and_is_recorded(world) -> None:
    client, _app, _db, home = world
    client.post("/api/flags/kill", json={"on": True}, headers=AGENT)
    decision = _kill_rows(client)[0]["decision_id"]
    done = client.post(f"/api/decisions/{decision}/undo", json={})
    assert done.status_code == 200, done.text
    assert not kill_switch_on(home, {})
    assert _kill_rows(client)[0]["undone"] is True


def test_undo_refuses_in_words_when_the_switch_was_changed_since(world) -> None:
    client, _app, _db, home = world
    client.post("/api/flags/kill", json={"on": True}, headers=AGENT)
    first = [r for r in _kill_rows(client) if r["action"] == "on"][0]["decision_id"]
    client.post("/api/flags/kill", json={"on": False}, headers=AGENT)
    refused = client.post(f"/api/decisions/{first}/undo", json={})
    assert refused.status_code == 409
    assert "changed again" in refused.text and not kill_switch_on(home, {})
