# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The doors to proposed changes: any caller reviews and decides; recorded and reversible (D-49)."""

from __future__ import annotations

import pytest

from superlocalmemory.compiler import proposals
from tests.test_server.test_per_request_profile import _daemon
from tests.test_truth.test_slot_pass import _fact


@pytest.fixture
def routed(engine_with_mock_deps):
    with _daemon(engine_with_mock_deps, profiles=("work",)) as pair:
        yield pair


@pytest.fixture
def proposed(routed):
    client, app = routed
    db = app.state.engine._db
    old = _fact(db, "The release cadence of SLM is one release every week", "2026-03-01T00:00:00Z")
    new = _fact(db, "The release cadence of SLM is one big release", "2026-11-14T00:00:00Z")
    with db.raw_connection() as conn:
        set_id = proposals.create_set(
            conn, profile_id="default", origin="dream", title="Tidy",
            proposals=[{"kind": "retire", "payload": {"predecessor": old, "successor": new}}])
    return client, app, set_id, old, new


def _states(app) -> list[str]:
    return [r["status"] for r in app.state.engine._db.execute("SELECT status FROM correction_cases")]


def test_the_list_and_the_detail_show_the_proposal_in_plain_words(proposed) -> None:
    client, _app, set_id, _old, _new = proposed
    body = client.get("/api/proposals").json()
    assert body["success"] and body["count"] == 1
    (found,) = body["sets"]
    assert found["set_id"] == set_id and found["state"] == "proposed" and found["title"] == "Tidy"
    assert found["proposals"][0]["summary"].startswith("Retire “")
    assert client.get(f"/api/proposals/{set_id}").json()["set"]["set_id"] == set_id
    assert client.get("/api/proposals/nope").status_code == 404
    assert client.get("/api/proposals", params={"state": "applied"}).json()["sets"] == []


def test_any_caller_applies_and_undoes_a_set_and_the_record_names_it(proposed) -> None:
    from superlocalmemory.core.agent_attribution import CHANNEL_HEADER, CONNECTION_HEADER

    client, app, set_id, _old, _new = proposed
    r = client.post(f"/api/proposals/{set_id}/apply", json={},
                    headers={CONNECTION_HEADER: "claude-code", CHANNEL_HEADER: "mcp"})
    assert r.status_code == 200, r.text
    done = r.json()["set"]
    assert done["state"] == "applied" and done["basis"]["kind"] == "decision"
    assert done["basis"]["ref"] == "agent:claude-code@mcp" and done["decided_channel"] == "mcp"
    assert done["decided_by"] == "agent:claude-code"
    assert _states(app) == ["applied"]
    assert client.get("/api/proposals").json()["count"] == 0
    undone = client.post(f"/api/proposals/{set_id}/rollback", json={})
    assert undone.status_code == 200 and undone.json()["set"]["state"] == "rolled_back"
    assert _states(app) == ["rolled_back"]
    rows = app.state.engine._db.execute(
        "SELECT action, actor, channel, reverted_by FROM decision_log ORDER BY rowid")
    assert [(r["action"], r["actor"], r["channel"]) for r in rows] == [
        ("apply", "agent:claude-code", "mcp"), ("rollback", "client:local", "http")]
    assert rows[0]["reverted_by"] == "client:local"  # the rollback marks the apply undone


def test_rejecting_changes_nothing(proposed) -> None:
    client, app, set_id, _old, _new = proposed
    r = client.post(f"/api/proposals/{set_id}/reject", json={})
    assert r.status_code == 200 and r.json()["set"]["state"] == "rejected"
    assert _states(app) == []


def test_a_set_cannot_be_decided_twice(proposed) -> None:
    client, _app, set_id, _old, _new = proposed
    assert client.post(f"/api/proposals/{set_id}/rollback", json={}).status_code == 409
    client.post(f"/api/proposals/{set_id}/apply", json={})
    assert client.post(f"/api/proposals/{set_id}/apply", json={}).status_code == 409
    assert client.post("/api/proposals/nope/apply", json={}).status_code == 404


def test_the_dashboard_door_is_recorded_as_the_dashboard(proposed) -> None:
    from fastapi.testclient import TestClient

    from superlocalmemory.core.agent_attribution import CHANNEL_HEADER
    from superlocalmemory.core.security_primitives import ensure_install_token

    _client, app, set_id, _old, _new = proposed
    page = TestClient(app, client=("127.0.0.1", 50000))
    page.headers["X-Install-Token"] = ensure_install_token()
    r = page.post(f"/api/proposals/{set_id}/apply", json={"channel": "agent"},
                  headers={CHANNEL_HEADER: "dashboard"})
    assert r.status_code == 200, r.text
    assert r.json()["set"]["basis"]["ref"] == "dashboard@dashboard"


def test_a_caller_view_parameter_decides_nothing(proposed) -> None:
    client, app, set_id, _old, _new = proposed
    remote = {"caller_view": "remote"}
    assert client.get("/api/proposals", params=remote).status_code == 200
    assert client.get(f"/api/proposals/{set_id}", params=remote).status_code == 200
    assert client.post(f"/api/proposals/{set_id}/apply", params=remote, json={}).status_code == 200
    assert _states(app) == ["applied"]


def test_a_remote_caller_without_the_role_cannot_decide(proposed, monkeypatch) -> None:
    """A-08/A-09: from another machine a viewer cannot apply, whatever it sends; a member can,
    on the ``http`` channel."""
    from fastapi.testclient import TestClient

    from superlocalmemory.access.rbac import RbacEngine
    from superlocalmemory.core.agent_attribution import CHANNEL_HEADER

    _client, app, set_id, _old, _new = proposed
    app.state.rbac = RbacEngine(str(app.state.engine._db.db_path))
    monkeypatch.setattr("superlocalmemory.infra.auth_middleware.verify_api_key",
                        lambda presented, key_file=None: presented == "good-key")

    def session(name, role):
        user = app.state.rbac.create_user(name, "password-1234")
        app.state.rbac.set_membership("default", user["user_id"], role)
        return {"X-SLM-User-Session": app.state.rbac.create_session(user["user_id"])}

    remote = TestClient(app, client=("203.0.113.9", 50000))
    base = {"X-SLM-API-Key": "good-key", CHANNEL_HEADER: "dashboard"}
    for action in ("apply", "reject", "rollback"):
        r = remote.post(f"/api/proposals/{set_id}/{action}", params={"caller_view": ""},
                        json={"channel": "dashboard"}, headers={**base, **session("vera", "viewer")}
                        if action == "apply" else {**base, **session(f"v-{action}", "viewer")})
        assert r.status_code == 403, action
    assert _states(app) == []
    r = remote.post(f"/api/proposals/{set_id}/apply", json={},
                    headers={**base, **session("mo", "member")})
    assert r.status_code == 200, r.text
    assert r.json()["set"]["basis"]["ref"].endswith("@http")
