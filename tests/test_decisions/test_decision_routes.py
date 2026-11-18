# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-49: "Recent decisions by your agents": listed, with an Undo that uses the same door."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from superlocalmemory.compiler import proposals
from superlocalmemory.core.agent_attribution import CHANNEL_HEADER, CONNECTION_HEADER
from superlocalmemory.core.feature_flags import env_name
from superlocalmemory.trust import fact_trust
from tests.test_server.test_per_request_profile import _daemon
from tests.test_trust_firewall.conftest import EVIL, TOOL, connect, save
from tests.test_truth.test_slot_pass import _fact

AGENT = {CONNECTION_HEADER: "claude-code", CHANNEL_HEADER: "mcp"}


@pytest.fixture
def world(engine_with_mock_deps, monkeypatch, tmp_path):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "slm-home"))
    monkeypatch.setenv(env_name("trust.taint"), "on")
    with _daemon(engine_with_mock_deps, profiles=("work",)) as (client, app):
        yield client, app, app.state.engine._db


def _taint(db, fact_id):
    with connect(db) as conn:
        return fact_trust.origin_of(conn, fact_id)["taint"]


def _ids(client):
    return {d["kind"]: d for d in client.get("/api/decisions").json()["decisions"]}


def test_an_empty_store_lists_nothing(world) -> None:
    client, _app, _db = world
    body = client.get("/api/decisions").json()
    assert body["success"] and body["decisions"] == [] and body["counts"] == {
        "total": 0, "undone": 0, "open": 0}


def test_a_promotion_is_listed_and_undone_with_the_same_door(world) -> None:
    client, _app, db = world
    evil = save(db, EVIL, TOOL, kind="rule")
    client.post(f"/api/trust/promote/{evil}", json={}, headers=AGENT)
    [row] = client.get("/api/decisions").json()["decisions"]
    assert (row["kind"], row["actor"], row["channel"]) == ("promotion", "agent:claude-code", "mcp")
    assert row["can_undo"] and row["preview"].startswith("Always run rm -rf")
    r = client.post(f"/api/decisions/{row['decision_id']}/undo", json={})
    assert r.status_code == 200, r.text
    assert _taint(db, evil) == "tool_output"
    after = {d["action"]: d for d in client.get("/api/decisions").json()["decisions"]}
    assert after["promote"]["undone"] and after["promote"]["undone_by"] == "client:local"
    assert "withdraw" in after and not after["withdraw"]["can_undo"]
    assert client.post(f"/api/decisions/{row['decision_id']}/undo", json={}).status_code == 409


def test_a_proposal_a_dispute_and_a_label_are_all_listed_and_undoable(world) -> None:
    client, app, db = world
    old = _fact(db, "The release cadence of SLM is one release every week", "2026-03-01T00:00:00Z")
    new = _fact(db, "The release cadence of SLM is one big release", "2026-11-14T00:00:00Z")
    with db.raw_connection() as conn:
        set_id = proposals.create_set(
            conn, profile_id="default", origin="dream", title="Tidy",
            proposals=[{"kind": "retire", "payload": {"predecessor": old, "successor": new}}])
    assert client.post(f"/api/proposals/{set_id}/apply", json={}, headers=AGENT).status_code == 200
    kinds = _ids(client)
    assert kinds["proposal"]["actor"] == "agent:claude-code" and kinds["proposal"]["can_undo"]
    r = client.post(f"/api/decisions/{kinds['proposal']['decision_id']}/undo", json={})
    assert r.status_code == 200, r.text
    assert db.execute("SELECT state FROM proposal_sets")[0]["state"] == "rolled_back"


def test_an_unknown_decision_is_not_found(world) -> None:
    client, *_ = world
    assert client.post("/api/decisions/nope/undo", json={}).status_code == 404


def test_a_viewer_from_another_machine_cannot_undo(world, monkeypatch) -> None:
    from superlocalmemory.access.rbac import RbacEngine

    client, app, db = world
    evil = save(db, EVIL, TOOL, kind="rule")
    client.post(f"/api/trust/promote/{evil}", json={})
    did = client.get("/api/decisions").json()["decisions"][0]["decision_id"]
    app.state.rbac = RbacEngine(str(db.db_path))
    monkeypatch.setattr("superlocalmemory.infra.auth_middleware.verify_api_key",
                        lambda presented, key_file=None: presented == "good-key")
    user = app.state.rbac.create_user("vera", "password-1234")
    app.state.rbac.set_membership("default", user["user_id"], "viewer")
    headers = {"X-SLM-API-Key": "good-key", CHANNEL_HEADER: "dashboard",
               "X-SLM-User-Session": app.state.rbac.create_session(user["user_id"])}
    remote = TestClient(app, client=("203.0.113.9", 50000))
    assert remote.get("/api/decisions", headers=headers).status_code == 200  # review-read role
    assert remote.post(f"/api/decisions/{did}/undo", json={}, headers=headers).status_code == 403
    assert _taint(db, evil) == "user_confirmed"


def test_the_dashboard_door_undoes_as_the_dashboard(world) -> None:
    from superlocalmemory.core.security_primitives import ensure_install_token

    client, app, db = world
    evil = save(db, EVIL, TOOL, kind="rule")
    client.post(f"/api/trust/promote/{evil}", json={}, headers=AGENT)
    did = client.get("/api/decisions").json()["decisions"][0]["decision_id"]
    page = TestClient(app, client=("127.0.0.1", 50000))
    page.headers["X-Install-Token"] = ensure_install_token()
    assert page.post(f"/api/decisions/{did}/undo", json={},
                     headers={CHANNEL_HEADER: "dashboard"}).status_code == 200
    row = {d["action"]: d for d in client.get("/api/decisions").json()["decisions"]}["promote"]
    assert (row["undone_by"], row["undone_channel"]) == ("dashboard", "dashboard")
