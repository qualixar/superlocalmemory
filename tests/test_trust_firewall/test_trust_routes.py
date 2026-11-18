# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The doors to the waiting list: any caller may promote on purpose; recorded, reversible (D-49)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from superlocalmemory.trust import fact_trust
from tests.test_server.test_per_request_profile import _daemon
from tests.test_trust_firewall.conftest import EVIL, PERSON, TAINT, TOOL, WEB, connect, save


@pytest.fixture
def world(engine_with_mock_deps, monkeypatch, tmp_path):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "slm-home"))
    monkeypatch.setenv(TAINT, "on")
    with _daemon(engine_with_mock_deps, profiles=("work",)) as (client, app):
        db = app.state.engine._db
        mine = save(db, "Run the tests before every commit.", PERSON, kind="rule")
        evil = save(db, EVIL, TOOL, kind="rule")
        page = save(db, "Pinned from a web page.", WEB, pinned=True)
        yield client, app, db, mine, evil, page


def _dashboard(app):
    """The dashboard page's door: the install token, no daemon capability."""
    from superlocalmemory.core.security_primitives import ensure_install_token

    c = TestClient(app, client=("127.0.0.1", 50000))
    c.headers["X-Install-Token"] = ensure_install_token()
    return c


def _taint(db, fact_id):
    with connect(db) as conn:
        return fact_trust.origin_of(conn, fact_id)["taint"]


# -- reading ----------------------------------------------------------------------------------------

def test_the_list_is_readable_by_anyone_with_the_review_role(world) -> None:
    client, _app, _db, mine, evil, page = world
    body = client.get("/api/trust/pending").json()
    assert body["success"] and body["acting"] is True and body["count"] == 2
    assert {i["fact_id"] for i in body["items"]} == {evil, page}
    assert mine not in {i["fact_id"] for i in body["items"]}
    assert client.get("/api/trust/pending", params={"limit": 1}).json()["items"].__len__() == 1


def test_status_says_what_is_on_and_how_many_wait(world) -> None:
    client, *_ = world
    body = client.get("/api/trust/status").json()
    assert body["taint"] == "on" and body["secret_screen"] == "off" and body["floor"] == "system"
    assert body["waiting"] == 2 and body["approved"] == 0


def test_a_caller_view_parameter_decides_nothing(world) -> None:
    """The old "remote app" parameter is a query string the caller sends: it grants and
    refuses nothing now (A-08). Locality is the connection's peer address."""
    client, _app, db, _mine, evil, _page = world
    for params in ({"caller_view": "remote"}, {"caller_view": ""}, {"caller_view": "local"}):
        assert client.get("/api/trust/pending", params=params).status_code == 200
        assert client.get("/api/trust/status", params=params).status_code == 200
    r = client.post(f"/api/trust/approve/{evil}", params={"caller_view": "remote"}, json={})
    assert r.status_code == 200 and _taint(db, evil) == "user_confirmed"


# -- promoting: one explicit call by any caller (D-49) ---------------------------------------------

def _decisions(db, **where):
    sql = "SELECT * FROM decision_log" + (
        " WHERE " + " AND ".join(f"{k} = ?" for k in where) if where else "")
    return db.execute(sql, tuple(where.values()))


def test_any_local_caller_promotes_and_it_is_recorded(world) -> None:
    client, _app, db, _mine, evil, _page = world
    r = client.post(f"/api/trust/promote/{evil}", json={"by": "my-script"})
    assert r.status_code == 200, r.text
    assert r.json()["now"] == "user_confirmed" and r.json()["was"] == "tool_output"
    assert _taint(db, evil) == "user_confirmed"
    row = db.execute("SELECT decided_by, decided_channel, decided_at FROM trust_approvals "
                     "WHERE fact_id = ?", (evil,))[0]
    assert (row["decided_by"], row["decided_channel"]) == ("client:my-script", "http")
    assert row["decided_at"]
    [dec] = _decisions(db, subject_id=evil, action="promote")
    assert (dec["actor"], dec["channel"]) == ("client:my-script", "http")


def test_an_agent_names_itself_and_its_channel(world) -> None:
    from superlocalmemory.core.agent_attribution import CHANNEL_HEADER, CONNECTION_HEADER

    client, _app, db, _mine, evil, _page = world
    r = client.post(f"/api/trust/approve/{evil}", json={},
                    headers={CONNECTION_HEADER: "claude-code", CHANNEL_HEADER: "mcp"})
    assert r.status_code == 200, r.text
    [dec] = _decisions(db, subject_id=evil)
    assert (dec["actor"], dec["channel"]) == ("agent:claude-code", "mcp")


def test_no_header_can_make_a_caller_the_dashboard(world) -> None:
    """The A-07 probes: a forged terminal header, or the dashboard's channel header with only the
    daemon capability, are recorded for what they are; nothing is refused and nothing is believed."""
    from superlocalmemory.core.agent_attribution import CHANNEL_HEADER

    client, _app, db, _mine, evil, page = world
    r = client.post(f"/api/trust/approve/{evil}", json={"channel": "cli"},
                    headers={"X-SLM-Client-Shell": "tty"})
    assert r.status_code == 200
    r = client.post(f"/api/trust/approve/{page}", json={"channel": "dashboard"},
                    headers={CHANNEL_HEADER: "dashboard"})
    assert r.status_code == 200
    channels = {d["channel"] for d in _decisions(db)}
    assert channels == {"http"} and "dashboard" not in {d["actor"] for d in _decisions(db)}


def test_the_dashboard_page_decides_through_its_own_door(world) -> None:
    from superlocalmemory.core.agent_attribution import CHANNEL_HEADER

    _client, app, db, _mine, evil, _page = world
    page = _dashboard(app)
    r = page.post(f"/api/trust/promote/{evil}", json={"by": "me <b>"},
                  headers={CHANNEL_HEADER: "dashboard"})
    assert r.status_code == 200, r.text
    assert _taint(db, evil) == "user_confirmed"
    row = db.execute("SELECT decided_by, decided_channel FROM trust_approvals "
                     "WHERE fact_id = ?", (evil,))[0]
    assert (row["decided_by"], row["decided_channel"]) == ("dashboard", "dashboard")


def test_a_remote_caller_is_governed_by_its_role_not_by_a_parameter(
        world, monkeypatch) -> None:
    """A-08: from another machine, with a key, a viewer cannot promote and a member can; either
    way it is the ``http`` channel, never the dashboard, whatever it sends."""
    from superlocalmemory.access.rbac import RbacEngine
    from superlocalmemory.core.agent_attribution import CHANNEL_HEADER

    _client, app, db, _mine, evil, page = world
    app.state.rbac = RbacEngine(str(db.db_path))
    monkeypatch.setattr("superlocalmemory.infra.auth_middleware.verify_api_key",
                        lambda presented, key_file=None: presented == "good-key")

    def session(name, role):
        user = app.state.rbac.create_user(name, "password-1234")
        app.state.rbac.set_membership("default", user["user_id"], role)
        return {"X-SLM-User-Session": app.state.rbac.create_session(user["user_id"])}

    remote = TestClient(app, client=("203.0.113.9", 50000))
    base = {"X-SLM-API-Key": "good-key", CHANNEL_HEADER: "dashboard"}
    viewer = {**base, **session("vera", "viewer")}
    for params in ({}, {"caller_view": ""}, {"caller_view": "local"}):
        r = remote.post(f"/api/trust/approve/{evil}", params=params, json={"channel": "dashboard"},
                        headers=viewer)
        assert r.status_code == 403, params
    assert _taint(db, evil) == "tool_output"
    assert remote.post(f"/api/trust/approve/{evil}", json={},
                       headers={CHANNEL_HEADER: "dashboard"}).status_code in (401, 403)  # no key
    member = {**base, **session("mo", "member")}
    r = remote.post(f"/api/trust/approve/{evil}", json={"by": "the-owner"}, headers=member)
    assert r.status_code == 200, r.text
    [dec] = _decisions(db, subject_id=evil)
    assert dec["channel"] == "http" and dec["actor"].startswith("api-key:")
    assert "owner" not in dec["actor"]


def test_a_promotion_is_taken_back_through_withdraw(world) -> None:
    client, _app, db, _mine, evil, _page = world
    client.post(f"/api/trust/promote/{evil}", json={})
    assert _taint(db, evil) == "user_confirmed"
    r = client.post(f"/api/trust/withdraw/{evil}", json={})
    assert r.status_code == 200 and r.json()["now"] == "tool_output"
    assert _taint(db, evil) == "tool_output"
    assert client.get("/api/trust/pending").json()["count"] == 2
    assert client.post(f"/api/trust/withdraw/{evil}", json={}).status_code == 404  # nothing left


def test_approve_all_and_the_one_that_is_not_there(world) -> None:
    client, _app, db, _mine, evil, page = world
    assert client.post("/api/trust/approve/ghost", json={}).status_code == 404
    r = client.post("/api/trust/approve-all", json={})
    assert r.status_code == 200 and r.json()["approved"] == 2
    assert {_taint(db, evil), _taint(db, page)} == {"user_confirmed"}
    assert client.get("/api/trust/pending").json()["count"] == 0
    assert client.get("/api/trust/status").json()["approved"] == 2


def test_a_memory_of_another_profile_is_not_found(world) -> None:
    client, _app, db, *_ = world
    with db.raw_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('work', 'work')")
    other = save(db, EVIL, TOOL, kind="rule")
    db.execute("UPDATE atomic_facts SET profile_id = 'work' WHERE fact_id = ?", (other,))
    r = client.post(f"/api/trust/approve/{other}", json={})
    assert r.status_code == 404
