# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The doors to the waiting list: agents read, only a person approves (D-37 pattern of ``slm audit``)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from superlocalmemory.core.agent_attribution import SHELL_HEADER
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


def test_a_remote_app_can_neither_read_nor_approve(world) -> None:
    client, _app, db, _mine, evil, _page = world
    remote = {"caller_view": "remote"}
    assert client.get("/api/trust/pending", params=remote).status_code == 403
    assert client.get("/api/trust/status", params=remote).status_code == 403
    for path in (f"/api/trust/approve/{evil}", "/api/trust/approve-all"):
        r = client.post(path, params=remote, json={"channel": "cli"},
                        headers={SHELL_HEADER: "tty"})
        assert r.status_code == 403
    assert _taint(db, evil) == "tool_output"


# -- approving: a person's doors only ---------------------------------------------------------------

def test_an_agent_without_a_terminal_cannot_approve(world) -> None:
    client, _app, db, _mine, evil, _page = world
    for headers in ({}, {SHELL_HEADER: "no_tty"}, {SHELL_HEADER: "person"}):
        r = client.post(f"/api/trust/approve/{evil}", json={"channel": "cli"}, headers=headers)
        assert r.status_code == 403, headers
    assert _taint(db, evil) == "tool_output"
    assert db.execute("SELECT COUNT(*) AS n FROM trust_approvals WHERE state='approved'")[0]["n"] == 0


def test_an_agent_cannot_borrow_the_dashboards_door(world) -> None:
    client, _app, db, _mine, evil, _page = world  # the client holds the daemon capability
    r = client.post(f"/api/trust/approve/{evil}", json={"channel": "dashboard"})
    assert r.status_code == 403 and _taint(db, evil) == "tool_output"
    assert client.post(f"/api/trust/approve/{evil}", json={"channel": "mcp"},
                       headers={SHELL_HEADER: "tty"}).status_code == 422


def test_a_person_at_a_terminal_approves_and_it_is_recorded(world) -> None:
    client, _app, db, _mine, evil, _page = world
    r = client.post(f"/api/trust/approve/{evil}", headers={SHELL_HEADER: "tty"},
                    json={"channel": "cli", "by": "varun"})
    assert r.status_code == 200, r.text
    assert r.json()["now"] == "user_confirmed" and r.json()["was"] == "tool_output"
    assert _taint(db, evil) == "user_confirmed"
    row = db.execute("SELECT decided_by, decided_channel, decided_at FROM trust_approvals "
                     "WHERE fact_id = ?", (evil,))[0]
    assert (row["decided_by"], row["decided_channel"]) == ("varun", "cli") and row["decided_at"]
    assert client.get("/api/trust/pending").json()["count"] == 1


def test_the_dashboard_page_approves_through_its_own_door(world) -> None:
    _client, app, db, _mine, evil, _page = world
    page = _dashboard(app)
    r = page.post(f"/api/trust/approve/{evil}", json={"channel": "dashboard", "by": "me <b>"})
    assert r.status_code == 200, r.text
    assert _taint(db, evil) == "user_confirmed"
    row = db.execute("SELECT decided_by, decided_channel FROM trust_approvals "
                     "WHERE fact_id = ?", (evil,))[0]
    assert (row["decided_by"], row["decided_channel"]) == ("me b", "dashboard")  # sanitised
    assert page.post(f"/api/trust/approve/{evil}", json={"channel": "cli"},
                     headers={SHELL_HEADER: "tty"}).status_code == 403  # no capability here


def test_approve_all_and_the_one_that_is_not_there(world) -> None:
    client, _app, db, _mine, evil, page = world
    tty = {SHELL_HEADER: "tty"}
    assert client.post("/api/trust/approve/ghost", json={"channel": "cli"},
                       headers=tty).status_code == 404
    r = client.post("/api/trust/approve-all", json={"channel": "cli"}, headers=tty)
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
    r = client.post(f"/api/trust/approve/{other}", json={"channel": "cli"},
                    headers={SHELL_HEADER: "tty"})
    assert r.status_code == 404
