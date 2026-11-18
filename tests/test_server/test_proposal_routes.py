# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The doors to proposed changes: read for review, answer only by a person on this computer (I-10)."""

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


def test_a_person_applies_and_undoes_a_set_and_the_record_names_them(proposed) -> None:
    client, app, set_id, _old, _new = proposed
    r = client.post(f"/api/proposals/{set_id}/apply", json={"channel": "dashboard"})
    assert r.status_code == 200, r.text
    done = r.json()["set"]
    assert done["state"] == "applied" and done["basis"]["kind"] == "person"
    assert done["basis"]["ref"].endswith("@dashboard") and done["decided_channel"] == "dashboard"
    assert _states(app) == ["applied"]
    assert client.get("/api/proposals").json()["count"] == 0
    undone = client.post(f"/api/proposals/{set_id}/rollback", json={"channel": "dashboard"})
    assert undone.status_code == 200 and undone.json()["set"]["state"] == "rolled_back"
    assert _states(app) == ["rolled_back"]


def test_rejecting_changes_nothing(proposed) -> None:
    client, app, set_id, _old, _new = proposed
    r = client.post(f"/api/proposals/{set_id}/reject", json={"channel": "cli"})
    assert r.status_code == 200 and r.json()["set"]["state"] == "rejected"
    assert _states(app) == []


def test_a_set_cannot_be_decided_twice_or_with_an_unknown_door(proposed) -> None:
    client, _app, set_id, _old, _new = proposed
    assert client.post(f"/api/proposals/{set_id}/apply", json={"channel": "agent"}).status_code == 403
    assert client.post(f"/api/proposals/{set_id}/rollback", json={"channel": "cli"}).status_code == 409
    client.post(f"/api/proposals/{set_id}/apply", json={"channel": "cli"})
    assert client.post(f"/api/proposals/{set_id}/apply", json={"channel": "cli"}).status_code == 409
    assert client.post("/api/proposals/nope/apply", json={"channel": "cli"}).status_code == 404


def test_a_remote_app_can_neither_read_nor_decide(proposed) -> None:
    client, app, set_id, _old, _new = proposed
    remote = {"caller_view": "remote"}
    assert client.get("/api/proposals", params=remote).status_code == 403
    assert client.get(f"/api/proposals/{set_id}", params=remote).status_code == 403
    for action in ("apply", "reject", "rollback"):
        r = client.post(f"/api/proposals/{set_id}/{action}", params=remote, json={"channel": "cli"})
        assert r.status_code == 403
    assert _states(app) == []
    assert client.get("/api/proposals").json()["count"] == 1
