# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The doors to what needs a person's answer: disputes and changed files (D-37..D-39)."""

from __future__ import annotations

import json

import pytest

from superlocalmemory.truth import leases
from superlocalmemory.truth.slot_pass import SlotPass
from tests.test_server.test_per_request_profile import _daemon
from tests.test_truth.test_slot_pass import _fact, _flags


@pytest.fixture
def routed(engine_with_mock_deps):
    with _daemon(engine_with_mock_deps, profiles=("work",)) as pair:
        yield pair


@pytest.fixture
def disputed(routed, tmp_path):
    client, app = routed
    db = app.state.engine._db
    old = _fact(db, "The release cadence of SLM is one release every week", "2026-03-01T00:00:00Z")
    new = _fact(db, "The release cadence of SLM is one big release", "2026-11-14T00:00:00Z")
    SlotPass(db, _flags(**{"truth.slots": "shadow"}), state_dir=tmp_path).run()
    return client, app, old, new


def _dispute_id(app) -> str:
    return app.state.engine._db.execute(
        "SELECT dispute_id FROM slot_disputes WHERE state = 'open'")[0]["dispute_id"]


def test_attention_lists_the_open_dispute_with_both_sides(disputed) -> None:
    client, _app, old, new = disputed
    body = client.get("/api/truth/attention").json()
    assert body["success"] and body["counts"] == {"disputes": 1, "changed_sources": 0}
    (view,) = body["disputes"]
    assert {s["fact_id"] for s in view["sides"]} == {old, new}
    assert view["next"] == "ask_user"
    assert client.get("/api/disputes").json()["disputes"][0]["dispute_id"] == view["dispute_id"]


def test_settling_keeps_one_side_and_closes_the_dispute(disputed) -> None:
    client, app, old, new = disputed
    did = _dispute_id(app)
    r = client.post(f"/api/disputes/{did}/settle", json={"keep": new, "channel": "dashboard"})
    assert r.status_code == 200, r.text
    assert r.json()["retired"] == [old] and r.json()["settled_by"].startswith("person:dashboard")
    assert client.get("/api/truth/attention").json()["counts"]["disputes"] == 0
    again = client.post(f"/api/disputes/{did}/settle", json={"keep": new})
    assert again.status_code == 404


def test_a_wrong_answer_is_refused_and_changes_nothing(disputed) -> None:
    client, app, _old, _new = disputed
    did = _dispute_id(app)
    r = client.post(f"/api/disputes/{did}/settle", json={"keep": "not-a-side"})
    assert r.status_code == 422 and "Nothing was changed" in r.text
    assert client.get("/api/truth/attention").json()["counts"]["disputes"] == 1


def test_an_unknown_dispute_is_not_found(routed) -> None:
    client, _app = routed
    assert client.post("/api/disputes/nope/settle", json={"keep": "both"}).status_code == 404


def test_a_changed_file_is_listed_and_still_true_clears_it(routed, tmp_path) -> None:
    client, app = routed
    db = app.state.engine._db
    src = tmp_path / "config.yaml"
    src.write_text("port: 8765\n")
    fact = _fact(db, "The daemon listens on port 8765", "2026-11-16T00:00:00Z")
    db.execute("UPDATE memories SET metadata_json = ? WHERE memory_id = "
               "(SELECT memory_id FROM atomic_facts WHERE fact_id = ?)",
               (json.dumps({leases.METADATA_KEY: leases.fingerprint(str(src)).as_meta()}), fact))
    SlotPass(db, _flags(**{"truth.slots": "shadow"}), state_dir=tmp_path).run()
    src.write_text("port: 9999\n")
    with db.raw_connection() as conn:
        leases.sweep(conn)
    body = client.get("/api/truth/attention").json()
    assert [c["fact_id"] for c in body["changed_sources"]] == [fact]
    r = client.post(f"/api/changed-sources/{fact}/still-true", json={})
    assert r.status_code == 200, r.text
    assert client.get("/api/truth/attention").json()["counts"]["changed_sources"] == 0
    assert client.post("/api/changed-sources/nope/still-true", json={}).status_code == 404


def test_a_remote_app_cannot_see_or_settle_a_dispute_with_a_side_it_may_not_see(disputed) -> None:
    client, app, old, new = disputed
    app.state.engine._db.execute(
        "UPDATE memories SET metadata_json = ? WHERE memory_id = "
        "(SELECT memory_id FROM atomic_facts WHERE fact_id = ?)",
        (json.dumps({"_slm_source": {"type": "folder", "source_id": "s1", "relpath": "a.md",
                                     "version": "1"}}), old))
    remote = {"caller_view": "remote"}
    assert client.get("/api/truth/attention", params=remote).json()["disputes"] == []
    did = _dispute_id(app)
    r = client.post(f"/api/disputes/{did}/settle", params=remote, json={"keep": new})
    assert r.status_code == 404
    assert client.get("/api/truth/attention").json()["counts"]["disputes"] == 1
