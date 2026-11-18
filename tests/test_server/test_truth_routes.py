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
    # the channel is the connection's: a body that says "dashboard" is not the dashboard
    assert r.json()["retired"] == [old] and r.json()["settled_by"] == "client:local@http"
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


def test_one_memorys_truth_view_shows_its_slot_values_and_dispute(disputed) -> None:
    client, _app, old, new = disputed
    body = client.get(f"/api/truth/facts/{new}").json()
    assert body["success"] and body["fact_id"] == new
    (slot,) = body["slots"]
    assert slot["legacy"] is False and slot["subject"] and slot["relation"]
    assert [v["fact_id"] for v in slot["values"]] == [new, old]  # newest first
    mine = slot["values"][0]
    assert mine["this_memory"] is True and mine["current"] is False
    assert slot["values"][1]["current"] is True and slot["values"][1]["content"]
    assert len(body["disputes"]) == 1 and body["source"] is None


def test_a_memory_in_no_slot_has_an_empty_view(routed) -> None:
    client, app = routed
    plain = _fact(app.state.engine._db, "Lunch was good", "2026-11-16T00:00:00Z")
    body = client.get(f"/api/truth/facts/{plain}").json()
    assert body["slots"] == [] and body["disputes"] == [] and body["source"] is None


def test_the_truth_checks_report_every_invariant_in_order(disputed) -> None:
    client, _app, _old, _new = disputed
    body = client.get("/api/truth/checks").json()
    assert body["success"] and body["checked_at"]
    ids = [r["invariant"] for r in body["results"]]
    assert ids[:2] == ["I-01", "I-02"] and "I-16" in ids
    assert all(r["status"] != "violated" for r in body["results"])
    assert all(set(r) == {"invariant", "title", "status", "count"} for r in body["results"])
    assert body["second_opinions"] == []


def test_the_truth_checks_also_report_signatures_and_the_audit_log_in_words(disputed) -> None:
    client, _app, _old, _new = disputed
    body = client.get("/api/truth/checks").json()
    sig, audit = body["signatures"], body["audit_chain"]
    assert sig["facts"] >= 2 and sig["signed"] == 0 and sig["words"]
    assert audit["present"] in (True, False) and audit["words"]
    assert "content" not in sig and "modified_ids" in sig


def test_a_remote_app_cannot_open_the_truth_view_of_a_memory_it_may_not_see(disputed) -> None:
    client, app, old, _new = disputed
    app.state.engine._db.execute(
        "UPDATE memories SET metadata_json = ? WHERE memory_id = "
        "(SELECT memory_id FROM atomic_facts WHERE fact_id = ?)",
        (json.dumps({"_slm_source": {"type": "folder", "source_id": "s1", "relpath": "a.md",
                                     "version": "1"}}), old))
    assert client.get(f"/api/truth/facts/{old}", params={"caller_view": "remote"}).status_code == 404
    assert client.get(f"/api/truth/facts/{old}").status_code == 200


@pytest.fixture
def audited(disputed):
    from superlocalmemory.core.feature_flags import FeatureFlags
    from superlocalmemory.core.judge_ladder import JudgeLadder
    from superlocalmemory.truth.dispute_pass import DisputePass
    from tests.test_truth.test_dispute_pass import Fixed

    client, app, old, new = disputed
    db = app.state.engine._db
    DisputePass(db, FeatureFlags(file_values={"judge.ladder": "shadow"}, environ={}),
                JudgeLadder([Fixed("same", 0.9)])).run()
    db.execute("UPDATE judge_receipts SET audit_selected = 1")
    return client, app, old, new


def test_the_audit_queue_answers_for_any_caller_and_a_parameter_decides_nothing(audited) -> None:
    client, app, old, new = audited
    body = client.get("/api/audit/pending", params={"caller_view": "remote"}).json()
    assert body["count"] == 1
    (item,) = body["items"]
    assert [m_["fact_id"] for m_ in item["memories"]] == [old, new] and "verdict" not in repr(item)
    url = f"/api/audit/{item['item_id']}/answer"
    assert client.post(url, json={"label": "maybe"}).status_code == 422
    r = client.post(url, json={"label": "replaces", "model": "some-model"})
    assert r.status_code == 200 and r.json()["labelled"] == 1
    assert r.json()["labeller_kind"] == "agent" and r.json()["counted"] == 1
    assert client.get("/api/audit/pending").json()["count"] == 0
    assert client.post(url, json={"label": "replaces"}).status_code == 404
    label = app.state.engine._db.execute(
        "SELECT labeller, blind, labeller_kind, labeller_model FROM judge_labels")[0]
    assert label["labeller"] == "client:local@http" and label["blind"] == 1
    assert (label["labeller_kind"], label["labeller_model"]) == ("agent", "some-model")


def test_only_the_dashboards_own_door_labels_as_a_person(audited) -> None:
    from fastapi.testclient import TestClient

    from superlocalmemory.core.agent_attribution import CHANNEL_HEADER
    from superlocalmemory.core.security_primitives import ensure_install_token

    client, app, _old, _new = audited
    (item,) = client.get("/api/audit/pending").json()["items"]
    url = f"/api/audit/{item['item_id']}/answer"
    page = TestClient(app, client=("127.0.0.1", 50000))
    page.headers["X-Install-Token"] = ensure_install_token()
    forged = client.post(url, json={"label": "replaces", "channel": "dashboard"},
                         headers={CHANNEL_HEADER: "dashboard"})
    assert forged.json()["labeller_kind"] == "agent" and forged.json()["counted"] == 0
    client.post(f"/api/audit/{item['item_id']}/retract", json={})
    r = page.post(url, json={"label": "replaces"}, headers={CHANNEL_HEADER: "dashboard"})
    assert r.json()["labeller_kind"] == "person" and r.json()["counted"] == 1


def test_a_label_can_be_retracted_and_a_settlement_reopened(audited) -> None:
    client, app, old, new = audited
    (item,) = client.get("/api/audit/pending").json()["items"]
    client.post(f"/api/audit/{item['item_id']}/answer", json={"label": "same"})
    assert client.post(f"/api/audit/{item['item_id']}/retract", json={}).json()["retracted"] == 1
    assert client.get("/api/audit/pending").json()["count"] == 1
    assert client.post(f"/api/audit/{item['item_id']}/retract", json={}).status_code == 404
    did = _dispute_id(app)
    assert client.post(f"/api/disputes/{did}/reopen").status_code == 404  # still open
    client.post(f"/api/disputes/{did}/settle", json={"keep": new})
    r = client.post(f"/api/disputes/{did}/reopen", json={})
    assert r.status_code == 200 and r.json()["restored"] == [old]
    assert client.get("/api/truth/attention").json()["counts"]["disputes"] == 1
