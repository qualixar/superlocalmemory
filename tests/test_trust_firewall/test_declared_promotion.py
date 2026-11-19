# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Owner decision 19 Nov (minimal firewall): the floor is ``agent_inferred``, so a rule or decision an
agent saves or confirms stands at session start with no promotion step and no decision-log entry.
Anything below the floor (web page, tool output, peer message, import, model text) stays held and the
ask goes on the waiting list. Everything here runs with ``trust.taint`` ON."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.core import standing_rules
from superlocalmemory.core.agent_attribution import CHANNEL_HEADER, CONNECTION_HEADER
from superlocalmemory.core.feature_flags import KILL_ENV
from superlocalmemory.storage.memory_kinds import MemoryKind
from superlocalmemory.trust import fact_trust, firewall
from tests.test_server.test_per_request_profile import _daemon
from tests.test_trust_firewall.conftest import (
    AGENT,
    FILE,
    MESH,
    MODEL,
    TAINT,
    TOOL,
    WEB,
    connect,
    save,
)

CALLER = {CONNECTION_HEADER: "claude-code", CHANNEL_HEADER: "mcp"}
RULE_TEXT = "Always run the full test suite once before every release."
DECISION_TEXT = "We decided that recall answers must arrive within three seconds."


@pytest.fixture
def world(engine_with_mock_deps, monkeypatch):
    monkeypatch.setenv(TAINT, "on")
    with _daemon(engine_with_mock_deps, profiles=("work",)) as (client, app):
        yield client, app, app.state.engine._db


def _remember(client, text, key, kind="", headers=CALLER, **extra):
    body = {"content": text, "idempotency_key": key, **extra}
    if kind:
        body["kind"] = kind
    return client.post("/remember", json=body, headers=headers)


def _taint(db, fact_id):
    with connect(db) as conn:
        return fact_trust.origin_of(conn, fact_id)["taint"]


def _promotions(client):
    rows = client.get("/api/decisions").json()["decisions"]
    return [r for r in rows if r["kind"] == "promotion"]


def _standing(db, kind=MemoryKind.RULE):
    return [f.fact_id for f in standing_rules.confirmed_facts(db, "default", kind, 10)]


def _agent_fact(db, text=RULE_TEXT):
    return save(db, text, AGENT)


# -- saving with a declared kind ---------------------------------------------------------------

def test_an_agent_saving_a_rule_reaches_session_start_with_the_firewall_on(world) -> None:
    client, _app, db = world
    resp = _remember(client, RULE_TEXT, "decl-rule-1", kind="rule")
    assert resp.status_code == 200, resp.text
    [fact_id] = resp.json()["fact_ids"]
    assert fact_id in _standing(db)
    assert _taint(db, fact_id) == "agent_inferred"  # trusted as it is: nothing was promoted


def test_an_agents_rule_needs_no_promotion_and_records_none(world) -> None:
    client, _app, db = world
    _remember(client, RULE_TEXT, "decl-rule-2", kind="rule")
    assert _promotions(client) == []
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM trust_approvals").fetchone()[0] == 0


def test_a_saved_decision_stands_too(world) -> None:
    client, _app, db = world
    [fact_id] = _remember(client, DECISION_TEXT, "decl-dec-1", kind="decision").json()["fact_ids"]
    assert fact_id in _standing(db, MemoryKind.DECISION)
    assert _promotions(client) == []


def test_the_ask_for_a_held_memory_names_the_actor_from_the_connection_never_the_body(
        world) -> None:
    client, _app, db = world
    fact_id = save(db, RULE_TEXT, WEB)
    resp = client.patch(f"/api/memory-kinds/fact/{fact_id}", json={"kind": "rule"},
                        headers={CONNECTION_HEADER: "claude-code", CHANNEL_HEADER: "cli"})
    assert resp.status_code == 200, resp.text
    with connect(db) as conn:
        [row] = conn.execute("SELECT requested_by, requested_channel, wanted FROM trust_approvals "
                             "WHERE fact_id = ?", (fact_id,)).fetchall()
    assert tuple(row) == ("agent:claude-code", "cli", "rule")


def test_a_held_rule_stands_after_one_recorded_promotion_and_undo_takes_it_back(world) -> None:
    client, _app, db = world
    fact_id = save(db, RULE_TEXT, WEB)
    client.patch(f"/api/memory-kinds/fact/{fact_id}", json={"kind": "rule"}, headers=CALLER)
    assert fact_id not in _standing(db)
    done = client.post(f"/api/trust/promote/{fact_id}", json={}, headers=CALLER)
    assert done.status_code == 200, done.text
    assert fact_id in _standing(db)
    [row] = _promotions(client)
    assert (row["actor"], row["channel"], row["subject_id"]) == (
        "agent:claude-code", "mcp", fact_id)
    undone = client.post(f"/api/decisions/{row['decision_id']}/undo", json={})
    assert undone.status_code == 200, undone.text
    assert fact_id not in _standing(db)
    assert _taint(db, fact_id) == "web"


def test_a_save_without_a_rule_or_decision_kind_is_not_promoted(world) -> None:
    client, _app, db = world
    [plain] = _remember(client, "The build takes about four minutes.", "decl-none").json()["fact_ids"]
    [pref] = _remember(client, "Varun prefers short answers.", "decl-pref",
                       kind="preference").json()["fact_ids"]
    assert _promotions(client) == []
    assert {_taint(db, plain), _taint(db, pref)} == {"agent_inferred"}


def test_saving_the_same_rule_again_changes_nothing(world) -> None:
    client, _app, db = world
    first = _remember(client, RULE_TEXT, "decl-rule-5", kind="rule").json()["fact_ids"]
    again = _remember(client, RULE_TEXT, "decl-rule-5", kind="rule").json()["fact_ids"]  # a retry
    assert first == again and _standing(db) == first and _promotions(client) == []


def test_a_rule_saved_while_the_writer_was_busy_stands_when_it_lands(
        engine_with_mock_deps, monkeypatch) -> None:
    """Accepted (202), committed in the background: it stands once it lands, with nothing promoted."""
    monkeypatch.setenv(TAINT, "on")
    engine = engine_with_mock_deps
    with _daemon(engine, profiles=("work",)) as (client, app):
        runtime = app.state.canonical_remember_runtime
        holder = sqlite3.connect(str(engine._db.db_path), timeout=5, isolation_level=None)
        holder.execute("BEGIN IMMEDIATE")
        try:
            accepted = _remember(client, RULE_TEXT, "decl-busy", kind="rule")
        finally:
            holder.execute("ROLLBACK")
            holder.close()
        assert accepted.status_code == 202, accepted.text
        assert runtime.wait_for_deferred(timeout=15.0)
        # Before anyone resends: the background commit itself is enough.
        [fact_id] = [r["fact_id"] for r in engine._db.execute(
            "SELECT fact_id FROM atomic_facts WHERE content = ?", (RULE_TEXT,))]
        assert fact_id in _standing(engine._db)
        assert _promotions(client) == []
        # The resend gets the final receipt.
        assert _remember(client, RULE_TEXT, "decl-busy", kind="rule").json()["fact_ids"] == [fact_id]


# -- confirming a kind on an existing memory ---------------------------------------------------

def test_confirming_an_agent_fact_as_a_decision_makes_it_stand_without_a_promotion(world) -> None:
    client, _app, db = world
    fact_id = _agent_fact(db, DECISION_TEXT)
    assert fact_id not in _standing(db, MemoryKind.DECISION)
    resp = client.patch(f"/api/memory-kinds/fact/{fact_id}", json={"kind": "decision"},
                        headers=CALLER)
    assert resp.status_code == 200, resp.text
    assert fact_id in _standing(db, MemoryKind.DECISION)
    assert _promotions(client) == []


def test_the_batch_confirm_makes_each_agent_rule_stand_and_skips_other_kinds(world) -> None:
    client, _app, db = world
    rule, pref = _agent_fact(db, RULE_TEXT), _agent_fact(db, "Varun likes dark themes.")
    resp = client.post("/api/memory-kinds/confirm", headers=CALLER, json={"items": [
        {"fact_id": rule, "kind": "rule"}, {"fact_id": pref, "kind": "preference"}]})
    assert resp.status_code == 200, resp.text
    assert _standing(db) == [rule]
    assert _promotions(client) == []
    assert _taint(db, pref) == "agent_inferred"


@pytest.mark.parametrize("origin, taint", [
    (WEB, "web"), (TOOL, "tool_output"), (MESH, "mesh_peer"), (FILE, "imported_file"),
    (MODEL, "model_generated")])
def test_a_rule_from_a_web_page_a_tool_a_peer_a_file_or_a_model_is_not_promoted(
        world, origin, taint) -> None:
    client, _app, db = world
    fact_id = save(db, RULE_TEXT, origin)
    resp = client.patch(f"/api/memory-kinds/fact/{fact_id}", json={"kind": "rule"}, headers=CALLER)
    assert resp.status_code == 200, resp.text
    assert _taint(db, fact_id) == taint
    assert fact_id not in _standing(db)
    assert _promotions(client) == []
    with connect(db) as conn:  # the ask is on the waiting list, for someone to promote on purpose
        assert [i["fact_id"] for i in firewall.pending(conn, "default")["items"]] == [fact_id]


def test_a_fact_that_is_already_trusted_needs_nothing(world) -> None:
    client, _app, db = world
    from tests.test_trust_firewall.conftest import PERSON

    fact_id = save(db, RULE_TEXT, PERSON)
    client.patch(f"/api/memory-kinds/fact/{fact_id}", json={"kind": "rule"}, headers=CALLER)
    assert fact_id in _standing(db)
    assert _promotions(client) == []


# -- never by a guess ----------------------------------------------------------------------------

def test_the_background_classifier_never_promotes(world) -> None:
    from superlocalmemory.storage.memory_kind_store import KindChange, MemoryKindStore
    from superlocalmemory.storage.memory_kinds import KindAssignment, KindSource

    client, _app, db = world
    fact_id = _agent_fact(db)
    store = MemoryKindStore(db)
    run = store.create_run("default", backend="rules", recipe_id="kinds-rules-v1",
                           mode="untyped", requested_by="test", total_estimate=1)
    assert store.set_run_status(run["run_id"], "running", expected=["queued"])
    run_id = run["run_id"]
    rowid = db.execute("SELECT rowid AS r FROM atomic_facts WHERE fact_id = ?",
                       (fact_id,))[0]["r"]
    change = KindChange(rowid=rowid, fact_id=fact_id, old_kind=None, old_source=None,
                        old_confidence=None, fact_type="semantic",
                        new=KindAssignment(MemoryKind.RULE, KindSource.MODEL_LLM, 0.9, "kinds-rules-v1"))
    result = store.apply_batch(run_id, "default", [change], new_cursor=rowid, actor="classifier")
    assert result.applied == 1
    assert db.execute("SELECT memory_kind FROM atomic_facts WHERE fact_id = ?",
                      (fact_id,))[0]["memory_kind"] == "rule"
    assert _taint(db, fact_id) == "agent_inferred"
    assert fact_id not in _standing(db)
    assert _promotions(client) == []


# -- off means off ---------------------------------------------------------------------------------

@pytest.mark.parametrize("state", ["off", "shadow"])
def test_with_the_firewall_off_or_shadow_nothing_is_promoted(
        engine_with_mock_deps, monkeypatch, state) -> None:
    monkeypatch.setenv(TAINT, state)
    with _daemon(engine_with_mock_deps, profiles=("work",)) as (client, app):
        [fact_id] = _remember(client, RULE_TEXT, f"decl-{state}", kind="rule").json()["fact_ids"]
        other = _agent_fact(app.state.engine._db, DECISION_TEXT)
        client.patch(f"/api/memory-kinds/fact/{other}", json={"kind": "decision"}, headers=CALLER)
        assert _promotions(client) == []
        with connect(app.state.engine._db) as conn:
            assert conn.execute("SELECT COUNT(*) FROM trust_approvals").fetchone()[0] == 0
        assert fact_id and not firewall.active()


def test_with_the_kill_switch_on_nothing_is_promoted(engine_with_mock_deps, monkeypatch) -> None:
    monkeypatch.setenv(TAINT, "on")
    monkeypatch.setenv(KILL_ENV, "1")
    with _daemon(engine_with_mock_deps, profiles=("work",)) as (client, _app):
        _remember(client, RULE_TEXT, "decl-kill", kind="rule")
        assert _promotions(client) == []


def test_the_promotion_step_itself_does_nothing_unless_the_firewall_acts(
        engine_with_mock_deps, monkeypatch) -> None:
    """Both guards: the one routes ask first (``wanted``) and the writer's own (``promote``)."""
    from superlocalmemory.trust import declared_promotion

    db = engine_with_mock_deps._db
    fact_id = _agent_fact(db)
    monkeypatch.setenv(TAINT, "off")
    assert not declared_promotion.wanted("rule")
    with db.raw_connection() as conn:
        done = declared_promotion.promote(conn, "default", [(fact_id, "rule")], by="agent:a",
                                          channel="mcp", how="saved")
    assert done == {"promoted": [], "held": []}
    monkeypatch.setenv(TAINT, "on")
    assert declared_promotion.wanted("Rule") and declared_promotion.wanted("decision")
    assert not declared_promotion.wanted("preference") and not declared_promotion.wanted("")


def test_a_memory_of_another_profile_is_never_promoted(world) -> None:
    from superlocalmemory.trust import declared_promotion

    _client, _app, db = world
    fact_id = _agent_fact(db)
    with db.raw_connection() as conn:
        done = declared_promotion.promote(conn, "work", [(fact_id, "rule")], by="agent:a",
                                          channel="mcp", how="saved")
    assert done == {"promoted": [], "held": []}
    assert _taint(db, fact_id) == "agent_inferred"


# -- a corrected rule keeps working, through the review route every door uses -------------------------

NEW_RULE = "Always run the full test suite twice before every release."


def _propose(client, fact_id, key):
    edited = client.patch(f"/api/memories/{fact_id}", json={"content": NEW_RULE},
                          headers={**CALLER, "X-Idempotency-Key": key})
    assert edited.status_code == 202, edited.text
    body = edited.json()
    return body["successor_fact_id"], body["correction_case"]


def _review(client, case, action, headers, version=None):
    return client.post(f"/api/corrections/{case['case_id']}/{action}", headers=headers,
                       json={"expected_version": case["version"] if version is None else version})


def test_applying_a_correction_of_a_rule_keeps_the_rule_standing_with_no_promotion(world) -> None:
    client, _app, db = world
    [rule] = _remember(client, RULE_TEXT, "corr-rule-1", kind="rule").json()["fact_ids"]
    successor, case = _propose(client, rule, "corr-edit-1")
    assert successor not in _standing(db)
    done = _review(client, case, "apply", {CONNECTION_HEADER: "reviewer", CHANNEL_HEADER: "cli"})
    assert done.status_code == 200, done.text
    assert successor in _standing(db)
    assert _taint(db, successor) == "agent_inferred"
    assert _promotions(client) == []


def test_rolling_the_correction_back_restores_the_old_rule(world) -> None:
    client, _app, db = world
    [rule] = _remember(client, RULE_TEXT, "corr-rule-2", kind="rule").json()["fact_ids"]
    successor, case = _propose(client, rule, "corr-edit-2")
    applied = _review(client, case, "apply", CALLER).json()["correction_case"]
    undone = _review(client, case, "rollback", CALLER, version=applied["version"])
    assert undone.status_code == 200, undone.text
    assert successor not in _standing(db) and rule in _standing(db)


def test_a_correction_of_a_held_rule_does_not_make_its_successor_stand(world) -> None:
    client, _app, db = world
    held = save(db, RULE_TEXT, WEB)  # a web page's sentence, confirmed as a rule but held
    db.execute("UPDATE atomic_facts SET memory_kind = 'rule', memory_kind_source = 'caller' "
               "WHERE fact_id = ?", (held,))
    assert held not in _standing(db)
    successor, case = _propose(client, held, "corr-edit-3")
    _review(client, case, "apply", CALLER)
    assert successor not in _standing(db)
    assert _taint(db, successor) == "web"  # the page's class travels to what replaced it
    assert _promotions(client) == []


def test_a_promotion_recorded_for_a_correction_is_taken_back_on_rollback(world) -> None:
    """Kept for a store whose earlier build recorded one: the rollback marks it undone."""
    from superlocalmemory.trust import declared_promotion

    _client, _app, db = world
    fact_id = save(db, RULE_TEXT, WEB)
    with db.raw_connection() as conn:
        firewall.approve(conn, fact_id, by="agent:a", channel="mcp",
                         via=declared_promotion.CORRECTION_VIA)
        assert declared_promotion.take_back(conn, "default", fact_id, by="agent:a",
                                            channel="mcp") is True
        assert declared_promotion.take_back(conn, "default", fact_id, by="agent:a",
                                            channel="mcp") is False  # nothing left to take back
    assert _taint(db, fact_id) == "web"


def test_with_the_firewall_off_a_correction_records_nothing(
        engine_with_mock_deps, monkeypatch) -> None:
    monkeypatch.setenv(TAINT, "on")
    with _daemon(engine_with_mock_deps, profiles=("work",)) as (client, app):
        [rule] = _remember(client, RULE_TEXT, "corr-rule-4", kind="rule").json()["fact_ids"]
        successor, case = _propose(client, rule, "corr-edit-4")
        monkeypatch.setenv(TAINT, "off")
        _review(client, case, "apply", CALLER)
        assert _promotions(client) == []
        assert _taint(app.state.engine._db, successor) == "agent_inferred"
