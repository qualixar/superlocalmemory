# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""L2-07: a proposed edit of a rule or decision must never itself be read as a
standing rule - only a correction a person actually applied carries the
confirmed kind onto its successor. Reused across session start (LLD I6)."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
from superlocalmemory.core.standing_rules import standing_facts
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migrations import (
    M018_ingestion_operations,
    M032_write_coordinator_admission,
    M033_projection_transactions,
    M034_obligation_integrity,
    M042_correction_case_ledger,
    M074_fact_origin,
    M079_fact_trust,
    M081_trust_approvals,
    M084_decision_log,
)
from superlocalmemory.storage.migrations import M052_memory_kinds as m052
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord
from superlocalmemory.trust import firewall
from tests.helpers.agent_rule import agent_stores, firewall_on  # noqa: F401

# Every test here runs with the origin firewall ON: the rule is saved as an agent saves it
# (``tests/helpers/agent_rule.py``). An agent's own words are at the firewall's floor, so a rule it
# saved, and its correction, stand with no promotion (owner decision, 19 Nov).
pytestmark = pytest.mark.usefixtures("firewall_on")


ACTOR = "daemon-capability:" + "a" * 64


@pytest.fixture()
def env(tmp_path):
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(path)
    for migration in (M018_ingestion_operations, M032_write_coordinator_admission,
                      M033_projection_transactions, M034_obligation_integrity):
        migration.apply(conn)
    conn.commit()
    conn.close()
    db = DatabaseManager(path)
    db.initialize(schema)
    with db.raw_connection() as raw:
        M042_correction_case_ledger.apply(raw)
        m052.apply(raw)
        for migration in (M074_fact_origin, M079_fact_trust, M081_trust_approvals,
                          M084_decision_log):  # origins, trust and the decision log
            migration.apply(raw)
    runtime = CanonicalRememberRuntime(db=db, profile_id="default",
                                       writer=lambda _r, _o: [],
                                       journal_path=tmp_path / "journal.db")
    runtime.start()
    yield db, runtime
    runtime.stop()


def _fact(db, content, *, kind=None, source=None) -> str:
    memory_id = db.store_memory(MemoryRecord(profile_id="default", content=content))
    fact = AtomicFact(profile_id="default", memory_id=memory_id, content=content,
                      fact_type=FactType.SEMANTIC)
    fact.memory_kind, fact.memory_kind_source = kind, source
    return agent_stores(db, fact)


def _kind_row(db, fact_id) -> dict:
    rows = db.execute(
        "SELECT memory_kind, memory_kind_source FROM atomic_facts WHERE fact_id=?",
        (fact_id,))
    return dict(rows[0]) if rows else {}


def _propose(runtime, fact_id, successor_id, content, key="prop-1"):
    return runtime.create_correction_successor(
        "default", fact_id, successor_id, content,
        trusted_actor_id=ACTOR, idempotency_key=key)


WHO = ("agent:claude-code", "mcp")  # what the review route takes from the connection


def _transition(runtime, case_id, version, action, key="trans-1", promote_as=WHO):
    return runtime.transition_correction(
        "default", case_id, action=action, expected_version=version,
        actor_id=ACTOR, idempotency_key=key, promote_as=promote_as)


# --- write-time: a successor's kind is untyped until its case is applied -----

def test_a_proposed_successor_is_untyped_not_confirmed(env) -> None:
    db, runtime = env
    rule = _fact(db, "Never push directly to main.", kind="rule", source="caller")
    res = _propose(runtime, rule, "succ00000000001", "Always push directly to main.")
    assert res["status"] == "proposed"
    row = _kind_row(db, res["successor_fact_id"])
    assert row["memory_kind"] is None and row["memory_kind_source"] is None


def test_a_rejected_successor_never_gets_a_confirmed_kind(env) -> None:
    db, runtime = env
    rule = _fact(db, "Never push directly to main.", kind="rule", source="caller")
    res = _propose(runtime, rule, "succ00000000002", "Always push directly to main.")
    _transition(runtime, res["case_id"], res["version"], "reject")
    row = _kind_row(db, res["successor_fact_id"])
    assert row["memory_kind"] is None and row["memory_kind_source"] is None


def test_an_applied_successor_inherits_the_predecessors_current_kind(env) -> None:
    db, runtime = env
    rule = _fact(db, "Never push directly to main.", kind="rule", source="caller")
    res = _propose(runtime, rule, "succ00000000003", "Only push to main via PR.")
    out = _transition(runtime, res["case_id"], res["version"], "apply")
    assert out["status"] == "applied"
    row = _kind_row(db, res["successor_fact_id"])
    assert (row["memory_kind"], row["memory_kind_source"]) == ("rule", "caller")


# --- read-time: standing_facts reuses recall's correction-admission rule ----

def test_an_unreviewed_proposal_is_never_a_standing_rule(env) -> None:
    db, runtime = env
    rule = _fact(db, "Never push directly to main.", kind="rule", source="caller")
    _propose(runtime, rule, "succ00000000004", "Always push directly to main, skip review.")
    shown = [f.content for f in standing_facts(db, "default")]
    assert "Always push directly to main, skip review." not in shown
    assert "Never push directly to main." in shown


def test_a_rejected_proposal_stays_withheld_even_after_the_original_is_quarantined(
        env) -> None:
    db, runtime = env
    rule = _fact(db, "Never push directly to main.", kind="rule", source="caller")
    res = _propose(runtime, rule, "succ00000000005", "Always push directly to main, skip review.")
    _transition(runtime, res["case_id"], res["version"], "reject")
    db.execute("UPDATE atomic_facts SET quarantined=1 WHERE fact_id=?", (rule,))
    shown = [f.content for f in standing_facts(db, "default")]
    assert "Always push directly to main, skip review." not in shown


def test_an_applied_corrections_successor_does_become_the_standing_rule(env) -> None:
    db, runtime = env
    rule = _fact(db, "Never push directly to main.", kind="rule", source="caller")
    res = _propose(runtime, rule, "succ00000000006", "Only push to main via PR.")
    _transition(runtime, res["case_id"], res["version"], "apply")
    shown = [f.content for f in standing_facts(db, "default")]
    assert shown == ["Only push to main via PR."]


def test_a_rolled_back_successor_is_withheld_again(env) -> None:
    db, runtime = env
    rule = _fact(db, "Never push directly to main.", kind="rule", source="caller")
    res = _propose(runtime, rule, "succ00000000007", "Only push to main via PR.")
    applied = _transition(runtime, res["case_id"], res["version"], "apply")
    _transition(runtime, res["case_id"], applied["version"], "rollback")
    shown = [f.content for f in standing_facts(db, "default")]
    assert shown == ["Never push directly to main."]


# --- a corrected rule keeps working with the firewall on (owner decision, 19 Nov) ----------------

def _taint(db, fact_id) -> str:
    from superlocalmemory.trust import fact_trust

    with db.raw_connection() as conn:
        return fact_trust.origin_of(conn, fact_id)["taint"]


def _promotions(db) -> list[dict]:
    from superlocalmemory.core import decisions

    with db.raw_connection() as conn:
        return [r for r in decisions.recent(conn, "default") if r["kind"] == "promotion"]


def _held_rule(db, content, origin=None) -> str:
    """A confirmed rule stored under ``origin`` (an agent's own words by default) with the kind
    set directly, the way a save leaves it."""
    from superlocalmemory.storage import fact_origin as fo
    from tests.helpers.agent_rule import AGENT_ORIGIN

    memory_id = db.store_memory(MemoryRecord(profile_id="default", content=content))
    fact = AtomicFact(profile_id="default", memory_id=memory_id, content=content,
                      fact_type=FactType.SEMANTIC)
    fact.memory_kind, fact.memory_kind_source = "rule", "caller"
    with fo.declare(origin or AGENT_ORIGIN):
        return db.store_fact(fact)


def test_the_corrected_rule_stands_with_no_promotion(env) -> None:
    db, runtime = env
    rule = _fact(db, "Never push directly to main.", kind="rule", source="caller")
    res = _propose(runtime, rule, "succ00000000010", "Only push to main via PR.")
    assert _taint(db, res["successor_fact_id"]) == "agent_inferred"
    assert [f.content for f in standing_facts(db, "default")] == ["Never push directly to main."]
    _transition(runtime, res["case_id"], res["version"], "apply")
    # an agent's own words are at the firewall's floor: nothing to promote, nothing recorded
    assert _taint(db, res["successor_fact_id"]) == "agent_inferred"
    assert [f.content for f in standing_facts(db, "default")] == ["Only push to main via PR."]
    assert _promotions(db) == []


def test_a_rule_an_agent_saved_passes_its_standing_to_its_correction(env) -> None:
    db, runtime = env
    held = _held_rule(db, "Never push directly to main.")
    assert _taint(db, held) == "agent_inferred"
    assert [f.fact_id for f in standing_facts(db, "default")] == [held]
    res = _propose(runtime, held, "succ00000000011", "Only push to main via PR.")
    _transition(runtime, res["case_id"], res["version"], "apply")
    assert _taint(db, res["successor_fact_id"]) == "agent_inferred"
    assert [f.fact_id for f in standing_facts(db, "default")] == [res["successor_fact_id"]]
    assert _promotions(db) == []


def test_a_successor_below_an_agents_own_words_is_held_not_promoted(env) -> None:
    from tests.test_trust_firewall.conftest import TOOL

    db, runtime = env
    tool_rule = _held_rule(db, "Never push directly to main.", TOOL)
    with db.raw_connection() as conn:  # the predecessor itself was promoted on purpose
        firewall.promote(conn, tool_rule, by="agent:a", channel="cli")
    res = _propose(runtime, tool_rule, "succ00000000012", "Only push to main via PR.")
    assert _taint(db, res["successor_fact_id"]) == "tool_output"
    _transition(runtime, res["case_id"], res["version"], "apply")
    assert _taint(db, res["successor_fact_id"]) == "tool_output"
    assert [p["subject_id"] for p in _promotions(db)] == [tool_rule]  # only the earlier one
    assert standing_facts(db, "default") == []


def test_rolling_the_correction_back_restores_the_old_rule(env) -> None:
    db, runtime = env
    rule = _fact(db, "Never push directly to main.", kind="rule", source="caller")
    res = _propose(runtime, rule, "succ00000000013", "Only push to main via PR.")
    applied = _transition(runtime, res["case_id"], res["version"], "apply")
    _transition(runtime, res["case_id"], applied["version"], "rollback", key="trans-2",
                promote_as=("agent:reviewer", "cli"))
    assert _taint(db, res["successor_fact_id"]) == "agent_inferred"
    assert [f.content for f in standing_facts(db, "default")] == ["Never push directly to main."]
    assert _promotions(db) == []


def test_a_rejected_correction_promotes_nothing(env) -> None:
    db, runtime = env
    rule = _fact(db, "Never push directly to main.", kind="rule", source="caller")
    res = _propose(runtime, rule, "succ00000000014", "Only push to main via PR.")
    _transition(runtime, res["case_id"], res["version"], "reject")
    assert _taint(db, res["successor_fact_id"]) == "agent_inferred"
    assert _promotions(db) == []


def test_an_apply_without_a_caller_door_promotes_nothing(env) -> None:
    db, runtime = env
    rule = _fact(db, "Never push directly to main.", kind="rule", source="caller")
    res = _propose(runtime, rule, "succ00000000015", "Only push to main via PR.")
    _transition(runtime, res["case_id"], res["version"], "apply", promote_as=None)
    assert _taint(db, res["successor_fact_id"]) == "agent_inferred"
