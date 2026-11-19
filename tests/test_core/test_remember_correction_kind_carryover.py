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
)
from superlocalmemory.storage.migrations import M052_memory_kinds as m052
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord


@pytest.fixture(autouse=True)
def _origin_firewall_off(monkeypatch):
    """These tests are about what session start selects, not the origin firewall: they save through agent
    doors (``agent_inferred``), which the firewall (on by default since 19 Nov) holds back until a promotion.
    tests/test_trust_firewall covers that rule."""
    monkeypatch.setenv("SLM_FEATURE_TRUST_TAINT", "off")

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
    return db.store_fact(fact)


def _kind_row(db, fact_id) -> dict:
    rows = db.execute(
        "SELECT memory_kind, memory_kind_source FROM atomic_facts WHERE fact_id=?",
        (fact_id,))
    return dict(rows[0]) if rows else {}


def _propose(runtime, fact_id, successor_id, content, key="prop-1"):
    return runtime.create_correction_successor(
        "default", fact_id, successor_id, content,
        trusted_actor_id=ACTOR, idempotency_key=key)


def _transition(runtime, case_id, version, action, key="trans-1"):
    return runtime.transition_correction(
        "default", case_id, action=action, expected_version=version,
        actor_id=ACTOR, idempotency_key=key)


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
