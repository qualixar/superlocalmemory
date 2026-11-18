# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M-15 / R-18: an automatic supersede needs origin(winner) >= origin(loser); else a dispute.

Poisoning scenario: a web page or a tool's output must never retire what the user said.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.core.judge_ladder import JudgeLadder
from superlocalmemory.storage import fact_origin as fo
from superlocalmemory.storage.models import AtomicFact, MemoryRecord
from superlocalmemory.trust import supersession
from superlocalmemory.truth import disputes, slot_store
from superlocalmemory.truth.dispute_pass import DisputePass
from superlocalmemory.truth.slot_pass import SlotPass
from superlocalmemory.truth.slot_resolver import reconcile_ledger
from tests.test_core.test_coverage_backfill import _db
from tests.test_truth.test_dispute_pass import Fixed, _calibrate
from tests.test_truth.test_slot_pass import _retired

OLD = "The release cadence of SLM is one release every week"
NEW = "The release cadence of SLM is one big release"
ON = {"judge.ladder": "on", "truth.slots": "on", "truth.slots.auto_apply": "on"}

PERSON = fo.Origin("remember_http", "verbatim", source_type="http", role="user",
                  agent_id_source="dashboard")
AGENT = fo.Origin("remember_http", "verbatim", source_type="http", role="assistant",
                  agent_id="claude")
WEB = fo.Origin("ingest_adapter", "verbatim", source_type="gmail")
TOOL = fo.Origin("hook_file_change", "system_built", source_type="http")
NONE = None  # no declaration: recorded as unknown, like a fact from before 4.2.30


def _flags(**states):
    return FeatureFlags(file_values=states, environ={})


def _fact(db, content, created, origin):
    record = MemoryRecord(content=content, session_date=created[:10])
    db.store_memory(record)
    fact = AtomicFact(memory_id=record.memory_id, content=content)
    if origin is None:
        db.store_fact(fact)
    else:
        with fo.declare(origin):
            db.store_fact(fact)
    db.execute("UPDATE atomic_facts SET created_at = ? WHERE fact_id = ?", (created, fact.fact_id))
    return fact.fact_id


def _pair(tmp_path, head_origin, new_origin, *, taint, old_text=OLD, new_text=NEW,
          head_when="2026-03-01T00:00:00Z", new_when="2026-11-14T00:00:00Z"):
    db = _db(tmp_path / "memory.db")
    head = _fact(db, old_text, head_when, head_origin)
    new = _fact(db, new_text, new_when, new_origin)
    flags = _flags(**{"truth.slots": "shadow", "trust.taint": taint}) if taint \
        else _flags(**{"truth.slots": "shadow"})
    counts = SlotPass(db, flags, state_dir=tmp_path).run()
    return db, head, new, counts


def _reason(db) -> str:
    return db.execute("SELECT reason_code FROM slot_disputes")[0]["reason_code"]


def _open(db) -> int:
    return db.execute("SELECT COUNT(*) AS n FROM slot_disputes WHERE state='open'")[0]["n"]


def _judge(db, tmp_path, verdict="replaces", *, taint=None, **extra):
    _calibrate(db, verdict)
    tier = Fixed(verdict)
    states = {**ON, **extra}
    if taint:
        states["trust.taint"] = taint
    counts = DisputePass(db, _flags(**states), JudgeLadder([tier])).run()
    return counts, tier


# -- binding: the dispute's reason ---------------------------------------------------------------

@pytest.mark.parametrize("head,new", [
    (PERSON, WEB), (PERSON, TOOL), (PERSON, AGENT), (AGENT, WEB), (AGENT, TOOL), (TOOL, WEB)])
def test_a_newer_value_from_a_lower_origin_opens_origin_below_head(tmp_path, head, new) -> None:
    db, _h, _n, counts = _pair(tmp_path, head, new, taint="on")
    assert _reason(db) == supersession.ORIGIN_BELOW_HEAD
    assert counts["origin_held"] == 1 and counts["dispute"] == 1


@pytest.mark.parametrize("head,new", [
    (PERSON, PERSON), (WEB, PERSON), (WEB, WEB), (AGENT, PERSON), (TOOL, AGENT), (WEB, TOOL)])
def test_an_equal_or_higher_origin_keeps_the_ordinary_dispute(tmp_path, head, new) -> None:
    db, _h, _n, counts = _pair(tmp_path, head, new, taint="on")
    assert _reason(db) == "pattern_key" and "origin_held" not in counts


@pytest.mark.parametrize("head,new", [(NONE, PERSON), (PERSON, NONE), (NONE, NONE)])
def test_unknown_on_either_side_always_disputes(tmp_path, head, new) -> None:
    db, _h, _n, counts = _pair(tmp_path, head, new, taint="on")
    assert _reason(db) == supersession.ORIGIN_BELOW_HEAD and counts["origin_held"] == 1


def test_taint_not_class_decides(tmp_path) -> None:
    """A fact cut from a web memory is `system` by class but `web` by taint."""
    db = _db(tmp_path / "memory.db")
    web_head = _fact(db, OLD, "2026-03-01T00:00:00Z", WEB)
    cut_id = _fact(db, NEW, "2026-11-14T00:00:00Z", fo.Origin(
        "derived", "split", source_type="coverage-backfill"))
    memory = db.execute("SELECT memory_id FROM atomic_facts WHERE fact_id = ?", (web_head,))[0][0]
    db.execute("UPDATE fact_origin SET parent_memory_id = ? WHERE fact_id = ?", (memory, cut_id))
    with db.raw_connection() as conn:
        assert supersession.decide(conn, winner=cut_id, loser=web_head).winner_class == "web"


def test_shadow_reports_what_it_would_hold_and_changes_nothing(tmp_path) -> None:
    db, _h, _n, counts = _pair(tmp_path, PERSON, WEB, taint="shadow")
    assert _reason(db) == "pattern_key"
    assert counts["origin_would_hold"] == 1 and "origin_held" not in counts


def test_the_kill_switch_means_off(tmp_path) -> None:
    db = _db(tmp_path / "memory.db")
    values = {"truth.slots": "shadow", "trust.taint": "on"}
    live = SlotPass(db, FeatureFlags(file_values=values, environ={}), state_dir=tmp_path)
    killed = SlotPass(db, FeatureFlags(file_values=values, environ={}, killed=True),
                      state_dir=tmp_path)
    assert live._taint_state() == "on" and killed._taint_state() == "off"


# -- the judge -----------------------------------------------------------------------------------

def test_flag_off_is_exactly_today_even_for_a_web_page_over_the_user(tmp_path) -> None:
    db, head, new, counts = _pair(tmp_path, PERSON, WEB, taint=None)
    assert _reason(db) == "pattern_key" and not any(k.startswith("origin") for k in counts)
    counts, tier = _judge(db, tmp_path)
    assert counts["resolved_replaces"] == 1 and _retired(db, head)[2] == new
    assert "origin_below_head" not in counts


def test_a_web_page_cannot_retire_what_the_user_said(tmp_path) -> None:
    db, head, _new, _c = _pair(tmp_path, PERSON, WEB, taint="on")
    counts, tier = _judge(db, tmp_path, taint="on")
    assert tier.calls == 0, "a refused dispute is no judge's to answer"
    assert _open(db) == 1 and _retired(db, head) == (None, None, None)
    assert not any(k.startswith("resolved") for k in counts)


def test_a_dispute_opened_before_the_flag_is_held_when_the_judge_comes(tmp_path) -> None:
    db, head, new, _c = _pair(tmp_path, PERSON, TOOL, taint=None)
    assert _reason(db) == "pattern_key"
    counts, tier = _judge(db, tmp_path, taint="on")
    assert counts["origin_below_head"] == 1 and "resolved_replaces" not in counts
    assert _reason(db) == supersession.ORIGIN_BELOW_HEAD
    assert _open(db) == 1 and _retired(db, head) == (None, None, None)
    assert db.execute("SELECT fact_id FROM slot_revisions r JOIN memory_slots s "
                      "ON s.head_revision_id = r.revision_id")[0][0] == head
    assert new != head


def test_an_equal_origin_is_retired_as_before(tmp_path) -> None:
    db, head, new, _c = _pair(tmp_path, PERSON, PERSON, taint="on")
    counts, _ = _judge(db, tmp_path, taint="on")
    assert counts["resolved_replaces"] == 1 and _retired(db, head)[2] == new


def test_a_higher_origin_retires_a_lower_one(tmp_path) -> None:
    db, head, new, _c = _pair(tmp_path, WEB, PERSON, taint="on")
    counts, _ = _judge(db, tmp_path, taint="on")
    assert counts["resolved_replaces"] == 1 and _retired(db, head)[2] == new


def test_unknown_head_is_never_retired_by_a_judge(tmp_path) -> None:
    db, head, _new, _c = _pair(tmp_path, NONE, PERSON, taint=None)
    counts, _ = _judge(db, tmp_path, taint="on")
    assert counts["origin_below_head"] == 1 and _retired(db, head) == (None, None, None)


def test_non_retiring_answers_are_not_blocked_by_origin(tmp_path) -> None:
    db, head, _new, _c = _pair(tmp_path, PERSON, WEB, taint=None)
    counts, _ = _judge(db, tmp_path, "both_true", taint="on")
    assert counts["resolved_both_true"] == 1 and _retired(db, head) == (None, None, None)


def test_an_older_value_retires_by_the_same_rule_with_the_head_as_winner(tmp_path) -> None:
    """The challenger is the older text; `replaces` keeps the head and retires the challenger."""
    kept = dict(head_when="2026-11-14T00:00:00Z", new_when="2026-03-01T00:00:00Z",
                old_text=NEW, new_text=OLD)
    (tmp_path / "a").mkdir()
    db, head, challenger, _c = _pair(tmp_path / "a", PERSON, WEB, taint="on", **kept)
    assert _reason(db) == "older_than_head"
    counts, _ = _judge(db, tmp_path / "a", taint="on")
    assert counts["resolved_replaces"] == 1 and _retired(db, challenger)[2] == head  # user wins

    (tmp_path / "b").mkdir()
    db, head, challenger, _c = _pair(tmp_path / "b", WEB, PERSON, taint="on", **kept)
    counts, _ = _judge(db, tmp_path / "b", taint="on")
    assert counts["origin_below_head"] == 1 and _retired(db, challenger) == (None, None, None)
    assert _reason(db) == "older_than_head"  # the reason its settling needs is kept


# -- a person decides ----------------------------------------------------------------------------

def test_a_person_can_settle_a_held_dispute_either_way(tmp_path) -> None:
    db, head, new, _c = _pair(tmp_path, PERSON, WEB, taint="on")
    with db.raw_connection() as conn:
        view = disputes.listing(conn, "default")[0]
        assert view["reason"] == supersession.ORIGIN_BELOW_HEAD
        assert view["question"] == disputes.ORIGIN_QUESTION and view["next"] == disputes.ASK_USER
        did = view["dispute_id"]
        done = disputes.settle(conn, profile_id="default", dispute_id=did, keep=new,
                               channel="dashboard", actor="alice")
    assert done["status"] == "settled" and done["retired"] == [head]
    assert _retired(db, head)[2] == new


# -- the ledger step -----------------------------------------------------------------------------

def _superseded_without_a_case(tmp_path, head_origin, new_origin):
    db, head, new, _c = _pair(tmp_path, head_origin, new_origin, taint=None)
    with db.raw_connection() as conn:
        did = conn.execute("SELECT dispute_id FROM slot_disputes").fetchone()[0]
        slot_store.resolve_dispute(conn, did, "b", resolved_by="test")
    return db, head, new


def test_reconcile_proposes_instead_of_applying_when_the_lattice_refuses(tmp_path) -> None:
    db, head, new = _superseded_without_a_case(tmp_path, PERSON, WEB)
    with db.raw_connection() as conn:
        counts = reconcile_ledger(conn, auto_apply=True, origin_gate=supersession.allows)
    assert counts["origin_below_head"] == 1 and counts["ledger_proposed"] == 1
    assert _retired(db, head) == (None, None, None)
    assert db.execute("SELECT status FROM correction_cases")[0]["status"] == "proposed"


def test_reconcile_applies_when_the_lattice_allows_and_when_there_is_no_gate(tmp_path) -> None:
    db, head, new = _superseded_without_a_case(tmp_path, WEB, PERSON)
    with db.raw_connection() as conn:
        counts = reconcile_ledger(conn, auto_apply=True, origin_gate=supersession.allows)
    assert counts["ledger_applied"] == 1 and "origin_below_head" not in counts
    assert _retired(db, head)[2] == new
    (tmp_path / "off").mkdir()
    db, head, new = _superseded_without_a_case(tmp_path / "off", PERSON, WEB)
    with db.raw_connection() as conn:
        assert reconcile_ledger(conn, auto_apply=True)["ledger_applied"] == 1  # flag off: today
    assert _retired(db, head)[2] == new


@pytest.mark.parametrize("taint,applied", [("on", 0), ("off", 1)])
def test_the_slot_pass_gates_its_ledger_step_only_while_taint_is_on(tmp_path, taint, applied) -> None:
    db, head, new = _superseded_without_a_case(tmp_path, PERSON, WEB)
    flags = _flags(**{"truth.slots": "on", "truth.slots.auto_apply": "on", "trust.taint": taint})
    counts = SlotPass(db, flags, state_dir=tmp_path).run()
    assert counts.get("ledger_applied", 0) == applied
    assert counts.get("origin_below_head", 0) == 1 - applied
    assert (_retired(db, head)[2] == new) is bool(applied)
