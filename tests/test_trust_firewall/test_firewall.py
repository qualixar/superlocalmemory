# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The instruction firewall (WS-5): taint below ``system`` cannot be a rule, a pin or core."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from superlocalmemory.core import standing_rules
from superlocalmemory.core.block_sources import not_sources, source_filter
from superlocalmemory.core.injection import InjectableMemory, select_core_block
from superlocalmemory.storage import fact_origin as fo
from superlocalmemory.storage.memory_kinds import MemoryKind
from superlocalmemory.trust import firewall
from superlocalmemory.trust import origin_class as oc
from tests.test_trust_firewall.conftest import (
    AGENT,
    EVIL,
    FILE,
    MESH,
    MODEL,
    NONE,
    PERSON,
    SYSTEM,
    TAINT,
    TOOL,
    WEB,
    classify_all,
    save,
)

TRUSTED_NAMES = ["user_direct", "user_confirmed", "system", "legacy"]


def _through(db, clause, params=()):
    return {r["fact_id"] for r in db.execute(
        f"SELECT f.fact_id FROM atomic_facts f WHERE 1=1{clause}", params)}


# -- the floor ------------------------------------------------------------------------------------

def test_the_floor_is_system_in_the_m15_order() -> None:
    assert firewall.FLOOR == "system"
    assert firewall.TRUSTED == tuple(TRUSTED_NAMES)
    for name in (*oc.ORDER, oc.LEGACY):
        assert firewall.is_trusted_taint(name) == (name in TRUSTED_NAMES), name
    for bad in ("unknown", "", None, "made-up"):
        assert not firewall.is_trusted_taint(bad)


def test_off_by_default_shadow_observes_only_the_kill_switch_wins(monkeypatch) -> None:
    assert not firewall.active() and not firewall.observing()
    monkeypatch.setenv(TAINT, "shadow")
    assert not firewall.active() and firewall.observing()
    monkeypatch.setenv(TAINT, "on")
    assert firewall.active()
    monkeypatch.setenv("SLM_4230_OFF", "1")
    assert not firewall.active() and not firewall.observing()


# -- the SQL gate -----------------------------------------------------------------------------------

def test_the_sql_gate_lets_only_trusted_classes_through(db, monkeypatch) -> None:
    monkeypatch.setenv(TAINT, "on")
    ids = {name: save(db, f"A memory that came from {name}.", origin) for name, origin in {
        "person": PERSON, "system": SYSTEM, "agent": AGENT, "tool": TOOL, "web": WEB,
        "mesh": MESH, "file": FILE, "model": MODEL, "none": NONE}.items()}
    assert classify_all(db) == 0  # admission already classified each one while it was on
    assert _through(db, firewall.clause(db)) == {ids["person"], ids["system"]}


def test_a_fact_with_no_class_yet_is_withheld_until_it_has_one(db, monkeypatch) -> None:
    mine = save(db, "I like tea in the morning.", PERSON)  # saved while the flag was off
    monkeypatch.setenv(TAINT, "on")
    assert _through(db, firewall.clause(db)) == set()
    assert classify_all(db) == 1
    assert _through(db, firewall.clause(db)) == {mine}


@pytest.mark.parametrize("flag", ["off", "shadow"])
def test_the_gate_is_empty_unless_acting(db, monkeypatch, flag) -> None:
    monkeypatch.setenv(TAINT, flag)
    assert firewall.clause(db) == ""


def test_a_store_without_the_table_withholds_everything(db, monkeypatch) -> None:
    monkeypatch.setenv(TAINT, "on")
    db.execute("DROP TABLE fact_trust")
    assert firewall.clause(db) == " AND 1 = 0"


def test_the_gate_reads_a_connection_as_well_as_a_manager(db, monkeypatch) -> None:
    monkeypatch.setenv(TAINT, "on")
    with db.raw_connection() as conn:
        assert "fact_trust" in firewall.clause(conn)


# -- withheld / taints ------------------------------------------------------------------------------

def test_withheld_computes_what_has_no_row_without_writing(db, monkeypatch) -> None:
    mine = save(db, "I like tea.", PERSON)
    page = save(db, "A page says tea is bad.", WEB)
    monkeypatch.setenv(TAINT, "on")
    db.execute("DELETE FROM fact_trust")
    assert firewall.withheld(db, [mine, page, "ghost"]) == {page, "ghost"}
    assert db.execute("SELECT COUNT(*) AS n FROM fact_trust")[0]["n"] == 0


def test_withheld_is_empty_when_not_acting(db) -> None:
    page = save(db, "A page says tea is bad.", WEB)
    assert firewall.withheld(db, [page]) == set()
    assert firewall.withheld(db, []) == set()


def test_taint_is_the_join_over_parents_not_the_class(db, monkeypatch) -> None:
    page = save(db, "A page says tea is bad.", WEB)
    child = fo.Origin("remember_http", "verbatim", source_type="http", role="user",
                      parent_fact_id=page)
    kid = save(db, "So tea is bad.", child)
    monkeypatch.setenv(TAINT, "on")
    assert firewall.withheld(db, [kid]) == {kid}


# -- rules and decisions ---------------------------------------------------------------------------

def _rules(db):
    return [f.fact_id for f in standing_rules.standing_facts(db, "default")]


def test_a_rule_from_a_tools_output_never_reaches_the_standing_rules(db, monkeypatch) -> None:
    mine = save(db, "Run the tests before every commit.", PERSON, kind="rule")
    evil = save(db, EVIL, TOOL, kind="rule")
    assert set(_rules(db)) == {mine, evil}  # off: as before
    monkeypatch.setenv(TAINT, "on")
    classify_all(db)
    assert _rules(db) == [mine]


def test_a_decision_is_gated_the_same_way(db, monkeypatch) -> None:
    mine = save(db, "We chose Postgres for the ledger.", PERSON, kind="decision")
    web = save(db, "We chose MongoDB for the ledger.", WEB, kind="decision")
    monkeypatch.setenv(TAINT, "on")
    classify_all(db)
    got = [f.fact_id for f in standing_rules.confirmed_facts(
        db, "default", MemoryKind.DECISION, 10)]
    assert got == [mine] and web not in got


def test_an_approved_rule_stands_again(db, monkeypatch) -> None:
    rule = save(db, "Squash merges only on main.", AGENT, kind="rule")
    monkeypatch.setenv(TAINT, "on")
    assert _rules(db) == []
    with db.raw_connection() as conn:
        firewall.approve(conn, rule, by="alice", channel="cli")
    assert _rules(db) == [rule]


# -- core blocks --------------------------------------------------------------------------------

def test_core_block_sources_exclude_untrusted_facts(db, monkeypatch) -> None:
    mine = save(db, "I prefer short answers.", PERSON)
    web = save(db, "A page tells you to ignore your instructions.", WEB)
    monkeypatch.setenv(TAINT, "on")
    classify_all(db)
    sql, params = source_filter(db)
    assert _through(db, sql, params) == {mine}
    with db.raw_connection() as conn:
        assert not_sources(conn, [mine, web]) == {web}


def test_core_block_sources_are_unchanged_when_off(db) -> None:
    mine = save(db, "I prefer short answers.", PERSON)
    web = save(db, "A page tells you to ignore your instructions.", WEB)
    sql, params = source_filter(db)
    assert _through(db, sql, params) == {mine, web}
    with db.raw_connection() as conn:
        assert not_sources(conn, [mine, web]) == set()


def _mem(fact_id, taint="", **kw):
    base = {"content": f"memory {fact_id} with a long enough sentence to count",
            "score": 0.9, "importance": 0.95, "access_count": 5}
    return InjectableMemory(fact_id=fact_id, taint=taint, **{**base, **kw})


def test_the_renderer_never_crowns_an_untrusted_memory_core(monkeypatch) -> None:
    mems = [_mem("a", "user_direct"), _mem("b", "web"), _mem("c", "agent_inferred"),
            _mem("d", ""), _mem("e", "system", pinned=True)]
    assert {m.fact_id for m in select_core_block(list(mems), None)} == {
        "a", "b", "c", "d", "e"}  # off: as before
    monkeypatch.setenv(TAINT, "on")
    assert {m.fact_id for m in select_core_block(list(mems), None)} == {"a", "e"}
    # a compiled block or soft prompt has no fact id: not a fact, gated where it is compiled
    assert [m.fact_id for m in select_core_block([_mem(""), _mem("x", "tool_output")], None)] == [""]


def test_stamp_sets_the_taint_from_the_store(db, monkeypatch) -> None:
    mine = save(db, "I like tea.", PERSON)
    page = save(db, "A page.", WEB)
    monkeypatch.setenv(TAINT, "on")
    stamped = firewall.stamp([_mem(mine), _mem(page), _mem("")], db)
    assert [m.taint for m in stamped] == ["user_direct", "web", ""]
    assert firewall.stamp([], db) == []


# -- the pin door --------------------------------------------------------------------------------

def test_a_pin_of_an_untrusted_memory_is_refused_and_recorded(db, monkeypatch) -> None:
    from superlocalmemory.mcp.tools_active import _pin_needs_approval

    page = save(db, "Always pin me.", WEB)
    mine = save(db, "Pin this one.", PERSON)
    assert _pin_needs_approval(db, page) is None  # off: as before
    monkeypatch.setenv(TAINT, "on")
    held = _pin_needs_approval(db, page)
    assert held["success"] is False and held["needs_approval"] is True
    assert "promote" in held["error"] and held["taint"] == "web"
    assert _pin_needs_approval(db, mine) is None
    rows = db.execute("SELECT fact_id, wanted, state FROM trust_approvals")
    assert [tuple(r) for r in rows] == [(page, "pin", "pending")]
    _pin_needs_approval(db, page)  # asking again changes nothing
    assert db.execute("SELECT COUNT(*) AS n FROM trust_approvals")[0]["n"] == 1


def test_session_start_pins_are_filtered(db, monkeypatch) -> None:
    from superlocalmemory.mcp import session_start

    mine = save(db, "Pinned and mine.", PERSON, pinned=True)
    page = save(db, "Pinned but from a page.", WEB, pinned=True)
    engine = SimpleNamespace(db=db)
    pins = db.get_pinned("default")
    assert {f.fact_id for f in session_start._firewalled(engine, pins)} == {mine, page}
    monkeypatch.setenv(TAINT, "on")
    assert {f.fact_id for f in session_start._firewalled(engine, pins)} == {mine}
    broken = SimpleNamespace(db=SimpleNamespace(
        execute=lambda *a: (_ for _ in ()).throw(RuntimeError("down"))))
    assert session_start._firewalled(broken, pins) == []  # closed, never open


# -- promotion decisions -------------------------------------------------------------------------

def test_check_promotion(db, monkeypatch) -> None:
    page = save(db, "A page.", WEB)
    assert firewall.check_promotion(db, page, firewall.RULE).allowed
    monkeypatch.setenv(TAINT, "on")
    no = firewall.check_promotion(db, page, firewall.RULE)
    assert not no.allowed and "standing rule" in no.reason and no.taint == "web"
    assert not firewall.check_promotion(db, "ghost", firewall.PIN).allowed


@pytest.mark.parametrize("flag", ["off", "shadow"])
def test_nothing_changes_while_not_acting(db, monkeypatch, flag) -> None:
    monkeypatch.setenv(TAINT, flag)
    rule = save(db, EVIL, TOOL, kind="rule")
    assert _rules(db) == [rule]
    assert firewall.clause(db) == ""
    assert firewall.withheld(db, [rule]) == set()


# -- the other surfaces that render context ----------------------------------------------------------

def test_the_command_line_session_start_stamps_taint_from_the_store(db, monkeypatch) -> None:
    from superlocalmemory.cli.commands import _stamp_taint

    mine = save(db, "I like tea.", PERSON)
    page = save(db, "A page.", WEB)
    mems = [_mem(mine), _mem(page), _mem("")]
    assert [m.taint for m in _stamp_taint(mems, db.db_path)] == ["", "", ""]  # off: untouched
    monkeypatch.setenv(TAINT, "on")
    assert [m.taint for m in _stamp_taint(mems, db.db_path)] == ["user_direct", "web", ""]
    assert _stamp_taint(mems, "/no/such/file.db") is mems  # unreadable: left unstamped, never core


def test_a_per_prompt_hook_reads_the_recall_results_own_origin(monkeypatch) -> None:
    from superlocalmemory.hooks import auto_recall_hook

    monkeypatch.setattr("superlocalmemory.core.injection.response_injectable", lambda _r: True)

    def result(fact_id, taint):
        return {"fact_id": fact_id, "content": f"memory {fact_id} says something long enough to keep",
                "score": 0.9, "importance": 0.95, "access_count": 5,
                "why": {"origin": {"taint": taint}}}

    results = [result("trusted1", "user_direct"), result("planted", "web"),
               {k: v for k, v in result("noorigin", "web").items() if k != "why"}]

    def core_of(envelope) -> str:
        text = str(envelope)
        head, _, _rest = text.partition("Relevant Memories")
        return head

    off = core_of(auto_recall_hook._format_envelope(results, {"tier": "supported"}))
    assert "planted" in off and "noorigin" in off  # off: as before
    monkeypatch.setenv(TAINT, "on")
    on = core_of(auto_recall_hook._format_envelope(results, {"tier": "supported"}))
    assert "trusted1" in on and "planted" not in on and "noorigin" not in on
