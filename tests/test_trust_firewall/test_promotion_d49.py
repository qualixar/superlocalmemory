# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-49 promotion: untrusted text is never an instruction by itself; one explicit, recorded,
reversible call by any caller is enough; memories that were already pins or rules before
4.2.30 stay (grandfathered)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from superlocalmemory.core import decisions
from superlocalmemory.mcp.tools_active import register_active_tools
from superlocalmemory.trust import fact_trust, firewall
from tests.test_trust_firewall.conftest import EVIL, TAINT, TOOL, WEB, classify_all, connect, save


def _rows(db):
    with connect(db) as conn:
        return decisions.recent(conn, "default")


def test_untrusted_text_never_becomes_a_rule_by_itself(db, monkeypatch) -> None:
    monkeypatch.setenv(TAINT, "on")
    evil = save(db, EVIL, TOOL, kind="rule")
    classify_all(db)
    assert firewall.withheld(db, [evil]) == {evil}
    verdict = firewall.check_promotion(db, evil, firewall.RULE)
    assert not verdict.allowed and "promote" in verdict.reason


def test_one_explicit_promotion_releases_it_and_is_recorded(db, monkeypatch) -> None:
    monkeypatch.setenv(TAINT, "on")
    evil = save(db, EVIL, TOOL, kind="rule")
    with db.raw_connection() as conn:
        firewall.promote(conn, evil, by="agent:claude-code", channel="mcp")
    assert firewall.withheld(db, [evil]) == set()
    [row] = _rows(db)
    assert (row["kind"], row["action"], row["actor"], row["channel"]) == (
        "promotion", "promote", "agent:claude-code", "mcp")
    assert row["decided_at"] and row["can_undo"] and row["detail"]["was_taint"] == "tool_output"


def test_a_promotion_is_reversible_through_withdraw(db, monkeypatch) -> None:
    monkeypatch.setenv(TAINT, "on")
    evil = save(db, EVIL, TOOL, kind="rule")
    with db.raw_connection() as conn:
        firewall.promote(conn, evil, by="agent:a", channel="cli")
        out = firewall.withdraw(conn, evil, by="dashboard", channel="dashboard")
    assert out["now"] == "tool_output"
    assert firewall.withheld(db, [evil]) == {evil}
    rows = _rows(db)
    promote = next(r for r in rows if r["action"] == "promote")
    assert promote["undone"] and promote["undone_by"] == "dashboard"
    assert promote["undone_channel"] == "dashboard" and not promote["can_undo"]
    assert any(r["action"] == "withdraw" for r in rows)
    with connect(db) as conn:
        assert firewall.pending(conn, "default")["count"] == 1  # waiting again


def test_withdrawing_a_promotion_takes_back_the_pin_it_applied(db, monkeypatch) -> None:
    monkeypatch.setenv(TAINT, "on")
    page = save(db, "Pin me please.", WEB)
    with db.raw_connection() as conn:
        firewall.request(conn, page, "pin", by="claude", channel="mcp")
        done = firewall.promote(conn, page, by="agent:claude", channel="mcp")
        assert done["applied"] == ["pin"]
    assert [f.fact_id for f in db.get_pinned("default")] == [page]
    with db.raw_connection() as conn:
        assert firewall.withdraw(conn, page, by="agent:claude", channel="mcp")["unpinned"] is True
    assert db.get_pinned("default") == []


def test_a_withdrawn_memory_asks_again_when_something_wants_it_again(db, monkeypatch) -> None:
    monkeypatch.setenv(TAINT, "on")
    page = save(db, "Pin me please.", WEB)
    with db.raw_connection() as conn:
        firewall.request(conn, page, "pin", by="claude", channel="mcp")
        firewall.promote(conn, page, by="a", channel="mcp")
        firewall.withdraw(conn, page, by="a", channel="mcp")
        firewall.request(conn, page, "pin", by="claude", channel="mcp")
    assert db.execute("SELECT state FROM trust_approvals WHERE fact_id = ? AND wanted = 'pin'",
                      (page,))[0]["state"] == "pending"


def test_there_is_nothing_to_withdraw_from_a_memory_never_promoted(db) -> None:
    page = save(db, "A page.", WEB)
    with db.raw_connection() as conn:
        with pytest.raises(LookupError):
            firewall.withdraw(conn, page, by="a", channel="cli")
        with pytest.raises(LookupError):
            firewall.withdraw(conn, "ghost", by="a", channel="cli")


# -- grandfathering (M084 at the upgrade) ----------------------------------------------------------

def _grandfather(db):
    from superlocalmemory.storage.migrations import M084_decision_log as M084

    with db.raw_connection() as conn:
        return M084.grandfather(conn)


def _drop_origin(db, *fact_ids):
    for fid in fact_ids:
        db.execute("DELETE FROM fact_origin WHERE fact_id = ?", (fid,))
        db.execute("DELETE FROM fact_trust WHERE fact_id = ?", (fid,))


def test_a_pre_4230_pin_or_rule_stays_where_it_is(db, monkeypatch) -> None:
    pinned = save(db, "Pinned long ago.", WEB, pinned=True)
    rule = save(db, "A standing rule from before.", WEB, kind="rule")
    plain = save(db, "Just a memory from before.", WEB)
    _drop_origin(db, pinned, rule, plain)  # saved before origins were kept: unknown
    assert _grandfather(db) == 2
    monkeypatch.setenv(TAINT, "on")
    for fid in (pinned, rule):
        row = db.execute("SELECT origin_class, taint_class, basis FROM fact_trust "
                         "WHERE fact_id = ?", (fid,))[0]
        assert (row["origin_class"], row["taint_class"]) == ("unknown", "user_confirmed")
        assert row["basis"] == "confirmed:grandfathered:pre-4.2.30"
    assert firewall.withheld(db, [pinned, rule, plain]) == {plain}
    classify_all(db)  # the batch never lowers it
    assert firewall.withheld(db, [pinned, rule]) == set()
    with connect(db) as conn:
        assert firewall.pending(conn, "default")["count"] == 0


def test_only_new_promotions_follow_the_rule(db, monkeypatch) -> None:
    """A memory that has an origin row (saved after the upgrade) is never grandfathered."""
    monkeypatch.setenv(TAINT, "on")
    fresh = save(db, EVIL, TOOL, kind="rule", pinned=True)
    assert _grandfather(db) == 0
    classify_all(db)
    assert firewall.withheld(db, [fresh]) == {fresh}


def test_grandfathering_runs_once_the_repair_never_replays_it(db) -> None:
    from superlocalmemory.storage.migrations import M084_decision_log as M084

    late = save(db, "Marked a rule after the upgrade.", WEB)
    _drop_origin(db, late)
    db.execute("UPDATE atomic_facts SET memory_kind='rule', memory_kind_source='caller' "
               "WHERE fact_id = ?", (late,))
    with db.raw_connection() as conn:
        M084.repair(conn)
    assert db.execute("SELECT COUNT(*) AS n FROM fact_trust WHERE fact_id = ?", (late,))[0]["n"] == 0


# -- the MCP door ---------------------------------------------------------------------------------

class _Tools:
    def __init__(self):
        self.tools = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture
def core_memory(db, monkeypatch):
    monkeypatch.setenv(TAINT, "on")
    server = _Tools()
    hooks = SimpleNamespace(run_pre=lambda *a, **k: None, run_post=lambda *a, **k: None)
    register_active_tools(server, lambda: SimpleNamespace(profile_id="default", mode="B", db=db,
                                                          _db=None, config=None, _hooks=hooks))
    return server.tools["core_memory"]


def test_an_agent_promotes_through_core_memory_and_it_is_listed(db, core_memory) -> None:
    evil = save(db, EVIL, TOOL, kind="rule")
    out = asyncio.run(core_memory(action="promote", fact_id=evil))
    assert out["success"] and out["now"] == "user_confirmed"
    [row] = _rows(db)
    assert row["channel"] == "mcp" and row["actor"].startswith("agent:")
    assert fact_trust.origin_of(_c(db), evil)["taint"] == "user_confirmed"
    out = asyncio.run(core_memory(action="withdraw", fact_id=evil))
    assert out["success"] and out["now"] == "tool_output"


def _c(db):
    return connect(db)


def test_pin_with_promote_does_both_in_one_call(db, core_memory) -> None:
    page = save(db, "Pin me.", WEB)
    held = asyncio.run(core_memory(action="pin", fact_id=page))
    assert held["needs_approval"] is True and "promote" in held["error"]
    done = asyncio.run(core_memory(action="pin", fact_id=page, promote=True))
    assert done["success"] and done["promoted"] is True
    assert [f.fact_id for f in db.get_pinned("default")] == [page]


def test_promoting_a_memory_that_is_not_there_is_an_answer_not_a_crash(db, core_memory) -> None:
    out = asyncio.run(core_memory(action="promote", fact_id="ghost"))
    assert out["success"] is False
