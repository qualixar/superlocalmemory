# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Open disputes as agents and people see and settle them (D-37, D-39)."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.truth import disputes
from superlocalmemory.truth.slot_pass import SlotPass
from tests.test_core.test_coverage_backfill import _db
from tests.test_truth.test_slot_pass import _fact, _flags, _retired


@pytest.fixture
def disputed(tmp_path):
    db = _db(tmp_path / "memory.db")
    old = _fact(db, "The release cadence of SLM is one release every week", "2026-03-01T00:00:00Z")
    new = _fact(db, "The release cadence of SLM is one big release", "2026-11-14T00:00:00Z")
    SlotPass(db, _flags(**{"truth.slots": "shadow"}), state_dir=tmp_path).run()
    return db, old, new


def _open(db) -> int:
    return db.execute("SELECT COUNT(*) AS n FROM slot_disputes WHERE state='open'")[0]["n"]


def _head(db) -> str:
    return db.execute("SELECT r.fact_id FROM memory_slots s JOIN slot_revisions r "
                      "ON r.revision_id = s.head_revision_id")[0]["fact_id"]


def _settle(db, keep, channel="mcp"):
    with db.raw_connection() as conn:
        did = conn.execute("SELECT dispute_id FROM slot_disputes WHERE state='open'").fetchone()[0]
        return disputes.settle(conn, profile_id="default", dispute_id=did, keep=keep,
                               channel=channel, actor="agent:claude-code")


def test_a_returned_fact_carries_both_sides_newest_first(disputed) -> None:
    db, old, new = disputed
    with db.raw_connection() as conn:
        found = disputes.for_facts(conn, "default", [new])
    assert list(found) == [new]
    (view,) = found[new]
    assert view["next"] == disputes.ASK_USER and view["question"] == disputes.QUESTION
    assert [s["fact_id"] for s in view["sides"]] == [new, old]
    assert [s["current"] for s in view["sides"]] == [False, True]  # the old value is still head
    assert all(s["visible"] and s["content"] for s in view["sides"])
    assert view["sides"][0]["said_at"].startswith("2026-11-14")


def test_a_side_the_caller_may_not_see_keeps_the_dispute_but_not_its_text(disputed) -> None:
    db, old, new = disputed
    with db.raw_connection() as conn:
        found = disputes.for_facts(conn, "default", [new], hidden=lambda ids: set(ids))
    other = next(s for s in found[new][0]["sides"] if s["fact_id"] == old)
    assert other["content"] is None and other["visible"] is False
    mine = next(s for s in found[new][0]["sides"] if s["fact_id"] == new)
    assert mine["visible"] and mine["content"]


def test_an_undisputed_fact_and_another_profile_get_nothing(disputed) -> None:
    db, old, new = disputed
    with db.raw_connection() as conn:
        assert disputes.for_facts(conn, "default", ["no-such-fact"]) == {}
        assert disputes.for_facts(conn, "someone-else", [new]) == {}
        assert disputes.listing(conn, "someone-else") == []


def test_the_listing_and_count_show_the_open_dispute(disputed) -> None:
    db, _old, _new = disputed
    with db.raw_connection() as conn:
        (view,) = disputes.listing(conn, "default")
        assert disputes.open_count(conn, "default") == 1
    assert view["state"] == "disputed" and len(view["sides"]) == 2


def test_keeping_the_newer_side_retires_the_older_through_the_ledger(disputed) -> None:
    db, old, new = disputed
    result = _settle(db, new)
    assert result["status"] == disputes.SETTLED
    assert result["resolution"] == "b" and result["retired"] == [old]
    assert result["settled_by"] == "agent:claude-code@mcp"
    assert _open(db) == 0 and _head(db) == new
    _until, expired, by = _retired(db, old)
    assert by == new and expired is not None
    row = db.execute("SELECT resolved_by FROM slot_disputes")[0]
    assert row["resolved_by"] == "agent:claude-code@mcp"


def test_keeping_the_older_side_retires_the_newer(disputed) -> None:
    db, old, new = disputed
    result = _settle(db, old, channel="dashboard")
    assert result["resolution"] == "a" and result["retired"] == [new]
    assert _head(db) == old
    _until, expired, _by = _retired(db, new)
    assert expired is not None


@pytest.mark.parametrize("keep,state", [("both", "compatible"), ("unrelated", "unbound")])
def test_both_or_unrelated_retires_nothing(disputed, keep, state) -> None:
    db, old, new = disputed
    result = _settle(db, keep, channel="cli")
    assert result["retired"] == [] and _open(db) == 0
    assert db.execute("SELECT state FROM slot_revisions WHERE fact_id = ?", (new,))[0][0] == state
    for fact in (old, new):
        assert _retired(db, fact)[1] is None


def test_a_wrong_answer_or_channel_changes_nothing(disputed) -> None:
    db, _old, _new = disputed
    with pytest.raises(disputes.SettleRefused):
        _settle(db, "some-other-fact")
    with pytest.raises(disputes.SettleRefused):
        _settle(db, "both", channel="agent")  # not one of mcp, cli, http, dashboard
    assert _open(db) == 1


def test_a_closed_or_missing_dispute_is_not_found(disputed) -> None:
    db, _old, new = disputed
    _settle(db, new)
    with db.raw_connection() as conn, pytest.raises(LookupError):
        disputes.settle(conn, profile_id="default", dispute_id="nope", keep=new, channel="cli")


def test_a_dispute_the_slot_moved_past_is_closed_and_nothing_else_changes(disputed) -> None:
    db, old, new = disputed
    with db.raw_connection() as conn:  # the slot's head moved since the dispute opened
        conn.execute("UPDATE memory_slots SET head_revision_id = 'elsewhere'")
    result = _settle(db, new)
    assert result["status"] == disputes.STALE and result["retired"] == []
    assert db.execute("SELECT state FROM slot_disputes")[0][0] == "dismissed"
    for fact in (old, new):
        assert _retired(db, fact)[1] is None


def test_a_store_without_slot_tables_has_no_disputes() -> None:
    conn = sqlite3.connect(":memory:")
    assert disputes.for_facts(conn, "default", ["f"]) == {}
    assert disputes.listing(conn, "default") == []
    assert disputes.open_count(conn, "default") == 0


def test_a_ledger_that_will_not_apply_leaves_everything_as_it_was(disputed) -> None:
    from tests.test_truth.test_slot_pass import _consolidator_proposal

    db, old, new = disputed
    rival = _fact(db, "Another memory a consolidator matched", "2026-05-01T00:00:00Z")
    _consolidator_proposal(db, "cons-rival", old, rival)  # an open case for another successor
    with pytest.raises(disputes.SettleRefused, match="did not apply"):
        _settle(db, new)
    assert _open(db) == 1 and _retired(db, old)[1] is None


# -- D-49: every answer is recorded and can be taken back -----------------------------------------

def _log(db):
    return [tuple(r) for r in db.execute(
        "SELECT action, actor, channel, reverted_by FROM decision_log ORDER BY rowid")]


def test_an_answer_is_recorded_with_who_channel_and_time(disputed) -> None:
    db, _old, new = disputed
    _settle(db, new)
    assert _log(db) == [("settle", "agent:claude-code", "mcp", None)]
    row = db.execute("SELECT summary, decided_at FROM decision_log")[0]
    assert "newer" in row["summary"] and row["decided_at"]


def _reopen(db, channel="dashboard", actor="dashboard"):
    with db.raw_connection() as conn:
        did = conn.execute("SELECT dispute_id FROM slot_disputes").fetchone()[0]
        return disputes.reopen(conn, profile_id="default", dispute_id=did, channel=channel,
                               actor=actor)


@pytest.mark.parametrize("keep", ["new", "old"])
def test_reopening_brings_the_retired_memory_back_and_the_question_open(disputed, keep) -> None:
    db, old, new = disputed
    snapshot = [tuple(r) for r in db.execute(
        "SELECT revision_id, state, superseded_by, reason_code FROM slot_revisions "
        "ORDER BY revision_id")]
    _settle(db, new if keep == "new" else old)
    loser = old if keep == "new" else new
    assert _retired(db, loser)[1] is not None
    out = _reopen(db)
    assert out["status"] == "open" and out["restored"] == [loser]
    assert _retired(db, loser)[1] is None and _open(db) == 1 and _head(db) == old
    assert [tuple(r) for r in db.execute(
        "SELECT revision_id, state, superseded_by, reason_code FROM slot_revisions "
        "ORDER BY revision_id")] == snapshot
    row = db.execute("SELECT state, resolution, resolved_by FROM slot_disputes")[0]
    assert (row["state"], row["resolution"], row["resolved_by"]) == ("open", None, None)
    assert _log(db) == [("settle", "agent:claude-code", "mcp", "dashboard"),
                        ("reopen", "dashboard", "dashboard", None)]
    again = _settle(db, new)  # and it can be answered again
    assert again["status"] == disputes.SETTLED


@pytest.mark.parametrize("keep", ["both", "unrelated"])
def test_reopening_a_both_or_unrelated_answer(disputed, keep) -> None:
    db, old, new = disputed
    _settle(db, keep)
    assert _reopen(db)["restored"] == []
    assert _open(db) == 1 and _head(db) == old
    assert db.execute("SELECT state FROM slot_revisions WHERE fact_id = ?", (new,))[0][0] == "disputed"


def test_a_dispute_that_is_not_settled_cannot_be_reopened(disputed) -> None:
    db, _old, _new = disputed
    with db.raw_connection() as conn:
        did = conn.execute("SELECT dispute_id FROM slot_disputes").fetchone()[0]
        with pytest.raises(LookupError):
            disputes.reopen(conn, profile_id="default", dispute_id=did, channel="cli")
        with pytest.raises(LookupError):
            disputes.reopen(conn, profile_id="default", dispute_id="nope", channel="cli")


def test_a_value_that_moved_on_cannot_be_reopened(disputed) -> None:
    db, _old, new = disputed
    _settle(db, new)
    with db.raw_connection() as conn:
        conn.execute("UPDATE memory_slots SET head_revision_id = 'elsewhere'")
        did = conn.execute("SELECT dispute_id FROM slot_disputes").fetchone()[0]
        with pytest.raises(disputes.ReopenRefused, match="changed since"):
            disputes.reopen(conn, profile_id="default", dispute_id=did, channel="cli")
    assert db.execute("SELECT state FROM slot_disputes")[0][0] == "resolved"
