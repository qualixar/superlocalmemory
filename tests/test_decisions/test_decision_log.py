# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-49: every decision a caller makes is recorded with who, channel and time, and listed."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.core import decisions as d
from tests.test_core.test_coverage_backfill import _db


@pytest.fixture
def conn(tmp_path):
    db = _db(tmp_path / "memory.db")
    c = sqlite3.connect(str(db.db_path))
    c.row_factory = sqlite3.Row
    yield c
    c.close()


def test_clean_actor_keeps_a_name_and_drops_markup() -> None:
    assert d.clean_actor("claude-code") == "claude-code"
    assert "<" not in d.clean_actor("me <b>x</b>") and " " not in d.clean_actor("me <b>x</b>")
    assert d.clean_actor("") == "client:local"
    assert len(d.clean_actor("a" * 500)) <= 80


def test_clean_channel_is_one_of_the_four_and_defaults_to_http() -> None:
    for ok in ("mcp", "cli", "http", "dashboard"):
        assert d.clean_channel(ok) == ok
    assert d.clean_channel("MCP ") == "mcp"
    assert d.clean_channel("agent") == "http" and d.clean_channel(None) == "http"


def test_a_decision_is_logged_with_actor_channel_and_time(conn) -> None:
    who = d.Decider("agent:claude-code", "mcp")
    did = d.log(conn, profile_id="default", kind=d.PROMOTION, subject_id="f1", action="promote",
                decider=who, summary="Promoted a memory", detail={"was": "tool_output"})
    row = d.recent(conn, "default")[0]
    assert row["decision_id"] == did and row["actor"] == "agent:claude-code"
    assert row["channel"] == "mcp" and row["decided_at"] and row["undone"] is False
    assert row["detail"] == {"was": "tool_output"} and row["can_undo"] is True


def test_recent_is_newest_first_and_bounded(conn) -> None:
    who = d.Decider("dashboard", "dashboard")
    for i in range(5):
        d.log(conn, profile_id="default", kind=d.DISPUTE, subject_id=f"d{i}", action="settle",
              decider=who)
    assert len(d.recent(conn, "default", limit=3)) == 3
    assert d.recent(conn, "default", limit=0)  # clamped to at least one
    assert d.recent(conn, "other") == []


def test_marking_a_decision_undone_records_who_and_keeps_the_row(conn) -> None:
    who = d.Decider("agent:a", "mcp")
    did = d.log(conn, profile_id="default", kind=d.LABEL, subject_id="i1", action="label",
                decider=who)
    assert d.mark_undone(conn, did, d.Decider("dashboard", "dashboard")) is True
    row = d.recent(conn, "default")[0]
    assert row["undone"] is True and row["undone_by"] == "dashboard"
    assert row["undone_channel"] == "dashboard" and row["can_undo"] is False
    assert d.mark_undone(conn, did, who) is False  # once


def test_the_latest_open_decision_for_a_subject(conn) -> None:
    who = d.Decider("agent:a", "cli")
    first = d.log(conn, profile_id="default", kind=d.PROMOTION, subject_id="f1", action="promote",
                  decider=who)
    d.mark_undone(conn, first, who)
    second = d.log(conn, profile_id="default", kind=d.PROMOTION, subject_id="f1", action="promote",
                   decider=who)
    got = d.open_decision(conn, "default", d.PROMOTION, "f1")
    assert got is not None and got["decision_id"] == second
    assert d.open_decision(conn, "default", d.PROMOTION, "nope") is None


def test_counts_for_memory_health(conn) -> None:
    who = d.Decider("agent:a", "http")
    a = d.log(conn, profile_id="default", kind=d.PROPOSAL, subject_id="s1", action="apply", decider=who)
    d.log(conn, profile_id="default", kind=d.DISPUTE, subject_id="d1", action="settle", decider=who)
    d.mark_undone(conn, a, who)
    assert d.counts(conn, "default") == {"total": 2, "undone": 1, "open": 1}


def test_a_store_before_the_log_lists_nothing_and_writes_nothing() -> None:
    bare = sqlite3.connect(":memory:")
    assert d.recent(bare, "default") == []
    assert d.counts(bare, "default") == {"total": 0, "undone": 0, "open": 0}
    assert d.log(bare, profile_id="default", kind=d.LABEL, subject_id="x", action="label",
                 decider=d.Decider("a", "cli")) == ""


def test_erasing_a_fact_erases_its_promotions_only(conn) -> None:
    who = d.Decider("agent:a", "cli")
    d.log(conn, profile_id="default", kind=d.PROMOTION, subject_id="f1", action="promote", decider=who)
    d.log(conn, profile_id="default", kind=d.DISPUTE, subject_id="f1", action="settle", decider=who)
    conn.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES ('m', 'default', 'x')")
    conn.execute("INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
                 "VALUES ('f1', 'm', 'default', 'x')")
    conn.execute("DELETE FROM atomic_facts WHERE fact_id = 'f1'")
    kinds = [r["kind"] for r in d.recent(conn, "default")]
    assert kinds == [d.DISPUTE]
