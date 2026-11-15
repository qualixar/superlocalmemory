# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Slot decisions: head, duplicate, compatible, historical, supersede, dispute."""

from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from superlocalmemory.storage.migrations import M069_memory_slots as M069
from superlocalmemory.truth import slot_store as ss
from superlocalmemory.truth.slot_keys import extract


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY)")
    M069.apply(c)
    c.executemany("INSERT INTO atomic_facts VALUES (?)", [(f"f{i}",) for i in range(1, 8)])
    return c


def _bind(conn, fact, text, when=None, key=None):
    key = key or extract(text)
    assert key is not None, text
    return ss.bind(conn, profile_id="p", scope="personal", fact_id=fact, key=key,
                   valid_from=when)


def _slot(conn, slot_id):
    return conn.execute("SELECT head_revision_id, state, version FROM memory_slots "
                        "WHERE slot_id = ?", (slot_id,)).fetchone()


def _state(conn, revision_id):
    return conn.execute("SELECT state, superseded_by FROM slot_revisions WHERE revision_id=?",
                        (revision_id,)).fetchone()


def test_first_value_becomes_head(conn) -> None:
    b = _bind(conn, "f1", "The release cadence of SLM is monthly", "2026-01-01")
    assert b.outcome == ss.HEAD
    assert _slot(conn, b.slot_id) == (b.revision_id, "current", 1)


def _explicit(value):
    return extract(value, explicit="SLM/release cadence")


def test_a_pattern_key_with_a_new_value_opens_a_dispute(conn) -> None:
    head = _bind(conn, "f1", "The release cadence of SLM is monthly", "2026-01-01")
    new = _bind(conn, "f2", "The release cadence of SLM is one big release", "2026-11-14")
    assert new.outcome == ss.DISPUTE and new.other_fact_id == "f1"
    assert _slot(conn, head.slot_id)[0:2] == (head.revision_id, "disputed")


def test_newer_value_supersedes_and_moves_the_head(conn) -> None:
    old = _bind(conn, "f1", "", "2026-01-01", key=_explicit("monthly"))
    new = _bind(conn, "f2", "", "2026-11-14", key=_explicit("one big release"))
    assert new.outcome == ss.SUPERSEDE and new.other_fact_id == "f1"
    assert _slot(conn, new.slot_id)[0:2] == (new.revision_id, "current")
    assert _state(conn, old.revision_id) == ("superseded", new.revision_id)


def test_same_value_again_is_a_duplicate(conn) -> None:
    first = _bind(conn, "f1", "The release cadence of SLM is monthly.", "2026-01-01")
    again = _bind(conn, "f2", "the release cadence of SLM is MONTHLY", "2026-02-01")
    assert again.outcome == ss.DUPLICATE
    assert _slot(conn, first.slot_id)[0] == first.revision_id


def test_plural_slots_keep_both(conn) -> None:
    a = _bind(conn, "f1", "Acme uses SQLite for storage", "2026-01-01")
    b = _bind(conn, "f2", "Acme uses LanceDB for storage", "2026-02-01")
    assert b.outcome == ss.COMPATIBLE and _slot(conn, a.slot_id)[0] == a.revision_id


def test_an_older_value_joins_history_without_replacing(conn) -> None:
    head = _bind(conn, "f1", "", "2026-11-14", key=_explicit("one big release"))
    past = _bind(conn, "f2", "", "2026-03-01", key=_explicit("weekly"))
    assert past.outcome == ss.HISTORICAL
    assert _slot(conn, head.slot_id)[0] == head.revision_id
    assert _state(conn, past.revision_id) == ("historical", head.revision_id)


def test_an_unconfident_key_opens_a_dispute_and_keeps_the_head(conn) -> None:
    head = _bind(conn, "f1", "The release cadence of SLM is monthly", "2026-01-01")
    weak = extract("The release cadence of SLM is weekly")  # a pattern key
    d = _bind(conn, "f2", "", "2026-02-01", key=weak)
    assert d.outcome == ss.DISPUTE and d.dispute_id
    assert _slot(conn, head.slot_id)[0:2] == (head.revision_id, "disputed")
    assert conn.execute("SELECT state FROM slot_disputes").fetchone() == ("open",)


@pytest.mark.parametrize("resolution,retire,head_is_b,state", [
    ("a", ("f2",), False, "current"), ("b", ("f1",), True, "current"),
    ("both", (), False, "current"), ("neither", ("f1", "f2"), None, "empty")])
def test_dispute_resolution(conn, resolution, retire, head_is_b, state) -> None:
    head = _bind(conn, "f1", "The release cadence of SLM is monthly", "2026-01-01")
    weak = extract("The release cadence of SLM is weekly")  # a pattern key
    d = _bind(conn, "f2", "", "2026-02-01", key=weak)
    assert ss.resolve_dispute(conn, d.dispute_id, resolution, resolved_by="user") == retire
    slot_head, slot_state, _ = _slot(conn, head.slot_id)
    assert slot_state == state
    if head_is_b is None:
        assert slot_head is None
    else:
        assert slot_head == (d.revision_id if head_is_b else head.revision_id)
    assert conn.execute("SELECT state, resolution FROM slot_disputes").fetchone() == (
        "resolved", resolution)
    with pytest.raises(LookupError):
        ss.resolve_dispute(conn, d.dispute_id, resolution, resolved_by="user")


def test_bad_resolution_is_refused(conn) -> None:
    with pytest.raises(ValueError):
        ss.resolve_dispute(conn, "nope", "maybe", resolved_by="user")


def test_binding_the_same_fact_twice_keeps_the_first_decision(conn) -> None:
    first = _bind(conn, "f1", "The release cadence of SLM is monthly", "2026-01-01")
    again = _bind(conn, "f1", "The release cadence of SLM is monthly", "2026-01-01")
    assert again.outcome == "head" and again.revision_id == first.revision_id
    assert conn.execute("SELECT COUNT(*) FROM slot_revisions").fetchone()[0] == 1


def test_scopes_and_profiles_are_separate_slots(conn) -> None:
    key = extract("The release cadence of SLM is monthly")
    a = ss.bind(conn, profile_id="p", scope="personal", fact_id="f1", key=key)
    b = ss.bind(conn, profile_id="p", scope="project", fact_id="f2", key=key)
    c = ss.bind(conn, profile_id="q", scope="personal", fact_id="f3", key=key)
    assert len({a.slot_id, b.slot_id, c.slot_id}) == 3
    assert {a.outcome, b.outcome, c.outcome} == {ss.HEAD}


def test_a_concurrent_change_is_detected(conn, monkeypatch) -> None:
    head = _bind(conn, "f1", "The release cadence of SLM is monthly", "2026-01-01")
    real = ss._ensure_slot

    def stale(*args, **kwargs):
        head_id, version = real(*args, **kwargs)
        conn.execute("UPDATE memory_slots SET version = version + 1")  # another writer
        return head_id, version

    monkeypatch.setattr(ss, "_ensure_slot", stale)
    with pytest.raises(ss.SlotConflictError):
        _bind(conn, "f2", "The release cadence of SLM is weekly", "2026-02-01")
    assert _slot(conn, head.slot_id)[0] == head.revision_id


def test_record_case_links_the_ledger(conn) -> None:
    _bind(conn, "f1", "", "2026-01-01", key=_explicit("monthly"))
    new = _bind(conn, "f2", "", "2026-02-01", key=_explicit("weekly"))
    ss.record_case(conn, new.revision_id, "case-9")
    assert conn.execute("SELECT case_id FROM slot_revisions WHERE revision_id = ?",
                        (new.revision_id,)).fetchone() == ("case-9",)


def test_a_dispute_against_a_superseded_head_cannot_move_the_head_back(conn) -> None:
    """Fable's reproduction: A head; B and C both dispute A; settle (A,B) as b, then (A,C) as same.

    Re-anchoring opens a NEW dispute for (B, C): a verdict on (A, C) must not stand for it."""
    a = _bind(conn, "f1", "", "2026-01-01", key=replace(_explicit("monthly"), key_source="pattern"))
    b = _bind(conn, "f2", "", "2026-02-01", key=replace(_explicit("weekly"), key_source="pattern"))
    c = _bind(conn, "f3", "", "2026-03-01", key=replace(_explicit("daily"), key_source="pattern"))
    assert (b.outcome, c.outcome) == (ss.DISPUTE, ss.DISPUTE)
    assert ss.resolve_dispute(conn, b.dispute_id, "b", resolved_by="user") == ("f1",)
    assert conn.execute("SELECT state FROM slot_disputes WHERE dispute_id = ?",
                        (c.dispute_id,)).fetchone() == ("dismissed",)
    new_id, anchored = conn.execute("SELECT dispute_id, revision_a FROM slot_disputes "
                                    "WHERE state = 'open'").fetchone()
    assert anchored == b.revision_id and new_id != c.dispute_id
    ss.resolve_dispute(conn, new_id, "same", resolved_by="user")
    assert _slot(conn, a.slot_id)[0] == b.revision_id  # never back on superseded A


def test_a_stale_dispute_is_dismissed_not_applied(conn) -> None:
    a = _bind(conn, "f1", "", "2026-01-01", key=replace(_explicit("monthly"), key_source="pattern"))
    b = _bind(conn, "f2", "", "2026-02-01", key=replace(_explicit("weekly"), key_source="pattern"))
    conn.execute("UPDATE memory_slots SET head_revision_id = ?", (b.revision_id,))
    with pytest.raises(ss.StaleDisputeError):
        ss.resolve_dispute(conn, b.dispute_id, "a", resolved_by="user")
    assert conn.execute("SELECT state FROM slot_disputes").fetchone() == ("open",)  # nothing written
    ss.dismiss_dispute(conn, b.dispute_id, by="user")
    assert conn.execute("SELECT state FROM slot_disputes").fetchone() == ("dismissed",)
    assert _slot(conn, a.slot_id)[0] == b.revision_id


def test_a_same_day_date_against_a_timestamp_opens_a_dispute_not_history(conn) -> None:
    head = _bind(conn, "f1", "", "2026-11-15T08:00:00+00:00", key=_explicit("monthly"))
    same_day = _bind(conn, "f2", "", "2026-11-15", key=_explicit("weekly"))
    assert same_day.outcome == ss.DISPUTE
    assert conn.execute("SELECT reason_code FROM slot_disputes").fetchone() == ("incomparable_time",)
    assert _slot(conn, head.slot_id)[0] == head.revision_id
    assert conn.execute("SELECT valid_from FROM slot_revisions WHERE fact_id='f2'").fetchone() == (
        "2026-11-15",)  # kept as a date
