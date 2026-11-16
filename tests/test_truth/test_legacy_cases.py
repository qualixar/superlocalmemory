# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Applied corrections from outside the slot layer, each on its own private slot (D-40)."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.truth import invariants, legacy_cases
from superlocalmemory.truth.slot_pass import SlotPass
from tests.test_core.test_coverage_backfill import _db
from tests.test_truth.test_slot_pass import _fact, _flags

CHECKED = ("I-01", "I-02", "I-03", "I-04", "I-05", "I-06", "I-07", "I-08", "I-12", "I-16")


def _case(db, case_id, pred, succ, *, reason="consolidation_supersede", status="applied",
          retire=True) -> None:
    conn = sqlite3.connect(db.db_path, isolation_level=None)
    conn.execute("INSERT INTO correction_cases (case_id, profile_id, scope, predecessor_fact_id, "
                 "successor_fact_id, reason_code, status, version, idempotency_key, "
                 "proposed_by_actor_id, proposed_by_actor_kind, proposed_by_trust_tier, "
                 "created_at, updated_at, applied_at) VALUES (?, 'default', 'personal', ?, ?, ?, "
                 "?, 1, ?, 'w', 'host_attested', 't', '2026-10-01T00:00:00+00:00', "
                 "'2026-10-01T00:00:00+00:00', '2026-10-01T00:00:00+00:00')",
                 (case_id, pred, succ, reason, status, f"k-{case_id}"))
    if retire:  # what the ledger did when it applied the case
        conn.execute("INSERT OR REPLACE INTO fact_temporal_validity (fact_id, profile_id, "
                     "valid_until, valid_until_source, system_expired_at, invalidated_by) VALUES "
                     "(?, 'default', '2026-11-14T00:00:00+00:00', 'successor_fact', "
                     "'2026-10-01T00:00:00+00:00', ?)",
                     (pred, succ))
    conn.close()


@pytest.fixture
def pair(tmp_path):
    db = _db(tmp_path / "memory.db")
    old = _fact(db, "Varun works from the old office in the city", "2026-03-01T00:00:00Z")
    new = _fact(db, "Varun works from home now", "2026-11-14T00:00:00Z")
    return db, old, new, tmp_path


def _pass(db, tmp):
    return SlotPass(db, _flags(**{"truth.slots": "shadow"}), state_dir=tmp).run()


def test_an_applied_case_gets_its_own_slot_with_its_interval(pair) -> None:
    db, old, new, tmp = pair
    _case(db, "case-1", old, new)
    assert _pass(db, tmp)["legacy_loaded"] == 1
    slot = db.execute("SELECT * FROM memory_slots WHERE key_source = 'legacy_case'")[0]
    assert (slot["subject_key"], slot["relation_key"], slot["qualifier"]) == ("legacy", "case",
                                                                            "case-1")
    assert slot["state"] == "current"
    revs = {r["fact_id"]: r for r in db.execute(
        "SELECT * FROM slot_revisions WHERE slot_id = ?", (slot["slot_id"],))}
    assert revs[new]["state"] == "head" and revs[new]["case_id"] == "case-1"
    assert slot["head_revision_id"] == revs[new]["revision_id"]
    assert revs[old]["state"] == "superseded"
    assert revs[old]["superseded_by"] == revs[new]["revision_id"]


def test_a_second_pass_loads_nothing_again(pair) -> None:
    db, old, new, tmp = pair
    _case(db, "case-1", old, new)
    _pass(db, tmp)
    assert "legacy_loaded" not in _pass(db, tmp)
    assert db.execute("SELECT COUNT(*) AS n FROM memory_slots "
                      "WHERE key_source = 'legacy_case'")[0]["n"] == 1


@pytest.mark.parametrize("reason,status", [("slot_supersede", "applied"),
                                           ("consolidation_supersede", "proposed"),
                                           ("consolidation_supersede", "rejected")])
def test_slot_made_or_unapplied_cases_are_left_alone(pair, reason, status) -> None:
    db, old, new, tmp = pair
    _case(db, "case-x", old, new, reason=reason, status=status, retire=False)
    assert "legacy_loaded" not in _pass(db, tmp)


def test_a_case_whose_successor_starts_first_is_counted_not_loaded(pair) -> None:
    db, old, new, tmp = pair
    _case(db, "case-inv", new, old)  # the newer fact "replaced" by the older one
    counts = _pass(db, tmp)
    assert counts["legacy_inverted"] == 1 and "legacy_loaded" not in counts


def test_the_invariants_hold_after_loading(pair) -> None:
    db, old, new, tmp = pair
    _case(db, "case-1", old, new)
    _pass(db, tmp)
    conn = invariants.open_read_only(str(db.db_path))
    try:
        results = {r.invariant: r for r in invariants.check(conn)}
    finally:
        conn.close()
    bad = {i: (results[i].status, results[i].count) for i in CHECKED
           if results[i].status == invariants.VIOLATED}
    assert bad == {}


def test_rolling_the_case_back_restores_the_old_value_as_a_question(pair) -> None:
    db, old, new, tmp = pair
    _case(db, "case-1", old, new)
    _pass(db, tmp)
    db.execute("UPDATE correction_cases SET status = 'rolled_back' WHERE case_id = 'case-1'")
    db.execute("DELETE FROM fact_temporal_validity WHERE fact_id = ?", (old,))
    counts = _pass(db, tmp)
    assert counts.get("repair_rolled_back") == 1
    (dispute,) = db.execute("SELECT reason_code, state FROM slot_disputes")
    assert (dispute["reason_code"], dispute["state"]) == ("rolled_back", "open")


def test_a_store_without_a_ledger_loads_nothing() -> None:
    assert legacy_cases.load(sqlite3.connect(":memory:")) == {}


def test_a_case_already_on_a_slot_revision_is_not_loaded_again(pair) -> None:
    db, old, new, tmp = pair
    _case(db, "case-1", old, new)
    conn = sqlite3.connect(db.db_path, isolation_level=None)
    conn.execute("INSERT INTO slot_revisions (revision_id, slot_id, profile_id, fact_id, value_hash, "
                 "state, decided_by, reason_code, case_id, recorded_at) VALUES ('rev-x', 'slot-x', "
                 "'default', ?, 'h', 'head', 'ledger', 'slot_supersede', 'case-1', '2026-10-01')",
                 (new,))
    conn.close()
    assert "legacy_loaded" not in _pass(db, tmp)


def test_a_chain_of_cases_is_one_timeline(pair) -> None:
    db, old, new, tmp = pair
    newest = _fact(db, "Varun works from a co-working space", "2026-11-20T00:00:00Z")
    _case(db, "case-1", old, new)
    _case(db, "case-2", new, newest)
    counts = _pass(db, tmp)
    assert counts["legacy_loaded"] == 2 and counts["legacy_chained"] == 1
    (slot,) = db.execute("SELECT slot_id, head_revision_id, state FROM memory_slots "
                         "WHERE key_source = 'legacy_case'")
    revs = {r["fact_id"]: r for r in db.execute(
        "SELECT * FROM slot_revisions WHERE slot_id = ?", (slot["slot_id"],))}
    assert slot["head_revision_id"] == revs[newest]["revision_id"] and slot["state"] == "current"
    assert revs[old]["superseded_by"] == revs[new]["revision_id"]
    assert revs[new]["superseded_by"] == revs[newest]["revision_id"]
    assert (revs[new]["state"], revs[new]["case_id"]) == ("superseded", "case-1")
    assert revs[newest]["case_id"] == "case-2"
    conn = invariants.open_read_only(str(db.db_path))
    try:
        bad = [r.invariant for r in invariants.check(conn)
               if r.invariant in CHECKED and r.status == invariants.VIOLATED]
    finally:
        conn.close()
    assert bad == []
