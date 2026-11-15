# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The slot pass on a fully migrated store: keys facts, moves heads, uses the ledger."""

from __future__ import annotations

import pytest

from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.storage.models import AtomicFact, MemoryRecord
from superlocalmemory.truth.slot_pass import SlotPass
from tests.test_core.test_coverage_backfill import _db

OLD = "The release cadence of SLM is one release every week"
NEW = "The release cadence of SLM is one big release"


def _fact(db, content: str, created: str) -> str:
    record = MemoryRecord(content=content, session_date=created[:10])
    db.store_memory(record)
    fact = AtomicFact(memory_id=record.memory_id, content=content)
    db.store_fact(fact)
    db.execute("UPDATE atomic_facts SET created_at = ? WHERE fact_id = ?", (created, fact.fact_id))
    return fact.fact_id


def _flags(**states: str) -> FeatureFlags:
    return FeatureFlags(file_values=states, environ={})


def _retired(db, fact_id: str) -> tuple:
    rows = db.execute("SELECT valid_until, system_expired_at, invalidated_by FROM "
                      "fact_temporal_validity WHERE fact_id = ?", (fact_id,))
    return tuple(rows[0]) if rows else (None, None, None)


@pytest.fixture
def store(tmp_path):
    db = _db(tmp_path / "memory.db")
    old = _fact(db, OLD, "2026-03-01T00:00:00+00:00")
    _fact(db, "Session notes about the launch went fine overall.", "2026-03-02T00:00:00+00:00")
    new = _fact(db, NEW, "2026-11-14T00:00:00+00:00")
    return db, old, new, tmp_path


def test_off_does_nothing(store) -> None:
    db, _old, _new, tmp = store
    assert SlotPass(db, _flags(), state_dir=tmp).run() == {}
    assert db.execute("SELECT COUNT(*) AS n FROM slot_revisions")[0]["n"] == 0


def test_shadow_keys_facts_and_opens_a_dispute_for_the_judge(store) -> None:
    db, old, new, tmp = store
    counts = SlotPass(db, _flags(**{"truth.slots": "shadow"}), state_dir=tmp).run()
    assert counts["head"] == 1 and counts["dispute"] == 1 and counts["unkeyed"] >= 1
    assert _retired(db, old) == (None, None, None)
    sides = db.execute("SELECT r.fact_id FROM slot_disputes d JOIN slot_revisions r "
                       "ON r.revision_id IN (d.revision_a, d.revision_b) ORDER BY r.fact_id")
    assert sorted(r["fact_id"] for r in sides) == sorted([old, new])


def test_pattern_keys_never_retire_a_fact_even_with_auto_apply(store) -> None:
    db, old, _new, tmp = store
    flags = _flags(**{"truth.slots": "on", "truth.slots.auto_apply": "on"})
    counts = SlotPass(db, flags, state_dir=tmp).run()
    assert counts["dispute"] == 1 and "ledger_applied" not in counts
    assert _retired(db, old) == (None, None, None)
    assert db.execute("SELECT COUNT(*) AS n FROM correction_cases")[0]["n"] == 0


def test_resumes_from_its_cursor_and_picks_up_new_saves(store) -> None:
    db, _old, _new, tmp = store
    flags = _flags(**{"truth.slots": "shadow"})
    SlotPass(db, flags, state_dir=tmp).run()
    assert SlotPass(db, flags, state_dir=tmp).run() == {}  # nothing new
    _fact(db, "The release cadence of SLM is two releases a month", "2026-11-20T00:00:00+00:00")
    assert SlotPass(db, flags, state_dir=tmp).run()["dispute"] == 1


def _resolve_explicit(db, fact_id, value, when, auto_apply=True):
    from superlocalmemory.truth.slot_resolver import resolve_fact

    with db.raw_connection() as conn:
        return resolve_fact(conn, fact_id=fact_id, profile_id="default", scope="personal",
                            content=value, valid_from=when, auto_apply=auto_apply,
                            explicit_key="SLM/release cadence")


def test_an_explicit_key_retires_the_old_value_through_the_ledger(store) -> None:
    db, old, new, _tmp = store
    assert _resolve_explicit(db, old, "weekly", "2026-03-01").outcome == "head"
    result = _resolve_explicit(db, new, "one big release", "2026-11-14T00:00:00+00:00")
    assert (result.outcome, result.ledger) == ("supersede", "applied")
    valid_until, expired, by = _retired(db, old)
    assert by == new and expired is not None
    assert valid_until == "2026-11-14T00:00:00+00:00"  # when the new value was saved
    case = db.execute("SELECT status, reason_code FROM correction_cases "
                      "WHERE predecessor_fact_id = ?", (old,))[0]
    assert (case["status"], case["reason_code"]) == ("applied", "slot_supersede")
    linked = db.execute("SELECT case_id FROM slot_revisions WHERE fact_id = ?", (new,))[0]
    assert linked["case_id"] == result.case_id


def _consolidator_proposal(db, case_id, predecessor, successor) -> None:
    from superlocalmemory.storage.correction_cases import CorrectionActor, propose_on_connection

    actor = CorrectionActor(actor_id="writer", actor_kind="host_attested",
                            trust_tier="canonical_writer")
    with db.raw_connection() as conn:
        propose_on_connection(conn, case_id=case_id, profile_id="default", scope="personal",
                              predecessor_fact_id=predecessor, successor_fact_id=successor,
                              reason_code="consolidation_supersede", actor=actor,
                              idempotency_key=case_id, is_profile_active=lambda p: True,
                              is_actor_trusted=lambda a: True)


def test_an_existing_consolidator_proposal_is_settled_not_duplicated(store) -> None:
    db, old, new, _tmp = store
    _consolidator_proposal(db, "cons-1", old, new)
    _resolve_explicit(db, old, "weekly", "2026-03-01")
    assert _resolve_explicit(db, new, "one big release", "2026-11-14").ledger == "applied"
    cases = db.execute("SELECT case_id, status FROM correction_cases")
    assert [(c["case_id"], c["status"]) for c in cases] == [("cons-1", "applied")]


def test_a_rival_proposal_is_left_for_review(store) -> None:
    db, old, new, _tmp = store
    other = _fact(db, "Unrelated memory that the consolidator matched.", "2026-05-01T00:00:00Z")
    _consolidator_proposal(db, "cons-2", old, other)
    _resolve_explicit(db, old, "weekly", "2026-03-01")
    assert _resolve_explicit(db, new, "one big release", "2026-11-14").ledger == (
        "deferred_to_review")
    assert _retired(db, old) == (None, None, None)


def test_a_retirement_already_applied_for_another_successor_is_not_reported_applied(
        store) -> None:
    from superlocalmemory.truth.slot_resolver import RETIRED_FOR_ANOTHER, retire_superseded

    db, old, new, _tmp = store
    other = _fact(db, "A different memory that retired the old value.", "2026-05-01T00:00:00Z")
    with db.raw_connection() as conn:
        assert retire_superseded(conn, profile_id="default", scope="personal",
                                 predecessor=old, successor=other)[0] == "applied"
    _resolve_explicit(db, old, "weekly", "2026-03-01")
    result = _resolve_explicit(db, new, "one big release", "2026-11-14")
    assert result.ledger == RETIRED_FOR_ANOTHER  # R-22: was "applied"
    assert _retired(db, old)[2] == other  # the earlier retirement stands untouched
    linked = db.execute("SELECT case_id FROM slot_revisions WHERE fact_id = ?", (new,))[0]
    assert linked["case_id"] is None




# -- B2 / R-19: the queue: rebinds after a head is erased, retries after a failure ------------

def _erase(db, fact_id: str) -> None:
    with db.raw_connection() as conn:
        conn.execute("DELETE FROM fact_temporal_validity WHERE fact_id = ?", (fact_id,))
        conn.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (fact_id,))


def _slot_state(db) -> list:
    return [(r["state"], r["n"]) for r in db.execute(
        "SELECT s.state, COUNT(r.revision_id) AS n FROM memory_slots s "
        "LEFT JOIN slot_revisions r ON r.slot_id = s.slot_id GROUP BY s.slot_id")]


def test_erasing_the_head_leaves_the_slot_as_if_it_was_never_saved(store) -> None:
    db, old, new, tmp = store
    flags = _flags(**{"truth.slots": "shadow"})
    SlotPass(db, flags, state_dir=tmp).run()
    assert _slot_state(db) == [("disputed", 2)]
    _erase(db, old)  # the head: the challenger is queued, the slot emptied and removed
    SlotPass(db, flags, state_dir=tmp).run()
    assert _slot_state(db) == [("current", 1)]
    head = db.execute("SELECT r.fact_id FROM memory_slots s JOIN slot_revisions r "
                      "ON r.revision_id = s.head_revision_id")
    assert [r["fact_id"] for r in head] == [new]
    assert db.execute("SELECT COUNT(*) AS n FROM slot_disputes")[0]["n"] == 0
    assert db.execute("SELECT COUNT(*) AS n FROM slot_pass_queue")[0]["n"] == 0


def test_erasing_the_challenger_leaves_the_head_current(store) -> None:
    db, old, new, tmp = store
    flags = _flags(**{"truth.slots": "shadow"})
    SlotPass(db, flags, state_dir=tmp).run()
    _erase(db, new)
    assert _slot_state(db) == [("current", 1)]


def _queue(db) -> list:
    return [tuple(r) for r in db.execute(
        "SELECT fact_id, reason, attempts, last_error FROM slot_pass_queue")]


def test_a_failed_bind_is_queued_with_backoff_and_retried_when_due(store, monkeypatch) -> None:
    import sqlite3

    from superlocalmemory.truth import slot_pass

    db, old, _new, tmp = store
    real = slot_pass.resolve_fact
    calls = []

    def flaky(conn, **kw):
        calls.append(kw["fact_id"])
        if kw["fact_id"] == old:
            raise sqlite3.OperationalError("database is locked")
        return real(conn, **kw)

    monkeypatch.setattr(slot_pass, "resolve_fact", flaky)
    flags = _flags(**{"truth.slots": "shadow"})
    counts = SlotPass(db, flags, state_dir=tmp).run()
    assert counts["error"] == 1 and counts["queued_retry"] == 1
    assert _queue(db) == [(old, "retry", 1, "OperationalError")]  # the class, never a message
    calls.clear()
    SlotPass(db, flags, state_dir=tmp).run()
    assert old not in calls  # not due yet: backoff holds
    monkeypatch.setattr(slot_pass, "resolve_fact", real)
    db.execute("UPDATE slot_pass_queue SET next_at = '2000-01-01T00:00:00+00:00'")
    counts = SlotPass(db, flags, state_dir=tmp).run()
    assert counts["queue_bound"] == 1 and _queue(db) == []
    assert db.execute("SELECT COUNT(*) AS n FROM slot_revisions WHERE fact_id = ?",
                      (old,))[0]["n"] == 1


def test_backoff_doubles_and_is_capped() -> None:
    from datetime import UTC, datetime

    from superlocalmemory.truth.slot_pass import RETRY_MAX_S, _retry_at

    t0 = datetime(2026, 11, 15, tzinfo=UTC)
    gaps = [(datetime.fromisoformat(_retry_at(n, t0)) - t0).total_seconds() for n in (1, 2, 3, 30)]
    assert gaps[:3] == [60.0, 120.0, 240.0] and gaps[3] == RETRY_MAX_S


def test_a_queued_fact_that_is_no_longer_live_leaves_the_queue(store) -> None:
    db, old, _new, tmp = store
    flags = _flags(**{"truth.slots": "shadow"})
    SlotPass(db, flags, state_dir=tmp).run()
    db.execute("UPDATE atomic_facts SET lifecycle = 'archived' WHERE fact_id = ?", (old,))
    db.execute("INSERT INTO slot_pass_queue (fact_id, reason, next_at, queued_at) "
               "VALUES (?, 'retry', '2000-01-01T00:00:00+00:00', '2000-01-01T00:00:00+00:00')",
               (old,))
    counts = SlotPass(db, flags, state_dir=tmp).run()
    assert counts["queue_dropped"] == 1 and _queue(db) == []



# -- R-07 / B5: a supersede decided while retiring was not allowed reaches the ledger later ---

def _cases(db) -> list:
    return [(r["status"], r["predecessor_fact_id"], r["successor_fact_id"]) for r in db.execute(
        "SELECT status, predecessor_fact_id, successor_fact_id FROM correction_cases "
        "ORDER BY created_at")]


def _link(db, fact_id):
    return db.execute("SELECT case_id FROM slot_revisions WHERE fact_id = ? "
                      "AND state != 'superseded'", (fact_id,))[0]["case_id"]


def _pending_supersede(store):
    db, old, new, _tmp = store
    assert _resolve_explicit(db, old, "weekly", "2026-03-01", auto_apply=False).outcome == "head"
    r = _resolve_explicit(db, new, "one big release", "2026-11-14T00:00:00+00:00",
                          auto_apply=False)
    assert (r.outcome, r.ledger) == ("supersede", "pending")
    return db, old, new


@pytest.mark.usefixtures("auto_apply_basis")
def test_a_stranded_supersede_is_proposed_then_applied_as_flags_allow(store) -> None:
    db, old, new = _pending_supersede(store)
    tmp = store[3]
    SlotPass(db, _flags(**{"truth.slots": "shadow"}), state_dir=tmp).run()
    assert _cases(db) == []  # shadow never writes the ledger (I-15)
    counts = SlotPass(db, _flags(**{"truth.slots": "on"}), state_dir=tmp).run()
    assert counts["ledger_proposed"] == 1
    assert _cases(db) == [("proposed", old, new)] and _retired(db, old) == (None, None, None)
    case = _link(db, new)
    assert case is not None
    counts = SlotPass(db, _flags(**{"truth.slots": "on", "truth.slots.auto_apply": "on"}),
                      state_dir=tmp).run()
    assert counts["ledger_applied"] == 1 and _cases(db) == [("applied", old, new)]
    assert _retired(db, old)[2] == new and _link(db, new) == case
    again = SlotPass(db, _flags(**{"truth.slots": "on", "truth.slots.auto_apply": "on"}),
                     state_dir=tmp).run()
    assert not any(k.startswith("ledger_") for k in again)  # settled: not read again


@pytest.mark.usefixtures("auto_apply_basis")
def test_a_decision_made_in_shadow_is_applied_directly_once_allowed(store) -> None:
    db, old, new = _pending_supersede(store)
    counts = SlotPass(db, _flags(**{"truth.slots": "on", "truth.slots.auto_apply": "on"}),
                      state_dir=store[3]).run()
    assert counts["ledger_applied"] == 1 and _cases(db) == [("applied", old, new)]


def test_a_rival_proposal_is_left_for_a_person_and_not_read_again(store) -> None:
    db, old, new = _pending_supersede(store)
    other = _fact(db, "The release cadence of SLM is quarterly", "2026-11-15T00:00:00+00:00")
    _consolidator_proposal(db, "rival", old, other)
    flags = _flags(**{"truth.slots": "on", "truth.slots.auto_apply": "on"})
    counts = SlotPass(db, flags, state_dir=store[3]).run()
    assert counts["ledger_deferred_to_review"] == 1
    assert _link(db, new) == "rival" and _retired(db, old) == (None, None, None)
    again = SlotPass(db, flags, state_dir=store[3]).run()
    assert "ledger_deferred_to_review" not in again


# -- D-23: rolling back everything the truth layer retired ------------------------------------

def test_rollback_slot_cases_restores_only_the_truth_layers_retirements(store) -> None:
    from superlocalmemory.truth.slot_resolver import rollback_slot_cases

    db, old, new, _tmp = store
    _resolve_explicit(db, old, "weekly", "2026-03-01")
    _resolve_explicit(db, new, "one big release", "2026-11-14T00:00:00+00:00")
    assert _retired(db, old)[2] == new
    other = _fact(db, "Some unrelated memory about a plant", "2026-05-01T00:00:00+00:00")
    third = _fact(db, "Another memory about the plant", "2026-06-01T00:00:00+00:00")
    _consolidator_proposal(db, "keep-me", other, third)
    with db.raw_connection() as conn:
        counts = rollback_slot_cases(conn)
    assert counts == {"rolled_back": 1}
    assert _retired(db, old) == (None, None, None)
    statuses = {r["case_id"]: r["status"] for r in db.execute(
        "SELECT case_id, status FROM correction_cases")}
    assert statuses["keep-me"] == "proposed"
    assert sorted(statuses.values()) == ["proposed", "rolled_back"]


def test_an_erasure_never_sends_the_pass_back_over_every_fact(store) -> None:
    db, _old, new, tmp = store
    flags = _flags(**{"truth.slots": "shadow"})
    first = SlotPass(db, flags, state_dir=tmp).run()
    assert first["keyed"] + first["unkeyed"] == 3
    _erase(db, new)  # a revision vanishes: R-38 used to restart from fact zero
    again = SlotPass(db, flags, state_dir=tmp).run()
    assert again.get("keyed", 0) + again.get("unkeyed", 0) == 0


def test_a_profile_fold_queues_its_facts_so_they_get_slots_in_the_target(store) -> None:
    from superlocalmemory.storage.profile_fold import fold_profile

    db, _old, _new, tmp = store
    with db.raw_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('work', 'work')")
        for table in ("memories", "atomic_facts"):
            conn.execute(f"UPDATE {table} SET profile_id = 'work'")  # noqa: S608
    flags = _flags(**{"truth.slots": "shadow"})
    SlotPass(db, flags, state_dir=tmp).run()
    assert {r["profile_id"] for r in db.execute("SELECT profile_id FROM memory_slots")} == {
        "work"}
    with db.raw_connection() as conn:
        fold_profile(conn, "work", "default")
    assert db.execute("SELECT COUNT(*) AS n FROM memory_slots")[0]["n"] == 0
    assert db.execute("SELECT COUNT(*) AS n FROM slot_pass_queue")[0]["n"] == 3
    SlotPass(db, flags, state_dir=tmp).run()
    assert {r["profile_id"] for r in db.execute("SELECT profile_id FROM memory_slots")} == {
        "default"}
    assert _slot_state(db) == [("disputed", 2)]
