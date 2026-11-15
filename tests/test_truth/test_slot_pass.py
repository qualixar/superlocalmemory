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


def _resolve_explicit(db, fact_id, value, when):
    from superlocalmemory.truth.slot_resolver import resolve_fact

    with db.raw_connection() as conn:
        return resolve_fact(conn, fact_id=fact_id, profile_id="default", scope="personal",
                            content=value, valid_from=when, auto_apply=True,
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


def test_vanished_revisions_restart_the_pass(store) -> None:
    db, _old, _new, tmp = store
    flags = _flags(**{"truth.slots": "shadow"})
    first = SlotPass(db, flags, state_dir=tmp).run()
    db.execute("DELETE FROM slot_revisions")  # what a profile fold does to the source
    again = SlotPass(db, flags, state_dir=tmp).run()
    assert again["keyed"] == first["keyed"] and again["head"] == 1
