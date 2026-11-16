# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Slots kept in step with the ledger and the fact lifecycle (R-09, Fable B8)."""

from __future__ import annotations

from superlocalmemory.truth import slot_repair
from superlocalmemory.truth.slot_pass import SlotPass
from tests.test_truth.test_slot_pass import (
    _dispute,
    _fact,
    _flags,
    _reconcile,
    _resolve_explicit,
    _retired,
    _settled_by_a_person,
)


def _repair(db):
    with db.raw_connection() as conn:
        return slot_repair.repair(conn)


def _slot(db):
    return tuple(db.execute("SELECT head_revision_id, state FROM memory_slots")[0])


def _states(db):
    return {r["fact_id"]: r["state"] for r in db.execute(
        "SELECT fact_id, state FROM slot_revisions")}


def _queue(db):
    return sorted(r["fact_id"] for r in db.execute("SELECT fact_id FROM slot_pass_queue"))


def _two_values(db, old, new):
    _resolve_explicit(db, old, "weekly", "2026-03-01")
    _resolve_explicit(db, new, "one big release", "2026-11-14T00:00:00+00:00")


def test_an_archived_head_empties_the_slot_and_its_live_values_bind_again(store) -> None:
    db, old, new, _tmp = store
    _two_values(db, old, new)
    db.execute("UPDATE atomic_facts SET lifecycle = 'archived' WHERE fact_id = ?", (old,))
    assert _repair(db)["head_gone"] == 1
    assert _slot(db) == (None, "empty")
    assert _states(db) == {old: "gone"}  # the history keeps it; the challenger is rebound
    assert _queue(db) == [new]
    assert db.execute("SELECT COUNT(*) AS n FROM slot_disputes WHERE state = 'open'")[0]["n"] == 0


def test_a_merged_head_queues_the_fact_it_was_merged_into(store) -> None:
    db, old, new, _tmp = store
    _resolve_explicit(db, old, "weekly", "2026-03-01")
    survivor = _fact(db, "SLM ships weekly, merged", "2026-03-02T00:00:00+00:00")
    db.execute("UPDATE atomic_facts SET merged_into = ? WHERE fact_id = ?", (survivor, old))
    assert _repair(db)["head_gone"] == 1
    assert _queue(db) == [survivor]


def test_a_quarantined_or_retired_head_is_gone_too(store) -> None:
    db, old, new, _tmp = store
    _resolve_explicit(db, old, "weekly", "2026-03-01")
    db.execute("UPDATE atomic_facts SET quarantined = 1 WHERE fact_id = ?", (old,))
    assert _repair(db)["head_gone"] == 1 and _slot(db) == (None, "empty")


def test_a_head_retired_through_the_ledger_elsewhere_is_gone(store) -> None:
    from superlocalmemory.truth.slot_resolver import retire_superseded

    db, old, new, _tmp = store  # replaces= and update_memory retire through the ledger
    _resolve_explicit(db, old, "weekly", "2026-03-01")
    other = _fact(db, "An unkeyed memory that replaced it", "2026-05-01T00:00:00+00:00")
    with db.raw_connection() as conn:
        assert retire_superseded(conn, profile_id="default", scope="personal",
                                 predecessor=old, successor=other)[0] == "applied"
    assert _repair(db)["head_gone"] == 1 and _slot(db) == (None, "empty")


def test_a_challenger_that_left_closes_its_dispute(store) -> None:
    db, old, new, _tmp = store
    _two_values(db, old, new)
    db.execute("UPDATE atomic_facts SET quarantined = 1 WHERE fact_id = ?", (new,))
    counts = _repair(db)
    assert counts["challenger_gone"] == 1 and "head_gone" not in counts
    assert _dispute(db)[2] == "dismissed"


def test_a_live_slot_is_left_alone(store) -> None:
    db, old, new, _tmp = store
    _two_values(db, old, new)
    before = (_slot(db), _states(db), _dispute(db))
    assert _repair(db) == {}
    assert (_slot(db), _states(db), _dispute(db)) == before


def test_a_rolled_back_retirement_puts_the_slot_back_and_waits_for_a_person(store) -> None:
    from superlocalmemory.truth.slot_resolver import rollback_slot_cases

    db, old, new, _tmp = store
    _settled_by_a_person(db, old, new)
    _reconcile(db)
    assert _retired(db, old)[2] == new
    db.execute("UPDATE slot_revisions SET receipt_id = 'jr-acted' WHERE fact_id = ?", (new,))
    db.execute("INSERT INTO judge_receipts (receipt_id, profile_id, question_id, question_version,"
               " subject_kind, subject_id, population, tier, model, verdict, threshold, accepted,"
               " calibration_id, inputs_hash, decided_at, audit_selected, batching_regime)"
               " VALUES ('jr-acted', 'default', 'SLOT_RELATION', '2', 'slot_dispute', 'd', "
               "'slot_dispute', 'rules', 'explicit-v1', 'replaces', 0.7, 1, 'c', 'h', 't', 0,"
               " 'unbatched')")
    with db.raw_connection() as conn:
        assert rollback_slot_cases(conn) == {"rolled_back": 1}
    assert _retired(db, old) == (None, None, None)  # the ledger restored the fact
    assert _repair(db)["rolled_back"] == 1
    head = db.execute("SELECT r.fact_id FROM memory_slots s JOIN slot_revisions r "
                      "ON r.revision_id = s.head_revision_id")[0]["fact_id"]
    assert head == old and _states(db) == {old: "head", new: "disputed"}
    opened = db.execute("SELECT reason_code, opened_by FROM slot_disputes WHERE state = 'open'")
    assert [tuple(r) for r in opened] == [("rolled_back", "ledger:rollback")]
    label = db.execute("SELECT label, source, blind FROM judge_labels")[0]
    assert tuple(label) == ("rolled_back", "rollback", 0)  # never certifies
    assert _repair(db) == {}  # settled: not read again


def test_the_dispute_pass_leaves_a_rolled_back_dispute_to_a_person(store) -> None:
    from superlocalmemory.core.feature_flags import FeatureFlags
    from superlocalmemory.core.judge_ladder import JudgeLadder, RulesTier
    from superlocalmemory.truth.dispute_pass import DisputePass

    db, old, new, _tmp = store
    _two_values(db, old, new)
    db.execute("UPDATE slot_disputes SET reason_code = 'rolled_back'")
    counts = DisputePass(db, FeatureFlags(file_values={"judge.ladder": "shadow"}, environ={}),
                         JudgeLadder([RulesTier()])).run()
    assert counts == {} and db.execute("SELECT COUNT(*) AS n FROM judge_receipts")[0]["n"] == 0


def test_the_slot_pass_runs_the_repair(store) -> None:
    db, old, new, tmp = store
    _two_values(db, old, new)
    db.execute("UPDATE atomic_facts SET lifecycle = 'archived' WHERE fact_id = ?", (old,))
    counts = SlotPass(db, _flags(**{"truth.slots": "shadow"}), state_dir=tmp).run()
    assert counts["repair_head_gone"] == 1
