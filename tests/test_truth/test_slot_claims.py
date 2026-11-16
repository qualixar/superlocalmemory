# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The slot a caller names on a save reaches the slot pass, on one fact only (D-36)."""

from __future__ import annotations

import pytest

from superlocalmemory.core.metadata_guard import strip_reserved_metadata
from superlocalmemory.storage.models import AtomicFact, MemoryRecord
from superlocalmemory.truth import slot_claims
from superlocalmemory.truth.slot_keys import SlotSpecError, parse_spec
from superlocalmemory.truth.slot_pass import SlotPass
from tests.test_core.test_coverage_backfill import _db
from tests.test_truth.test_slot_pass import _flags


def _memory(db, facts, claim, created):
    record = MemoryRecord(content=" ".join(facts), session_date=created[:10],
                          metadata={slot_claims.METADATA_KEY: claim} if claim else {})
    db.store_memory(record)
    ids = []
    for text in facts:
        fact = AtomicFact(memory_id=record.memory_id, content=text)
        db.store_fact(fact)
        db.execute("UPDATE atomic_facts SET created_at = ? WHERE fact_id = ?",
                   (created, fact.fact_id))
        ids.append(fact.fact_id)
    return ids


def _revisions(db):
    return {r["fact_id"]: (r["subject_key"], r["value_hash"][:7]) for r in db.execute(
        "SELECT r.fact_id, s.subject_key, r.value_hash FROM slot_revisions r "
        "JOIN memory_slots s ON s.slot_id = r.slot_id")}


def test_a_claim_keys_the_memorys_first_fact_and_no_other(tmp_path) -> None:
    db = _db(tmp_path / "memory.db")
    first, second = _memory(db, ["We ship SLM every week", "Alex runs the release"],
                            slot_claims.encode("SLM/release cadence", "weekly"),
                            "2026-03-01T00:00:00+00:00")
    SlotPass(db, _flags(**{"truth.slots": "shadow"}), state_dir=tmp_path).run()
    revisions = _revisions(db)
    assert revisions[first] == ("name:slm", "stated:")
    assert second not in revisions  # a split memory never disputes with itself


def test_two_saves_naming_the_slot_open_an_explicit_dispute(tmp_path) -> None:
    db = _db(tmp_path / "memory.db")
    _memory(db, ["We ship SLM every week"], slot_claims.encode("SLM/release cadence", "weekly"),
            "2026-03-01T00:00:00+00:00")
    _memory(db, ["From now on SLM ships one big release"],
            slot_claims.encode("slm/Release cadence", "one big release"),
            "2026-11-14T00:00:00+00:00")
    counts = SlotPass(db, _flags(**{"truth.slots": "shadow"}), state_dir=tmp_path).run()
    assert counts["dispute"] == 1
    assert db.execute("SELECT reason_code FROM slot_disputes")[0][0] == "explicit_value_changed"


def test_a_callers_own_metadata_can_never_set_a_claim() -> None:
    assert strip_reserved_metadata({slot_claims.METADATA_KEY: {"spec": "a/b"}, "x": 1}) == {"x": 1}


def test_a_malformed_claim_in_the_store_is_ignored(tmp_path) -> None:
    db = _db(tmp_path / "memory.db")
    [fact] = _memory(db, ["We ship SLM every week"], {"spec": 7}, "2026-03-01T00:00:00+00:00")
    with db.raw_connection() as conn:
        assert slot_claims.claim_for_fact(conn, fact) is None


@pytest.mark.parametrize("spec,value", [
    ("only-one-part", None), ("a//c", None), ("a/b/c/d", None), ("", "weekly"),
    ("x" * 201 + "/b", None), ("a/b", "v" * 241)])
def test_the_door_refuses_a_malformed_slot(spec, value) -> None:
    with pytest.raises(SlotSpecError):
        parse_spec(spec, value)


def test_no_slot_is_no_claim() -> None:
    assert slot_claims.encode(None, None) is None and slot_claims.encode("  ", None) is None
    assert slot_claims.encode("SLM/release cadence", " ") == {"spec": "SLM/release cadence"}
