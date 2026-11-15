# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Open slot disputes settled by the ladder: shadow records, on acts, ledger retires."""

from __future__ import annotations

import pytest

from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.core.judge_ladder import JudgeLadder, TierAnswer
from superlocalmemory.truth.dispute_pass import DisputePass
from superlocalmemory.truth.slot_pass import SlotPass
from tests.test_core.test_coverage_backfill import _db
from tests.test_truth.test_slot_pass import _fact, _flags, _retired


class Fixed:
    name = "laya"

    def __init__(self, verdict, confidence=0.95):
        self.verdict, self.confidence, self.calls = verdict, confidence, 0

    def answer(self, question, items):
        self.calls += 1
        return [TierAnswer(self.verdict, self.confidence, model="fake")] * len(items)


@pytest.fixture
def disputed(tmp_path):
    db = _db(tmp_path / "memory.db")
    old = _fact(db, "The release cadence of SLM is one release every week", "2026-03-01T00:00:00Z")
    new = _fact(db, "The release cadence of SLM is one big release", "2026-11-14T00:00:00Z")
    SlotPass(db, _flags(**{"truth.slots": "shadow"}), state_dir=tmp_path).run()
    assert db.execute("SELECT COUNT(*) AS n FROM slot_disputes WHERE state='open'")[0]["n"] == 1
    return db, old, new


def _open(db) -> int:
    return db.execute("SELECT COUNT(*) AS n FROM slot_disputes WHERE state='open'")[0]["n"]


def _run(db, verdict, *, confidence=0.95, **flags):
    tier = Fixed(verdict, confidence)
    counts = DisputePass(db, FeatureFlags(file_values=flags, environ={}),
                         JudgeLadder([tier])).run()
    return counts, tier


def test_off_asks_nothing(disputed) -> None:
    db, _old, _new = disputed
    counts, tier = _run(db, "replaces")
    assert counts == {} and tier.calls == 0


def test_shadow_records_a_receipt_and_changes_nothing(disputed) -> None:
    db, old, _new = disputed
    counts, _ = _run(db, "replaces", **{"judge.ladder": "shadow"})
    assert counts == {"laya:replaces": 1}
    assert _open(db) == 1 and _retired(db, old) == (None, None, None)
    assert db.execute("SELECT COUNT(*) AS n FROM judge_receipts")[0]["n"] == 1


def test_a_pair_already_asked_is_not_asked_again(disputed) -> None:
    db, _old, _new = disputed
    _run(db, "replaces", confidence=0.5, **{"judge.ladder": "shadow"})
    counts, tier = _run(db, "replaces", **{"judge.ladder": "shadow"})
    assert counts == {} and tier.calls == 0


def test_replaces_without_auto_apply_is_held_for_review(disputed) -> None:
    db, old, _new = disputed
    counts, _ = _run(db, "replaces", **{"judge.ladder": "on"})
    assert counts["held_for_review"] == 1 and _open(db) == 1
    assert _retired(db, old) == (None, None, None)


def test_replaces_with_auto_apply_retires_through_the_ledger(disputed) -> None:
    db, old, new = disputed
    counts, _ = _run(db, "replaces", **{"judge.ladder": "on", "truth.slots.auto_apply": "on"})
    assert counts["resolved_replaces"] == 1 and _open(db) == 0
    _until, expired, by = _retired(db, old)
    assert by == new and expired is not None
    head = db.execute("SELECT r.fact_id FROM memory_slots s JOIN slot_revisions r "
                      "ON r.revision_id = s.head_revision_id")[0]["fact_id"]
    assert head == new


@pytest.mark.parametrize("verdict,state", [("same", "duplicate"), ("both_true", "compatible"),
                                           ("unrelated", "compatible")])
def test_non_retiring_answers_settle_without_touching_facts(disputed, verdict, state) -> None:
    db, old, new = disputed
    counts, _ = _run(db, verdict, **{"judge.ladder": "on"})
    assert counts[f"resolved_{verdict}"] == 1 and _open(db) == 0
    assert _retired(db, old) == (None, None, None)
    assert db.execute("SELECT state FROM slot_revisions WHERE fact_id = ?",
                      (new,))[0]["state"] == state


def test_an_unsure_judge_leaves_the_dispute_open(disputed) -> None:
    db, old, _new = disputed
    counts, _ = _run(db, "replaces", confidence=0.6,
                     **{"judge.ladder": "on", "truth.slots.auto_apply": "on"})
    assert counts == {"none:abstain": 1} and _open(db) == 1
    assert _retired(db, old) == (None, None, None)
