# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W0-24 (I-13): a verdict acts only on the pair it was computed on, checked in the write step."""

from __future__ import annotations

import pytest

from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.core.judge_ladder import JudgeLadder, TierAnswer
from superlocalmemory.truth.dispute_pass import DisputePass
from tests.test_truth import test_dispute_pass as _dispute_tests
from tests.test_truth.test_dispute_pass import _calibrate, _open, _retired

disputed = _dispute_tests.disputed  # the store these tests share (fixture by alias)

ON = {"judge.ladder": "on", "truth.slots": "on", "truth.slots.auto_apply": "on"}


class EditsDuringTheCall:
    """A judge that answers ``replaces`` while someone edits one side of the pair."""

    name = "laya"

    def __init__(self, db, fact_id, new_text):
        self.db, self.fact_id, self.new_text = db, fact_id, new_text

    def answer(self, question, items):
        if self.new_text is not None:
            self.db.execute("UPDATE atomic_facts SET content = ? WHERE fact_id = ?",
                            (self.new_text, self.fact_id))
        return [TierAnswer("replaces", 0.95, model="fake")] * len(items)


def _run(db, tier):
    _calibrate(db, "replaces")
    dp = DisputePass(db, FeatureFlags(file_values=ON, environ={}), JudgeLadder([tier]))
    return dp.run()


@pytest.mark.parametrize("side", ["old", "new"])
def test_a_side_edited_during_the_judge_call_is_not_acted_on(disputed, side) -> None:
    db, old, new = disputed
    fact = old if side == "old" else new
    counts = _run(db, EditsDuringTheCall(db, fact, "The release cadence is now every two weeks"))
    assert counts["changed_during_judge"] == 1
    assert "resolved_replaces" not in counts
    assert _open(db) == 1 and _retired(db, old) == (None, None, None)
    # the judgement itself is on record, with no action taken
    (receipt,) = [r for r in db.execute(
        "SELECT action_taken, action_ref FROM judge_receipts WHERE receipt_id NOT LIKE 'seed-%'")]
    assert receipt["action_taken"] is None and receipt["action_ref"] is None


def test_the_unchanged_pair_still_acts(disputed) -> None:
    db, old, new = disputed
    counts = _run(db, EditsDuringTheCall(db, old, None))
    assert counts["resolved_replaces"] == 1 and "changed_during_judge" not in counts
    assert _retired(db, old)[2] == new


def test_a_write_of_the_same_text_is_not_an_edit(disputed) -> None:
    db, old, new = disputed
    same = db.execute("SELECT content FROM atomic_facts WHERE fact_id = ?", (new,))[0]["content"]
    counts = _run(db, EditsDuringTheCall(db, new, same))
    assert counts["resolved_replaces"] == 1


def test_a_dispute_settled_during_the_judge_call_is_not_acted_on_again(disputed) -> None:
    db, old, _new = disputed

    class Settles(EditsDuringTheCall):
        def answer(self, question, items):
            self.db.execute("UPDATE slot_disputes SET state = 'dismissed'")
            return super().answer(question, items)

    counts = _run(db, Settles(db, old, None))
    assert counts["changed_during_judge"] == 1 and "resolved_replaces" not in counts
    assert _retired(db, old) == (None, None, None)
