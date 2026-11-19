# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W0-24 for the backlog pass: a verdict closes a case only if the pair is still the pair
it was computed on, checked in the write step (I-13)."""

from __future__ import annotations

import pytest

from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.core.judge_ladder import JudgeLadder, TierAnswer
from superlocalmemory.truth.backlog_pass import BacklogPass
from tests.helpers.flag_baseline import off_except
from tests.test_truth import test_backlog_pass as _backlog_tests
from tests.test_truth.test_backlog_pass import _calibrated, _status

backlog = _backlog_tests.backlog  # the store these tests share (fixture by alias)


class Edits:
    """A judge that answers ``both_true`` while something else happens to the case."""

    name = "laya"

    def __init__(self, db, sql, *args):
        self.db, self.sql, self.args = db, sql, args

    def answer(self, question, items):
        if self.sql:
            self.db.execute(self.sql, self.args)
        return [TierAnswer("both_true", 0.95, model="fake")] * len(items)


def _run(db, tier):
    _calibrated(db, "both_true")
    return BacklogPass(db, FeatureFlags(file_values=off_except({"judge.ladder": "on"}), environ={}),
                       JudgeLadder([tier])).run()


@pytest.mark.parametrize("column", ["predecessor_fact_id", "successor_fact_id"])
def test_a_side_edited_during_the_judge_call_does_not_close_the_case(backlog, column) -> None:
    fact = backlog.execute(f"SELECT {column} AS f FROM correction_cases")[0]["f"]  # noqa: S608
    tier = Edits(backlog, "UPDATE atomic_facts SET content = ? WHERE fact_id = ?",
                 "The standup moved to 11 on Fridays only", fact)
    counts = _run(backlog, tier)
    assert counts["changed_during_judge"] == 1 and "closed" not in counts
    assert _status(backlog) == "proposed"
    (receipt,) = backlog.execute("SELECT action_taken, action_ref FROM judge_receipts "
                                 "WHERE receipt_id NOT LIKE 'seed-%'")
    assert receipt["action_taken"] is None and receipt["action_ref"] is None


def test_a_case_that_is_no_longer_open_is_not_acted_on(backlog) -> None:
    tier = Edits(backlog, "UPDATE correction_cases SET status = 'rejected'")
    counts = _run(backlog, tier)
    assert counts["changed_during_judge"] == 1 and "closed" not in counts


def test_an_unchanged_pair_still_closes(backlog) -> None:
    counts = _run(backlog, Edits(backlog, "", ))
    assert counts["closed"] == 1 and "changed_during_judge" not in counts


def test_rewriting_the_same_text_is_not_an_edit(backlog) -> None:
    fact = backlog.execute("SELECT successor_fact_id AS f FROM correction_cases")[0]["f"]
    same = backlog.execute("SELECT content FROM atomic_facts WHERE fact_id = ?", (fact,))[0][0]
    counts = _run(backlog, Edits(backlog, "UPDATE atomic_facts SET content = ? WHERE fact_id = ?",
                                 same, fact))
    assert counts["closed"] == 1
