# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""WS-7 writer-lock audit: the truth passes make their slow calls with no lock and no write open.

A slow call is a judge (a model, possibly a network request) or a file check (a hash of a file on
disk). The rule (owner, 18 Nov): no background pass holds the writer lock across one. Each test
runs the real pass on a real store and asks, from inside the slow call itself, whether this
thread holds the store's writer lock (an RLock: ``_is_owned``) and whether a write transaction is
open on the connection the pass used (the pass has let go of it: none is bound to this thread).
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.core.judge_ladder import JudgeLadder, TierAnswer
from superlocalmemory.truth import leases
from superlocalmemory.truth.backlog_pass import BacklogPass
from superlocalmemory.truth.dispute_pass import DisputePass
from superlocalmemory.truth.slot_pass import SlotPass
from tests.test_truth.test_backlog_pass import backlog as backlog_store  # noqa: F401 - fixture
from tests.test_truth.test_dispute_pass import disputed  # noqa: F401 - fixture
from tests.test_truth.test_leases import _leased_fact, leased  # noqa: F401 - fixture
from tests.test_truth.test_slot_pass import _flags


class Spy:
    """A judge that records, at the moment it is asked, what the pass was still holding."""

    name = "laya"

    def __init__(self, db) -> None:
        self.db, self.asked = db, []

    def answer(self, question, items):
        self.asked.append({"lock": self.db._lock._is_owned(),
                           "write_open": getattr(self.db._txn_state, "conn", None) is not None})
        return [TierAnswer("both_true", 0.95, model="fake")] * len(items)


def test_the_dispute_pass_judges_with_nothing_held(disputed) -> None:  # noqa: F811
    db, _old, _new = disputed
    spy = Spy(db)
    DisputePass(db, FeatureFlags(file_values={"judge.ladder": "shadow", "truth.slots": "shadow"},
                                 environ={}), JudgeLadder([spy])).run()
    assert spy.asked == [{"lock": False, "write_open": False}]


def test_the_backlog_pass_judges_with_nothing_held(backlog_store) -> None:  # noqa: F811
    spy = Spy(backlog_store)
    BacklogPass(backlog_store, FeatureFlags(file_values={"judge.ladder": "shadow"}, environ={}),
                JudgeLadder([spy])).run()
    assert spy.asked == [{"lock": False, "write_open": False}]


def test_a_judge_that_takes_a_long_time_does_not_block_a_save(disputed) -> None:  # noqa: F811
    """What the rule is for: while the judge is thinking, another thread can write."""
    import threading

    db, _old, _new = disputed
    done = threading.Event()

    class Slow(Spy):
        def answer(self, question, items):
            writer = threading.Thread(target=lambda: (db.execute(
                "UPDATE atomic_facts SET confidence = confidence WHERE 1 = 0"), done.set()))
            writer.start()
            writer.join(timeout=5)
            return super().answer(question, items)

    DisputePass(db, FeatureFlags(file_values={"judge.ladder": "shadow", "truth.slots": "shadow"},
                                 environ={}), JudgeLadder([Slow(db)])).run()
    assert done.is_set()


def test_the_file_check_for_leases_runs_with_nothing_held(tmp_path, monkeypatch) -> None:
    from tests.test_core.test_coverage_backfill import _db

    db = _db(tmp_path / "memory.db")
    src = tmp_path / "config.yaml"
    src.write_text("port: 8765\n")
    _leased_fact(db, src)
    seen: list[dict] = []
    real = leases.recheck

    def recheck(due):
        seen.append({"lock": db._lock._is_owned(),
                     "write_open": getattr(db._txn_state, "conn", None) is not None})
        return real(due)

    monkeypatch.setattr(leases, "recheck", recheck)
    SlotPass(db, _flags(**{"truth.slots": "shadow"}), state_dir=tmp_path).run()
    assert seen == [{"lock": False, "write_open": False}]


@pytest.mark.parametrize("pass_name", ["dispute", "backlog"])
def test_the_write_step_is_one_short_hold_after_the_judge(pass_name, request) -> None:
    """Order: read (hold), judge (none), write (hold): two holds, the judge between them."""
    events: list[str] = []
    if pass_name == "dispute":
        db, _o, _n = request.getfixturevalue("disputed")
        runner = lambda ladder: DisputePass(  # noqa: E731
            db, FeatureFlags(file_values={"judge.ladder": "shadow", "truth.slots": "shadow"},
                             environ={}), ladder).run()
    else:
        db = request.getfixturevalue("backlog_store")
        runner = lambda ladder: BacklogPass(  # noqa: E731
            db, FeatureFlags(file_values={"judge.ladder": "shadow"}, environ={}), ladder).run()
    real = db.raw_connection

    def tracked():
        events.append("hold")
        return real()

    db.raw_connection = tracked

    class Marker(Spy):
        def answer(self, question, items):
            events.append("judge")
            return super().answer(question, items)

    runner(JudgeLadder([Marker(db)]))
    assert events == ["hold", "judge", "hold"]
