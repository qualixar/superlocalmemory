# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M-06 / R-30: shadow escalation of confident answers to the next tier up."""

from __future__ import annotations

import math

import pytest

from superlocalmemory.core import shadow_escalation as shadow
from superlocalmemory.core.judge_calibration import ActingGate
from superlocalmemory.core.judge_ladder import (
    SHADOW_SUFFIX,
    Context,
    JudgeLadder,
    TierAnswer,
    write_receipts,
)
from superlocalmemory.core.judge_questions import SLOT_RELATION
from tests.test_core.test_audit_sampler import KEY, _items, conn  # noqa: F401 - fixture


class Tier:
    def __init__(self, name, verdict, conf, per_item=True):
        self.name, self.verdict, self.conf = name, verdict, conf
        self.per_item = per_item
        self.asked: list[str] = []

    def answer(self, question, items):
        self.asked += [it.subject_id for it in items]
        return [TierAnswer(self.verdict, self.conf, model=self.name) for _ in items]


def _ctx(eps):
    return Context(population="slot_dispute", hash_key=KEY, audit_probability=0.1,
                   escalation_probability=eps)


def test_every_decided_pair_goes_up_at_eps_one_in_shadow_only() -> None:
    low, high = Tier("laya", "same", 0.95), Tier("jev", "replaces", 0.99)
    decisions, rows = JudgeLadder([low, high]).decide(SLOT_RELATION, _items(4), _ctx(1.0))
    assert [d.tier for d in decisions] == ["laya"] * 4  # the decision is the lower tier's
    assert {d.verdict for d in decisions} == {"same"}
    assert sorted(high.asked) == ["d0", "d1", "d2", "d3"]
    up = [r for r in rows if r.tier == "jev"]
    assert len(up) == 4 and {r.population for r in up} == {"slot_dispute" + SHADOW_SUFFIX}
    assert not any(r.accepted for r in up)  # a shadow answer is never a decision
    assert not {r.receipt_id for r in up} & {rid for d in decisions for rid in d.receipts}


def test_off_by_default_and_when_none() -> None:
    low, high = Tier("laya", "same", 0.95), Tier("jev", "same", 0.99)
    JudgeLadder([low, high]).decide(SLOT_RELATION, _items(20), _ctx(None))
    JudgeLadder([low, high]).decide(SLOT_RELATION, _items(20), Context(hash_key=KEY))
    assert high.asked == []


def test_the_draw_is_one_per_pair_near_eps_and_repeatable() -> None:
    n, eps = 4000, 0.05
    first, again = Tier("jev", "same", 0.9), Tier("jev", "same", 0.9)
    JudgeLadder([Tier("laya", "same", 0.95), first]).decide(SLOT_RELATION, _items(n), _ctx(eps))
    JudgeLadder([Tier("laya", "same", 0.95), again]).decide(SLOT_RELATION, _items(n), _ctx(eps))
    assert first.asked == again.asked
    assert abs(len(first.asked) / n - eps) < 4 * math.sqrt(eps * (1 - eps) / n)


def test_the_top_tier_and_abstentions_never_escalate() -> None:
    low, top = Tier("laya", "same", 0.10), Tier("jev", "same", 0.95)  # laya unsure: jev decides
    decisions, rows = JudgeLadder([low, top]).decide(SLOT_RELATION, _items(3), _ctx(1.0))
    assert [d.tier for d in decisions] == ["jev"] * 3 and len(top.asked) == 3
    assert not any(r.population.endswith(SHADOW_SUFFIX) for r in rows)
    nobody = Tier("jev", "same", 0.10)
    decisions, rows = JudgeLadder([Tier("laya", "same", 0.1), nobody]).decide(
        SLOT_RELATION, _items(3), _ctx(1.0))
    assert all(d.abstained for d in decisions) and len(nobody.asked) == 3


def test_only_one_tier_up() -> None:
    low, mid, top = Tier("rules", "same", 0.95), Tier("laya", "same", 0.95), Tier("jev", "x", 0.9)
    JudgeLadder([low, mid, top]).decide(SLOT_RELATION, _items(3), _ctx(1.0))
    assert len(mid.asked) == 3 and top.asked == []


def test_an_eps_above_one_is_refused() -> None:
    with pytest.raises(ValueError):
        JudgeLadder([Tier("laya", "same", 0.95), Tier("jev", "same", 0.9)]).decide(
            SLOT_RELATION, _items(1), _ctx(1.5))


def _run(conn, high_verdict, high_conf, n=10):  # noqa: F811
    low, high = Tier("laya", "same", 0.95), Tier("jev", high_verdict, high_conf)
    _d, rows = JudgeLadder([low, high]).decide(SLOT_RELATION, _items(n), _ctx(1.0))
    write_receipts(conn, rows)


def test_a_sure_different_upper_answer_is_a_disagreement(conn) -> None:  # noqa: F811
    _run(conn, "replaces", 0.95)
    (step,) = shadow.disagreements(conn, "p")
    assert (step.lower_tier, step.upper_tier, step.pairs, step.disagreed) == ("laya", "jev", 10, 10)
    assert step.rate_lower > 0.5 and step.rate_upper == 1.0


def test_an_unsure_or_agreeing_upper_answer_is_not(conn) -> None:  # noqa: F811
    _run(conn, "replaces", 0.50)  # below the 0.85 routing threshold for replaces: no opinion
    (step,) = shadow.disagreements(conn, "p")
    assert (step.pairs, step.disagreed) == (10, 0) and step.rate_upper < 0.30


def test_a_pair_is_counted_once(conn) -> None:  # noqa: F811
    _run(conn, "same", 0.95)
    _run(conn, "same", 0.95)
    (step,) = shadow.disagreements(conn, "p")
    assert (step.pairs, step.disagreed) == (10, 0)


def test_shadow_receipts_never_feed_a_real_cell_or_a_reuse(conn) -> None:  # noqa: F811
    _run(conn, "same", 0.99, n=200)
    stored = conn.execute("SELECT population, accepted, COUNT(*) FROM judge_receipts "
                          "WHERE tier = 'jev' GROUP BY 1, 2").fetchall()
    assert stored == [("slot_dispute" + SHADOW_SUFFIX, 0, 200)]
    gate = ActingGate(conn, profile_id="p", question_id=SLOT_RELATION.question_id,
                      version=SLOT_RELATION.version)
    assert gate.allows("jev", "jev", "same", 0.99, population="slot_dispute",
                       regime="single") is False


def test_no_table_no_report() -> None:
    import sqlite3

    assert shadow.disagreements(sqlite3.connect(":memory:"), "p") == []


def test_only_the_accepted_lower_answer_is_compared(conn) -> None:  # noqa: F811
    class Failing(Tier):
        def answer(self, question, items):
            raise RuntimeError("down")

    high = Tier("jev", "same", 0.95)
    _d, rows = JudgeLadder([Failing("laya", "same", 0.9), high]).decide(
        SLOT_RELATION, _items(5), _ctx(1.0))  # laya failed: jev decided for real, no shadow
    write_receipts(conn, rows)
    _run(conn, "same", 0.95, n=5)  # later laya is sure; jev agrees in shadow
    (step,) = shadow.disagreements(conn, "p")
    assert (step.pairs, step.disagreed) == (5, 0)
