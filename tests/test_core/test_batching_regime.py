# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M-01: the batching regime is part of a certificate cell's key.

A judge asked about one pair alone and the same judge asked about sixteen at
once are different instruments (R-23: answers in a batch can depend on their
peers). Labels earned in one regime must never let the other act.
"""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.core.audit_sampler import record_audit_label
from superlocalmemory.core.judge_calibration import ActingGate
from superlocalmemory.core.judge_ladder import (
    BATCHING_REGIMES,
    MAX_BATCH,
    Context,
    JudgeLadder,
    TierAnswer,
    batching_regime,
    write_receipts,
)
from superlocalmemory.core.judge_questions import SLOT_RELATION
from superlocalmemory.storage.migrations import M072_judge_receipts as M072
from superlocalmemory.storage.migrations import M077_judge_certificates as M077
from tests.test_core.test_audit_sampler import KEY, Fixed, _items


class Rule(Fixed):
    name = "rules"
    is_model = False


def _decide(tier, n, conn=None, model="m", verdict="same", conf=0.95):
    tier = tier(lambda items: [TierAnswer(verdict, conf, model=model)] * len(items))
    ctx = Context(population="slot_dispute", hash_key=KEY, audit_probability=1.0)
    decisions, rows = JudgeLadder([tier]).decide(SLOT_RELATION, _items(n), ctx)
    if conn is not None:
        write_receipts(conn, rows)
    return decisions, rows


def test_the_regime_names_how_the_decision_was_asked() -> None:
    assert batching_regime(1, model=True) == "single"
    assert batching_regime(2, model=True) == "batched"
    assert batching_regime(1, model=False) == batching_regime(9, model=False) == "unbatched"
    assert set(BATCHING_REGIMES) == {"single", "batched", "unbatched"}


@pytest.mark.parametrize(("tier", "n", "regime"), [
    (Fixed, 1, "single"), (Fixed, 3, "batched"), (Rule, 1, "unbatched"), (Rule, 3, "unbatched")])
def test_every_receipt_and_decision_carries_its_regime(tier, n, regime) -> None:
    decisions, rows = _decide(tier, n)
    assert {r.batching_regime for r in rows} == {regime}
    assert {d.regime for d in decisions} == {regime}


@pytest.mark.parametrize("n", [MAX_BATCH + 1, 2 * MAX_BATCH + 1, 3 * MAX_BATCH + 1, 50])
def test_no_item_is_left_alone_by_the_arithmetic_of_the_batches(n) -> None:
    # W0-25: 17 items were a batch of 16 and a batch of 1, so one item's regime (single, the
    # certifiable one) came from the remainder. The batches are now balanced: 9 + 8.
    _, rows = _decide(Fixed, n)
    assert {r.batching_regime for r in rows} == {"batched"}
    sizes = sorted({r.batch_size for r in rows})
    assert sizes[0] >= 2 and sizes[-1] <= MAX_BATCH and sizes[-1] - sizes[0] <= 1
    assert sum(1 for _ in rows) == n


def test_a_lone_item_with_no_peers_is_still_single() -> None:
    _, rows = _decide(Fixed, 1)
    assert [r.batching_regime for r in rows] == ["single"]


def test_a_tier_that_asks_one_at_a_time_stays_single_at_any_count() -> None:
    class OneAtATime(Fixed):
        def batch_limit(self, question):
            return 1

    _, rows = _decide(OneAtATime, 5)
    assert {r.batching_regime for r in rows} == {"single"}


def test_a_rule_is_never_batched_whatever_the_count() -> None:
    _, rows = _decide(Rule, MAX_BATCH + 1)
    assert {r.batching_regime for r in rows} == {"unbatched"}


def _gate(conn) -> ActingGate:
    return ActingGate(conn, profile_id="p", question_id=SLOT_RELATION.question_id,
                      version=SLOT_RELATION.version, alpha_for=lambda v: 0.05, delta=0.05)


def _decide_alone(n, conn, model="m", verdict="same", conf=0.95):
    """n decisions, each asked on its own (regime ``single``)."""
    rows = []
    for i in range(n):
        tier = Fixed(lambda items: [TierAnswer(verdict, conf, model=model)] * len(items))
        ctx = Context(population="slot_dispute", hash_key=KEY, audit_probability=1.0)
        item = _items(n)[i]
        rows += JudgeLadder([tier]).decide(SLOT_RELATION, [item], ctx)[1]
    write_receipts(conn, rows)
    return rows


def _label_all(conn, rows, label):
    for r in rows:
        record_audit_label(conn, r.receipt_id, label, labeller="alex")


def test_a_batched_cell_never_acts_and_never_certifies_a_single_ask() -> None:
    conn = sqlite3.connect(":memory:")
    M072.apply(conn)
    M077.apply(conn)
    _, rows = _decide(Fixed, 112, conn)  # seven full batches of sixteen
    _, inbox = _decide(Fixed, 304, conn, model="m2", verdict="replaces", conf=0.6)
    _label_all(conn, inbox, "replaces")  # real replacements in the tier's inbox (M-02)
    _label_all(conn, rows, "same")
    gate = _gate(conn)
    for regime in ("batched", "single", "unknown"):  # W0-04: labels in a batch are not
        assert not gate.allows("laya", "m", "same", 0.95,  # independent trials
                               population="slot_dispute", regime=regime)


def test_a_single_ask_cell_certifies_on_its_own_labels() -> None:
    conn = sqlite3.connect(":memory:")
    M072.apply(conn)
    M077.apply(conn)
    rows = _decide_alone(112, conn)
    inbox = _decide_alone(300, conn, model="m2", verdict="replaces", conf=0.6)
    _label_all(conn, inbox, "replaces")
    _label_all(conn, rows, "same")
    gate = _gate(conn)
    assert gate.allows("laya", "m", "same", 0.95, population="slot_dispute", regime="single")
    assert not gate.allows("laya", "m", "same", 0.95, population="slot_dispute",
                           regime="batched")


def test_the_regime_is_a_required_part_of_the_gate_key() -> None:
    conn = sqlite3.connect(":memory:")
    M072.apply(conn)
    M077.apply(conn)
    with pytest.raises(TypeError):
        _gate(conn).allows("laya", "m", "same", 0.95, population="slot_dispute")


def test_balanced_batches_keep_order_respect_the_limit_and_leave_no_remainder() -> None:
    from superlocalmemory.core.judge_ladder import balanced_batches

    assert balanced_batches([], 16) == []
    assert balanced_batches([7], 16) == [[7]]
    assert balanced_batches(list(range(16)), 16) == [list(range(16))]
    assert [len(b) for b in balanced_batches(list(range(33)), 32)] == [17, 16]
    assert [len(b) for b in balanced_batches(list(range(17)), 16)] == [9, 8]
    assert [len(b) for b in balanced_batches(list(range(5)), 1)] == [1] * 5
    for n, limit in ((100, 16), (31, 16), (49, 16), (2, 16)):
        batches = balanced_batches(list(range(n)), limit)
        assert [i for b in batches for i in b] == list(range(n))
        assert max(map(len, batches)) <= limit and min(map(len, batches)) >= 2
