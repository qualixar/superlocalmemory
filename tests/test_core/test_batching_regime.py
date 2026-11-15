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


def test_a_lone_tail_after_full_batches_is_single() -> None:
    _, rows = _decide(Fixed, MAX_BATCH + 1)
    assert [r.batching_regime for r in rows].count("single") == 1


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
    with pytest.raises(TypeError):
        _gate(conn).allows("laya", "m", "same", 0.95, population="slot_dispute")
