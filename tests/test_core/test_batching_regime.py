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


def test_labels_earned_in_batches_never_certify_a_single_ask() -> None:
    conn = sqlite3.connect(":memory:")
    M072.apply(conn)
    _, rows = _decide(Fixed, 112, conn)  # seven full batches of sixteen
    _, inbox = _decide(Fixed, 304, conn, model="m2", verdict="replaces", conf=0.6)
    for r in inbox:  # real replacements in the tier's inbox (M-02 base rate)
        record_audit_label(conn, r.receipt_id, "replaces", labeller="alex")
    for r in rows:
        record_audit_label(conn, r.receipt_id, "same", labeller="alex")
    gate = _gate(conn)
    assert gate.allows("laya", "m", "same", 0.95, population="slot_dispute", regime="batched")
    assert not gate.allows("laya", "m", "same", 0.95, population="slot_dispute",
                           regime="single")
    assert not gate.allows("laya", "m", "same", 0.95, population="slot_dispute",
                           regime="unknown")


def test_the_regime_is_a_required_part_of_the_gate_key() -> None:
    conn = sqlite3.connect(":memory:")
    M072.apply(conn)
    with pytest.raises(TypeError):
        _gate(conn).allows("laya", "m", "same", 0.95, population="slot_dispute")


def test_m072_repair_brings_an_earlier_store_up_to_date_as_unknown() -> None:
    conn = sqlite3.connect(":memory:")
    old = M072.DDL.replace(
        "    batching_regime   TEXT NOT NULL DEFAULT 'unknown'\n"
        "        CHECK (batching_regime IN ('single', 'batched', 'unbatched', 'unknown')),\n", "")
    assert old != M072.DDL
    for statement in old.split(";"):
        if statement.strip():
            conn.execute(statement)
    conn.execute("INSERT INTO judge_receipts (receipt_id, profile_id, question_id, "
                 "question_version, subject_kind, subject_id, tier, model, verdict, threshold, "
                 "accepted, calibration_id, inputs_hash, decided_at) VALUES "
                 "('r', 'p', 'Q', '2', 's', 's', 'jev', 'm', 'same', 0.7, 1, 'c', 'h', 't')")
    assert not M072.verify(conn)
    M072.repair(conn)
    assert M072.verify(conn)
    assert conn.execute("SELECT batching_regime FROM judge_receipts").fetchall() == [("unknown",)]
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE judge_receipts SET batching_regime = 'pooled'")
