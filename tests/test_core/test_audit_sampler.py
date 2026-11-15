# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The audit sampler (R-13, M-04): a fixed-probability draw, blind of score and label."""

from __future__ import annotations

import dataclasses
import math
import sqlite3

import pytest

from superlocalmemory.core import audit_sampler as audit
from superlocalmemory.core.judge_ladder import Context, Item, JudgeLadder, TierAnswer, write_receipts
from superlocalmemory.core.judge_questions import SLOT_RELATION
from superlocalmemory.storage.migrations import M072_judge_receipts as M072

KEY = b"k" * 32


class Fixed:
    name = "laya"

    def __init__(self, answers):
        self._answers = answers

    def answer(self, question, items):
        return self._answers(items)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    M072.apply(c)
    return c


def _items(n):
    return [Item("slot_dispute", f"d{i}", "p", {"older": f"o{i}", "newer": f"n{i}"},
                 fact_ids=(f"a{i}", f"b{i}")) for i in range(n)]


def test_the_draw_is_fixed_by_key_and_id() -> None:
    assert audit.draw(KEY, "jr_1") == audit.draw(KEY, "jr_1")
    assert audit.draw(KEY, "jr_1") != audit.draw(b"other" * 7, "jr_1")
    assert 0.0 <= audit.draw(None, "jr_1") < 1.0


def test_the_selection_rate_matches_the_probability() -> None:
    n, p = 20_000, 0.10
    hits = sum(audit.select(KEY, f"jr_{i}", "same", p)[0] for i in range(n))
    sigma = math.sqrt(p * (1 - p) / n)
    assert abs(hits / n - p) < 4 * sigma


@pytest.mark.parametrize("verdict", ["no_answer", "tier_failed", "abstain"])
def test_non_answers_are_never_audited(verdict) -> None:
    assert audit.select(KEY, "jr_1", verdict, 1.0) == (False, None)


def test_without_a_probability_nothing_is_sampled() -> None:
    assert audit.select(KEY, "jr_1", "same", None) == (False, None)
    with pytest.raises(ValueError):
        audit.select(KEY, "jr_1", "same", 1.5)


def test_the_ladder_logs_the_draw_on_every_real_answer_whatever_its_score() -> None:
    scores = [0.55, 0.99]  # below and above the routing floor: both are candidates
    tier = Fixed(lambda items: [TierAnswer("same", s) for s in scores])
    ctx = Context(population="slot_dispute", hash_key=KEY, audit_probability=1.0)
    _, rows = JudgeLadder([tier]).decide(SLOT_RELATION, _items(2), ctx)
    assert [(r.accepted, r.audit_selected, r.audit_probability) for r in rows] == [
        (False, True, 1.0), (True, True, 1.0)]
    _, none = JudgeLadder([tier]).decide(SLOT_RELATION, _items(2),
                                         dataclasses.replace(ctx, audit_probability=0.0))
    assert [(r.audit_selected, r.audit_probability) for r in none] == [(False, 0.0)] * 2


def _receipts(conn, n, p=1.0, verdict="same", conf=0.95, model="m"):
    tier = Fixed(lambda items: [TierAnswer(verdict, conf, model=model)] * len(items))
    ctx = Context(population="slot_dispute", hash_key=KEY, audit_probability=p)
    _, rows = JudgeLadder([tier]).decide(SLOT_RELATION, _items(n), ctx)
    write_receipts(conn, rows)
    return rows


def _receipts_alone(conn, n, verdict="same", conf=0.95, model="m"):
    """n receipts, each asked on its own (regime ``single``: certifiable)."""
    rows = []
    for item in _items(n):
        tier = Fixed(lambda items: [TierAnswer(verdict, conf, model=model)] * len(items))
        ctx = Context(population="slot_dispute", hash_key=KEY, audit_probability=1.0)
        rows += JudgeLadder([tier]).decide(SLOT_RELATION, [item], ctx)[1]
    write_receipts(conn, rows)
    return rows


def test_the_queue_is_selected_and_unlabelled_and_hides_the_verdict(conn) -> None:
    rows = _receipts(conn, 3)
    audit.record_audit_label(conn, rows[0].receipt_id, "same", labeller="alex")
    items = audit.queue(conn, "p")
    assert [i.receipt_id for i in items] == [r.receipt_id for r in rows[1:]]
    fields = {f.name for f in dataclasses.fields(audit.AuditItem)}
    assert not fields & {"verdict", "confidence", "model", "tier", "probabilities_json"}


def test_an_audit_label_on_an_unselected_receipt_is_refused(conn) -> None:
    [row] = _receipts(conn, 1, p=0.0)
    with pytest.raises(audit.NotSelectedError):
        audit.record_audit_label(conn, row.receipt_id, "same", labeller="alex")


def test_the_writer_checks_the_label_and_the_labeller(conn) -> None:
    [row] = _receipts(conn, 1)
    with pytest.raises(ValueError):
        audit.record_audit_label(conn, row.receipt_id, "maybe", labeller="alex")
    with pytest.raises(ValueError):
        audit.record_audit_label(conn, row.receipt_id, "same", labeller="")
    with pytest.raises(LookupError):
        audit.record_audit_label(conn, "jr_missing", "same", labeller="alex")


def test_labels_written_through_the_sampler_earn_a_certificate(conn) -> None:
    from superlocalmemory.core.judge_calibration import ActingGate

    rows = _receipts_alone(conn, 110)  # a closing cell at alpha 0.05 needs 100 (delta/3)
    inbox = [r for r in _receipts_alone(conn, 300, verdict="replaces", conf=0.6, model="m2")
             if r.verdict == "replaces"]
    for r in inbox:  # real replacements in the tier's inbox (M-02 base rate)
        audit.record_audit_label(conn, r.receipt_id, "replaces", labeller="alex")
    gate = ActingGate(conn, profile_id="p", question_id=SLOT_RELATION.question_id,
                      version=SLOT_RELATION.version, alpha_for=lambda v: 0.05, delta=0.05)
    assert not gate.allows("laya", "m", "same", 0.95, population="slot_dispute",
                           regime="single")
    for r in rows:
        audit.record_audit_label(conn, r.receipt_id, "same", labeller="alex")
    gate = ActingGate(conn, profile_id="p", question_id=SLOT_RELATION.question_id,
                      version=SLOT_RELATION.version, alpha_for=lambda v: 0.05, delta=0.05)
    assert gate.allows("laya", "m", "same", 0.95, population="slot_dispute",
                           regime="single")
