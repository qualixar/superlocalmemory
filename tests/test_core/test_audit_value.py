# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W2 innovation 5: the audit queue asks first where an answer is worth most."""

from __future__ import annotations

import json
import sqlite3

import pytest

from superlocalmemory.core import audit_value
from superlocalmemory.core.judge_questions import SLOT_RELATION
from superlocalmemory.storage.migrations import M072_judge_receipts as M072


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    M072.apply(c)
    c.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, access_count INTEGER)")
    return c


_N = iter(range(10_000))


def _r(conn, h, *, verdict="same", conf=0.9, model="m", regime="single", minute=None,
       facts=("fa", "fb"), labelled=False, drawn=1):
    i = next(_N)
    rid = f"r{i}"
    minute = i if minute is None else minute
    conn.execute(
        "INSERT INTO judge_receipts (receipt_id, profile_id, question_id, question_version, "
        "subject_kind, subject_id, population, fact_ids_json, tier, model, verdict, confidence, "
        "threshold, accepted, calibration_id, inputs_hash, decided_at, audit_selected, "
        "audit_probability, batching_regime, inbox_selected) VALUES (?, 'p', 'SLOT_RELATION', ?, "
        "'seed', ?, 'slot_dispute', ?, 'laya', ?, ?, ?, 0.7, 1, 'c', ?, ?, ?, 0.1, ?, ?)",
        (rid, SLOT_RELATION.version, rid, json.dumps(list(facts)), model, verdict, conf, h,
         f"2026-11-16T00:{minute // 60:02d}:{minute % 60:02d}+00:00", drawn, regime,
         1 - drawn))
    if labelled:
        conn.execute("INSERT INTO judge_labels (receipt_id, label, source, labeller, blind, "
                     "labelled_at) VALUES (?, ?, 'audit', 'person', 1, '2026-11-16')",
                     (rid, verdict))
    return (h, "SLOT_RELATION", json.dumps(list(facts)), f"t{minute:05d}")


def _order(conn, rows):
    return [r[0] for r in audit_value.order(conn, "p", rows)]


def test_a_cells_oldest_unanswered_question_comes_before_riskier_later_ones(conn) -> None:
    _r(conn, "done", labelled=True)
    head = _r(conn, "head", conf=0.99)  # the next one this cell's certificate waits for
    later = _r(conn, "later", conf=0.10)  # riskier, but a label here changes nothing yet
    assert _order(conn, [head, later]) == ["head", "later"]
    assert _order(conn, [later, head]) == ["head", "later"]


def test_the_cell_nearest_a_certificate_goes_first(conn) -> None:
    for _ in range(50):
        _r(conn, f"near-{next(_N)}", model="near", labelled=True)
    for _ in range(80):  # labels on base-rate-only receipts do not move a certificate
        _r(conn, f"far-inbox-{next(_N)}", model="far", labelled=True, drawn=0)
    near = _r(conn, "near-head", model="near")
    far = _r(conn, "far-head", model="far")
    assert _order(conn, [far, near]) == ["near-head", "far-head"]


def test_a_batched_cell_cannot_certify_so_its_head_is_not_preferred(conn) -> None:
    batched = _r(conn, "batched-head", regime="batched", conf=0.99)
    single = _r(conn, "single-head", conf=0.99)
    assert _order(conn, [batched, single]) == ["single-head", "batched-head"]


def test_among_the_rest_harm_times_doubt_orders(conn) -> None:
    _r(conn, "h", model="x")  # takes the head of cell x/same
    low = _r(conn, "sure-same", model="x", conf=0.95)
    _r(conn, "hr", model="x", verdict="replaces")  # head of cell x/replaces
    risky = _r(conn, "unsure-replaces", model="x", verdict="replaces", conf=0.6)
    assert _order(conn, [low, risky]) == ["unsure-replaces", "sure-same"]
    same_doubt = _r(conn, "unsure-same", model="x", conf=0.6)
    assert _order(conn, [same_doubt, risky]) == ["unsure-replaces", "unsure-same"]  # harm


def test_more_used_memories_go_first_when_all_else_is_equal(conn) -> None:
    _r(conn, "h", model="y")
    conn.execute("INSERT INTO atomic_facts VALUES ('hot', 500), ('cold', 0)")
    cold = _r(conn, "cold-pair", model="y", facts=("cold", "fb"))
    hot = _r(conn, "hot-pair", model="y", facts=("hot", "fb"))
    assert _order(conn, [cold, hot]) == ["hot-pair", "cold-pair"]


def test_a_rule_without_a_score_counts_as_half_sure(conn) -> None:
    _r(conn, "h1", model="z")
    rule = _r(conn, "rule", model="z", conf=None)
    sure = _r(conn, "sure", model="z", conf=0.9)
    assert _order(conn, [sure, rule]) == ["rule", "sure"]


def test_without_receipts_the_order_is_kept() -> None:
    rows = [("a", "SLOT_RELATION", "[]", "t1"), ("b", "SLOT_RELATION", "[]", "t2")]
    assert [r[0] for r in audit_value.order(sqlite3.connect(":memory:"), "p", rows)] == ["a", "b"]
