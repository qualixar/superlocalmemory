# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W0-11: the fixed sequence over the threshold grid starts where the cell first tested.

The start depends on the scores only (never the labels) and is stored once per cell
(M077 ``judge_certificate_starts``), so the family-wise error stays at delta and a
certificate at a loose threshold is not voided when a stricter one becomes testable.
"""

from __future__ import annotations

import random
import sqlite3

import pytest

from superlocalmemory.core import judge_certificates as records
from superlocalmemory.core.judge_calibration import (
    ActingGate,
    eprocess_certificate,
    first_testable_start,
)
from superlocalmemory.storage.migrations import M072_judge_receipts as M072
from superlocalmemory.storage.migrations import M077_judge_certificates as M077

ALPHA = DELTA = 0.05
_INSERT = ("INSERT INTO judge_receipts (receipt_id, profile_id, question_id, question_version,"
           " subject_kind, subject_id, population, tier, model, verdict, confidence, threshold,"
           " accepted, calibration_id, inputs_hash, decided_at, audit_selected,"
           " audit_probability, batching_regime, inbox_selected)"
           " VALUES (?, 'p', 'Q', '2', 's', ?, 'slot_dispute', 'jev', 'm', 'replaces', ?, 0.7, 1,"
           " 'c', ?, ?, 1, 0.1, 'single', 1)")


def _store() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    M072.apply(conn)
    M077.apply(conn)
    return conn


def _add(conn, seq, *, first=0) -> None:
    """Drawn, labelled receipts in decision order; ``seq`` is (confidence, wrong) pairs."""
    for i, (conf, wrong) in enumerate(seq, first):
        rid = f"r{i}"
        conn.execute(_INSERT, (rid, rid, conf, f"h{i}",
                               f"2026-11-16T{i // 3600:02d}:{i // 60 % 60:02d}:{i % 60:02d}"))
        conn.execute("INSERT INTO judge_labels (receipt_id, label, source, labeller, blind,"
                     " labelled_at) VALUES (?, ?, 'audit', 'a', 1, 't')",
                     (rid, "unrelated" if wrong else "replaces"))


def _gate(conn) -> ActingGate:
    return ActingGate(conn, profile_id="p", question_id="Q", version="2",
                      alpha_for=lambda verdict: ALPHA, delta=DELTA)


def _permit(conn, conf=0.85) -> bool:
    return _gate(conn).allows("jev", "m", "replaces", conf, population="slot_dispute",
                              regime="single")


#: 600 clean mid-score items with 100 high-score ones (6 wrong) mixed in after the first 200.
def _flap_sequence():
    rng = random.Random(7)
    mid = [(0.85, False)] * 200
    rest = [(0.85, False)] * 400 + [(0.96, i % 17 == 0) for i in range(100)]
    rng.shuffle(rest)
    return mid + rest


def test_the_certificate_names_its_start_and_a_given_start_is_the_first_threshold_tested() -> None:
    scores, wrong = [0.85] * 120, [False] * 120
    cert = eprocess_certificate(scores, wrong, alpha=ALPHA, delta=DELTA)
    assert cert.start == 0.85 and cert.acts  # the chain runs on to the loosest grid point
    # A start that was fixed earlier is used even when a stricter threshold is testable now.
    both = [0.96] * 100 + [0.85] * 100
    clean = [False] * 200
    assert eprocess_certificate(both, clean, alpha=ALPHA, delta=DELTA).start == 0.95
    pinned = eprocess_certificate(both, clean, alpha=ALPHA, delta=DELTA, start=0.85)
    assert pinned.start == 0.85 and [t[0] for t in pinned.tested][0] == 0.85


def test_a_stricter_threshold_that_becomes_testable_does_not_void_a_looser_certificate() -> None:
    seq = _flap_sequence()
    scores, wrong = [s for s, _ in seq], [w for _, w in seq]
    first = eprocess_certificate(scores[:200], wrong[:200], alpha=ALPHA, delta=DELTA)
    assert first.start == 0.85 and first.acts
    moving = eprocess_certificate(scores, wrong, alpha=ALPHA, delta=DELTA)
    assert moving.threshold is None  # a start chosen afresh tests 0.95 first, and it fails
    fixed = eprocess_certificate(scores, wrong, alpha=ALPHA, delta=DELTA, start=first.start)
    assert fixed.acts and fixed.start == 0.85


def test_the_gate_stores_the_start_at_its_first_test_and_does_not_flap() -> None:
    conn = _store()
    seq = _flap_sequence()
    _add(conn, seq[:200])
    assert _permit(conn) is True
    stored = conn.execute("SELECT start_threshold FROM judge_certificate_starts").fetchall()
    assert stored == [(0.85,)]
    _add(conn, seq[200:], first=200)  # 0.95 is testable now, and fails on its own
    assert _permit(conn) is True
    (row,) = conn.execute("SELECT voided_at, threshold FROM judge_certificates").fetchall()
    assert row[0] is None


def test_a_threshold_is_testable_at_exactly_the_clean_run_it_needs() -> None:
    assert first_testable_start([0.85] * 73, alpha=ALPHA, delta=DELTA) == 0.85  # 73 clean labels
    assert first_testable_start([0.85] * 72, alpha=ALPHA, delta=DELTA) is None


def test_no_start_is_stored_before_any_threshold_can_be_tested() -> None:
    conn = _store()
    _add(conn, [(0.85, False)] * 10)
    assert _permit(conn) is False
    assert conn.execute("SELECT COUNT(*) FROM judge_certificate_starts").fetchone() == (0,)


def test_a_stored_start_cannot_move() -> None:
    conn = _store()
    _add(conn, _flap_sequence()[:200])
    _permit(conn)
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE judge_certificate_starts SET start_threshold = 0.95")
    cell = records.Cell("p", "Q", "2", "jev", "m", "replaces", "slot_dispute", "single")
    assert records.fix_start(conn, cell, 0.95) == 0.85  # the second claim is ignored
    assert records.start_of(conn, cell) == 0.85


def test_a_store_that_ran_the_earlier_m077_is_repaired() -> None:
    conn = sqlite3.connect(":memory:")
    M072.apply(conn)
    conn.executescript(M077.DDL.split("CREATE TABLE IF NOT EXISTS judge_certificate_starts")[0])
    assert not M077.verify(conn)
    M077.repair(conn)
    assert M077.verify(conn)


def _simulate(cells: int, truth, seed: int, looks=(100, 160, 240)) -> float:
    """Share of cells that certify a threshold whose true error rate exceeds alpha."""
    rng = random.Random(seed)
    false = 0
    for _ in range(cells):
        scores = [rng.uniform(0.5, 1.0) for _ in range(looks[-1])]
        wrong = [rng.random() < truth(s) for s in scores]
        start = None
        for k in looks:
            cert = eprocess_certificate(scores[:k], wrong[:k], alpha=ALPHA, delta=DELTA,
                                        start=start)
            start = start if start is not None else cert.start
            if cert.threshold is not None and _true_rate(truth, cert.threshold) > ALPHA:
                false += 1
                break
    return false / cells


def _true_rate(truth, threshold: float) -> float:
    """The exact error rate among scores >= threshold, scores uniform on [0.5, 1]."""
    steps = 2000
    xs = [threshold + (1.0 - threshold) * (i + 0.5) / steps for i in range(steps)]
    return sum(truth(x) for x in xs) / steps


#: 1500 cells: standard error of a 0.05 rate is 0.0056; the bound is delta + 3 SE.
TOLERANCE = DELTA + 3 * (DELTA * (1 - DELTA) / 1500) ** 0.5


def test_simulation_every_threshold_at_the_boundary_stays_within_delta() -> None:
    rate = _simulate(1500, lambda s: ALPHA - 1e-9, seed=20261118)
    assert rate <= TOLERANCE


def test_simulation_graded_error_rates_stay_within_delta() -> None:
    # Strict scores are reliable (1%), loose ones are not (12%): thresholds from 0.85 up are
    # truly below alpha, 0.80 and looser are truly above it and must almost never certify.
    rate = _simulate(1500, lambda s: 0.01 if s >= 0.9 else 0.12, seed=20261119)
    assert rate <= TOLERANCE


def test_a_development_store_with_the_earlier_m077_is_upgraded_not_failed(tmp_path) -> None:
    from superlocalmemory.storage import migration_runner as mr
    from tests.test_core.test_coverage_backfill import _db

    memory = tmp_path / "memory.db"
    _db(memory)
    learning = tmp_path / "memory-learning.db"
    earlier = "e63eec887f8b8025de4702031abce29ecb25813dc9857dc33d01a72029f0ee22"
    with sqlite3.connect(memory) as conn:
        conn.execute("DROP TABLE judge_certificate_starts")
        conn.execute("UPDATE migration_log SET ddl_sha256 = ? WHERE name = ?",
                     (earlier, M077.NAME))
        assert M077.verify(conn) is False
    result = mr.apply_deferred(learning, memory)
    assert result["failed"] == [], result
    with sqlite3.connect(memory) as conn:
        assert M077.verify(conn) is True
