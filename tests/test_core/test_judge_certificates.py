# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Certificates as records: issued, voided on labels, voided on drift (I-14, R-26, M-07)."""

from __future__ import annotations

import pytest

from superlocalmemory.core import judge_certificates as records
from superlocalmemory.core.judge_calibration import ActingGate
from superlocalmemory.core.judge_questions import SLOT_RELATION
from tests.test_core.test_coverage_backfill import _db
from tests.test_truth.test_dispute_pass import _seed

CELL = dict(population="slot_dispute", regime="single")


@pytest.fixture
def db(tmp_path):
    return _db(tmp_path / "memory.db")


def _gate(conn):
    return ActingGate(conn, profile_id="default", question_id=SLOT_RELATION.question_id,
                      version=SLOT_RELATION.version)


def _allows(db, verdict="replaces", confidence=0.95):
    with db.raw_connection() as conn:
        return _gate(conn).allows("laya", "fake", verdict, confidence, **CELL)


def _rows(db):
    return [dict(r) for r in db.execute("SELECT * FROM judge_certificates ORDER BY issued_at")]


def _recent(db, n, confidence):
    with db.raw_connection() as conn:
        for i in range(n):
            conn.execute(
                "INSERT INTO judge_receipts (receipt_id, profile_id, question_id, question_version, "
                "subject_kind, subject_id, population, tier, model, verdict, confidence, threshold, "
                "accepted, calibration_id, inputs_hash, decided_at, audit_selected, "
                "audit_probability, batching_regime, inbox_selected) VALUES (?, 'default', "
                "'SLOT_RELATION', ?, 'seed', ?, 'slot_dispute', 'laya', 'fake', 'replaces', ?, 0.7, "
                "1, 'c', ?, '2099-01-01T00:00:00+00:00', 0, 0.1, 'single', 0)",
                (f"recent-{i}", SLOT_RELATION.version, f"recent-{i}", confidence, f"h-recent-{i}"))


def test_a_certificate_is_recorded_once_when_it_first_holds(db) -> None:
    _seed(db, "replaces", 80, tier="laya", model="fake")
    assert _allows(db) is True and _allows(db) is True
    (row,) = _rows(db)
    assert (row["tier"], row["model"], row["verdict"], row["method"]) == ("laya", "fake", "replaces",
                                                                         "eprocess")
    assert row["n"] > 0 and row["errors"] == 0 and row["voided_at"] is None
    assert row["label_set_hash"] == records.label_set_hash([f"seed-replaces-{i}" for i in range(80)])
    assert row["question_version"] == SLOT_RELATION.version


def test_no_labels_no_record(db) -> None:
    assert _allows(db) is False and _rows(db) == []


def test_labels_that_stop_supporting_it_void_it(db) -> None:
    _seed(db, "replaces", 80, tier="laya", model="fake")
    assert _allows(db)
    db.execute("UPDATE judge_labels SET label = 'unrelated' WHERE receipt_id IN "
               "('seed-replaces-0', 'seed-replaces-1', 'seed-replaces-2', 'seed-replaces-3')")
    assert _allows(db) is False
    (row,) = _rows(db)
    assert row["void_reason"] == "labels" and row["voided_at"]


def test_scores_that_shift_void_it_for_drift_and_old_labels_cannot_reissue_it(db) -> None:
    _seed(db, "replaces", 80, tier="laya", model="fake")  # audited at 0.95
    assert _allows(db)
    _recent(db, records.MIN_RECENT + 10, 0.55)  # the judge now decides at much lower scores
    assert _allows(db) is False
    (row,) = _rows(db)
    assert row["void_reason"] == "drift"
    assert _allows(db) is False and len(_rows(db)) == 1  # the old labels no longer count


def test_too_few_recent_decisions_are_not_tested_for_drift(db) -> None:
    _seed(db, "replaces", 80, tier="laya", model="fake")
    assert _allows(db)
    _recent(db, records.MIN_RECENT - 1, 0.55)
    assert _allows(db) is True and _rows(db)[0]["voided_at"] is None


def test_recent_scores_like_the_audited_ones_do_not_void_it(db) -> None:
    _seed(db, "replaces", 80, tier="laya", model="fake")
    assert _allows(db)
    _recent(db, records.MIN_RECENT + 10, 0.95)
    assert _allows(db) is True and _rows(db)[0]["voided_at"] is None


def test_a_new_model_version_starts_with_no_certificate(db) -> None:
    _seed(db, "replaces", 80, tier="laya", model="fake")
    assert _allows(db)
    with db.raw_connection() as conn:
        assert _gate(conn).allows("laya", "fake-v2", "replaces", 0.95, **CELL) is False


def test_ks_statistic_and_pvalue() -> None:
    same = [0.1 * i for i in range(10)]
    assert records.ks_statistic(same, same) == 0.0
    assert records.ks_pvalue(0.0, 50, 50) == 1.0
    assert records.ks_statistic([0.1] * 20, [0.9] * 20) == 1.0
    assert records.ks_pvalue(1.0, 40, 40) < 1e-6
    assert records.ks_pvalue(0.1, 40, 40) > 0.5


def test_the_records_listing_shows_the_newest_first(db) -> None:
    _seed(db, "replaces", 80, tier="laya", model="fake")
    _allows(db)
    with db.raw_connection() as conn:
        (rec,) = records.records(conn, "default")
    assert rec["verdict"] == "replaces" and rec["voided_at"] is None
