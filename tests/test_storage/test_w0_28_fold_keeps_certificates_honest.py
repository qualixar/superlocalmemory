# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W0-28: folding profile A into B must not let B inherit a certificate earned on A's labels,
nor spoil one B earned itself; and no evidence is lost."""

from __future__ import annotations

import pytest

from superlocalmemory.core.judge_calibration import ActingGate
from superlocalmemory.core.judge_questions import SLOT_RELATION
from superlocalmemory.storage.profile_fold import fold_profile
from tests.test_core.test_coverage_backfill import _db
from tests.test_truth.test_dispute_pass import _seed

CELL = dict(population="slot_dispute", regime="single")


@pytest.fixture
def db(tmp_path):
    return _db(tmp_path / "memory.db")


def _gate(conn, profile):
    return ActingGate(conn, profile_id=profile, question_id=SLOT_RELATION.question_id,
                      version=SLOT_RELATION.version)


def _allows(db, profile) -> bool:
    with db.raw_connection() as conn:
        return _gate(conn, profile).allows("laya", "fake", "replaces", 0.95, **CELL)


def _two_profiles(db, *, source_wrong: int):
    """default earns a certificate on 80 clean audits; 'work' has its own audited receipts."""
    _seed(db, "replaces", 80, tier="laya", model="fake", tag="own")
    _seed(db, "replaces", 120, tier="laya", model="fake", tag="src", wrong=source_wrong)
    with db.raw_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('work', 'work')")
        conn.execute("UPDATE judge_receipts SET profile_id = 'work' WHERE receipt_id LIKE "
                     "'seed-src-%'")


def _fold(db):
    with db.raw_connection() as conn:
        fold_profile(conn, "work", "default")


def test_the_target_keeps_its_certificate_when_the_source_had_errors(db) -> None:
    _two_profiles(db, source_wrong=40)
    assert _allows(db, "default") is True
    (before,) = db.execute("SELECT cert_id, n, errors FROM judge_certificates "
                           "WHERE voided_at IS NULL")
    _fold(db)
    assert _allows(db, "default") is True  # the source's wrong labels are not in its cell
    (after,) = db.execute("SELECT cert_id, n, errors, voided_at FROM judge_certificates "
                          "WHERE voided_at IS NULL")
    assert tuple(after)[:3] == tuple(before)  # same certificate, same n, no errors added


def test_the_target_cannot_inherit_a_certificate_the_source_earned(db) -> None:
    _seed(db, "replaces", 120, tier="laya", model="fake", tag="src")  # the source's clean audits
    with db.raw_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('work', 'work')")
        conn.execute("UPDATE judge_receipts SET profile_id = 'work'")
    assert _allows(db, "work") is True and _allows(db, "default") is False
    _fold(db)
    assert _allows(db, "default") is False  # it has earned nothing itself


def test_no_evidence_is_lost_receipts_and_labels_move_and_are_marked_folded(db) -> None:
    _two_profiles(db, source_wrong=10)
    _fold(db)
    moved = db.execute("SELECT profile_id, population FROM judge_receipts "
                       "WHERE receipt_id LIKE 'seed-src-%'")
    assert len(moved) == 120
    assert {tuple(r) for r in moved} == {("default", "folded")}
    assert db.execute("SELECT COUNT(*) AS n FROM judge_labels WHERE receipt_id LIKE "
                      "'seed-src-%'")[0]["n"] == 120
    own = db.execute("SELECT DISTINCT population FROM judge_receipts WHERE receipt_id LIKE "
                     "'seed-own-%'")
    assert [r["population"] for r in own] == ["slot_dispute"]


def test_the_stale_comment_is_gone() -> None:
    import pathlib

    text = (pathlib.Path(__file__).resolve().parents[2]
            / "src/superlocalmemory/storage/profile_fold.py").read_text(encoding="utf-8")
    assert "Slot ids are\n    # derived from the profile" not in text
    assert "ids name the profile" not in text
