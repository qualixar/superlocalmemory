# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-13: the correction backlog judged by the ladder; closing only with a certificate."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.core import overtaken_cases as ot
from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.core.judge_ladder import JudgeLadder
from superlocalmemory.truth.backlog_pass import POPULATION, BacklogPass
from tests.helpers.flag_baseline import off_except
from tests.test_core.test_coverage_backfill import _db
from tests.test_truth.test_dispute_pass import Fixed, _calibrate
from tests.test_truth.test_slot_pass import _fact


def _propose(db, case_id, pred, succ, *, kind="host_attested", reason="temporal_contradiction"):
    from superlocalmemory.storage.correction_cases import CorrectionActor, propose_on_connection

    actor = CorrectionActor(actor_id="canonical-writer", actor_kind=kind,
                            trust_tier="canonical_writer" if kind == "host_attested" else "trusted")
    path = db.db_path if hasattr(db, "db_path") else db._db_path
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")
    propose_on_connection(conn, case_id=case_id, profile_id="default", scope="personal",
                          predecessor_fact_id=pred, successor_fact_id=succ, reason_code=reason,
                          actor=actor, idempotency_key=f"k-{case_id}",
                          is_profile_active=lambda p: True, is_actor_trusted=lambda a: True)
    conn.execute("COMMIT")
    conn.close()


@pytest.fixture
def backlog(tmp_path):
    db = _db(tmp_path / "memory.db")
    old = _fact(db, "The team standup is at 9 every morning", "2026-03-01T00:00:00Z")
    new = _fact(db, "The team standup moved to 10 in the morning", "2026-11-14T00:00:00Z")
    _propose(db, "case-1", old, new)
    return db


def _status(db, case_id="case-1"):
    rows = db.execute("SELECT status FROM correction_cases WHERE case_id = ?", (case_id,))
    return rows[0]["status"] if rows else None


def _calibrated(db, verdict):
    _calibrate(db, verdict)
    db.execute("UPDATE judge_receipts SET population = ? WHERE receipt_id LIKE 'seed-%'",
               (POPULATION,))


def _run(db, verdict, confidence=0.95, **flags):
    tier = Fixed(verdict, confidence)
    counts = BacklogPass(db, FeatureFlags(file_values=off_except(flags), environ={}),
                         JudgeLadder([tier])).run()
    return counts, tier


def test_off_asks_nothing(backlog) -> None:
    counts, tier = _run(backlog, "both_true")
    assert counts == {} and tier.calls == 0


def test_shadow_receipts_under_backlog_and_changes_nothing(backlog) -> None:
    counts, _ = _run(backlog, "both_true", **{"judge.ladder": "shadow"})
    assert counts == {"laya:both_true": 1} and _status(backlog) == "proposed"
    (row,) = backlog.execute("SELECT population, subject_kind, subject_id FROM judge_receipts")
    assert tuple(row) == (POPULATION, "correction_case", "case-1")


def test_on_without_a_certificate_holds(backlog) -> None:
    counts, _ = _run(backlog, "both_true", **{"judge.ladder": "on"})
    assert counts["held_uncalibrated"] == 1 and _status(backlog) == "proposed"


def test_a_certified_closing_verdict_closes_the_case_and_it_can_come_back(backlog) -> None:
    _calibrated(backlog, "both_true")
    counts, _ = _run(backlog, "both_true", **{"judge.ladder": "on"})
    assert counts["closed"] == 1 and _status(backlog) is None
    (moved,) = backlog.execute("SELECT user_action, actor_id, closed_reason FROM "
                               "correction_cases_overtaken WHERE case_id = 'case-1'")
    assert moved["user_action"] == "judge_closed" and moved["actor_id"] == "judge_ladder"
    assert "both_true" in moved["closed_reason"]
    acted = backlog.execute("SELECT action_taken FROM judge_receipts WHERE population = ? "
                            "AND receipt_id NOT LIKE 'seed-%'", (POPULATION,))
    assert [r["action_taken"] for r in acted] == ["closed_both_true"]
    with backlog.raw_connection() as conn:
        ot.restore(conn, "case-1")
    assert _status(backlog) == "proposed"


def test_a_certificate_for_disputes_does_not_close_backlog_cases(backlog) -> None:
    _calibrate(backlog, "both_true")  # labels earned under slot_dispute only
    counts, _ = _run(backlog, "both_true", **{"judge.ladder": "on"})
    assert counts["held_uncalibrated"] == 1 and _status(backlog) == "proposed"


def test_replaces_is_always_left_for_a_person(backlog) -> None:
    _calibrated(backlog, "replaces")
    counts, _ = _run(backlog, "replaces", **{"judge.ladder": "on"})
    assert counts["left_for_review"] == 1 and "closed" not in counts
    assert _status(backlog) == "proposed"


def test_a_pair_judged_once_is_not_asked_again(backlog) -> None:
    _run(backlog, "both_true", confidence=0.4, **{"judge.ladder": "shadow"})
    counts, tier = _run(backlog, "both_true", **{"judge.ladder": "shadow"})
    assert counts == {} and tier.calls == 0


def test_a_case_reviewed_meanwhile_is_left_alone(backlog, monkeypatch) -> None:
    _calibrated(backlog, "both_true")

    def raced(*args, **kwargs):
        raise ot.OvertakeRaced("changed")

    monkeypatch.setattr(ot, "overtake", raced)
    counts, _ = _run(backlog, "both_true", **{"judge.ladder": "on"})
    assert counts["already_reviewed"] == 1 and _status(backlog) == "proposed"


def test_only_machine_proposals_are_judged(tmp_path) -> None:
    db = _db(tmp_path / "memory.db")
    a = _fact(db, "The office wifi name is Atlas", "2026-03-01T00:00:00Z")
    b = _fact(db, "The office wifi name is Atlas-5G", "2026-11-14T00:00:00Z")
    _propose(db, "user-case", a, b, kind="trusted_user")
    counts, tier = _run(db, "both_true", **{"judge.ladder": "shadow"})
    assert counts == {} and tier.calls == 0


def test_no_ledger_no_backlog(tmp_path) -> None:
    class Bare:
        def raw_connection(self):
            return sqlite3.connect(tmp_path / "bare.db")

    counts, tier = _run(Bare(), "both_true", **{"judge.ladder": "shadow"})
    assert counts == {} and tier.calls == 0


def test_a_machine_case_for_another_reason_is_not_judged(tmp_path) -> None:
    db = _db(tmp_path / "memory.db")
    a = _fact(db, "The deploy key rotates every month", "2026-03-01T00:00:00Z")
    b = _fact(db, "The deploy key rotates every quarter", "2026-11-14T00:00:00Z")
    _propose(db, "direct-case", a, b, reason="direct_content_correction")
    counts, tier = _run(db, "both_true", **{"judge.ladder": "shadow"})
    assert counts == {} and tier.calls == 0
