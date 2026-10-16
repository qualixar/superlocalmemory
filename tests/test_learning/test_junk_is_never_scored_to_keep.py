# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Every writer of a retention score holds junk at the keep threshold.

Four writers can put a memory above 0.8: the decay pass, a genuine access
(``advance_clock``), a used-fact outcome (``record_use_clock``) and the
scheduler's on-access refresh. None read the content, which is how
"2024-01-15 File was modified" came to be scored as worth keeping.
"""

from __future__ import annotations

from superlocalmemory.core.config import ForgettingConfig
from superlocalmemory.core.keep_worthiness import KEEP_THRESHOLD
from superlocalmemory.learning.forgetting_scheduler import ForgettingScheduler
from superlocalmemory.math.ebbinghaus import EbbinghausCurve
from superlocalmemory.storage.access_log import AccessLog
from tests.test_learning.test_forgetting_scheduler import db_with_facts as _fixture

PID = "test_profile"
JUNK = "2024-01-15 File was modified"
REAL = "Varun prefers the dark theme on the dashboard"

db_with_facts = _fixture


def _scheduler(db) -> ForgettingScheduler:
    cfg = ForgettingConfig()
    return ForgettingScheduler(db, EbbinghausCurve(cfg), cfg)


def _label(db) -> None:
    """fact_000 is junk, fact_001 is real; both were used minutes ago."""
    db.execute("UPDATE atomic_facts SET content=? WHERE fact_id='fact_000'", (JUNK,))
    db.execute("UPDATE atomic_facts SET content=? WHERE fact_id='fact_001'", (REAL,))


def _state(db, fact_id: str) -> tuple[float, str, str]:
    row = dict(db.execute(
        "SELECT fr.retention_score AS s, fr.lifecycle_zone AS z, af.lifecycle AS l "
        "FROM fact_retention fr JOIN atomic_facts af ON af.fact_id = fr.fact_id "
        "WHERE fr.fact_id = ?", (fact_id,))[0])
    return float(row["s"]), row["z"], row["l"]


def _fact(content: str | None) -> dict:
    fact = {"fact_id": "f", "access_count": 3, "importance": 0.9,
            "confirmation_count": 2, "emotional_salience": 0.0,
            "last_accessed_at": "2999-01-01T00:00:00+00:00", "trust_score": 1.0}
    if content is not None:
        fact["content"] = content
    return fact


class TestCurve:
    def test_batch_caps_junk_at_the_threshold(self) -> None:
        curve = EbbinghausCurve(ForgettingConfig())
        junk, real, unknown = curve.batch_compute_retention(
            [_fact(JUNK), _fact(REAL), _fact(None)])
        assert (junk["retention"], junk["zone"]) == (KEEP_THRESHOLD, "warm")
        assert (real["retention"], real["zone"]) == (1.0, "active")
        assert (unknown["retention"], unknown["zone"]) == (1.0, "active")


class TestWriters:
    def test_decay_pass_never_scores_junk_to_keep(self, db_with_facts) -> None:
        _label(db_with_facts)
        _scheduler(db_with_facts).run_decay_cycle(PID, force=True)
        score, zone, mirror = _state(db_with_facts, "fact_000")
        assert score <= KEEP_THRESHOLD and zone != "active" and mirror == zone
        assert _state(db_with_facts, "fact_001")[0] > KEEP_THRESHOLD

    def test_access_refreshes_real_to_active_and_junk_to_warm(self, db_with_facts) -> None:
        _label(db_with_facts)
        _scheduler(db_with_facts).run_decay_cycle(PID, force=True)
        log = AccessLog(db_with_facts)
        assert log.store_access("fact_000", PID, "search")
        assert log.store_access("fact_001", PID, "search")
        assert _state(db_with_facts, "fact_000") == (KEEP_THRESHOLD, "warm", "warm")
        assert _state(db_with_facts, "fact_001") == (1.0, "active", "active")

    def test_used_fact_outcome_holds_junk_at_warm(self, db_with_facts) -> None:
        import sqlite3

        from superlocalmemory.storage.retention_access import record_use_clock

        _label(db_with_facts)
        _scheduler(db_with_facts).run_decay_cycle(PID, force=True)
        conn = sqlite3.connect(str(db_with_facts.db_path))
        try:
            record_use_clock(conn, PID, ["fact_000", "fact_001"], "oc_1",
                             "2026-10-16T10:00:00+00:00")
            conn.commit()
        finally:
            conn.close()
        assert _state(db_with_facts, "fact_000") == (KEEP_THRESHOLD, "warm", "warm")
        assert _state(db_with_facts, "fact_001") == (1.0, "active", "active")

    def test_on_access_event_holds_junk_at_warm(self, db_with_facts) -> None:
        _label(db_with_facts)
        sched = _scheduler(db_with_facts)
        sched.run_decay_cycle(PID, force=True)
        sched.on_access_event("fact_000", PID)
        sched.on_access_event("fact_001", PID)
        score, zone, mirror = _state(db_with_facts, "fact_000")
        assert score <= KEEP_THRESHOLD and zone == mirror == "warm"
        assert _state(db_with_facts, "fact_001")[1] == "active"
