# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Issue #106 — one canonical feedback store, one signal count.

Issue #102's fix routed explicit feedback into ``learning_feedback``. That is
the pre-v3.4.22 table: ``legacy_migration`` copies it FORWARD into
``learning_signals``, and the dashboard reports it as ``legacy_feedback_rows``
beside a pending-migration card. Meanwhile every phase-resolving surface reads
``learning_signals``. So the fix wrote to a table no phase counter consumes.

Two user-visible failures followed, and they share one cause — the write store
and the read stores were different tables:

* the dashboard showed ``Feedback signals: 0`` after 39 rows were written; and
* ``report_feedback`` returned ``success`` beside a plausible, incrementing
  ``total_signals`` even when the durable write never happened, because the
  reported count fell back to ``feedback_records`` in memory.db — a third
  table, which increments on every call and which nothing reads.

These tests pin the contract: one explicit-feedback event becomes one
canonical ``learning_signals`` row paired with its ``learning_features`` row,
and every surface that reports or gates on a signal count reports the same
number from that one table.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.learning.feedback import FeedbackCollector


@pytest.fixture()
def learning_db(tmp_path: Path) -> Path:
    return tmp_path / "learning.db"


def _count(db: Path, table: str, profile_id: str = "p1") -> int:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE profile_id = ?",  # noqa: S608
            (profile_id,),
        ).fetchone()[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# The canonical write
# ---------------------------------------------------------------------------


def test_explicit_feedback_writes_the_canonical_signal_pair(
    learning_db: Path,
) -> None:
    """One feedback event must become one signal row + one feature row.

    ``learning_signals`` is what the recall phase gate, the dashboard Living
    Brain panel and the ranker-phase card all count. A feedback event that
    does not appear there influences nothing, however many legacy rows it
    writes.
    """
    collector = FeedbackCollector(learning_db)

    write = collector.record_explicit_event(
        profile_id="p1", fact_id="f-1", signal_type="user_positive",
        value=1.0, query="q",
    )

    assert write.canonical is True
    assert write.signal_row_id is not None
    assert _count(learning_db, "learning_signals") == 1
    assert _count(learning_db, "learning_features") == 1


def test_canonical_feature_row_is_paired_and_flagged_synthetic(
    learning_db: Path,
) -> None:
    """The feature row must point at its signal and stay out of training.

    ``learning_features`` is 1:1 with ``learning_signals`` — an unpaired
    signal row breaks that invariant for every consumer that joins them.
    Feedback arrives out of band with no ranked candidate list, so there is no
    real feature vector; the row must carry ``is_synthetic=1`` so the LightGBM
    retrainer (which selects ``WHERE is_synthetic=0``) never trains on a
    vector of zeros.
    """
    collector = FeedbackCollector(learning_db)
    write = collector.record_explicit_event(
        profile_id="p1", fact_id="f-1", signal_type="user_negative",
        value=0.0, query="q",
    )

    conn = sqlite3.connect(str(learning_db))
    conn.row_factory = sqlite3.Row
    try:
        feature = conn.execute(
            "SELECT signal_id, is_synthetic, label, query_id "
            "FROM learning_features",
        ).fetchone()
        signal = conn.execute(
            "SELECT id, query_id, signal_type FROM learning_signals",
        ).fetchone()
        orphans = conn.execute(
            "SELECT COUNT(*) FROM learning_features f "
            "LEFT JOIN learning_signals s ON s.id = f.signal_id "
            "WHERE s.id IS NULL",
        ).fetchone()[0]
    finally:
        conn.close()

    assert orphans == 0
    assert feature["signal_id"] == signal["id"] == write.signal_row_id
    assert feature["is_synthetic"] == 1
    assert feature["query_id"] == signal["query_id"]
    assert feature["label"] == 0.0


def test_dashboard_feedback_reaches_the_canonical_store(
    learning_db: Path,
) -> None:
    """A thumbs-up from the dashboard must move the dashboard's own counter.

    The Living Brain panel counts ``learning_signals``. Writing the dashboard's
    own button presses only to ``learning_feedback`` is why the reporter saw
    39 rows on disk and ``Feedback signals: 0`` on screen.
    """
    collector = FeedbackCollector(learning_db)

    collector.record_dashboard_feedback(
        memory_id="f-1", query="q", feedback_type="thumbs_up",
        profile_id="p1",
    )

    assert _count(learning_db, "learning_signals") == 1
    assert collector.get_signal_count("p1") == 1


def test_feedback_event_is_atomic_across_both_stores(
    learning_db: Path,
) -> None:
    """A failed canonical insert must not leave a stranded legacy row.

    A partial write is how the two stores drift apart permanently: the legacy
    table advances, the phase counter does not, and nothing ever reconciles
    them.
    """
    collector = FeedbackCollector(learning_db)
    collector.record_explicit_event(
        profile_id="p1", fact_id="f-0", signal_type="user_positive", value=1.0,
    )
    before_legacy = _count(learning_db, "learning_feedback")
    before_signals = _count(learning_db, "learning_signals")

    with patch.object(
        FeedbackCollector, "_insert_canonical_pair",
        side_effect=sqlite3.OperationalError("canonical insert exploded"),
    ), pytest.raises(sqlite3.OperationalError):
        collector.record_explicit_event(
            profile_id="p1", fact_id="f-1", signal_type="user_positive",
            value=1.0,
        )

    assert _count(learning_db, "learning_feedback") == before_legacy
    assert _count(learning_db, "learning_signals") == before_signals


# ---------------------------------------------------------------------------
# One count, every surface
# ---------------------------------------------------------------------------


def test_every_surface_reports_the_same_signal_count(
    learning_db: Path,
) -> None:
    """MCP, the recall gate and the dashboard must agree on one number.

    Before the fix the MCP surface counted ``learning_feedback`` while the
    dashboard counted ``learning_signals``, so the phase a user was shown and
    the phase applied to their results were computed from different tables and
    could disagree without limit.
    """
    import superlocalmemory.mcp.tools_active as ta
    from superlocalmemory.core.recall_pipeline import _ReadOnlyLearningView

    collector = FeedbackCollector(learning_db)
    # Deliberately the long-standing ``record_explicit`` API, so this pins
    # behaviour rather than the presence of a new method name.
    for i in range(4):
        collector.record_explicit(
            profile_id="p1", fact_id=f"f-{i}", signal_type="user_positive",
            value=1.0, query="q",
        )

    with patch.object(ta, "state_path", lambda n: learning_db.parent / n):
        mcp_count = ta._canonical_feedback_count("p1")

    gate_count = _ReadOnlyLearningView(learning_db).count_signals("p1")
    dashboard_count = _count(learning_db, "learning_signals")

    assert mcp_count == gate_count == dashboard_count == 4


def test_recall_phase_gate_unlocks_on_canonical_signals(
    tmp_path: Path,
) -> None:
    """The gate must read ``learning_signals``, like every other surface.

    A profile sitting on 60 canonical signals is Phase 2 everywhere the user
    can see. Reading ``learning_feedback`` here meant recall stayed on the
    Phase 1 path while the dashboard reported Phase 2 — the same table split,
    seen from the ranking side.
    """
    from superlocalmemory.core.recall_pipeline import apply_adaptive_ranking
    from superlocalmemory.learning.database import LearningDatabase
    from superlocalmemory.learning.ranker import PHASE_2_THRESHOLD

    db_path = tmp_path / "learning.db"
    learning = LearningDatabase(db_path)
    for i in range(PHASE_2_THRESHOLD + 10):
        learning.store_signal(
            profile_id="p1", query="q", fact_id=f"f-{i}",
            signal_type="candidate", value=1.0,
        )
    # The legacy table stays empty: only the canonical count may unlock.
    assert _count(db_path, "learning_signals") >= PHASE_2_THRESHOLD

    response = _fake_recall_response()
    config = MagicMock()

    with patch(
        "superlocalmemory.infra.data_root.state_path",
        lambda *parts: db_path,
    ):
        result = apply_adaptive_ranking(response, "q", "p1", config=config)

    assert result is not response, (
        "reranking must run once the canonical signal count clears Phase 2"
    )


def test_recall_phase_gate_stays_cold_below_the_threshold(
    tmp_path: Path,
) -> None:
    """Below the threshold the response must be returned untouched."""
    from superlocalmemory.core.recall_pipeline import apply_adaptive_ranking
    from superlocalmemory.learning.database import LearningDatabase

    db_path = tmp_path / "learning.db"
    learning = LearningDatabase(db_path)
    for i in range(3):
        learning.store_signal(
            profile_id="p1", query="q", fact_id=f"f-{i}",
            signal_type="candidate", value=1.0,
        )

    response = _fake_recall_response()
    with patch(
        "superlocalmemory.infra.data_root.state_path",
        lambda *parts: db_path,
    ):
        result = apply_adaptive_ranking(response, "q", "p1", config=MagicMock())

    assert result is response


def _fake_recall_response():
    """Build a minimal RecallResponse the ranker can walk."""
    from superlocalmemory.storage.models import (
        AtomicFact,
        Mode,
        RecallResponse,
        RetrievalResult,
    )

    fact = AtomicFact(fact_id="f-1", profile_id="p1", content="hello")
    result = RetrievalResult(fact=fact, score=0.5)
    return RecallResponse(
        query="q", mode=Mode.B, results=[result], query_type="factual",
    )


# ---------------------------------------------------------------------------
# Honest reporting — issue #106 Issue A
# ---------------------------------------------------------------------------


class _MockServer:
    """Minimal mock that captures @server.tool() decorated functions."""

    def __init__(self) -> None:
        self._tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self._tools[fn.__name__] = fn
            return fn
        return decorator


def _report_feedback_tool(engine):
    from superlocalmemory.mcp.tools_active import register_active_tools

    srv = _MockServer()
    register_active_tools(srv, lambda: engine)
    return srv._tools["report_feedback"]


def _engine_mock(feedback_records_count: int = 4096):
    engine = MagicMock()
    engine.profile_id = "p1"
    engine._adaptive_learner.record_feedback.return_value = MagicMock(
        feedback_id="fb-1",
    )
    # A deliberately distinctive number: if it ever reaches the caller, the
    # response is quoting ``feedback_records`` in memory.db, which no phase
    # counter reads.
    engine._adaptive_learner.get_feedback_count.return_value = (
        feedback_records_count
    )
    return engine


def test_report_feedback_reports_the_canonical_count(
    learning_db: Path,
) -> None:
    """``total_signals`` must be the number the gate and dashboard use."""
    import superlocalmemory.mcp.tools_active as ta

    collector = FeedbackCollector(learning_db)
    for i in range(3):
        collector.record_explicit_event(
            profile_id="p1", fact_id=f"seed-{i}",
            signal_type="user_positive", value=1.0,
        )

    engine = _engine_mock()
    tool = _report_feedback_tool(engine)
    with patch.object(ta, "state_path", lambda n: learning_db.parent / n):
        out = asyncio.run(tool(fact_id="f-9", feedback="relevant"))

    assert out["success"] is True
    assert out["durable"] is True
    # 3 seeded + this call, counted in learning_signals — NOT 4096.
    assert out["total_signals"] == 4
    assert _count(learning_db, "learning_signals") == 4


def test_report_feedback_never_fabricates_a_count_from_another_store(
    learning_db: Path,
) -> None:
    """A failed canonical write must not return success beside a fake count.

    This is the reported symptom exactly: every call returned
    ``"success": true`` with ``total_signals`` climbing, while the learning
    database was never touched. The count came from ``feedback_records``,
    which increments on every call regardless of whether the durable write
    happened, so a total failure was indistinguishable from success.
    """
    import superlocalmemory.mcp.tools_active as ta

    engine = _engine_mock(feedback_records_count=4096)
    tool = _report_feedback_tool(engine)

    with patch.object(ta, "state_path", lambda n: learning_db.parent / n), \
         patch(
             "superlocalmemory.learning.feedback.FeedbackCollector",
             side_effect=OSError("learning.db is gone"),
         ):
        out = asyncio.run(tool(fact_id="f-1", feedback="relevant"))

    assert out["success"] is False, (
        "a call that wrote nothing durable must not report success"
    )
    assert out["durable"] is False
    assert out.get("total_signals") != 4096, (
        "total_signals must never be sourced from feedback_records"
    )
    assert "error" in out


def test_session_init_and_report_feedback_agree_on_the_signal_count(
    learning_db: Path,
) -> None:
    """Two tools, one profile, one session — one number.

    ``session_init`` used to read ``feedback_records`` while
    ``report_feedback`` read learning.db, so a single session could be told it
    had 10 signals by one tool and thousands by the other.
    """
    import superlocalmemory.mcp.tools_active as ta
    from superlocalmemory.mcp.tools_active import register_active_tools

    collector = FeedbackCollector(learning_db)
    for i in range(6):
        collector.record_explicit_event(
            profile_id="p1", fact_id=f"seed-{i}",
            signal_type="user_positive", value=1.0,
        )

    engine = _engine_mock(feedback_records_count=4096)
    srv = _MockServer()
    register_active_tools(srv, lambda: engine)

    with patch.object(ta, "state_path", lambda n: learning_db.parent / n):
        canonical = ta._canonical_feedback_count("p1")
        feedback_out = asyncio.run(
            srv._tools["report_feedback"](fact_id="f-9", feedback="relevant"),
        )

    assert canonical == 6
    assert feedback_out["total_signals"] == 7
    # session_init resolves its learning block from the same helper, so the
    # number it publishes cannot come from feedback_records any more.
    with patch.object(ta, "state_path", lambda n: learning_db.parent / n):
        assert ta._canonical_feedback_count("p1") == 7


# ---------------------------------------------------------------------------
# The batch migration must not double-count eager writes
# ---------------------------------------------------------------------------


def test_legacy_migration_does_not_recopy_eagerly_written_feedback(
    learning_db: Path,
) -> None:
    """Both writers share one identity per event, so neither duplicates it.

    ``FeedbackCollector`` now carries each event forward at write time, and
    ``legacy_migration`` still copies the historic table in batch. Without a
    shared ``query_id`` the same event would be counted twice in the store
    that gates the ranking phase.
    """
    from superlocalmemory.learning.legacy_migration import (
        migrate_legacy_feedback,
    )
    from superlocalmemory.storage.migrations import M003_migration_log as m003

    collector = FeedbackCollector(learning_db)
    conn = sqlite3.connect(str(learning_db))
    conn.executescript(m003.DDL)
    # A row from before this release: legacy only, never carried forward.
    conn.execute(
        "INSERT INTO learning_feedback "
        "(profile_id, fact_id, signal_type, signal_value, query_hash, "
        " created_at) "
        "VALUES ('p1', 'historic', 'user_positive', 1.0, 'abc', "
        "'2026-01-01T00:00:00+00:00')",
    )
    conn.commit()
    conn.close()

    collector.record_explicit(
        profile_id="p1", fact_id="fresh", signal_type="user_positive",
        value=1.0, query="q",
    )
    # The fresh event is canonical immediately — no migration run required.
    assert _count(learning_db, "learning_signals") == 1

    stats = migrate_legacy_feedback(learning_db)

    assert stats["copied"] == 1, "only the historic row needed copying"
    assert _count(learning_db, "learning_signals") == 2, (
        "the eagerly-written event must not be copied a second time"
    )

    conn = sqlite3.connect(str(learning_db))
    try:
        duplicates = conn.execute(
            "SELECT COUNT(*) FROM (SELECT query_id FROM learning_signals "
            "GROUP BY query_id HAVING COUNT(*) > 1)",
        ).fetchone()[0]
    finally:
        conn.close()
    assert duplicates == 0
