# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Issue #102 — explicit feedback must reach the store the system reads.

Before 3.8.11, MCP ``report_feedback`` wrote only to ``feedback_records`` in
memory.db. The single production reader of that table is AdaptiveLearner's own
``get_feedback_count`` (which just returns a number) and its ``train()``,
which nothing in the running system calls. So feedback returned success and an
incrementing counter while the phase gate, the pattern miner and the dashboard
all saw nothing.

These tests pin the contract: an explicit feedback write must land in
learning.db and be visible to every downstream consumer.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def learning_root(tmp_path: Path) -> Path:
    return tmp_path


def test_explicit_feedback_lands_in_learning_db(learning_root: Path) -> None:
    import superlocalmemory.mcp.tools_active as ta

    with patch.object(ta, "state_path", lambda n: learning_root / n):
        assert ta._record_canonical_feedback(
            profile_id="p1", fact_id="f-1", feedback="relevant", query="q",
        ) is True

    db = learning_root / "learning.db"
    assert db.exists(), "explicit feedback must create/populate learning.db"
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT profile_id, fact_id, signal_type, signal_value, channel "
            "FROM learning_feedback"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("p1", "f-1", "user_positive", 1.0, "explicit")]


def test_feedback_is_visible_to_the_phase_gate(learning_root: Path) -> None:
    """The gate that unlocks adaptive ranking must count this write.

    ``apply_adaptive_ranking`` skips reranking entirely below 50 rows, reading
    ``learning_feedback`` through ``_ReadOnlyLearningView.count_feedback``. If
    explicit feedback does not land there, no amount of user feedback ever
    unlocks Phase 2/3 — the exact symptom in issue #102.
    """
    import superlocalmemory.mcp.tools_active as ta
    from superlocalmemory.core.recall_pipeline import _ReadOnlyLearningView

    with patch.object(ta, "state_path", lambda n: learning_root / n):
        for i in range(3):
            ta._record_canonical_feedback(
                profile_id="p1", fact_id=f"f-{i}", feedback="relevant",
            )

    gate = _ReadOnlyLearningView(learning_root / "learning.db")
    assert gate.count_feedback("p1") == 3


def test_feedback_query_text_is_never_stored_in_plaintext(
    learning_root: Path,
) -> None:
    import superlocalmemory.mcp.tools_active as ta

    secret = "my social security number is 123-45-6789"
    with patch.object(ta, "state_path", lambda n: learning_root / n):
        ta._record_canonical_feedback(
            profile_id="p1", fact_id="f-1", feedback="relevant", query=secret,
        )

    raw = (learning_root / "learning.db").read_bytes()
    assert secret.encode() not in raw, "query text must be hashed, never stored"


def test_feedback_types_map_to_distinct_signals(learning_root: Path) -> None:
    import superlocalmemory.mcp.tools_active as ta

    with patch.object(ta, "state_path", lambda n: learning_root / n):
        for fid, fb in (("a", "relevant"), ("b", "irrelevant"), ("c", "partial")):
            ta._record_canonical_feedback(
                profile_id="p1", fact_id=fid, feedback=fb,
            )

    conn = sqlite3.connect(str(learning_root / "learning.db"))
    try:
        got = dict(conn.execute(
            "SELECT fact_id, signal_type FROM learning_feedback"
        ).fetchall())
    finally:
        conn.close()
    assert got == {
        "a": "user_positive",
        "b": "user_negative",
        "c": "user_correction",
    }


def test_canonical_write_failure_is_reported_not_swallowed(
    learning_root: Path,
) -> None:
    """A failed learning write must return False so callers can tell the truth.

    The whole class of bug in #102/#103 is success being reported for work that
    did not happen. Best-effort must still be honest.
    """
    import superlocalmemory.mcp.tools_active as ta

    with patch.object(ta, "state_path", lambda n: learning_root / n), \
         patch(
             "superlocalmemory.learning.feedback.FeedbackCollector",
             side_effect=OSError("disk gone"),
         ):
        assert ta._record_canonical_feedback(
            profile_id="p1", fact_id="f-1", feedback="relevant",
        ) is False


def test_pattern_miner_channel_query_works_on_a_fresh_database(
    learning_root: Path,
) -> None:
    """pattern_miner groups on learning_feedback.channel.

    That column was read by the miner but defined by no schema, so every fresh
    database raised "no such column: channel" — swallowed at debug level, which
    silently killed channel mining AND the co-retrieval mining that shared its
    try block.
    """
    from superlocalmemory.learning.feedback import FeedbackCollector

    db = learning_root / "learning.db"
    collector = FeedbackCollector(db)
    collector.record_explicit(
        profile_id="p1", fact_id="f-1", signal_type="user_positive",
        value=1.0, channel="semantic",
    )

    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT channel, COUNT(*) AS cnt, AVG(signal_value) AS avg_signal "
            "FROM learning_feedback WHERE profile_id = ? "
            "GROUP BY channel ORDER BY cnt DESC",
            ("p1",),
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("semantic", 1, 1.0)]


def test_legacy_database_without_channel_is_self_healed(
    learning_root: Path,
) -> None:
    """A pre-3.8.11 learning.db must gain the column, keeping its rows."""
    from superlocalmemory.learning.feedback import FeedbackCollector

    db = learning_root / "learning.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE learning_feedback ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id TEXT NOT NULL,"
        " fact_id TEXT NOT NULL, signal_type TEXT NOT NULL,"
        " signal_value REAL NOT NULL, query_hash TEXT,"
        " created_at TEXT NOT NULL, metadata TEXT)"
    )
    conn.execute(
        "INSERT INTO learning_feedback "
        "(profile_id, fact_id, signal_type, signal_value, created_at) "
        "VALUES ('p1', 'old-fact', 'user_positive', 1.0, '2026-01-01')"
    )
    conn.commit()
    conn.close()

    FeedbackCollector(db)  # _ensure_schema must migrate in place

    conn = sqlite3.connect(str(db))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(learning_feedback)")}
        preserved = conn.execute(
            "SELECT fact_id, channel FROM learning_feedback"
        ).fetchall()
    finally:
        conn.close()

    assert "channel" in cols
    assert preserved == [("old-fact", "unknown")], "existing rows must survive"
