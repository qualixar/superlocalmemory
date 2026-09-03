"""Regression tests for host-scoped pending-outcome settlement.

Hermes emits ``on_session_end`` after every turn, so it needs an idempotent
MCP operation that settles only that host session's real pending recalls.  It
must reuse ``EngagementRewardModel`` rather than synthesising feedback.
"""

from __future__ import annotations

import json
import sqlite3
import uuid


def _db(tmp_path):
    path = tmp_path / "memory.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE pending_outcomes (
                outcome_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL,
                session_id TEXT NOT NULL, recall_query_id TEXT NOT NULL,
                fact_ids_json TEXT NOT NULL, query_text_hash TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL, expires_at_ms INTEGER NOT NULL,
                signals_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending'
            );
            CREATE TABLE action_outcomes (
                outcome_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL DEFAULT 'default',
                query TEXT NOT NULL DEFAULT '', fact_ids_json TEXT NOT NULL DEFAULT '[]',
                outcome TEXT NOT NULL DEFAULT '', context_json TEXT NOT NULL DEFAULT '{}',
                timestamp TEXT NOT NULL DEFAULT (datetime('now')), reward REAL,
                settled INTEGER NOT NULL DEFAULT 0, settled_at TEXT,
                recall_query_id TEXT
            );
            """
        )
    return path


def _pending(path, *, profile_id="default", session_id="hermes-a", signals=None):
    outcome_id = str(uuid.uuid4())
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO pending_outcomes "
            "(outcome_id, profile_id, session_id, recall_query_id, fact_ids_json, "
            "query_text_hash, created_at_ms, expires_at_ms, signals_json, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
            (outcome_id, profile_id, session_id, "query", "[\"fact\"]", "hash", 1, 2,
             json.dumps(signals if signals is not None else {"cite": True})),
        )
    return outcome_id


def _status(path, outcome_id):
    with sqlite3.connect(path) as conn:
        return conn.execute(
            "SELECT status FROM pending_outcomes WHERE outcome_id=?", (outcome_id,)
        ).fetchone()[0]


def test_settle_session_outcomes_is_scoped_idempotent_and_uses_real_reward_model(tmp_path):
    from superlocalmemory.mcp.tools_learning import settle_pending_session_outcomes

    path = _db(tmp_path)
    mine = _pending(path, session_id="hermes-a")
    other_session = _pending(path, session_id="hermes-b")
    other_profile = _pending(path, profile_id="other", session_id="hermes-a")

    first = settle_pending_session_outcomes(path, profile_id="default", session_id="hermes-a")

    assert first == {"selected": 1, "settled": 1}
    assert _status(path, mine) == "settled"
    assert _status(path, other_session) == "pending"
    assert _status(path, other_profile) == "pending"
    with sqlite3.connect(path) as conn:
        assert conn.execute(
            "SELECT reward FROM action_outcomes WHERE outcome_id=?", (mine,)
        ).fetchone()[0] == 0.9

    assert settle_pending_session_outcomes(
        path, profile_id="default", session_id="hermes-a"
    ) == {"selected": 0, "settled": 0}


def test_settle_session_outcomes_rejects_missing_host_session_id(tmp_path):
    from superlocalmemory.mcp.tools_learning import settle_pending_session_outcomes

    path = _db(tmp_path)
    _pending(path)

    assert settle_pending_session_outcomes(path, profile_id="default", session_id=" ") == {
        "selected": 0,
        "settled": 0,
    }


def test_per_turn_settlement_preserves_evidence_free_recalls_until_finalization(tmp_path):
    from superlocalmemory.mcp.tools_learning import settle_pending_session_outcomes

    path = _db(tmp_path)
    evidenced = _pending(path, session_id="hermes-a", signals={"cite": True})
    later_feedback = _pending(path, session_id="hermes-a", signals={})

    assert settle_pending_session_outcomes(
        path, profile_id="default", session_id="hermes-a", evidence_only=True,
    ) == {"selected": 1, "settled": 1}
    assert _status(path, evidenced) == "settled"
    assert _status(path, later_feedback) == "pending"
    assert settle_pending_session_outcomes(
        path, profile_id="default", session_id="hermes-a", evidence_only=False,
    ) == {"selected": 1, "settled": 1}
    assert _status(path, later_feedback) == "settled"
