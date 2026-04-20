# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21 — Stage 9 W1

"""Stage 9 W1 regressions — LLD-10 producer wiring (C1).

Stage 9 architect persona flagged that ``feed_recall_settled`` and
``attach_candidate`` had ZERO production callers — the LLD-10 A/B loop
consumer was wired (SB-1 via 7df7609) but the producer side was floating.
These tests lock in that:

1. ``_persist_candidate`` now calls ``shadow_router.attach_candidate``
   after the INSERT commits, so the candidate is registered with the
   router and a ShadowTest starts collecting paired observations.
2. ``EngagementRewardModel.finalize_outcome`` now calls
   ``recall_pipeline.feed_recall_settled`` after the reward row lands,
   so each settled recall feeds its reward (NDCG@10 proxy) into the
   active ShadowTest or post-promotion ModelRollback watch.

Both calls are fail-soft — they use ``logger.debug`` on exception so
that the A/B loop never breaks the core write path.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from unittest import mock

import pytest

from superlocalmemory.core import shadow_router as sr
from superlocalmemory.learning.reward import EngagementRewardModel


@pytest.fixture
def reset_router() -> None:
    """Clear singleton between tests."""
    sr.reset_for_testing()
    yield
    sr.reset_for_testing()


def _seed_pending_row(
    memory_db: Path, *, profile_id: str = "default",
    session_id: str = "s1", recall_query_id: str = "q1",
    fact_ids: list[str] | None = None,
    signals: str = '{"cited": true}',
) -> str:
    """Insert one pending_outcomes row; return the outcome_id."""
    outcome_id = uuid.uuid4().hex
    facts = fact_ids if fact_ids is not None else ["fact-1"]
    import json as _j
    fids_json = _j.dumps(facts)
    conn = sqlite3.connect(memory_db)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS pending_outcomes (
                outcome_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                recall_query_id TEXT,
                fact_ids_json TEXT NOT NULL,
                signals_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at_ms INTEGER NOT NULL DEFAULT 0,
                expires_at_ms INTEGER
            );
            CREATE TABLE IF NOT EXISTS action_outcomes (
                outcome_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL DEFAULT '',
                query TEXT NOT NULL DEFAULT '',
                fact_ids_json TEXT NOT NULL DEFAULT '[]',
                outcome TEXT NOT NULL DEFAULT 'pending',
                context_json TEXT NOT NULL DEFAULT '{}',
                timestamp TEXT NOT NULL DEFAULT '',
                reward REAL,
                settled INTEGER NOT NULL DEFAULT 0,
                settled_at TEXT,
                recall_query_id TEXT
            );
        """)
        conn.execute(
            "INSERT INTO pending_outcomes "
            "(outcome_id, profile_id, session_id, recall_query_id, "
            " fact_ids_json, signals_json, status, created_at_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending', 0)",
            (outcome_id, profile_id, session_id, recall_query_id,
             fids_json, signals),
        )
        conn.commit()
    finally:
        conn.close()
    return outcome_id


def test_finalize_outcome_feeds_shadow_router(
    tmp_path: Path, reset_router: None,
) -> None:
    """C1: finalize_outcome must call feed_recall_settled with reward
    as the NDCG@10 proxy. Verified by mocking feed_recall_settled and
    checking it was invoked with the right kwargs."""
    memory_db = tmp_path / "memory.db"
    outcome_id = _seed_pending_row(memory_db, recall_query_id="q-123")

    model = EngagementRewardModel(memory_db)
    with mock.patch(
        "superlocalmemory.core.recall_pipeline.feed_recall_settled"
    ) as mock_feed:
        reward = model.finalize_outcome(outcome_id=outcome_id)

    model.close()

    assert 0.0 <= reward <= 1.0
    mock_feed.assert_called_once()
    call_kwargs = mock_feed.call_args.kwargs
    assert call_kwargs["profile_id"] == "default"
    assert call_kwargs["query_id"] == "q-123"
    assert call_kwargs["ndcg_at_10"] == pytest.approx(reward)
    # memory_db / learning_db paths should be side-by-side.
    assert call_kwargs["memory_db"].endswith("memory.db")
    assert call_kwargs["learning_db"].endswith("learning.db")


def test_finalize_outcome_survives_router_exception(
    tmp_path: Path, reset_router: None,
) -> None:
    """C1 fail-soft: a raised exception inside feed_recall_settled must
    NOT propagate — the reward return contract is more important than
    the A/B loop signal."""
    memory_db = tmp_path / "memory.db"
    outcome_id = _seed_pending_row(memory_db)

    model = EngagementRewardModel(memory_db)
    with mock.patch(
        "superlocalmemory.core.recall_pipeline.feed_recall_settled",
        side_effect=RuntimeError("simulated router crash"),
    ):
        reward = model.finalize_outcome(outcome_id=outcome_id)

    model.close()
    assert 0.0 <= reward <= 1.0


def test_finalize_outcome_skips_feed_when_no_query_id(
    tmp_path: Path, reset_router: None,
) -> None:
    """If the pending row has no recall_query_id (legacy / edge), the
    router feed is skipped — nothing to correlate against."""
    memory_db = tmp_path / "memory.db"
    outcome_id = _seed_pending_row(memory_db, recall_query_id="")

    model = EngagementRewardModel(memory_db)
    with mock.patch(
        "superlocalmemory.core.recall_pipeline.feed_recall_settled"
    ) as mock_feed:
        model.finalize_outcome(outcome_id=outcome_id)
    model.close()

    mock_feed.assert_not_called()


def test_persist_candidate_attaches_to_router(
    tmp_path: Path, reset_router: None,
) -> None:
    """C1: _persist_candidate must invoke ShadowRouter.attach_candidate
    after the INSERT commits, so a ShadowTest starts collecting paired
    observations for this candidate."""
    from superlocalmemory.learning import ranker_retrain_online as rro

    learning_db = tmp_path / "learning.db"
    # Minimal learning_model_state schema for the insert to land.
    conn = sqlite3.connect(learning_db)
    try:
        conn.executescript("""
            CREATE TABLE learning_model_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL,
                model_version TEXT NOT NULL DEFAULT '3.4.21',
                state_bytes BLOB NOT NULL,
                bytes_sha256 TEXT NOT NULL DEFAULT '',
                trained_on_count INTEGER NOT NULL DEFAULT 0,
                feature_names TEXT NOT NULL DEFAULT '[]',
                metrics_json TEXT NOT NULL DEFAULT '{}',
                is_active INTEGER NOT NULL DEFAULT 0,
                is_candidate INTEGER NOT NULL DEFAULT 0,
                shadow_results_json TEXT NOT NULL DEFAULT '{}',
                trained_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX idx_model_candidate_one
                ON learning_model_state(profile_id)
                WHERE is_candidate = 1;
        """)
        conn.commit()
    finally:
        conn.close()

    with mock.patch.object(
        sr.ShadowRouter, "attach_candidate"
    ) as mock_attach:
        cand_id = rro._persist_candidate(
            str(learning_db),
            profile_id="default",
            state_bytes=b"model-blob" * 32,
            feature_names=["f1", "f2"],
            trained_on_count=100,
            metrics={"val_ndcg": 0.72},
            shadow_results=None,
        )

    assert cand_id > 0
    mock_attach.assert_called_once_with(cand_id)
