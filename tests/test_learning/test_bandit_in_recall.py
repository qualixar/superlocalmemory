# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-03 §7.5

"""Integration tests for the bandit + ensemble hook inside
``core/recall_pipeline.apply_v2_bandit_ensemble``.

Uses real LLD-07 M005 schema in a temp DB + synthetic ``RecallResponse``
objects. No LightGBM booster required (we drive the cold-start path).
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from superlocalmemory.learning.arm_catalog import ARM_CATALOG
from superlocalmemory.retrieval.engine import apply_channel_weights
from superlocalmemory.storage.migration_runner import apply_all
from superlocalmemory.storage.models import (
    AtomicFact, Mode, RecallResponse, RetrievalResult,
)


def _mk_result(fact_id: str, score: float,
               channel_scores: dict | None = None) -> RetrievalResult:
    fact = AtomicFact(
        fact_id=fact_id,
        content="fact content",
        confidence=0.8,
    )
    return RetrievalResult(
        fact=fact,
        score=score,
        channel_scores=channel_scores or {"semantic": 0.5, "bm25": 0.5},
        confidence=0.8,
        evidence_chain=[],
        trust_score=0.5,
    )


def _mk_response(n: int = 3) -> RecallResponse:
    return RecallResponse(
        query="q",
        mode=Mode.A,
        results=[_mk_result(f"f{i}", 1.0 - i * 0.1) for i in range(n)],
        query_type="single_hop",
        channel_weights={},
        total_candidates=n,
        retrieval_time_ms=5.0,
    )


@pytest.fixture()
def bandit_db(tmp_path: Path) -> Path:
    learning = tmp_path / "learning.db"
    memory = tmp_path / "memory.db"
    stats = apply_all(learning, memory)
    assert "M005_bandit_tables" in stats["applied"]
    return learning


# ---------------------------------------------------------------------------
# apply_channel_weights — retrieval/engine.py export
# ---------------------------------------------------------------------------


def test_apply_channel_weights_returns_new_instances():
    """Must not mutate input; returns a fresh list of RetrievalResult."""
    original = _mk_response(3).results
    weights = {"semantic": 2.0, "bm25": 1.0, "entity_graph": 1.0,
               "temporal": 1.0, "cross_encoder_bias": 1.0}
    out = apply_channel_weights(original, weights)
    assert out is not original
    # Channel scores doubled on semantic.
    assert out[0].channel_scores["semantic"] == pytest.approx(1.0)
    # Originals untouched.
    assert original[0].channel_scores["semantic"] == 0.5


def test_apply_channel_weights_empty_input_returns_empty():
    assert apply_channel_weights([], {"semantic": 1.0}) == []


def test_apply_channel_weights_none_weights_identity():
    original = _mk_response(2).results
    out = apply_channel_weights(original, None)
    # Content identity — new list, equal contents.
    assert len(out) == len(original)
    for a, b in zip(out, original):
        assert a.fact.fact_id == b.fact.fact_id


def test_apply_channel_weights_cross_encoder_bias_applied():
    original = _mk_response(1).results
    weights = {"semantic": 1.0, "bm25": 1.0, "entity_graph": 1.0,
               "temporal": 1.0, "cross_encoder_bias": 2.0}
    out = apply_channel_weights(original, weights)
    # base = sum of channel weighted = 1.0; × ce_bias 2.0 = 2.0
    assert out[0].score == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# apply_v2_bandit_ensemble — integration
# ---------------------------------------------------------------------------


def test_recall_creates_bandit_play_row(bandit_db: Path, monkeypatch):
    """A full recall path creates one bandit_plays row and returns response."""
    from superlocalmemory.core.recall_pipeline import apply_v2_bandit_ensemble

    response = _mk_response(5)
    out = apply_v2_bandit_ensemble(
        response, query="hello", profile_id="px",
        query_id="qid-1", learning_db_path=bandit_db,
    )
    # Never drops candidates.
    assert len(out.results) == 5
    # One play row inserted.
    conn = sqlite3.connect(str(bandit_db))
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM bandit_plays WHERE profile_id = ?",
            ("px",),
        ).fetchone()[0]
    finally:
        conn.close()
    assert n == 1


def test_recall_bandit_disabled_env_skips(bandit_db: Path, monkeypatch):
    """SLM_BANDIT_DISABLED=1 → identity; no play row created."""
    from superlocalmemory.core.recall_pipeline import apply_v2_bandit_ensemble

    monkeypatch.setenv("SLM_BANDIT_DISABLED", "1")
    response = _mk_response(3)
    out = apply_v2_bandit_ensemble(
        response, query="q", profile_id="off",
        query_id="qid-off", learning_db_path=bandit_db,
    )
    assert out is response
    conn = sqlite3.connect(str(bandit_db))
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM bandit_plays"
        ).fetchone()[0]
    finally:
        conn.close()
    assert n == 0


def test_recall_empty_response_passthrough(bandit_db: Path):
    """Empty results → identity, no play row."""
    from superlocalmemory.core.recall_pipeline import apply_v2_bandit_ensemble

    empty = RecallResponse(
        query="q", mode=Mode.A, results=[],
        query_type="single_hop", channel_weights={},
        total_candidates=0, retrieval_time_ms=0.0,
    )
    out = apply_v2_bandit_ensemble(
        empty, query="q", profile_id="p",
        query_id="qid-empty", learning_db_path=bandit_db,
    )
    assert out is empty


def test_recall_missing_db_returns_response_unchanged(tmp_path: Path):
    """learning.db absent → identity, no crash."""
    from superlocalmemory.core.recall_pipeline import apply_v2_bandit_ensemble

    response = _mk_response(2)
    out = apply_v2_bandit_ensemble(
        response, query="q", profile_id="p",
        query_id="q-none",
        learning_db_path=tmp_path / "nope.db",
    )
    assert out is response


def test_recall_does_not_raise_on_missing_tables(tmp_path: Path):
    """DB exists but bandit tables missing → response unchanged, no exception."""
    from superlocalmemory.core.recall_pipeline import apply_v2_bandit_ensemble

    db = tmp_path / "empty.db"
    sqlite3.connect(str(db)).close()
    response = _mk_response(3)
    out = apply_v2_bandit_ensemble(
        response, query="q", profile_id="p",
        query_id="q-e",
        learning_db_path=db,
    )
    # No crash; results preserved (may be reweighted with channel=fallback).
    assert len(out.results) == 3
