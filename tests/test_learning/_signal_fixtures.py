# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Shared fixture helpers for LLD-02 signal pipeline tests.

Keeps the conftest lean and lets every test module import the same
schema-setup + candidate-builder helpers.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from superlocalmemory.learning.database import LearningDatabase
from superlocalmemory.learning.signals import (
    SignalBatch,
    SignalCandidate,
)
from superlocalmemory.storage.migration_runner import apply_all


def make_db_with_migrations(tmp_path: Path) -> LearningDatabase:
    """Build a fresh learning.db + memory.db with v3.4.22 migrations applied.

    Returns a ``LearningDatabase`` pointing at the learning.db so tests can
    write and query without touching ``Path.home()``.
    """
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    db = LearningDatabase(learning_db)
    # Ensure memory.db exists (migration runner opens it too).
    sqlite3.connect(memory_db).close()
    stats = apply_all(learning_db, memory_db)
    assert not stats["failed"], f"migration failed: {stats['details']}"
    return db


def make_candidate(
    fact_id: str,
    *,
    channel_scores: dict | None = None,
    cross_encoder_score: float | None = None,
) -> SignalCandidate:
    return SignalCandidate(
        fact_id=fact_id,
        channel_scores=channel_scores or {"semantic": 0.7, "bm25": 0.3},
        cross_encoder_score=cross_encoder_score,
        result_dict={"fact": {"age_days": 1, "access_count": 2}},
    )


def make_batch(
    *,
    profile_id: str = "p1",
    query_id: str = "q-0001",
    query_text: str = "what did varun do yesterday",
    n_candidates: int = 3,
    query_context: dict | None = None,
) -> SignalBatch:
    candidates = tuple(
        make_candidate(f"fact-{i:03d}", cross_encoder_score=0.9 - i * 0.1)
        for i in range(n_candidates)
    )
    return SignalBatch(
        profile_id=profile_id,
        query_id=query_id,
        query_text=query_text,
        candidates=candidates,
        query_context=dict(query_context or {"query_type": "single_hop"}),
    )


def open_conn(db: LearningDatabase) -> sqlite3.Connection:
    """Open a WAL connection matching signal_worker's setup."""
    conn = sqlite3.connect(db._db_path, isolation_level=None, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.row_factory = sqlite3.Row
    return conn
