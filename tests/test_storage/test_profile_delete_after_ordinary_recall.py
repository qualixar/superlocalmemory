# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Ordinary recall creates learning tables; they must not make a profile undeletable (ENG-30)."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.learning.database import LearningDatabase
from superlocalmemory.learning.signals import LearningSignals
from superlocalmemory.storage import profile_fold_sidecars as sidecars

#: An older store's table: no longer created or written since 4.2.30 (W3-6), but a
#: profile erasure must still clear its rows.
_OLD_CHANNEL_CREDITS = ("CREATE TABLE IF NOT EXISTS channel_credits (id INTEGER PRIMARY KEY "
                        "AUTOINCREMENT, profile_id TEXT NOT NULL, query_type TEXT NOT NULL, "
                        "channel TEXT NOT NULL, hits INTEGER DEFAULT 0, total INTEGER DEFAULT 0, "
                        "updated_at TEXT NOT NULL)")


def _recall_learns(path, profile: str) -> None:
    """What recall leaves in learning.db (an older store's channel credits too)."""
    signals = LearningSignals(str(path))
    conn = sqlite3.connect(path)
    conn.execute(_OLD_CHANNEL_CREDITS)
    conn.execute("INSERT INTO channel_credits (profile_id, query_type, channel, hits, total, "
                 "updated_at) VALUES (?, 'q', 'bm25', 1, 2, 'now')", (profile,))
    conn.execute("INSERT INTO co_retrieval_edges (profile_id, fact_id_a, fact_id_b, updated_at) "
                 "VALUES (?, 'fact-a', 'fact-b', 'now')", (profile,))
    conn.commit()
    conn.close()
    assert signals is not None


@pytest.fixture
def learning(tmp_path):
    path = tmp_path / "learning.db"
    LearningDatabase(path)
    return path


def test_the_producers_tables_have_a_recorded_decision(learning) -> None:
    _recall_learns(learning, "other")
    sidecars.check_learning(learning)  # raised "no recorded decision for ... channel_credits"


def test_deleting_a_profile_removes_only_its_recall_learning(learning) -> None:
    _recall_learns(learning, "gone")
    _recall_learns(learning, "default")

    sidecars.purge_learned_state(learning, "gone")

    conn = sqlite3.connect(learning)
    try:
        for table in ("channel_credits", "co_retrieval_edges"):
            owners = {r[0] for r in conn.execute(f"SELECT profile_id FROM {table}")}
            assert owners == {"default"}, table
    finally:
        conn.close()


def test_an_unknown_table_still_refuses_the_delete(learning) -> None:
    conn = sqlite3.connect(learning)
    conn.execute("CREATE TABLE future_feature (profile_id TEXT, x TEXT)")
    conn.commit()
    conn.close()
    with pytest.raises(sidecars.SidecarFoldError, match="future_feature"):
        sidecars.check_learning(learning)


def test_every_learning_table_a_producer_creates_is_decided(tmp_path) -> None:
    """The inventory is read from the real producers, not from a hand-kept list."""
    path = tmp_path / "learning.db"
    LearningDatabase(path)
    LearningSignals(str(path))
    conn = sqlite3.connect(path)
    try:
        scoped = sidecars._scoped(conn)
    finally:
        conn.close()
    undecided = [t for t in scoped if t not in sidecars.LEARNING_DECISIONS]
    assert not undecided, undecided


def test_a_gdpr_erasure_removes_the_co_retrieved_ids_too(learning) -> None:
    _recall_learns(learning, "gone")
    _recall_learns(learning, "default")

    LearningDatabase(learning).reset("gone")

    conn = sqlite3.connect(learning)
    try:
        for table in ("channel_credits", "co_retrieval_edges"):
            assert {r[0] for r in conn.execute(f"SELECT profile_id FROM {table}")} == {"default"}
    finally:
        conn.close()
