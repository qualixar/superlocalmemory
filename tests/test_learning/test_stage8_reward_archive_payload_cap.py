# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Stage 8 F5 (Mediums/Lows)

"""Stage 8 F5 regression — SEC-L3 payload_json soft cap (256 KB).

A pathological consolidator could write multi-MB blobs into
``memory_archive.payload_json`` and blow through the I4 disk budget.
The write path now truncates oversize payloads to a stub and tags the
row with ``reason='reward_gated_ebbinghaus_truncated'`` — the fact
remains recoverable via ``atomic_facts`` (archive never DELETEs).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.learning import reward_archive as ra


def _bootstrap_schema(db_path: Path) -> None:
    """Minimal schema mirroring post-M011 memory.db for this test."""
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE atomic_facts (
                fact_id                   TEXT PRIMARY KEY,
                profile_id                TEXT NOT NULL,
                content                   TEXT,
                canonical_entities_json   TEXT,
                importance                REAL,
                confidence                REAL,
                embedding                 TEXT,
                created_at                TEXT,
                archive_status            TEXT DEFAULT 'live',
                archive_reason            TEXT,
                merged_into               TEXT,
                retrieval_prior           REAL DEFAULT 0.0
            );
            CREATE TABLE memory_archive (
                archive_id    TEXT PRIMARY KEY,
                fact_id       TEXT NOT NULL,
                profile_id    TEXT NOT NULL,
                payload_json  TEXT NOT NULL,
                archived_at   TEXT NOT NULL,
                reason        TEXT NOT NULL
            );
            CREATE TABLE action_outcomes (
                outcome_id       TEXT PRIMARY KEY,
                profile_id       TEXT,
                fact_ids_json    TEXT,
                reward           REAL,
                settled          INTEGER,
                settled_at       TEXT
            );
            """
        )


def _seed_fact(db_path: Path, fid: str, *, content: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, profile_id, content, "
            "canonical_entities_json, importance, confidence, embedding, "
            "created_at, archive_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (fid, "default", content, "[]", 0.0, 0.5, None, "2026-01-01", "live"),
        )


def test_sec_l3_large_payload_truncated_to_stub(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _bootstrap_schema(db)

    # Content is bigger than PAYLOAD_JSON_MAX_BYTES (256 KB).
    huge = "x" * (ra.PAYLOAD_JSON_MAX_BYTES + 1024)
    _seed_fact(db, "fact-huge-1", content=huge)

    archived = ra.run_reward_gated_archive(
        db, "default", candidate_fact_ids=["fact-huge-1"],
    )
    assert archived == ["fact-huge-1"]

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT payload_json, reason FROM memory_archive "
            "WHERE fact_id=?", ("fact-huge-1",),
        ).fetchone()

    assert row is not None
    payload = json.loads(row["payload_json"])
    assert payload.get("truncated") is True
    assert payload.get("fact_id") == "fact-huge-1"
    # Row is tagged so the operator can see what happened.
    assert row["reason"] == "reward_gated_ebbinghaus_truncated"
    # The stub itself is well under the cap.
    assert len(row["payload_json"].encode("utf-8")) < ra.PAYLOAD_JSON_MAX_BYTES


def test_sec_l3_normal_payload_not_truncated(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    _bootstrap_schema(db)
    _seed_fact(db, "fact-small-1", content="hello world")

    archived = ra.run_reward_gated_archive(
        db, "default", candidate_fact_ids=["fact-small-1"],
    )
    assert archived == ["fact-small-1"]

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT payload_json, reason FROM memory_archive "
            "WHERE fact_id=?", ("fact-small-1",),
        ).fetchone()
    payload = json.loads(row["payload_json"])
    assert payload.get("truncated") is not True
    assert payload["content"] == "hello world"
    assert row["reason"] == "reward_gated_ebbinghaus"
