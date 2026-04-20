# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — F4.A Stage-8 H-03/H-06 fix

"""Tests for ``superlocalmemory.learning.fact_outcome_joins``.

Covers:
  * Exact-match retrieval under JSON1 ``json_each``.
  * No substring false positives across overlapping fact_id prefixes
    (this is the H-06 regression proof).
  * Profile_id scoping — outcomes in other profiles are invisible.
  * Empty/None inputs produce empty iterables.
  * Parameterised query — SQL-injection-shaped fact_id probes.

Contract refs:
  - Stage 8 H-03 (architect) + H-06 (skeptic) — see
    ``.backup/active-brain/audit/stage8-CONSOLIDATED.md`` §HIGH.
  - LLD-12 §5 — reward-gated archive criterion 2.
  - LLD-00 §1.4 — action_outcomes schema.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Schema fixture — action_outcomes shape that mirrors post-M007/M010
# ---------------------------------------------------------------------------


def _bootstrap(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE action_outcomes (
                outcome_id       TEXT PRIMARY KEY,
                profile_id       TEXT NOT NULL DEFAULT 'default',
                fact_ids_json    TEXT NOT NULL DEFAULT '[]',
                reward           REAL,
                settled          INTEGER NOT NULL DEFAULT 0,
                settled_at       TEXT
            );
            """
        )


def _seed(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    fact_ids: list[str],
    reward: float = 0.5,
    settled_at: str | None = "2026-04-19T00:00:00+00:00",
) -> str:
    oid = str(uuid.uuid4())
    import json as _json
    conn.execute(
        "INSERT INTO action_outcomes "
        "(outcome_id, profile_id, fact_ids_json, reward, settled, settled_at) "
        "VALUES (?, ?, ?, ?, 1, ?)",
        (oid, profile_id, _json.dumps(fact_ids), reward, settled_at),
    )
    conn.commit()
    return oid


@pytest.fixture
def db(tmp_path: Path) -> Path:
    p = tmp_path / "memory.db"
    _bootstrap(p)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_iter_outcomes_for_fact_returns_rows_with_exact_match(db: Path) -> None:
    """Outcomes whose fact_ids_json array contains the fact_id are
    returned.
    """
    from superlocalmemory.learning.fact_outcome_joins import (
        iter_outcomes_for_fact,
    )

    with sqlite3.connect(db) as conn:
        _seed(conn, profile_id="p1", fact_ids=["fact-abc"], reward=0.8)
        _seed(conn, profile_id="p1", fact_ids=["fact-xyz"], reward=0.1)
        rows = list(iter_outcomes_for_fact(conn, "p1", "fact-abc"))
    assert len(rows) == 1


def test_iter_outcomes_for_fact_does_not_substring_match(db: Path) -> None:
    """H-06 regression: overlapping fact_id substrings MUST NOT collide.

    Under the old LIKE '%"<fid>"%' approach, searching for "abc" would
    match outcomes referencing "abc_extra". JSON1 json_each with an
    equality predicate eliminates this entire class of false positive.
    """
    from superlocalmemory.learning.fact_outcome_joins import (
        iter_outcomes_for_fact,
    )

    with sqlite3.connect(db) as conn:
        # Three outcomes: fact-abc (target), fact-abc_extra (prefix-superset),
        # fact-abcd (no delimiter).
        _seed(conn, profile_id="p1", fact_ids=["fact-abc"], reward=0.9)
        _seed(conn, profile_id="p1", fact_ids=["fact-abc_extra"], reward=0.9)
        _seed(conn, profile_id="p1", fact_ids=["fact-abcd"], reward=0.9)
        rows = list(iter_outcomes_for_fact(conn, "p1", "fact-abc"))
    # Exactly one — only fact-abc matches.
    assert len(rows) == 1, (
        f"substring leak: expected 1 exact match, got {len(rows)}"
    )


def test_iter_outcomes_for_fact_respects_profile_id_scope(db: Path) -> None:
    """Outcomes belonging to a different profile MUST NOT appear in the
    result set, even when the fact_id array is identical.
    """
    from superlocalmemory.learning.fact_outcome_joins import (
        iter_outcomes_for_fact,
    )

    with sqlite3.connect(db) as conn:
        _seed(conn, profile_id="p1", fact_ids=["fact-shared"], reward=1.0)
        _seed(conn, profile_id="p2", fact_ids=["fact-shared"], reward=0.0)
        rows_p1 = list(iter_outcomes_for_fact(conn, "p1", "fact-shared"))
        rows_p2 = list(iter_outcomes_for_fact(conn, "p2", "fact-shared"))
    assert len(rows_p1) == 1
    assert len(rows_p2) == 1
    # And scoping is real — not a cross-profile union.
    rows_ghost = list(iter_outcomes_for_fact(conn, "p-missing", "fact-shared"))
    assert rows_ghost == []


def test_iter_outcomes_for_fact_empty_when_missing(db: Path) -> None:
    """Unknown fact_id returns empty iterable (never raises)."""
    from superlocalmemory.learning.fact_outcome_joins import (
        iter_outcomes_for_fact,
    )

    with sqlite3.connect(db) as conn:
        _seed(conn, profile_id="p1", fact_ids=["other"], reward=0.5)
        rows = list(iter_outcomes_for_fact(conn, "p1", "never-inserted"))
    assert rows == []


def test_iter_outcomes_for_fact_parameterized_query(db: Path) -> None:
    """Fact_id containing SQL meta-characters must not break the query
    nor execute injection. The JSON1 helper MUST pass fact_id as a bind
    param — never concatenated into SQL.
    """
    from superlocalmemory.learning.fact_outcome_joins import (
        iter_outcomes_for_fact,
    )

    with sqlite3.connect(db) as conn:
        evil = "fact'); DROP TABLE action_outcomes;--"
        _seed(conn, profile_id="p1", fact_ids=[evil], reward=0.5)
        # Query with the evil id — should match exactly one row, the
        # action_outcomes table must still exist afterwards.
        rows = list(iter_outcomes_for_fact(conn, "p1", evil))
        assert len(rows) == 1
        # Prove the table is intact.
        still_there = conn.execute(
            "SELECT COUNT(*) FROM action_outcomes",
        ).fetchone()[0]
        assert still_there == 1
