# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M073 / R-11: every valid_until records where it came from (I-12)."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS
from superlocalmemory.storage.migrations import M073_valid_until_source as M073


def _bare() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, valid_from TEXT, created_at TEXT);
        CREATE TABLE fact_temporal_validity (fact_id TEXT, profile_id TEXT, valid_from TEXT,
            valid_until TEXT, system_created_at TEXT, system_expired_at TEXT,
            invalidated_by TEXT, invalidation_reason TEXT);
        CREATE TABLE correction_cases (case_id TEXT PRIMARY KEY, profile_id TEXT,
            predecessor_fact_id TEXT, successor_fact_id TEXT, status TEXT,
            event_valid_until TEXT, applied_at TEXT, predecessor_valid_from TEXT);
    """)
    return conn


def _case(conn, case, pred, succ, *, until, applied_at="2026-11-10T00:00:00+00:00",
          succ_from=None, succ_created=None, reviewer=None):
    conn.execute("INSERT INTO atomic_facts VALUES (?, NULL, '2026-01-01T00:00:00+00:00')",
                 (pred,))
    conn.execute("INSERT INTO atomic_facts VALUES (?, ?, ?)", (succ, succ_from, succ_created))
    conn.execute("INSERT INTO fact_temporal_validity (fact_id, profile_id, valid_from, "
                 "valid_until, system_expired_at) VALUES (?, 'p', '2026-01-01', ?, ?)",
                 (pred, until, applied_at))
    conn.execute("INSERT INTO correction_cases VALUES (?, 'p', ?, ?, 'applied', ?, ?, NULL)",
                 (case, pred, succ, reviewer, applied_at))


def _source(conn, fact_id):
    return conn.execute("SELECT valid_until_source FROM fact_temporal_validity WHERE fact_id = ?",
                        (fact_id,)).fetchone()[0]


def test_registered_deferred_and_additive() -> None:
    assert _MODULES[M073.NAME] is M073
    assert M073.NAME in {m.name for m in DEFERRED_MIGRATIONS}
    assert M073.DOWNGRADE_FLOOR == 51 and M073.DB_TARGET == "memory"


def test_the_backfill_names_the_evidence_or_says_recorded() -> None:
    conn = _bare()
    _case(conn, "c1", "a1", "b1", until="2026-05-01T00:00:00+00:00",
          succ_from="2026-05-01T00:00:00+00:00")                       # the successor's own time
    _case(conn, "c2", "a2", "b2", until="2026-11-10T00:00:00+00:00")   # only when applied
    _case(conn, "c3", "a3", "b3", until="2026-02-02T00:00:00+00:00",
          succ_from="2026-05-01T00:00:00+00:00")                       # not reproduced
    _case(conn, "c4", "a4", "b4", until="2026-03-03T00:00:00+00:00",
          reviewer="2026-03-03T00:00:00+00:00")                        # a person's time
    conn.execute("INSERT INTO fact_temporal_validity (fact_id, profile_id, valid_until) "
                 "VALUES ('z', 'p', '2026-09-09')")                    # not from a correction
    assert not M073.verify(conn)
    M073.apply(conn)
    assert [_source(conn, f) for f in ("a1", "a2", "a3", "a4", "z")] == [
        "successor_fact", "applied", "recorded", "reviewer", "recorded"]
    assert M073.verify(conn)
    M073.apply(conn)  # again: nothing changes
    assert _source(conn, "a2") == "applied"


def test_the_column_refuses_an_unknown_source() -> None:
    conn = _bare()
    M073.apply(conn)
    conn.execute("INSERT INTO fact_temporal_validity (fact_id, profile_id) VALUES ('x', 'p')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE fact_temporal_validity SET valid_until_source = 'guess'")


def test_a_deleted_case_takes_its_snapshot_with_it() -> None:
    conn = _bare()
    M073.apply(conn)
    conn.execute("INSERT INTO correction_case_until_source VALUES ('c9', 'stated')")
    conn.execute("INSERT INTO correction_cases (case_id) VALUES ('c9')")
    conn.execute("DELETE FROM correction_cases WHERE case_id = 'c9'")
    assert conn.execute("SELECT COUNT(*) FROM correction_case_until_source").fetchone()[0] == 0


def test_the_ledger_columns_m042_pins_are_untouched() -> None:
    from superlocalmemory.storage.migrations import M042_correction_case_ledger as M042

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, valid_from TEXT, "
                 "created_at TEXT)")
    conn.execute("CREATE TABLE fact_temporal_validity (fact_id TEXT, profile_id TEXT, "
                 "valid_from TEXT, valid_until TEXT)")
    for statement in M042.DDL.split(";"):
        if statement.strip():
            conn.execute(statement)
    M073.apply(conn)
    assert M042.verify(conn)
