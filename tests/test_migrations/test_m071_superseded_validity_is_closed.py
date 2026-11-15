# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M071: applied corrections get their predecessor's valid_until, with an undo record."""

from __future__ import annotations

import sqlite3

from superlocalmemory.storage import data_repair_undo
from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS
from superlocalmemory.storage.migrations import M071_superseded_validity_is_closed as M071


def _store() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
    CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, profile_id TEXT, valid_from TEXT,
                               created_at TEXT);
    CREATE TABLE fact_temporal_validity (fact_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL,
        valid_from TEXT, valid_until TEXT, system_created_at TEXT, system_expired_at TEXT,
        invalidated_by TEXT, invalidation_reason TEXT);
    CREATE TABLE correction_cases (case_id TEXT PRIMARY KEY, profile_id TEXT,
        predecessor_fact_id TEXT, successor_fact_id TEXT, status TEXT, applied_at TEXT,
        event_valid_until TEXT);
    """)
    return conn


def _case(conn, case_id, pred, succ, *, status="applied", applied="2026-10-01T00:00:00+00:00",
          event_until=None, pred_from=None, succ_tv_from=None, succ_created=None,
          expired="2026-10-01T00:00:00+00:00") -> None:
    conn.execute("INSERT INTO atomic_facts VALUES (?, 'p', NULL, '2026-01-01T00:00:00+00:00')",
                 (pred,))
    conn.execute("INSERT INTO atomic_facts VALUES (?, 'p', NULL, ?)", (succ, succ_created))
    conn.execute("INSERT INTO fact_temporal_validity (fact_id, profile_id, valid_from, "
                 "system_expired_at) VALUES (?, 'p', ?, ?)", (pred, pred_from, expired))
    if succ_tv_from is not None:
        conn.execute("INSERT INTO fact_temporal_validity (fact_id, profile_id, valid_from) "
                     "VALUES (?, 'p', ?)", (succ, succ_tv_from))
    conn.execute("INSERT INTO correction_cases VALUES (?, 'p', ?, ?, ?, ?, ?)",
                 (case_id, pred, succ, status, applied, event_until))


def _until(conn, fact_id):
    return conn.execute("SELECT valid_until FROM fact_temporal_validity WHERE fact_id = ?",
                        (fact_id,)).fetchone()[0]


def test_registered_deferred_with_floor_and_no_repair() -> None:
    assert _MODULES[M071.NAME] is M071
    assert M071.NAME in {m.name for m in DEFERRED_MIGRATIONS}
    assert M071.DOWNGRADE_FLOOR == 51 and M071.REPAIR_NOT_APPLICABLE.strip()


def test_evidence_order_reviewer_then_successor_validity_then_save_then_apply() -> None:
    conn = _store()
    _case(conn, "c1", "a1", "b1", event_until="2026-03-01T00:00:00+00:00",
          succ_tv_from="2026-02-01T00:00:00+00:00")
    _case(conn, "c2", "a2", "b2", succ_tv_from="2026-02-01", succ_created="2026-05-01T00:00:00Z")
    _case(conn, "c3", "a3", "b3", succ_created="2026-05-01T00:00:00Z")
    _case(conn, "c4", "a4", "b4")
    M071.apply(conn)
    assert _until(conn, "a1") == "2026-03-01T00:00:00+00:00"
    assert _until(conn, "a2") == "2026-02-01T00:00:00+00:00"
    assert _until(conn, "a3") == "2026-05-01T00:00:00+00:00"
    assert _until(conn, "a4") == "2026-10-01T00:00:00+00:00"


def test_leaves_proposed_open_inverted_and_already_closed_alone() -> None:
    conn = _store()
    _case(conn, "c1", "a1", "b1", status="proposed")
    _case(conn, "c2", "a2", "b2", pred_from="2026-12-01T00:00:00+00:00")  # would invert
    _case(conn, "c3", "a3", "b3", expired=None)  # not retired
    _case(conn, "c4", "a4", "b4")
    conn.execute("UPDATE fact_temporal_validity SET valid_until = '2026-09-09T00:00:00+00:00' "
                 "WHERE fact_id = 'a4'")
    M071.apply(conn)
    assert [_until(conn, f) for f in ("a1", "a2", "a3")] == [None, None, None]
    assert _until(conn, "a4") == "2026-09-09T00:00:00+00:00"


def test_idempotent_and_undoable() -> None:
    conn = _store()
    _case(conn, "c1", "a1", "b1")
    M071.apply(conn)
    M071.apply(conn)
    assert data_repair_undo.summary(conn, M071.NAME)
    rows = conn.execute("SELECT COUNT(*) FROM data_repair_undo WHERE repair = ?",
                        (M071.NAME,)).fetchone()[0]
    assert rows == 1
    conn.commit()
    assert data_repair_undo.undo(conn, M071.NAME) == 1
    assert _until(conn, "a1") is None


def test_a_store_without_the_ledger_is_untouched() -> None:
    conn = sqlite3.connect(":memory:")
    M071.apply(conn)
    assert not data_repair_undo.has_table(conn)
