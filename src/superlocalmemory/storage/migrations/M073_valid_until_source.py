# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M073: every ``valid_until`` says where it came from (4.2.30, R-11, I-12).

"What was true in March?" can only be answered honestly if SLM knows whether
an end time is evidence about the world or only about when SLM learned of a
change. A correction's end time comes from a chain: the reviewer's time, else
when the replacing memory became true, else when it was saved, else when the
correction was applied (``truth/validity.closing_time_sourced``). The last two
are system time: an upper bound on when the old value stopped being true,
never the time itself (blueprint M-10).

Adds ``fact_temporal_validity.valid_until_source`` (one of
``truth/validity.SOURCES``) and a side table ``correction_case_until_source``
holding the predecessor's source at apply time, so a rollback restores the
source with the time. A side table, not a column on ``correction_cases``: M042
verifies that ledger's exact columns and must not change once shipped. Backfills existing ends:
the predecessor of an applied correction gets the source its end is
reproduced from by the same chain on the same inputs; anything not
reproduced, or not from a correction, is ``recorded`` (set, evidence unknown),
never a guess. Only a new column is filled, so an older build is unaffected.
"""

from __future__ import annotations

import sqlite3

NAME = "M073_valid_until_source"
DB_TARGET = "memory"
#: Additive: two new nullable columns an older build never reads.
DOWNGRADE_FLOOR = 51

_SOURCES = ("stated", "reviewer", "successor_validity", "successor_fact", "successor_saved",
            "applied", "recorded")

DDL = """
-- apply() adds valid_until_source and backfills it (see the module docstring).
"""

_SIDE = """
CREATE TABLE IF NOT EXISTS correction_case_until_source (
    case_id TEXT PRIMARY KEY,
    predecessor_valid_until_source TEXT
)
"""
#: A case deleted (erasure, overtaken proposals) takes its snapshot with it.
_SIDE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_correction_case_until_source_gone
AFTER DELETE ON correction_cases
BEGIN
    DELETE FROM correction_case_until_source WHERE case_id = OLD.case_id;
END
"""

_CASE_ENDS = """
SELECT tv.fact_id, tv.profile_id, tv.valid_until, cc.event_valid_until, cc.applied_at,
       COALESCE(cc.predecessor_valid_from, tv.valid_from) AS pred_from,
       sv.valid_from AS succ_tv_from, sf.valid_from AS succ_af_from, sf.created_at AS succ_created
  FROM fact_temporal_validity tv
  JOIN correction_cases cc
    ON cc.predecessor_fact_id = tv.fact_id AND cc.profile_id = tv.profile_id
   AND cc.status = 'applied'
  LEFT JOIN fact_temporal_validity sv ON sv.fact_id = cc.successor_fact_id
  LEFT JOIN atomic_facts sf ON sf.fact_id = cc.successor_fact_id
 WHERE tv.valid_until IS NOT NULL AND tv.valid_until_source IS NULL
"""


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")}  # noqa: S608


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    return bool(conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                             (table,)).fetchone())


def _add_columns(conn: sqlite3.Connection) -> None:
    if "valid_until_source" not in _columns(conn, "fact_temporal_validity"):
        allowed = ", ".join(f"'{s}'" for s in _SOURCES)
        conn.execute("ALTER TABLE fact_temporal_validity ADD COLUMN valid_until_source TEXT "
                     f"CHECK (valid_until_source IS NULL OR valid_until_source IN ({allowed}))")
    conn.execute(_SIDE)
    if _has_table(conn, "correction_cases"):
        conn.execute(_SIDE_TRIGGER)


def _backfill(conn: sqlite3.Connection) -> int:
    from superlocalmemory.truth.validity import closing_time_sourced, normalise

    filled = 0
    if _has_table(conn, "correction_cases") and _has_table(conn, "atomic_facts"):
        for row in conn.execute(_CASE_ENDS).fetchall():
            fact_id, profile_id, until = row[0], row[1], row[2]
            close, source = closing_time_sourced(
                event_valid_until=row[3], successor_validity_from=row[6],
                successor_fact_valid_from=row[7], successor_created_at=row[8],
                applied_at=row[4], predecessor_valid_from=row[5])
            if close is None or normalise(until) != close:
                source = "recorded"
            conn.execute("UPDATE fact_temporal_validity SET valid_until_source = ? "
                         "WHERE fact_id = ? AND profile_id = ?", (source, fact_id, profile_id))
            filled += 1
    filled += conn.execute("UPDATE fact_temporal_validity SET valid_until_source = 'recorded' "
                           "WHERE valid_until IS NOT NULL AND valid_until_source IS NULL").rowcount
    return filled


def apply(conn: sqlite3.Connection) -> None:
    if not _has_table(conn, "fact_temporal_validity"):
        return
    _add_columns(conn)
    _backfill(conn)


def verify(conn: sqlite3.Connection) -> bool:
    if not _has_table(conn, "fact_temporal_validity"):
        return True
    if "valid_until_source" not in _columns(conn, "fact_temporal_validity"):
        return False
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    if "correction_case_until_source" not in names:
        return False
    if _has_table(conn, "correction_cases") and (
            "trg_correction_case_until_source_gone" not in names):
        return False
    return conn.execute("SELECT 1 FROM fact_temporal_validity WHERE valid_until IS NOT NULL "
                        "AND valid_until_source IS NULL LIMIT 1").fetchone() is None


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)
