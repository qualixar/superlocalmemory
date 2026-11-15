# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Memories a correction replaced get the date they stopped being true, once.

WHAT WAS WRONG
Applying a correction retired the older memory (``system_expired_at``, the
moment SLM learned) but left its ``valid_until`` empty unless a reviewer typed
one, so "what was true at T" could not be answered from valid time. On one
real store all 340 applied corrections were open-ended.

WHAT THIS DOES
For every applied correction whose predecessor's validity row is retired but
open-ended, it sets ``valid_until`` from the best evidence available
(``truth/validity.closing_time``: the reviewer's time, else when the successor
became valid, else when it was saved, else when the correction was applied).
A close that would end before the memory began is skipped. Every change goes
to ``data_repair_undo`` in the same transaction, so
``data_repair_undo.undo(conn, NAME)`` reverses it. Rolling a correction back
still restores the exact snapshot taken when it was applied.

WHY IT IS SAFE
Default recall already hides retired memories by ``system_expired_at``; this
only fills the valid-time column historical questions read. One transaction;
running it again finds nothing open.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

NAME = "M071_superseded_validity_is_closed"
DB_TARGET = "memory"
#: Data plus the shared undo table; an older build reads both as before.
DOWNGRADE_FLOOR = 51

DDL = """
-- apply() closes the validity of memories applied corrections retired.
"""

REPAIR_NOT_APPLICABLE = (
    "A one-time data repair with no schema end state: replaying it would find "
    "nothing left to change, and its undo record is the way back."
)

_OPEN_RETIRED = """
SELECT cc.predecessor_fact_id AS fact_id, cc.profile_id, cc.event_valid_until,
       cc.applied_at, tv.valid_from AS pred_from, tv.valid_until, tv.system_expired_at,
       sv.valid_from AS succ_tv_from, sf.valid_from AS succ_af_from,
       sf.created_at AS succ_created
  FROM correction_cases cc
  JOIN fact_temporal_validity tv
    ON tv.fact_id = cc.predecessor_fact_id AND tv.profile_id = cc.profile_id
  LEFT JOIN fact_temporal_validity sv ON sv.fact_id = cc.successor_fact_id
  LEFT JOIN atomic_facts sf ON sf.fact_id = cc.successor_fact_id
 WHERE cc.status = 'applied' AND tv.valid_until IS NULL
   AND tv.system_expired_at IS NOT NULL
"""


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def apply(conn: sqlite3.Connection) -> None:
    from superlocalmemory.storage.data_repair_undo import UndoWriter
    from superlocalmemory.truth.validity import closing_time

    if not {"correction_cases", "fact_temporal_validity", "atomic_facts"} <= _tables(conn):
        logger.debug("M071: no correction ledger yet; nothing to close")
        return
    previous = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(_OPEN_RETIRED).fetchall()
    finally:
        conn.row_factory = previous
    undo = UndoWriter(conn, NAME)
    closed = skipped = 0
    for row in rows:
        close = closing_time(
            event_valid_until=row["event_valid_until"],
            successor_validity_from=row["succ_tv_from"],
            successor_fact_valid_from=row["succ_af_from"],
            successor_created_at=row["succ_created"],
            applied_at=row["applied_at"],
            predecessor_valid_from=row["pred_from"],
        )
        if close is None:
            skipped += 1
            continue
        undo.record("fact_temporal_validity", row["fact_id"], row["fact_id"],
                    row["profile_id"], {"valid_until": None}, {"valid_until": close})
        conn.execute("UPDATE fact_temporal_validity SET valid_until = ? "
                     "WHERE fact_id = ? AND profile_id = ? AND valid_until IS NULL",
                     (close, row["fact_id"], row["profile_id"]))
        closed += 1
    logger.info("M071: closed the validity of %d memory(ies) applied corrections retired; "
                "skipped %d with no safe end; every change is in data_repair_undo",
                closed, skipped)
