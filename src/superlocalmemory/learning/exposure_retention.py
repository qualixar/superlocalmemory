# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Keep only the exposure rows a ranker can still learn from (W3-6c, D-45).

With ranking off, recall now logs its top 20 as exposures in shadow (about
575 bytes each with their features, F-48 snapshot). Kept forever that is on
the order of a gigabyte a year on a busy store, most of it rows that can never
be labelled. Two rules, run by the consolidation timer before a retrain:

* an exposure is **unlabellable** once its outcome window and a day have passed
  and no delivery (``injection_deliveries.recall_query_id``) and no outcome
  report (``action_outcomes.recall_query_id``) names its recall: deleted;
* of the rest, only the newest ``KEEP_LABELLABLE`` per profile are kept, four
  times the rows the trainer reads (``ranker_retrain_online`` fetches 5,000).

Explicit feedback rows (``legacy_feedback``) are never touched. Deletes run in
batches of ``BATCH`` so the learning store is never held for long; memory.db
is attached read-only.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

#: Routing only: hours after which an exposure no delivery or outcome names can
#: no longer be labelled (the 24 h outcome window plus a day of slack).
UNLABELLED_GRACE_H = 48.0
#: Routing only: labellable exposures kept per profile (4 x the trainer's 5,000).
KEEP_LABELLABLE = 20_000
#: Routing only: rows deleted per statement.
BATCH = 5_000

_EXPOSURE_KINDS = "('candidate', 'shown')"


def _has(conn: sqlite3.Connection, schema: str, table: str, column: str) -> bool:
    try:
        return column in {r[1] for r in conn.execute(f"PRAGMA {schema}.table_info({table})")}
    except sqlite3.Error:
        return False


def _delete(conn: sqlite3.Connection, select_ids: str, params: tuple) -> int:
    """Delete one batch of exposures (features first) chosen by ``select_ids``."""
    ids = [r[0] for r in conn.execute(select_ids + f" LIMIT {BATCH}", params)]
    if not ids:
        return 0
    marks = ",".join("?" * len(ids))
    with conn:
        conn.execute(f"DELETE FROM learning_features WHERE signal_id IN ({marks})", ids)
        conn.execute(f"DELETE FROM learning_signals WHERE id IN ({marks})", ids)
    return len(ids)


def _unlabellable_sql(deliveries: bool, outcomes: bool) -> str:
    named = []
    if deliveries:
        named.append("EXISTS (SELECT 1 FROM mem.injection_deliveries d WHERE "
                     "d.profile_id = s.profile_id AND d.recall_query_id = s.query_id)")
    if outcomes:
        named.append("EXISTS (SELECT 1 FROM mem.action_outcomes o WHERE "
                     "o.profile_id = s.profile_id AND o.recall_query_id = s.query_id)")
    nobody = " AND NOT (" + " OR ".join(named) + ")"
    return ("SELECT s.id FROM learning_signals s WHERE s.profile_id = ? "
            f"AND s.signal_type IN {_EXPOSURE_KINDS} "
            "AND julianday(s.created_at) + ? / 24.0 < julianday('now')" + nobody)


_OVER_CAP = (f"SELECT id FROM learning_signals WHERE profile_id = ? "
             f"AND signal_type IN {_EXPOSURE_KINDS} "
             "AND id NOT IN (SELECT id FROM learning_signals WHERE profile_id = ? "
             f"AND signal_type IN {_EXPOSURE_KINDS} ORDER BY created_at DESC, id DESC LIMIT ?)")


def prune(learning_db: str | Path, memory_db: str | Path | None, profile_id: str) -> dict:
    """Delete exposures that can no longer be labelled and those over the cap.
    Returns the counts; never raises."""
    out = {"unlabellable": 0, "over_cap": 0}
    try:
        conn = sqlite3.connect(str(learning_db), timeout=5.0, uri=True)
    except sqlite3.Error as exc:
        logger.debug("exposure retention: %s", exc)
        return out
    try:
        if not _has(conn, "main", "learning_signals", "query_id"):
            return out
        attached = False
        if memory_db is not None and Path(memory_db).exists():
            conn.execute("ATTACH DATABASE ? AS mem",
                         (f"{Path(memory_db).resolve().as_uri()}?mode=ro",))
            attached = True
        deliveries = attached and _has(conn, "mem", "injection_deliveries", "recall_query_id")
        outcomes = attached and _has(conn, "mem", "action_outcomes", "recall_query_id")
        if deliveries or outcomes:  # never judge "unlabellable" without seeing a label source
            sql = _unlabellable_sql(deliveries, outcomes)
            while n := _delete(conn, sql, (profile_id, UNLABELLED_GRACE_H)):
                out["unlabellable"] += n
        while n := _delete(conn, _OVER_CAP, (profile_id, profile_id, KEEP_LABELLABLE)):
            out["over_cap"] += n
    except sqlite3.Error as exc:
        logger.warning("exposure retention stopped: %s", exc)
    finally:
        conn.close()
    return out


__all__ = ["BATCH", "KEEP_LABELLABLE", "UNLABELLED_GRACE_H", "prune"]
