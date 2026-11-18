# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""How long delivery receipts are kept (W3 open item from W3-1).

Since W3-1 every recall an agent asks for adds a delivery row and one receipt
per result, so a busy agent adds thousands of rows a day. What reads them, and
how far back:

* **causal estimates (M-12) and the harm gate** read the *randomised*
  deliveries (session start) for as long as the store lives: they are the only
  evidence there is, and they are few (F-47). **Never deleted.** Their
  ``reask`` outcome is a later *pull* in the same conversation within 24 h, so
  a pull inside that window after a randomised delivery is kept as well;
* the **position-attention model** (D-46) reads pulls that carry a logged
  swap: kept;
* **use rates (M-13)** weight a delivery 2^(−age / 30 d): past
  ``KEEP_PULL_DAYS`` = 180 days (six half-lives) one delivery weighs under 1/64;
* the **need gate** (M-16) reads 30 days; the **ranker's labels** come from
  exposures that keep only their newest 20,000 (``exposure_retention``).

So pulls older than ``KEEP_PULL_DAYS`` that nothing above needs are deleted
with their receipts and usage rows, and need-gate turns older than that too.
Deletes run in batches (``BATCH`` rows per statement) on the consolidation
timer; the store's writer lock is taken for one batch at a time and let go between them. Ids and numbers only: there
is no memory text in any of these tables.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from superlocalmemory.outcomes import effects, utility
from superlocalmemory.outcomes import randomiser as rz
from superlocalmemory.storage.write_lock import get_write_lock

logger = logging.getLogger(__name__)

#: Six M-13 half-lives: an older delivery would weigh under 1/64 in a use rate.
KEEP_PULL_DAYS = 6 * utility.HALF_LIFE_DAYS
#: Routing only: deliveries deleted per statement.
BATCH = 2_000

_EXPIRED_PULLS = """
SELECT p.delivery_id FROM injection_deliveries p
 WHERE p.profile_id = ? AND p.path IN ({pulls})
   AND julianday(p.delivered_at) < julianday('now') - ?
   AND p.swap_at IS NULL
   AND NOT EXISTS (
       SELECT 1 FROM injection_deliveries r
        WHERE r.profile_id = p.profile_id AND r.session_id = p.session_id
          AND r.path IN ({randomised})
          AND julianday(p.delivered_at) > julianday(r.delivered_at)
          AND julianday(p.delivered_at) <= julianday(r.delivered_at) + ? / 24.0)
 LIMIT ?
"""


def _sql() -> tuple[str, list]:
    pulls, randomised = sorted(rz.PULL_PATHS), sorted(rz.RANDOMISED_PATHS)
    sql = _EXPIRED_PULLS.format(pulls=", ".join("?" * len(pulls)),
                                randomised=", ".join("?" * len(randomised)))
    return sql, [*pulls, *randomised]


def _batch(conn: sqlite3.Connection, profile_id: str, lock) -> int:
    sql, names = _sql()
    pulls = names[:len(rz.PULL_PATHS)]
    randomised = names[len(rz.PULL_PATHS):]
    ids = [r[0] for r in conn.execute(sql, [profile_id, *pulls, KEEP_PULL_DAYS, *randomised,
                                            effects.OUTCOME_WINDOW_H, BATCH])]
    if not ids:
        return 0
    marks = ", ".join("?" * len(ids))
    with lock, conn:  # the id read above ran outside it: one short hold per batch
        conn.execute(f"DELETE FROM injection_usage WHERE delivery_id IN ({marks})", ids)
        conn.execute(f"DELETE FROM injection_receipts WHERE delivery_id IN ({marks})", ids)
        conn.execute(f"DELETE FROM injection_deliveries WHERE delivery_id IN ({marks})", ids)
    return len(ids)


def prune(db_path: str | Path, profile_id: str) -> dict:
    """Delete expired pull deliveries and old need-gate turns. Never raises."""
    out = {"pull_deliveries": 0, "need_turns": 0}
    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
    except sqlite3.Error as exc:
        logger.debug("receipt retention: %s", exc)
        return out
    lock = get_write_lock(db_path)  # the store file's writer lock, taken per batch (WS-7)
    try:
        while n := _batch(conn, profile_id, lock):
            out["pull_deliveries"] += n
        while True:
            with lock, conn:
                n = conn.execute(
                    "DELETE FROM need_turns WHERE turn_id IN (SELECT turn_id FROM need_turns"
                    " WHERE profile_id = ? AND julianday(at) < julianday('now') - ? LIMIT ?)",
                    (profile_id, KEEP_PULL_DAYS, BATCH)).rowcount
            if not n:
                break
            out["need_turns"] += n
    except sqlite3.Error as exc:  # a missing table (older store) ends the pass quietly
        logger.debug("receipt retention stopped: %s", exc)
    finally:
        conn.close()
    return out


__all__ = ["BATCH", "KEEP_PULL_DAYS", "prune"]
