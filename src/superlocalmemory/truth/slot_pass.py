# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The background pass that binds facts to slots, oldest first, resumably.

One path for every fact: the 43k already stored and each new save, which the
next maintenance cycle picks up. Oldest first, so a store's history builds its
slots in the order it happened (a newer value supersedes, an out-of-order older
one joins history). The save hot path is untouched.

Bounded: ``batch`` facts per transaction, ``max_batches`` per run, a cursor
file so a stop loses nothing, and ``stop()`` honoured between batches. Off
unless ``truth.slots`` is shadow or on; retiring facts additionally needs
``truth.slots.auto_apply``.

The queue (``slot_pass_queue``, M069) is drained before the cursor moves on:
facts to bind again because erasing a slot's head rebuilt it (``rebind``), and
facts whose binding failed (``retry``), each retried with exponential backoff
(``RETRY_BASE_S`` doubling, at most ``RETRY_MAX_S``), so no fact is skipped
silently (R-19). A queued fact that is no longer eligible (retired, archived,
quarantined, erased) simply leaves the queue.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from superlocalmemory.core.feature_flags import FeatureFlags
from superlocalmemory.truth.slot_keys import MAX_CHARS
from superlocalmemory.truth.slot_resolver import resolve_fact

logger = logging.getLogger(__name__)

STATE = "slot-pass.json"

_FIELDS = """f.rowid AS rid, f.fact_id, f.profile_id, f.scope, f.content,
       COALESCE(v.valid_from, f.valid_from, f.created_at) AS valid_from"""
_LIVE = f"""f.quarantined = 0 AND f.non_answer = 0
   AND f.lifecycle != 'archived' AND f.merged_into IS NULL
   AND v.system_expired_at IS NULL AND length(f.content) <= {MAX_CHARS}"""

_ELIGIBLE = f"""
SELECT {_FIELDS}
  FROM atomic_facts f
  LEFT JOIN fact_temporal_validity v ON v.fact_id = f.fact_id
 WHERE f.rowid > ? AND {_LIVE}
 ORDER BY f.rowid
 LIMIT ?
"""

#: Due queue rows, oldest fact first; ``eligible`` 0 means the fact left the pass.
_DUE = f"""
SELECT {_FIELDS}, q.attempts, CASE WHEN {_LIVE} THEN 1 ELSE 0 END AS eligible
  FROM slot_pass_queue q
  JOIN atomic_facts f ON f.fact_id = q.fact_id
  LEFT JOIN fact_temporal_validity v ON v.fact_id = f.fact_id
 WHERE q.next_at <= ?
 ORDER BY f.rowid
 LIMIT ?
"""

#: Routing only (when to try again), never a memory decision.
RETRY_BASE_S = 60.0
RETRY_MAX_S = 86_400.0


def _now() -> datetime:
    return datetime.now(UTC)


def _retry_at(attempts: int, now: datetime) -> str:
    delay = min(RETRY_BASE_S * (2 ** max(attempts - 1, 0)), RETRY_MAX_S)
    return (now + timedelta(seconds=delay)).isoformat()


class SlotPass:
    def __init__(self, db: Any, flags: FeatureFlags, *, state_dir: Path, batch: int = 200,
                 max_batches: int | None = 25) -> None:
        self._db = db
        self._flags = flags
        self._state = Path(state_dir) / STATE
        self._batch = batch
        self._max_batches = max_batches
        self._stop = threading.Event()
        self._last_error = ""
        self.counts: Counter[str] = Counter()

    def stop(self) -> None:
        self._stop.set()

    def _revisions(self) -> int:
        rows = self._db.execute("SELECT COUNT(*) AS n FROM slot_revisions", ())
        return int(rows[0]["n"]) if rows else 0

    def _cursor(self) -> int:
        """Where the last run stopped; 0 when revisions vanished since then.

        A profile fold deletes the source profile's slots (they cannot move,
        their ids name the profile) and an erasure deletes a fact's revisions,
        so fewer revisions than recorded means facts behind the cursor may have
        lost theirs. Starting over is safe: binding a fact twice changes nothing.
        """
        try:
            state = json.loads(self._state.read_text(encoding="utf-8"))
            after, seen = int(state["after_rowid"]), int(state.get("revisions", 0))
        except (OSError, ValueError, KeyError, TypeError):
            return 0
        if self._revisions() >= seen:
            return after
        # Starting over binds every fact again, so queued rebinds would be done twice.
        self._db.execute("DELETE FROM slot_pass_queue WHERE reason = 'rebind'", ())
        return 0

    def _save(self, after_rowid: int) -> None:
        self._state.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"after_rowid": after_rowid, "revisions": self._revisions()}),
                       encoding="utf-8")
        os.replace(tmp, self._state)

    def _one_batch(self, after: int, auto_apply: bool) -> int | None:
        with self._db.raw_connection() as conn:
            rows = conn.execute(_ELIGIBLE, (after, self._batch)).fetchall()
            if not rows:
                return None
            for row in rows:
                if not self._one_fact(conn, row, auto_apply):
                    self._enqueue_retry(conn, row[1], attempts=1)
            return int(rows[-1][0])

    def _queue_batch(self, auto_apply: bool) -> int:
        """Bind up to ``batch`` due queued facts. Returns how many were due."""
        now = _now()
        with self._db.raw_connection() as conn:
            rows = conn.execute(_DUE, (now.isoformat(), self._batch)).fetchall()
            for row in rows:
                attempts, eligible = int(row[6]), bool(row[7])
                if not eligible:
                    self.counts["queue_dropped"] += 1
                elif not self._one_fact(conn, row, auto_apply):
                    self._enqueue_retry(conn, row[1], attempts=attempts + 1, now=now)
                    continue
                else:
                    self.counts["queue_bound"] += 1
                conn.execute("DELETE FROM slot_pass_queue WHERE fact_id = ?", (row[1],))
        return len(rows)

    def _enqueue_retry(self, conn: Any, fact_id: str, *, attempts: int,
                       now: datetime | None = None) -> None:
        now = now or _now()
        conn.execute(
            "INSERT INTO slot_pass_queue (fact_id, reason, attempts, next_at, last_error, "
            "queued_at) VALUES (?, 'retry', ?, ?, ?, ?) ON CONFLICT(fact_id) DO UPDATE SET "
            "reason = 'retry', attempts = excluded.attempts, next_at = excluded.next_at, "
            "last_error = excluded.last_error",
            (fact_id, attempts, _retry_at(attempts, now), self._last_error, now.isoformat()))
        self.counts["queued_retry"] += 1

    def _one_fact(self, conn: Any, row: Any, auto_apply: bool) -> bool:
        """One fact in its own savepoint: a failure undoes only that fact."""
        conn.execute("SAVEPOINT slot_fact")
        try:
            result = resolve_fact(
                conn, fact_id=row[1], profile_id=row[2], scope=row[3], content=row[4],
                valid_from=row[5], auto_apply=auto_apply)
        except Exception as exc:  # noqa: BLE001 - one bad fact must not stall the pass
            conn.execute("ROLLBACK TO slot_fact")
            conn.execute("RELEASE slot_fact")
            self.counts["error"] += 1
            # The class only: a message could quote the memory.
            self._last_error = type(exc).__name__
            logger.info("slot pass: fact %s failed (%s); queued for retry", row[1],
                        self._last_error)
            return False
        conn.execute("RELEASE slot_fact")
        self.counts["keyed" if result.keyed else "unkeyed"] += 1
        if result.outcome:
            self.counts[result.outcome] += 1
        if result.ledger:
            self.counts[f"ledger_{result.ledger}"] += 1
        return True

    def run(self) -> dict[str, int]:
        """Bind up to ``max_batches`` batches. Returns the outcome counts."""
        if not self._flags.observing("truth.slots"):
            return {}
        # A child flag never acts beyond its parent: shadow slots retire nothing.
        auto_apply = (self._flags.enabled("truth.slots.auto_apply")
                      and self._flags.enabled("truth.slots"))
        after = self._cursor()
        batches = 0
        while not self._stop.is_set():
            if self._max_batches is not None and batches >= self._max_batches:
                break
            if self._queue_batch(auto_apply):  # queued facts first; failures move to later
                batches += 1
                continue
            last = self._one_batch(after, auto_apply)
            if last is None:
                break
            after = last
            self._save(after)
            batches += 1
        if self._flags.enabled("truth.slots") and not self._stop.is_set():
            # Decisions made in shadow or with auto_apply off reach the ledger (R-07).
            # Never in shadow: shadow writes only the truth layer's own tables.
            from superlocalmemory.truth.slot_resolver import reconcile_ledger

            with self._db.raw_connection() as conn:
                self.counts.update(reconcile_ledger(conn, auto_apply=auto_apply,
                                                    limit=self._batch))
        if self.counts:
            logger.info("Slot pass: %s", dict(self.counts))
        return dict(self.counts)


__all__ = ["STATE", "SlotPass"]
