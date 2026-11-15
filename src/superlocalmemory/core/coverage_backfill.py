# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Memories saved before 4.2.5 get a fact for every sentence.

Since 4.2.5 a save keeps a verbatim fact for each sentence its extracted facts
miss (``encoding/coverage_facts.py``). This pass gives older memories the same,
in the daemon, in bounded batches that yield to recall.

It only reaches memories whose whole-memory fact is live. That fact already
makes every sentence retrievable, so the pass never exposes text recall could
not already find; it makes it findable one sentence at a time. A memory with an
erasure tombstone, or whose whole fact is archived, quarantined, merged,
expired or a stub, is left as it is. Coverage is judged against every fact the
memory ever had, live or not, so a corrected or forgotten sentence is not
brought back.

Each added fact is appended to ``coverage-backfill.jsonl`` (fact and memory
ids) so the pass can be undone exactly. Newest memories first: they are the
ones being asked about. ``coverage-backfill.json`` holds the last memory
reached; a stop loses nothing and a restart resumes there.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from superlocalmemory.encoding.coverage_facts import DEFAULT_LIMIT, coverage_facts
from superlocalmemory.storage.models import AtomicFact

logger = logging.getLogger(__name__)

BATCH = 25
LEDGER = "coverage-backfill.jsonl"
LOCK = "coverage-backfill.lock"
STATE = "coverage-backfill.json"
#: Longer memories are stored truncated with a source pointer; left as they are.
MAX_CHARS = 24000
#: Above any rowid: the newest-first pass starts here.
_TOP = 2**62

_CANDIDATES = """
SELECT m.rowid AS rid, m.memory_id, m.profile_id, m.content, m.session_id,
       m.session_date, m.scope, m.shared_with
FROM memories m
WHERE m.rowid < ? AND length(m.content) BETWEEN 40 AND ?
  AND EXISTS (SELECT 1 FROM atomic_facts s
              WHERE s.memory_id = m.memory_id AND s.content = m.content
                AND s.quarantined = 0 AND s.non_answer = 0
                AND s.lifecycle != 'archived' AND s.merged_into IS NULL
                AND s.valid_until IS NULL {retired})
  {erased}
ORDER BY m.rowid DESC
LIMIT ?
"""
_ERASED = "AND NOT EXISTS (SELECT 1 FROM projection_tombstones t WHERE t.memory_id = m.memory_id)"


#: A memory whose matching fact was replaced must not be re-extracted: that
#: would bring the replaced value back. ``atomic_facts.valid_until`` above is a
#: mirror that nothing fills; retirement lives in ``fact_temporal_validity``.
_RETIRED = ("AND NOT EXISTS (SELECT 1 FROM fact_temporal_validity v "
            "WHERE v.fact_id = s.fact_id AND v.system_expired_at IS NOT NULL)")


def _has_table(execute: Callable, name: str) -> bool:
    return bool(execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)))


def _candidates_sql(execute: Callable) -> str:
    # M035 creates the tombstone table; a store without it has no erasures.
    erased = _ERASED if _has_table(execute, "projection_tombstones") else ""
    retired = _RETIRED if _has_table(execute, "fact_temporal_validity") else ""
    return _CANDIDATES.format(erased=erased, retired=retired)
_HELD = "SELECT content FROM atomic_facts WHERE memory_id = ? AND content != ?"


def plan(execute: Callable, row, limit: int = DEFAULT_LIMIT) -> list[AtomicFact]:
    """The facts *row*'s memory is missing, ready for the writer."""
    content = row["content"]
    held = [AtomicFact(content=r["content"]) for r in execute(_HELD, (row["memory_id"], content))]
    shared = json.loads(row["shared_with"] or "null")
    return [
        replace(fact, memory_id=row["memory_id"], profile_id=row["profile_id"],
                scope=row["scope"] or "personal", shared_with=shared)
        for fact in coverage_facts(content, held, session_id=row["session_id"] or "",
                                   session_date=row["session_date"], limit=limit)
    ]


def pending(execute: Callable, limit: int = DEFAULT_LIMIT,
            before_rowid: int = _TOP) -> tuple[int, int]:
    """(memories, facts) the pass would add, without writing anything."""
    memories = facts = 0
    sql = _candidates_sql(execute)
    while True:
        rows = execute(sql, (before_rowid, MAX_CHARS, 500))
        if not rows:
            return memories, facts
        for row in rows:
            n = len(plan(execute, row, limit))
            memories += 1 if n else 0
            facts += n
        before_rowid = rows[-1]["rid"]


def _try_lock(fd: int) -> bool:
    try:
        if os.name == "nt":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


class CoverageBackfill:
    """Bounded, resumable, undoable. ``write`` stores one prebuilt fact with
    full enrichment (the engine's ``run_store_fact_direct``)."""

    def __init__(self, db, write: Callable[[AtomicFact], object], *, state_dir: Path,
                 batch: int = BATCH, max_batches: int | None = None,
                 limit: int = DEFAULT_LIMIT) -> None:
        self._db = db
        self._write = write
        self._dir = Path(state_dir)
        self._batch = batch
        self._max_batches = max_batches
        self._limit = limit
        self._stop = threading.Event()
        self.written = 0
        self.finished = False

    def stop(self, timeout: float = 0.0) -> None:
        """Stop after the memory in progress."""
        self._stop.set()

    def _state(self) -> dict:
        try:
            return json.loads((self._dir / STATE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"before_rowid": _TOP, "finished": False}

    def _save(self, before_rowid: int, finished: bool) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._dir / (STATE + ".tmp")
        tmp.write_text(json.dumps({"before_rowid": before_rowid, "finished": finished,
                                   "written": self.written}), encoding="utf-8")
        tmp.replace(self._dir / STATE)

    def _record(self, fact: AtomicFact) -> None:
        with (self._dir / LEDGER).open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps({"fact_id": fact.fact_id, "memory_id": fact.memory_id,
                                     "profile_id": fact.profile_id,
                                     "at": datetime.now(UTC).isoformat()}) + "\n")

    def run_until_done(self) -> int:
        """Run in the calling thread; return the facts added. One runner at a
        time across processes: a CLI engine and the daemon never both add."""
        self._dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._dir / LOCK, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if not _try_lock(fd):
                return 0
            try:
                return self._run_locked()
            finally:
                _unlock(fd)
        finally:
            os.close(fd)

    def _run_locked(self) -> int:
        from superlocalmemory.core.recall_gate import background_work

        state = self._state()
        if state.get("finished"):
            self.finished = True  # newer memories were covered when saved
            return 0
        before, batches = int(state.get("before_rowid") or _TOP), 0
        sql = _candidates_sql(self._db.execute)
        while not self._stop.is_set() and (self._max_batches is None
                                           or batches < self._max_batches):
            rows = self._db.execute(sql, (before, MAX_CHARS, self._batch))
            if not rows:
                self.finished = True
                break
            with background_work():
                for row in rows:
                    if self._stop.is_set():
                        break
                    for fact in plan(self._db.execute, row, self._limit):
                        self._write(fact)
                        self._record(fact)
                        self.written += 1
                    before = row["rid"]
            self._save(before, False)
            batches += 1
        self._save(before, self.finished)
        return self.written
