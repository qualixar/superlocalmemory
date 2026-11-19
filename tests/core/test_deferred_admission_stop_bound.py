# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""H2 (2026-10-06): ``remember_runtime.stop()`` can wait 5 s. Why, and bound it.

Root cause: ``DeferredCommitter._finish()`` calls ``mark_committed`` /
``mark_rejected`` / ``quarantine`` -- pure background bookkeeping stamps for
an entry whose underlying remember was *already* durably committed by
``self._commit()`` moments earlier -- with no deadline at all. The admission
journal's ``GroupCommitWriter._begin()`` treats "no deadline" as "wait the
full unbounded busy window" (``journal_writer._UNBOUNDED_BUSY_SECONDS``,
5 s) before giving up on ``BEGIN IMMEDIATE``. Since ``remember_runtime.stop()``
joins this worker thread (via ``DeferredCommitter.stop()``) before releasing
the writer lease, any journal-file contention at exactly the wrong moment
(very plausible right after a write burst, which is exactly when a stop is
also likely to be attempted) stalls the whole stop path for up to 5 real
seconds, once per retry.

The fix: these bookkeeping stamps now carry a short bounded deadline
(``_BOOKKEEPING_DEADLINE_SECONDS``, 2 s). Nothing about the user's data is at
risk from giving up early here -- the remember itself is already committed;
on a bookkeeping failure the entry simply stays "prepared"/"dispatched" and
is retried (by this same worker, with backoff) or, if the process stops
first, redone idempotently by ``replay_pending`` at the next start (the
coordinator dedupes by journal id, so a replayed stamp costs nothing).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from superlocalmemory.core.deferred_admission import (
    _BOOKKEEPING_DEADLINE_SECONDS,
    DeferredCommitter,
)
from superlocalmemory.storage.admission_journal import (
    Actor,
    AdmissionJournal,
    RememberRequest,
)
from superlocalmemory.storage.journal_writer import _UNBOUNDED_BUSY_SECONDS


@dataclass(frozen=True)
class _TestCodec:
    prefix: bytes = b"h2-stop-bound:"

    def encrypt(self, plaintext: bytes) -> bytes:
        return self.prefix + plaintext[::-1]

    def decrypt(self, ciphertext: bytes) -> bytes:
        assert ciphertext.startswith(self.prefix)
        return ciphertext[len(self.prefix):][::-1]


_ACTOR = Actor("daemon:test", frozenset({"default"}), frozenset({"personal"}))


def _request(key: str) -> RememberRequest:
    return RememberRequest(
        content=f"H2 stop-bound evidence {key}.",
        profile_id="default",
        source_type="test",
        idempotency_key=f"h2-stop-bound:{key}",
    )


def test_stop_does_not_ride_out_the_unbounded_busy_wait(tmp_path: Path) -> None:
    """A journal-file lock held across a deferred commit's bookkeeping stamp
    must not block the worker thread (and therefore ``DeferredCommitter.stop()``)
    for the writer's full unbounded busy window."""
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    entry = journal.prepare(_request("one"), _ACTOR)

    committed = threading.Event()

    def _commit(entry_, request) -> dict:
        committed.set()  # proves the real commit already happened
        return {"operation_id": "op-1", "fact_ids": ["f1"], "commit_sequence": 1}

    committer = DeferredCommitter(journal, _commit)

    # Hold BEGIN IMMEDIATE on the journal's own file for longer than the old
    # unbounded busy window, so the pre-fix code path (no deadline on the
    # bookkeeping stamp) is forced to ride out the full 5 s wait before its
    # first retry -- a short hold would let BEGIN IMMEDIATE succeed as soon
    # as the lock clears, which would not distinguish the bug from the fix.
    blocker = sqlite3.connect(journal.path)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        committer.defer(entry)
        assert committed.wait(timeout=2.0), "the real commit never ran"
        committer.stop()  # bounded: see the assertions below
        elapsed = time.monotonic() - started
    finally:
        blocker.rollback()
        blocker.close()

    # The bookkeeping stamp must give up near its own short deadline, not ride
    # out the writer's full unbounded busy window. One deadline here is about
    # 2.2 s; a slow CI runner overshoots SQLite's busy-handler sleeps to about
    # 3.9 s, so the slack is 2.5 s. The bug itself is caught by the assertion
    # below (never as slow as the old unbounded wait).
    assert elapsed < _BOOKKEEPING_DEADLINE_SECONDS + 2.5, (
        f"stop() took {elapsed:.2f}s -- still riding out the unbounded "
        f"({_UNBOUNDED_BUSY_SECONDS}s) busy wait"
    )
    assert elapsed < _UNBOUNDED_BUSY_SECONDS, (
        f"stop() took {elapsed:.2f}s, not faster than the old unbounded "
        f"{_UNBOUNDED_BUSY_SECONDS}s ceiling it is supposed to beat"
    )

    # No loss: the entry is still durable (not silently dropped) and, now
    # that the lock is free, a fresh attempt commits it for good.
    journal.close()
    journal2 = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    try:
        reread = journal2.get(entry.journal_id)
        assert reread.state in {"prepared", "dispatched", "committed"}
    finally:
        journal2.close()
