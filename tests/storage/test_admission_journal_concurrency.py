# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Concurrency contracts for the daemon-owned admission journal.

One writer thread group-commits journal mutations; reads use a bounded pool of
persistent connections. These tests pin what callers rely on: concurrent saves
share commits instead of queueing on a lock, a deadline is honoured and a
refused save is never written, a full queue is refused at once and honestly,
and a crash in the middle of a batch keeps all of it or none of it.
"""

from __future__ import annotations

import multiprocessing
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from superlocalmemory.storage import journal_writer
from superlocalmemory.storage.admission_journal import (
    Actor,
    AdmissionJournal,
    AdmissionJournalOverloaded,
    AdmissionJournalUnavailable,
    IdempotencyConflict,
    RememberRequest,
)
from superlocalmemory.storage.journal_writer import GroupCommitWriter


@dataclass(frozen=True)
class _TestCodec:
    prefix: bytes = b"journal-concurrency:"

    def encrypt(self, plaintext: bytes) -> bytes:
        return self.prefix + plaintext[::-1]

    def decrypt(self, ciphertext: bytes) -> bytes:
        assert ciphertext.startswith(self.prefix)
        return ciphertext[len(self.prefix) :][::-1]


_ACTOR = Actor("daemon:test", frozenset({"default"}), frozenset({"personal"}))


def _request(key: str, content: str | None = None) -> RememberRequest:
    return RememberRequest(
        content=content or f"Concurrent journal evidence {key}.",
        profile_id="default",
        source_type="test",
        idempotency_key=f"journal-concurrency:{key}",
    )


def _journal(tmp_path: Path) -> AdmissionJournal:
    return AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())


def _scratch_writer(tmp_path: Path, **kwargs: Any) -> GroupCommitWriter:
    path = tmp_path / "scratch.db"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS t(k TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()
    return GroupCommitWriter(path, **kwargs)


def _rows(path: Path) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {row[0] for row in conn.execute("SELECT k FROM t")}
    finally:
        conn.close()


def test_concurrent_prepares_share_commits_on_one_writer(tmp_path) -> None:
    """32 parallel saves all land, through batched commits, never racing BEGIN."""
    journal = _journal(tmp_path)
    batches: list[int] = []
    journal._writer.before_commit = lambda live: batches.append(len(live))
    try:
        with ThreadPoolExecutor(max_workers=32) as pool:
            futures = [
                pool.submit(journal.prepare, _request(str(i)), _ACTOR,
                            deadline=time.monotonic() + 5.0)
                for i in range(128)
            ]
            for future in futures:
                future.result(timeout=10.0)
        assert journal.count() == 128
        assert sum(batches) == 128
        assert len(batches) < 128, "every save paid its own commit"
        assert max(batches) <= journal_writer.MAX_BATCH
    finally:
        journal.close()


def test_a_save_opens_no_connection_in_steady_state(tmp_path, monkeypatch) -> None:
    """Persistent writer and reader connections: no per-save sqlite3.connect."""
    journal = _journal(tmp_path)
    receipt = {"operation_id": "op", "fact_ids": ["f"], "commit_sequence": 1}
    try:
        warm = journal.prepare(_request("warm"), _ACTOR)
        journal.mark_committed(warm.journal_id, receipt)
        connects: list[str] = []
        original = journal_writer.sqlite3.connect

        def counting_connect(*args, **kwargs):
            connects.append(str(args[0]))
            return original(*args, **kwargs)

        monkeypatch.setattr(journal_writer.sqlite3, "connect", counting_connect)
        for i in range(20):
            entry = journal.prepare(_request(f"steady-{i}"), _ACTOR)
            journal.mark_committed(entry.journal_id, receipt)
        assert connects == []
    finally:
        journal.close()


def test_one_failing_operation_does_not_spoil_its_batch(tmp_path) -> None:
    """Each operation runs in its own SAVEPOINT inside the shared transaction."""
    writer = _scratch_writer(tmp_path, linger_seconds=0.2)
    batches: list[int] = []
    writer.before_commit = lambda live: batches.append(len(live))

    def failing(conn):
        conn.execute("INSERT INTO t VALUES ('half-done')")
        raise IdempotencyConflict("refused after writing")

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            bad = pool.submit(writer.submit, failing)
            good = pool.submit(writer.submit, lambda c: c.execute("INSERT INTO t VALUES ('ok')"))
            with pytest.raises(IdempotencyConflict):
                bad.result(timeout=5.0)
            good.result(timeout=5.0)
        assert batches == [2]
        assert _rows(tmp_path / "scratch.db") == {"ok"}
    finally:
        writer.close()


def test_full_queue_is_refused_at_once_with_retry_after(tmp_path) -> None:
    writer = _scratch_writer(tmp_path, queue_cap=2)
    gate = threading.Event()
    entered_held = threading.Event()

    def held(conn):
        entered_held.set()
        gate.wait(5.0)
        conn.execute("INSERT INTO t VALUES ('held')")

    def _wait_for_queue_depth(depth: int, *, timeout: float = 2.0) -> None:
        """Poll the writer's own queue instead of guessing a sleep duration:
        the two `queued` threads below must actually be enqueued (not merely
        started) before the test submits the one that should overflow it."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with writer._cond:
                if len(writer._queue) >= depth:
                    return
            time.sleep(0.005)
        with writer._cond:
            actual = len(writer._queue)
        raise AssertionError(f"queue never reached depth {depth} (stuck at {actual})")

    try:
        first = threading.Thread(target=writer.submit, args=(held,))
        first.start()
        assert entered_held.wait(timeout=2.0), "writer never started executing held"
        queued = [
            threading.Thread(
                target=writer.submit,
                args=(lambda c, k=k: c.execute("INSERT INTO t VALUES (?)", (k,)),),
            )
            for k in ("q1", "q2")
        ]
        for thread in queued:
            thread.start()
        _wait_for_queue_depth(2)
        started = time.monotonic()
        with pytest.raises(AdmissionJournalOverloaded) as refused:
            writer.submit(lambda c: c.execute("INSERT INTO t VALUES ('over')"))
        # A liveness bound, not a latency benchmark: a full queue must refuse
        # synchronously (a length check + raise, see GroupCommitWriter.submit)
        # rather than block or poll, so this only needs enough headroom for
        # scheduler jitter on a shared machine -- consistent with the other
        # "refused without waiting" bounds in this file (0.2s, below).
        assert time.monotonic() - started < 0.2
        assert refused.value.retry_after_seconds >= 1
        gate.set()
        first.join(5.0)
        for thread in queued:
            thread.join(5.0)
        assert _rows(tmp_path / "scratch.db") == {"held", "q1", "q2"}
    finally:
        gate.set()
        writer.close()


def test_deadline_refusal_is_never_written_later(tmp_path) -> None:
    """A caller told 'not saved' at its deadline is never saved afterwards."""
    path = tmp_path / "admission_journal.db"
    journal = AdmissionJournal(path, codec=_TestCodec())
    blocker = sqlite3.connect(path)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        with pytest.raises(AdmissionJournalUnavailable, match="busy"):
            journal.prepare(_request("blocked"), _ACTOR, deadline=started + 0.05)
        # Ignoring the deadline means waiting out the writer's own busy wait
        # (5 s). A fifth of that proves the deadline was kept without asking a
        # shared CI machine to wake within 50 ms of being told to.
        assert time.monotonic() - started < journal_writer._UNBOUNDED_BUSY_SECONDS / 5
    finally:
        blocker.rollback()
        blocker.close()
    time.sleep(0.2)  # give the writer every chance to (wrongly) commit it
    assert journal.count() == 0
    # The journal is healthy again for the next caller.
    assert journal.prepare(_request("after"), _ACTOR).state == "prepared"
    journal.close()


def test_withdrawal_during_execution_reruns_batch_without_it(tmp_path) -> None:
    """Cancel before commit: a withdrawn operation is rolled out of its batch."""
    writer = _scratch_writer(tmp_path, linger_seconds=0.2)
    deadline = time.monotonic() + 1.0
    slow_runs = 0

    def slow(conn):
        # Still executing when its caller's deadline passes.
        nonlocal slow_runs
        slow_runs += 1
        conn.execute("INSERT INTO t VALUES ('slow')")
        time.sleep(max(0.0, deadline - time.monotonic()) + 0.2)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            keeper = pool.submit(
                writer.submit, lambda c: c.execute("INSERT INTO t VALUES ('keep')"),
            )
            quitter = pool.submit(writer.submit, slow, deadline=deadline)
            with pytest.raises(AdmissionJournalUnavailable):
                quitter.result(timeout=10.0)
            keeper.result(timeout=10.0)
        assert slow_runs == 1, "the withdrawal must have happened mid-execution"
        assert _rows(tmp_path / "scratch.db") == {"keep"}
    finally:
        writer.close()


def test_claimed_commit_reports_the_truth_past_the_deadline(tmp_path) -> None:
    """Once COMMIT is under way the caller waits for it, never a false 'no'."""
    writer = _scratch_writer(tmp_path)
    writer.before_commit = lambda _live: time.sleep(0.1)
    try:
        started = time.monotonic()
        writer.submit(
            lambda c: c.execute("INSERT INTO t VALUES ('late')"),
            deadline=started + 0.05,
        )
        assert _rows(tmp_path / "scratch.db") == {"late"}
    finally:
        writer.close()


def test_read_busy_is_a_typed_unavailable(tmp_path, monkeypatch) -> None:
    """A journal read cannot leak a raw SQLite lock error."""
    journal = _journal(tmp_path)
    prepared = journal.prepare(_request("read"), _ACTOR)

    class BusyReadConnection:
        in_transaction = False

        def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> Any:
            if sql.startswith("PRAGMA busy_timeout"):
                return None
            raise sqlite3.OperationalError("database is locked")

        def close(self) -> None:
            pass

    monkeypatch.setattr(journal._readers, "_acquire", lambda _deadline: BusyReadConnection())
    with pytest.raises(AdmissionJournalUnavailable, match="busy"):
        journal.request_for(prepared, deadline=time.monotonic() + 0.05)
    journal.close()


def test_reader_pool_is_bounded_and_honours_the_deadline(tmp_path) -> None:
    journal = _journal(tmp_path)
    holders: list[Any] = []
    try:
        for _ in range(journal_writer.READ_POOL_SIZE):
            context = journal._read_connection()
            context.__enter__()
            holders.append(context)
        started = time.monotonic()
        with pytest.raises(AdmissionJournalUnavailable):
            with journal._read_connection(deadline=started + 0.05):
                pass
        assert time.monotonic() - started < 0.2
    finally:
        for context in holders:
            context.__exit__(None, None, None)
        journal.close()


# -- crash in the middle of a batch ------------------------------------------

_CRASH_BATCH = 8


def _crash_child(path: str, when: str) -> None:
    import os
    import signal

    journal = AdmissionJournal(Path(path), codec=_TestCodec())
    journal._writer._linger = 0.5  # every save below lands in one batch

    def kill(live) -> None:
        if len(live) == _CRASH_BATCH:
            os.kill(os.getpid(), signal.SIGKILL)

    if when == "before":
        journal._writer.before_commit = kill
    else:
        journal._writer.after_commit = kill
    threads = [
        threading.Thread(target=journal.prepare, args=(_request(f"crash-{i}"), _ACTOR))
        for i in range(_CRASH_BATCH)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10.0)
    os._exit(3)  # the batch never reached the kill: the test must fail


@pytest.mark.parametrize(("when", "expected"), [("before", 0), ("after", _CRASH_BATCH)])
def test_kill_mid_batch_keeps_all_of_it_or_none(tmp_path, when, expected) -> None:
    import signal

    path = tmp_path / "admission_journal.db"
    AdmissionJournal(path, codec=_TestCodec()).close()
    child = multiprocessing.get_context("spawn").Process(
        target=_crash_child, args=(str(path), when),
    )
    child.start()
    child.join(30.0)
    assert child.exitcode == -signal.SIGKILL
    survivor = AdmissionJournal(path, codec=_TestCodec())
    try:
        assert survivor.count() == expected
    finally:
        survivor.close()
