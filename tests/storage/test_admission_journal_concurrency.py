# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Concurrency contracts for the daemon-owned admission journal."""

from __future__ import annotations

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import pytest

from superlocalmemory.storage.admission_journal import (
    Actor,
    AdmissionJournal,
    AdmissionJournalUnavailable,
    RememberRequest,
)


@dataclass(frozen=True)
class _TestCodec:
    prefix: bytes = b"journal-concurrency:"

    def encrypt(self, plaintext: bytes) -> bytes:
        return self.prefix + plaintext[::-1]

    def decrypt(self, ciphertext: bytes) -> bytes:
        assert ciphertext.startswith(self.prefix)
        return ciphertext[len(self.prefix) :][::-1]


class _TransactionProbe:
    """Observe overlapping BEGIN attempts without changing SQLite semantics."""

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._second_begin_attempted = threading.Event()
        self._first_transaction = True
        self.active_transactions = 0
        self.max_active_transactions = 0

    def wrap(self, connection: Any) -> "_ObservedConnection":
        return _ObservedConnection(connection, self)

    def begin(self, connection: Any, sql: str, parameters: tuple[Any, ...]) -> Any:
        with self._state_lock:
            is_first = self._first_transaction
            self._first_transaction = False
            self.active_transactions += 1
            self.max_active_transactions = max(
                self.max_active_transactions,
                self.active_transactions,
            )
            if not is_first:
                self._second_begin_attempted.set()
        try:
            result = connection.execute(sql, parameters)
        except BaseException:
            self.finish()
            raise
        if is_first:
            self._second_begin_attempted.wait(timeout=0.25)
        return result

    def finish(self) -> None:
        with self._state_lock:
            self.active_transactions -= 1


class _ObservedConnection:
    def __init__(self, connection: Any, probe: _TransactionProbe) -> None:
        self._connection = connection
        self._probe = probe
        self._transaction_open = False

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> Any:
        if sql == "BEGIN IMMEDIATE":
            result = self._probe.begin(self._connection, sql, parameters)
            self._transaction_open = True
            return result
        return self._connection.execute(sql, parameters)

    def commit(self) -> None:
        try:
            self._connection.commit()
        finally:
            self._finish()

    def rollback(self) -> None:
        try:
            self._connection.rollback()
        finally:
            self._finish()

    def _finish(self) -> None:
        if self._transaction_open:
            self._transaction_open = False
            self._probe.finish()


def test_concurrent_prepare_serializes_journal_write_transactions(
    tmp_path,
    monkeypatch,
) -> None:
    """Parallel remember calls never race BEGIN on admission_journal.db."""
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    actor = Actor(
        "daemon:test",
        frozenset({"default"}),
        frozenset({"personal"}),
    )
    original_connection = journal._connection
    probe = _TransactionProbe()

    @contextmanager
    def observed_connection(*, timeout: float = 1.0):
        with original_connection(timeout=timeout) as connection:
            yield probe.wrap(connection)

    monkeypatch.setattr(journal, "_connection", observed_connection)

    def prepare(sequence: int) -> None:
        journal.prepare(
            RememberRequest(
                content=f"Concurrent journal evidence {sequence}.",
                profile_id="default",
                source_type="test",
                idempotency_key=f"journal-concurrency:{sequence}",
            ),
            actor,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(prepare, sequence) for sequence in range(2)]
        for future in futures:
            future.result(timeout=3.0)

    assert probe.max_active_transactions == 1
    assert journal.count() == 2


def test_prepare_deadline_bounds_process_lock_wait(tmp_path, monkeypatch) -> None:
    """A queued journal mutation fails with a typed error inside its budget."""
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    actor = Actor("daemon:test", frozenset({"default"}), frozenset({"personal"}))
    observed_timeouts: list[float] = []

    class UnavailableRecordingLock:
        def acquire(self, *, timeout: float = -1.0) -> bool:
            observed_timeouts.append(timeout)
            return False

        def release(self) -> None:
            raise AssertionError("an unacquired journal lock must not be released")

    monkeypatch.setattr(journal, "_write_lock", UnavailableRecordingLock())

    with pytest.raises(AdmissionJournalUnavailable, match="deadline"):
        journal.prepare(
            RememberRequest(
                content="A bounded process lock wait.",
                profile_id="default",
                source_type="test",
                idempotency_key="journal-deadline:process-lock",
            ),
            actor,
            deadline=time.monotonic() + 0.05,
        )

    assert len(observed_timeouts) == 1
    assert 0 < observed_timeouts[0] <= 0.05
    assert journal.count() == 0


def test_prepare_deadline_caps_external_sqlite_busy_wait(tmp_path, monkeypatch) -> None:
    """A foreign SQLite writer cannot force remember past its journal budget."""
    path = tmp_path / "admission_journal.db"
    journal = AdmissionJournal(path, codec=_TestCodec())
    actor = Actor("daemon:test", frozenset({"default"}), frozenset({"personal"}))
    blocker = sqlite3.connect(path)
    blocker.execute("BEGIN IMMEDIATE")
    original_connection = journal._connection
    observed_busy_timeout_ms: list[int] = []

    @contextmanager
    def observed_connection(*, timeout: float = 1.0):
        with original_connection(timeout=timeout) as connection:
            observed_busy_timeout_ms.append(
                int(connection.execute("PRAGMA busy_timeout").fetchone()[0])
            )
            yield connection

    monkeypatch.setattr(journal, "_connection", observed_connection)
    try:
        with pytest.raises(AdmissionJournalUnavailable, match="busy"):
            journal.prepare(
                RememberRequest(
                    content="A bounded external SQLite wait.",
                    profile_id="default",
                    source_type="test",
                    idempotency_key="journal-deadline:sqlite",
                ),
                actor,
                deadline=time.monotonic() + 0.05,
            )
    finally:
        blocker.rollback()
        blocker.close()

    assert len(observed_busy_timeout_ms) == 1
    assert 1 <= observed_busy_timeout_ms[0] <= 50
    assert journal.count() == 0


def test_request_read_translates_sqlite_busy_to_typed_unavailable(
    tmp_path,
    monkeypatch,
) -> None:
    """The encrypted command read cannot leak a raw SQLite lock error."""
    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_TestCodec())
    actor = Actor("daemon:test", frozenset({"default"}), frozenset({"personal"}))
    prepared = journal.prepare(
        RememberRequest(
            content="A typed busy error protects the journal read.",
            profile_id="default",
            source_type="test",
            idempotency_key="journal-deadline:read",
        ),
        actor,
    )
    original_connection = journal._connection

    class BusyReadConnection:
        def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> Any:
            if sql.startswith("SELECT command_json"):
                raise sqlite3.OperationalError("database is locked")
            raise AssertionError(f"unexpected SQL: {sql}")

    @contextmanager
    def busy_connection(*, timeout: float = 1.0):
        with original_connection(timeout=timeout):
            yield BusyReadConnection()

    monkeypatch.setattr(journal, "_connection", busy_connection)

    with pytest.raises(AdmissionJournalUnavailable, match="busy"):
        journal.request_for(prepared, deadline=time.monotonic() + 0.05)
