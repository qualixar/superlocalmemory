# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""Physically read-only SQLite snapshots for canonical ``memory.db``.

The factory is deliberately small: a query path must receive a connection
which SQLite itself refuses to mutate.  This is the storage half of the
Command-Query Separation contract; higher layers must not use it to perform
telemetry, access tracking, or any other side effect.
"""
from __future__ import annotations

import sqlite3
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Iterator


class ReadConnectionError(RuntimeError):
    """Raised when a read snapshot cannot be opened safely."""


class ReadConnectionFactory:
    """Open short-lived query-only snapshots of one canonical database path."""

    def __init__(self, memory_db: str | Path, timeout_ms: int = 250) -> None:
        if timeout_ms < 0:
            raise ValueError("timeout_ms must be greater than or equal to zero")
        self._memory_db = Path(memory_db).expanduser().resolve()
        self._timeout_ms = timeout_ms

    @property
    def memory_db(self) -> Path:
        """Return the resolved canonical database path."""
        return self._memory_db

    @contextmanager
    def snapshot(self) -> Iterator[sqlite3.Connection]:
        """Yield a SQLite connection that cannot issue a write statement.

        ``immutable=1`` is intentionally not used: a live WAL database may
        have valid state in its WAL file.  ``mode=ro`` retains normal WAL
        visibility while prohibiting a writable open, and ``query_only`` is a
        second SQLite-enforced guard against accidental DML or DDL.
        """
        if not self._memory_db.exists():
            raise ReadConnectionError(f"canonical database does not exist: {self._memory_db}")

        uri = f"{self._memory_db.as_uri()}?mode=ro"
        try:
            conn = sqlite3.connect(
                uri,
                uri=True,
                timeout=self._timeout_ms / 1000.0,
            )
        except sqlite3.Error as exc:
            raise ReadConnectionError(
                f"could not open read-only snapshot for {self._memory_db}"
            ) from exc

        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(f"PRAGMA busy_timeout={self._timeout_ms}")
            yield conn
        finally:
            conn.close()

    def open(self) -> "ReadConnectionLease":
        """Return a manually-closeable read-only lease for legacy query code.

        New code should use :meth:`snapshot`; the lease is a small compatibility
        bridge for client paths that still use ``try/finally: conn.close()``.
        Closing it exits this same context manager, preserving both ``mode=ro``
        and ``PRAGMA query_only`` rather than reopening SQLite directly.
        """
        return ReadConnectionLease(self.snapshot())


class ReadConnectionLease:
    """A legacy-compatible facade around one read-only snapshot."""

    def __init__(self, snapshot: AbstractContextManager[sqlite3.Connection]) -> None:
        object.__setattr__(self, "_snapshot", snapshot)
        object.__setattr__(self, "_connection", snapshot.__enter__())
        object.__setattr__(self, "_closed", False)

    def __getattr__(self, name: str):
        return getattr(self._connection, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._connection, name, value)

    def __enter__(self) -> sqlite3.Connection:
        return self._connection

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._close(exc_type, exc, traceback)

    def close(self) -> None:
        """Close this lease once, including its context-manager cleanup."""
        self._close(None, None, None)

    def _close(self, exc_type, exc, traceback) -> None:
        if not self._closed:
            object.__setattr__(self, "_closed", True)
            self._snapshot.__exit__(exc_type, exc, traceback)


__all__ = ["ReadConnectionError", "ReadConnectionFactory", "ReadConnectionLease"]
