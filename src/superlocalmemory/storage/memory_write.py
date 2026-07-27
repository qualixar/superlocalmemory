# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""Canonical short-lived WRITE / READ connections for memory.db.

Why this exists
---------------
memory.db is a WAL SQLite database with exactly one writer allowed at any
instant.  Historically many subsystems opened their own bare
``sqlite3.connect(...)`` and wrote without any coordination, which produced
two failure modes under real multi-agent load:

1. **In-process races** — daemon worker threads racing each other at the WAL
   layer, retrying ``SQLITE_BUSY`` until they time out.
2. **Cross-process races** — Claude Code hooks and the ``slm ingest`` CLI run
   as *separate OS processes* and write memory.db directly.  An in-process
   ``threading`` lock cannot serialise those; only SQLite's own
   ``PRAGMA busy_timeout`` makes them WAIT for the writer instead of erroring.

``memory_write()`` closes BOTH gaps with one helper that every writer uses:

* Acquires the process-level :func:`get_write_lock` (the OUTERMOST lock — see
  ``write_lock.py``) so in-process writers serialise cleanly with **no**
  ``SQLITE_BUSY`` spin.
* Opens the connection with ``PRAGMA busy_timeout`` so that a *different
  process* (hook / CLI) writing at the same instant WAITS rather than failing.
* Commits on success, rolls back on error, always closes — a short critical
  section.

The single hard rule for callers
---------------------------------
**Never hold this connection across a slow operation** — no embedding /
network call / large unbounded transaction inside the ``with`` block.  Do the
slow work first, then open ``memory_write()`` only for the fast INSERT/UPDATE.
Holding the writer across a multi-second op is exactly what starves everyone
else (SQLITE_BUSY after retries).  Batch large writes into bounded chunks that
each commit quickly.

Reads
-----
Use :func:`memory_read` for read-only access.  WAL allows concurrent readers
without blocking the writer, so reads do NOT take the write lock.  The helper
opens SQLite with ``mode=ro`` and ``PRAGMA query_only=ON``; it uses the
3.8.6 read-path budget of at most 250ms during a brief checkpoint window.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from superlocalmemory.storage.read_connection import ReadConnectionFactory
from superlocalmemory.storage.write_lock import get_write_lock


def _busy_timeout_ms() -> int:
    """Busy-timeout in ms, env-overridable, matching DatabaseManager's default.

    Kept in sync with ``storage/database.py::_BUSY_TIMEOUT_MS`` (env
    ``SLM_DB_BUSY_TIMEOUT_MS``, default 10_000) so every memory.db connection
    in the process — and in hook/CLI child processes that import this — waits
    the same amount for the single writer.
    """
    try:
        return max(0, int(os.environ.get("SLM_DB_BUSY_TIMEOUT_MS", "10000")))
    except (TypeError, ValueError):
        return 10000


@contextmanager
def memory_write(db_path: str | Path) -> Generator[sqlite3.Connection, None, None]:
    """Yield a serialised, busy-timeout-guarded WRITE connection to memory.db.

    Acquires the process write lock (outermost), opens a short-lived
    connection with ``busy_timeout``, commits on success / rolls back on
    error, and always closes.

    HARD RULE: keep the ``with`` block short — never embed / call the network /
    run an unbounded transaction while holding it.
    """
    ms = _busy_timeout_ms()
    lock = get_write_lock(db_path)
    with lock:
        conn = sqlite3.connect(str(db_path), timeout=ms / 1000.0)
        try:
            conn.execute(f"PRAGMA busy_timeout={ms}")
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            conn.close()


@contextmanager
def memory_read(db_path: str | Path) -> Generator[sqlite3.Connection, None, None]:
    """Yield a physically read-only connection with no write lock.

    WAL permits concurrent readers, so this deliberately does NOT take the
    write lock.  A 250ms bounded wait protects the query path from a brief
    checkpoint without converting a dashboard recall into a long hang.
    """
    with ReadConnectionFactory(db_path, timeout_ms=min(_busy_timeout_ms(), 250)).snapshot() as conn:
        yield conn


__all__ = ["memory_write", "memory_read"]
