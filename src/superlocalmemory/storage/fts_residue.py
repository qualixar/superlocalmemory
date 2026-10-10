# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Deleted words must leave the keyword index, not just stop matching.

An FTS5 delete only records a "deleted" marker; the term itself stays in the
index segments until they are merged. Measured through a real daemon: after a
memory was fully erased, ``atomic_facts_fts_data`` still held its unique word,
readable by anyone who opens the file, though no query could match it.

Two fixes, both idempotent:

* ``ensure_secure_delete``: FTS5's ``secure-delete`` option (SQLite 3.42+)
  makes every later delete remove the term from the index at once. It is a
  persistent setting stored in the index itself, so it covers every connection
  and process. On an older SQLite the option is unknown; that is reported, not
  raised, and the repair's ``optimize`` remains the way to purge.
  It is also left off on SQLite 3.44.0 up to (not including) 3.46.1, where
  ``secure-delete`` corrupts the index: after a content update (an FTS
  'delete' plus insert) ``PRAGMA quick_check`` reports "malformed inverted
  index for FTS5 table main.atomic_facts_fts" and later SQLite versions read
  it as real corruption. Upgrade SQLite or Python to get the immediate purge.
  An index that already has the option on there (a store made before this was
  known) gets it turned off at the next start and is reported "disabled" once;
  if its index is already malformed a single warning points to
  ``slm db repair``, which rebuilds it from the stored memories.
* ``purge_deleted_terms``: ``optimize`` rewrites the index without the words of
  rows deleted before ``secure-delete`` was on (the existing-store repair).
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

#: The keyword indexes that hold memory text.
FTS_TABLES: tuple[str, ...] = ("atomic_facts_fts", "fact_expansion_fts")
#: FTS5 learned ``secure-delete`` in SQLite 3.42. Older builds (Ubuntu 22.04: 3.37.2)
#: answer the attempt with a bare "SQL logic error" (GitHub #153).
SECURE_DELETE_MIN_SQLITE = (3, 42, 0)
#: Half-open range ``[first broken, first fixed)``. 3.44.0 and 3.44.1 are untested
#: and treated as broken to be safe; 3.43.1 and 3.46.1 were measured clean.
SECURE_DELETE_BROKEN_SQLITE = ((3, 44, 0), (3, 46, 1))
_old_sqlite_reported = False


def secure_delete_supported(version: tuple[int, int, int] | None = None) -> bool:
    """True when FTS5 ``secure-delete`` exists and is safe in this SQLite."""
    version = tuple(sqlite3.sqlite_version_info if version is None else version)
    if version < SECURE_DELETE_MIN_SQLITE:
        return False
    first_broken, first_fixed = SECURE_DELETE_BROKEN_SQLITE
    return not first_broken <= version < first_fixed


def _report_unsupported_once() -> None:
    """One INFO line naming why, with the fallback; never a warning."""
    global _old_sqlite_reported
    if _old_sqlite_reported:
        return
    _old_sqlite_reported = True
    if sqlite3.sqlite_version_info < SECURE_DELETE_MIN_SQLITE:
        why = "is older than 3.42"
    else:
        why = ("has a known FTS5 secure-delete corruption (3.44.0 to 3.46.0); "
               "upgrade SQLite or Python")
    logger.info("SQLite %s %s: deleted words leave the keyword index at the next "
                "'slm db repair' instead of at once", sqlite3.sqlite_version, why)


def _run(target: Any, sql: str, params: tuple = ()) -> list:
    result = target.execute(sql, params)
    return list(result.fetchall() if hasattr(result, "fetchall") else result)


def _exists(target: Any, table: str) -> bool:
    return bool(_run(target, "SELECT 1 FROM sqlite_master WHERE name = ?", (table,)))


def secure_delete_on(target: Any, table: str) -> bool:
    """True when the index already removes deleted terms at once."""
    rows = _run(target, f"SELECT v FROM {table}_config WHERE k = 'secure-delete'")  # noqa: S608
    return bool(rows) and str(tuple(rows[0])[0]) == "1"


def _switch_off_if_broken(target: Any, table: str) -> str:
    """Under a SQLite that corrupts the index with the option on: turn it off.

    Only the range from 3.42 up can have it on. Never raises: a store must open.
    """
    if sqlite3.sqlite_version_info < SECURE_DELETE_MIN_SQLITE:
        return "unsupported"
    try:
        if not secure_delete_on(target, table):
            return "unsupported"
        _run(target, f"INSERT INTO {table}({table}, rank) VALUES('secure-delete', 0)")  # noqa: S608
    except Exception as exc:
        logger.warning("keyword index %s secure delete could not be turned off: %s",
                       table, exc)
        return "unsupported"
    if keyword_index_damaged(target, table):
        logger.warning("keyword index %s is damaged (a known SQLite %s problem); run "
                       "'slm db repair' to rebuild it from your memories; nothing is lost",
                       table, sqlite3.sqlite_version)
    return "disabled"


def keyword_index_damaged(target: Any, table: str) -> bool:
    """True when FTS5's own integrity check finds the index malformed."""
    try:
        _run(target, f"INSERT INTO {table}({table}) VALUES('integrity-check')")  # noqa: S608
    except sqlite3.DatabaseError:  # includes IntegrityError and "malformed"
        return True
    return False


def rebuild_keyword_index(target: Any, table: str) -> None:
    """Rebuild one index from its content table (``atomic_facts``); no memory is lost."""
    _run(target, f"INSERT INTO {table}({table}) VALUES('rebuild')")  # noqa: S608


def ensure_secure_delete(target: Any) -> dict[str, str]:
    """Turn ``secure-delete`` on for every keyword index. Cheap when already on.

    ``target`` is a ``sqlite3.Connection`` or a ``DatabaseManager``. Returns
    ``{table: "on" | "enabled" | "disabled" | "absent" | "unsupported"}``.
    "disabled": this SQLite corrupts the index when the option is on (see the
    module docstring) and it was on, so it was just turned off. The next call
    reports "unsupported".
    """
    state: dict[str, str] = {}
    supported = secure_delete_supported()
    for table in FTS_TABLES:
        if not _exists(target, table):
            state[table] = "absent"
            continue
        if not supported:
            # A known limit with a fallback: ``slm db repair`` purges deleted words.
            _report_unsupported_once()
            state[table] = _switch_off_if_broken(target, table)
            continue
        if secure_delete_on(target, table):
            state[table] = "on"
            continue
        try:
            _run(target, f"INSERT INTO {table}({table}, rank) "  # noqa: S608
                 "VALUES('secure-delete', 1)")
            state[table] = "enabled"
        except Exception as exc:  # SQLite older than 3.42
            if "secure-delete" not in str(exc).lower() and "unknown" not in str(exc).lower():
                raise
            logger.warning("keyword index %s cannot purge deleted words at once: %s",
                           table, exc)
            state[table] = "unsupported"
    return state


def enable_quietly(conn: Any) -> None:
    """At store creation: never stops a store from opening."""
    try:
        ensure_secure_delete(conn)
    except Exception as exc:
        logger.warning("keyword index secure delete not enabled: %s", exc)


def purge_deleted_terms(conn: Any, table: str) -> None:
    """Rewrite one index without the words of already-deleted rows."""
    _run(conn, f"INSERT INTO {table}({table}) VALUES('optimize')")  # noqa: S608


__all__ = [
    "FTS_TABLES", "SECURE_DELETE_BROKEN_SQLITE", "enable_quietly",
    "ensure_secure_delete", "keyword_index_damaged", "purge_deleted_terms",
    "rebuild_keyword_index", "secure_delete_on", "secure_delete_supported",
]
