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
  ``slm db repair``, which turns the setting off first (also when an earlier
  repair had already purged the index), rebuilds the index from the stored
  memories, checks it again, and recreates the table if the rebuild was not
  enough (GitHub #204).
* ``purge_deleted_terms``: ``optimize`` rewrites the index without the words of
  rows deleted before ``secure-delete`` was on (the existing-store repair).
"""

from __future__ import annotations

import logging
import re
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


def _switch_off_if_broken(target: Any, table: str, *, warn: bool = True) -> str:
    """Under a SQLite that corrupts the index with the option on: turn it off.

    Only the range from 3.42 up can have it on. Never raises: a store must open.
    ``warn=False`` is for ``slm db repair``, which is about to rebuild the index.
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
    if warn and keyword_index_damaged(target, table):
        logger.warning("keyword index %s is damaged (a known SQLite %s problem); run "
                       "'slm db repair' to rebuild it from your memories; nothing is lost",
                       table, sqlite3.sqlite_version)
    return "disabled"


def keyword_index_damaged(target: Any, table: str) -> bool:
    """True when FTS5's own integrity check finds the index malformed.

    Only a corruption error counts. A read-only connection (``slm db health``)
    cannot run the integrity-check command at all; there the database-wide
    ``quick_check``, which includes FTS5 indexes on the SQLite versions that can
    damage them, is read for this table instead. Any other error is not proof
    of damage.
    """
    try:
        _run(target, f"INSERT INTO {table}({table}) VALUES('integrity-check')")  # noqa: S608
    except sqlite3.DatabaseError as exc:
        name = getattr(exc, "sqlite_errorname", "") or ""
        if name.startswith("SQLITE_CORRUPT"):
            return True
        if name == "SQLITE_READONLY":
            return _quick_check_names(target, table)
        logger.debug("keyword index %s integrity check did not run: %s", table, exc)
    return False


def _quick_check_names(target: Any, table: str) -> bool:
    """Whether ``PRAGMA quick_check`` reports this FTS5 table malformed."""
    try:
        rows = _run(target, "PRAGMA quick_check")
    except sqlite3.DatabaseError:
        return False
    markers = (f"fts5 table main.{table}".lower(), f'from table "{table}"'.lower())
    return any(m in str(tuple(row)[0]).lower() for row in rows for m in markers)


def rebuild_keyword_index(target: Any, table: str) -> None:
    """Rebuild one index from its content table (``atomic_facts``); no memory is lost."""
    _run(target, f"INSERT INTO {table}({table}) VALUES('rebuild')")  # noqa: S608


def disable_where_damaging(target: Any) -> dict[str, str]:
    """Turn ``secure-delete`` off in every keyword index when this SQLite damages
    the index with it on (3.44.0 up to, not including, 3.46.1). Nothing else is touched.

    ``slm db repair`` calls this before it rebuilds an index: a rebuild with the
    setting still on is damaged again by the next edit (GitHub #204). Returns
    ``{table: "disabled" | "unsupported"}`` for the indexes that exist; empty on a
    SQLite where the setting is safe.
    """
    if secure_delete_supported() or sqlite3.sqlite_version_info < SECURE_DELETE_MIN_SQLITE:
        return {}
    return {table: _switch_off_if_broken(target, table, warn=False)
            for table in FTS_TABLES if _exists(target, table)}


def _index_columns(target: Any, table: str) -> list[str]:
    return [str(row[1]) for row in _run(target, f"PRAGMA table_info({table})")]


def _is_external_content(ddl: str) -> bool:
    """``content='atomic_facts'``: the text lives in another table, not in the index."""
    return re.search(r"content\s*=\s*['\"][^'\"]+['\"]", ddl, re.IGNORECASE) is not None


def recreate_keyword_index(target: Any, table: str) -> None:
    """Drop one keyword index and create it again from the stored text.

    The fallback for an index that a ``rebuild`` could not repair. An index whose
    text lives in another table (``atomic_facts_fts``) is refilled from it. A
    standalone index (``fact_expansion_fts``) is its own text, so its rows are
    read out before the drop and put back after it. All or nothing: a failure
    leaves the old table as it was. ``secure-delete`` is off in the new table.
    """
    rows = _run(target, "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,))
    if not rows or not rows[0][0]:
        raise sqlite3.OperationalError(f"no keyword index {table} to recreate")
    ddl = str(rows[0][0])
    external = _is_external_content(ddl)
    _run(target, "SAVEPOINT slm_recreate_keyword_index")
    try:
        columns = _index_columns(target, table)
        kept = [] if external else [tuple(r) for r in _run(
            target, f"SELECT rowid, {', '.join(columns)} FROM {table}")]  # noqa: S608
        _run(target, f"DROP TABLE {table}")
        _run(target, ddl)
        if external:
            _run(target, f"INSERT INTO {table}({table}) VALUES('rebuild')")  # noqa: S608
        else:
            marks = ", ".join("?" for _ in range(len(columns) + 1))
            for row in kept:
                _run(target, f"INSERT INTO {table}(rowid, {', '.join(columns)}) "  # noqa: S608
                     f"VALUES ({marks})", row)
    except BaseException:
        _run(target, "ROLLBACK TO slm_recreate_keyword_index")
        _run(target, "RELEASE slm_recreate_keyword_index")
        raise
    _run(target, "RELEASE slm_recreate_keyword_index")


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
    "FTS_TABLES", "SECURE_DELETE_BROKEN_SQLITE", "disable_where_damaging", "enable_quietly",
    "ensure_secure_delete", "keyword_index_damaged", "purge_deleted_terms",
    "rebuild_keyword_index", "recreate_keyword_index", "secure_delete_on",
    "secure_delete_supported",
]
