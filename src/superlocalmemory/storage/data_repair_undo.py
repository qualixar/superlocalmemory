# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The undo record of the automatic data repairs, and how to play it back.

4.2.8 repairs data on its first start with no command (M065 made-up dates, M066
wrongly archived memories). Every row such a repair changes or removes is
written here first, in the repair's own transaction, so the record and the
change go together or not at all.

One row per changed row: the table, the key, the values before and after (a
removed row has no "after"). :func:`undo` plays a repair back newest first and
leaves alone any row changed since the repair (it is counted, never
overwritten). :func:`summary` is what ``slm doctor`` shows.

ERASURE
Rows quote memory text, so they go with the memory: ``fact_id`` cascades from
``atomic_facts`` and is in ``fact_dependents.DERIVED_FACT_ROWS`` for delete paths
that run without foreign keys; ``profile_id`` lets a profile erasure find them.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

TABLE = "data_repair_undo"

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    repair      TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    fact_id     TEXT NOT NULL REFERENCES atomic_facts (fact_id) ON DELETE CASCADE,
    profile_id  TEXT NOT NULL,
    target      TEXT NOT NULL,
    key_column  TEXT NOT NULL,
    key_value   TEXT NOT NULL,
    before_json TEXT NOT NULL,
    after_json  TEXT,
    created_at  TEXT NOT NULL,
    undone_at   TEXT,
    PRIMARY KEY (repair, seq)
);
CREATE INDEX IF NOT EXISTS idx_data_repair_undo_fact ON {TABLE} (fact_id);
"""

#: Tables a repair may change, and the column that names one row. Undo touches
#: nothing else, so a damaged record cannot be turned into an arbitrary write.
KEYS: dict[str, str] = {
    "atomic_facts": "fact_id",
    "temporal_events": "event_id",
    "fact_temporal_validity": "fact_id",
    "bm25_tokens": "fact_id",
    "fact_context": "fact_id",
    "fact_retention": "fact_id",
}


def ensure_table(conn: sqlite3.Connection) -> None:
    for statement in DDL.split(";"):
        if statement.strip():
            conn.execute(statement)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def has_table(conn: sqlite3.Connection, name: str = TABLE) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (name,)).fetchone() is not None


class UndoWriter:
    """Records changes for one repair; call inside the repair's transaction."""

    def __init__(self, conn: sqlite3.Connection, repair: str) -> None:
        ensure_table(conn)
        self._conn = conn
        self._repair = repair
        self._seq = int(conn.execute(f"SELECT COALESCE(MAX(seq), 0) FROM {TABLE} "  # noqa: S608
                                     "WHERE repair = ?", (repair,)).fetchone()[0])

    def record(self, target: str, key: str, fact_id: str, profile_id: str,
               before: dict[str, Any], after: dict[str, Any] | None) -> None:
        """``after`` is None when the row is removed; ``before`` is then the whole row."""
        if target not in KEYS:
            raise ValueError(f"undo record cannot name table {target!r}")
        self._seq += 1
        self._conn.execute(
            f"INSERT INTO {TABLE} (repair, seq, fact_id, profile_id, target, key_column, "  # noqa: S608
            "key_value, before_json, after_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (self._repair, self._seq, fact_id, profile_id, target, KEYS[target], key,
             json.dumps(before, default=_blob), None if after is None
             else json.dumps(after, default=_blob), _now()))


def _blob(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__hex__": bytes(value).hex()}
    raise TypeError(f"cannot record {type(value).__name__}")


def _value(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__hex__"}:
        return bytes.fromhex(value["__hex__"])
    return value


def _current(conn: sqlite3.Connection, target: str, key_col: str, key: str,
             cols: list[str]) -> tuple | None:
    return conn.execute(f"SELECT {','.join(cols)} FROM {target} WHERE {key_col} = ?",  # noqa: S608
                        (key,)).fetchone()


def _undo_update(conn, target, key_col, key, before, after) -> bool:
    cols = list(after)
    current = _current(conn, target, key_col, key, cols)
    if current is None or list(current) != [_value(after[c]) for c in cols]:
        return False  # changed (or removed) since the repair: leave it
    sets = ",".join(f"{c} = ?" for c in before)
    conn.execute(f"UPDATE {target} SET {sets} WHERE {key_col} = ?",  # noqa: S608
                 (*[_value(v) for v in before.values()], key))
    if target == "atomic_facts" and "content" in before:
        conn.execute("UPDATE atomic_facts SET embedding = NULL WHERE fact_id = ?", (key,))
    return True


def _undo_delete(conn, target, key_col, key, before) -> bool:
    if _current(conn, target, key_col, key, [key_col]) is not None:
        return False  # something new took the key since: leave it
    cols = list(before)
    conn.execute(f"INSERT INTO {target} ({','.join(cols)}) "  # noqa: S608
                 f"VALUES ({','.join('?' * len(cols))})",
                 [_value(before[c]) for c in cols])
    return True


def undo(conn: sqlite3.Connection, repair: str) -> int:
    """Play ``repair`` back, newest change first. Returns the rows put back.

    One transaction. Rows already undone are not replayed, so a second call
    returns 0. A row changed since the repair is left as it is.
    """
    if not has_table(conn):
        return 0
    restored = 0
    conn.execute("BEGIN IMMEDIATE")
    try:
        rows = conn.execute(
            f"SELECT seq, target, key_column, key_value, before_json, after_json FROM {TABLE} "  # noqa: S608
            "WHERE repair = ? AND undone_at IS NULL ORDER BY seq DESC", (repair,)).fetchall()
        for seq, target, key_col, key, before_json, after_json in rows:
            if KEYS.get(target) != key_col:
                raise ValueError(f"undo record {repair}#{seq} names {target}.{key_col}")
            before = json.loads(before_json)
            done = (_undo_delete(conn, target, key_col, key, before) if after_json is None
                    else _undo_update(conn, target, key_col, key, before, json.loads(after_json)))
            restored += int(done)
            conn.execute(f"UPDATE {TABLE} SET undone_at = ? WHERE repair = ? AND seq = ?",  # noqa: S608
                         (_now(), repair, seq))
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    logger.info("%s undone: %d of %d recorded change(s) put back", repair, restored, len(rows))
    return restored


def summary(conn: sqlite3.Connection, repair: str) -> dict[str, Any]:
    """What ``repair`` changed: facts, text rewrites, rows removed, rows undone."""
    empty = {"facts": 0, "text_changed": 0, "rows_removed": 0, "lifecycle_restored": 0,
             "undone": 0, "applied_at": None}
    if not has_table(conn):
        return empty
    row = conn.execute(
        f"SELECT COUNT(DISTINCT fact_id), "  # noqa: S608
        "SUM(target = 'atomic_facts' AND before_json LIKE '%\"content\"%'), "
        "SUM(after_json IS NULL), SUM(target = 'fact_retention'), "
        f"SUM(undone_at IS NOT NULL), MIN(created_at) FROM {TABLE} WHERE repair = ?",
        (repair,)).fetchone()
    return {"facts": int(row[0] or 0), "text_changed": int(row[1] or 0),
            "rows_removed": int(row[2] or 0), "lifecycle_restored": int(row[3] or 0),
            "undone": int(row[4] or 0), "applied_at": row[5]}


__all__ = ["DDL", "KEYS", "TABLE", "UndoWriter", "ensure_table", "has_table", "summary",
           "undo"]
