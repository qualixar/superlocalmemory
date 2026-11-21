# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""The ``slm doctor`` row for JSON text columns that hold binary values.

A writer that binds ``bytes`` to a TEXT column leaves a BLOB there. When those
bytes were not UTF-8 text, every save whose enrichment read the row failed with
"'utf-8' codec can't decode ..." and was dead-lettered. The readers now treat
such a value as empty (``storage/json_columns.py``), so saves keep working;
this row says where the values are, per ``table.column``, so an agent or a
person can look. It opens the store read-only and never repairs anything.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from superlocalmemory.storage.json_columns import TOLERATED_COLUMNS

_SHOWN = 6


def _readonly(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=5)


def binary_json_values(db_path: str | Path, *, include_zero: bool = False) -> dict[str, int]:
    """``{"table.column": count}`` of BLOB values in the tolerated JSON columns.

    A table or column the store does not have is skipped. ``include_zero``
    keeps the columns with none (to show what was checked).
    """
    counts: dict[str, int] = {}
    conn = _readonly(Path(db_path))
    try:
        for table, column in TOLERATED_COLUMNS:
            present = {str(r[1]) for r in conn.execute(f'PRAGMA table_info("{table}")')}
            if column not in present:
                continue
            n = int(conn.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE typeof("{column}") = \'blob\'',
            ).fetchone()[0])
            if n or include_zero:
                counts[f"{table}.{column}"] = n
    finally:
        conn.close()
    return counts


def _show_query(db_path: Path, key: str) -> str:
    table, column = key.split(".", 1)
    return (f"sqlite3 -readonly '{db_path}' \"SELECT rowid FROM {table} "
            f"WHERE typeof({column})='blob'\"")


def binary_json_values_finding(db_path: str | Path) -> tuple[str, str, str]:
    """``(status, detail, fix)``: PASS when there are none, WARN when there are some."""
    db_path = Path(db_path)
    counts = binary_json_values(db_path)
    if not counts:
        return ("PASS", f"no binary values in the {len(TOLERATED_COLUMNS)} JSON columns saves read", "")
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = ", ".join(f"{key}: {n}" for key, n in ranked[:_SHOWN])
    more = f", and {len(ranked) - _SHOWN} more columns" if len(ranked) > _SHOWN else ""
    detail = (f"{total} stored {'value is' if total == 1 else 'values are'} binary, not JSON "
              f"text ({shown}{more})")
    fix = ("These rows are read safely: a value that is not text counts as empty, so saves "
           "keep working, and nothing is changed. To see them: " + _show_query(db_path, ranked[0][0])
           + " (the same query with the other table and column names shows the rest)")
    return ("WARN", detail, fix)


__all__ = ["binary_json_values", "binary_json_values_finding"]
