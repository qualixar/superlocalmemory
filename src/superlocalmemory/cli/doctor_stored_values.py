# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""The ``slm doctor`` row for JSON text columns that hold binary values.

A writer that binds ``bytes`` to a TEXT column leaves a BLOB there. When those
bytes were not UTF-8 text, every save whose enrichment read the row failed with
"'utf-8' codec can't decode ..." and was dead-lettered. The readers now treat
such a value as empty (``storage/json_columns.py``), so saves keep working, and
a later save that writes the value either rebuilds it from its source and keeps
the original bytes, or skips the row, per column (the fix text says which). A
BLOB that holds UTF-8 text reads normally and is only mentioned. This row says
where the values are, per ``table.column``, so an agent or a person can look. It opens the store read-only and never
repairs anything.
"""

from __future__ import annotations

import shlex
import sqlite3
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from superlocalmemory.storage.json_columns import TOLERATED_COLUMNS

_SHOWN = 6


def _readonly(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=5)


def _is_text(value: object) -> bool:
    try:
        bytes(value).decode("utf-8")  # type: ignore[arg-type]
    except UnicodeDecodeError:
        return False
    return True


def binary_json_value_kinds(
    db_path: str | Path, *, include_zero: bool = False,
) -> dict[str, tuple[int, int]]:
    """``{"table.column": (holds_utf8_text, not_text)}`` for the BLOBs in the tolerated columns.

    One scan per table, reading only the rows that have a BLOB in one of its
    columns. A table or column the store does not have is skipped.
    ``include_zero`` keeps the columns with none (to show what was checked).
    The media store (``media.db``) is scanned too when it sits next to the database.
    """
    by_table: dict[str, list[str]] = {}
    for table, column in TOLERATED_COLUMNS:
        by_table.setdefault(table, []).append(column)
    db_path = Path(db_path)
    # ``doc_pages`` lives in the media store (media.db, next to the memory database).
    media = db_path.parent / "media.db"
    kinds: dict[str, tuple[int, int]] = {}
    for path in (db_path, *([media] if media.exists() and media != db_path else [])):
        kinds.update(_scan(path, by_table, include_zero))
    return kinds


def _scan(db_path: Path, by_table: dict[str, list[str]],
          include_zero: bool) -> dict[str, tuple[int, int]]:
    kinds: dict[str, tuple[int, int]] = {}
    conn = _readonly(db_path)
    try:
        for table, wanted in by_table.items():
            present = {str(r[1]) for r in conn.execute(f'PRAGMA table_info("{table}")')}
            columns = [c for c in wanted if c in present]
            if not columns:
                continue
            quoted = ", ".join(f'"{c}"' for c in columns)
            where = " OR ".join(f'typeof("{c}") = \'blob\'' for c in columns)
            counts = {c: [0, 0] for c in columns}
            for row in conn.execute(f'SELECT {quoted} FROM "{table}" WHERE {where}'):
                for column, value in zip(columns, row):
                    if isinstance(value, (bytes, bytearray, memoryview)):
                        counts[column][0 if _is_text(value) else 1] += 1
            for column in columns:
                readable, unreadable = counts[column]
                if readable or unreadable or include_zero:
                    kinds[f"{table}.{column}"] = (readable, unreadable)
    finally:
        conn.close()
    return kinds


def binary_json_values(db_path: str | Path, *, include_zero: bool = False) -> dict[str, int]:
    """``{"table.column": count}`` of every BLOB in the tolerated JSON columns."""
    return {key: readable + unreadable for key, (readable, unreadable)
            in binary_json_value_kinds(db_path, include_zero=include_zero).items()}


def _is_windows() -> bool:
    return sys.platform == "win32"


def _quote_path(path: str) -> str:
    """A path as one shell word: ``shlex.quote`` on POSIX, double quotes on Windows."""
    return subprocess.list2cmdline([path]) if _is_windows() else shlex.quote(path)


def _database_of(db_path: Path, table: str) -> Path:
    """The file a tolerated table lives in: ``doc_pages`` is in ``media.db``, the rest in the memory database."""
    return db_path.parent / "media.db" if table == "doc_pages" else db_path


def _show_query(db_path: Path, key: str) -> str:
    table, column = key.split(".", 1)
    return (f"sqlite3 -readonly {_quote_path(str(_database_of(db_path, table)))} "
            f"\"SELECT rowid FROM {table} WHERE typeof({column})='blob'\"")


#: What a writer does with a value that is not text, per column. "rebuilt" means the
#: new value is made from its source and the original bytes are kept; "left" means
#: the writer skips that row and the bytes stay exactly as they are.
_REBUILT_LEDGER = "rebuilt from the fact when it is next written; the original bytes are kept in the change ledger"
_FATE: dict[str, str] = {
    "atomic_facts.entities_json": _REBUILT_LEDGER,
    "atomic_facts.canonical_entities_json": _REBUILT_LEDGER,
    "atomic_facts.source_turn_ids_json": _REBUILT_LEDGER,
    "atomic_facts.langevin_position": _REBUILT_LEDGER,
    "entity_profiles.fact_ids_json":
        "rebuilt from the facts that carry the entity when the profile is next written; "
        "the bytes are kept in a version row",
    "memory_scenes.entity_ids_json":
        "rebuilt from the scene's facts when the scene is next written; the bytes are kept in a version row",
    "memory_scenes.fact_ids_json":
        "the scene is left as it is and a new fact starts a scene of its own",
    "memories.metadata_json": "left as it is: writers skip that row",
    "memories.shared_with": "left as it is: writers skip that row",
    "atomic_facts.shared_with": "left as it is: writers skip that row",
    "doc_pages.fact_ids_json": "read as no facts and left as it is",
}


def _fates(keys: Sequence[str]) -> str:
    return "; ".join(f"{key}: {_FATE.get(key, 'read as empty and left as it is')}" for key in keys)


def _ranked(counts: dict[str, int]) -> str:
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = ", ".join(f"{key}: {n}" for key, n in ranked[:_SHOWN])
    more = f", and {len(ranked) - _SHOWN} more columns" if len(ranked) > _SHOWN else ""
    return f"{shown}{more}"


def binary_json_values_finding(db_path: str | Path) -> tuple[str, str, str]:
    """``(status, detail, fix)``: WARN when a value is not text, PASS otherwise."""
    db_path = Path(db_path)
    kinds = binary_json_value_kinds(db_path)
    if not kinds:
        return ("PASS", f"no binary values in the {len(TOLERATED_COLUMNS)} JSON columns saves read", "")
    unreadable = {k: u for k, (_r, u) in kinds.items() if u}
    readable = {k: r for k, (r, _u) in kinds.items() if r}
    n_read = sum(readable.values())
    holds = "holds" if n_read == 1 else "hold"
    reads = "reads" if n_read == 1 else "read"
    if not unreadable:
        return ("PASS", f"{n_read} stored {'value is' if n_read == 1 else 'values are'} bytes in a "
                f"text column that {holds} UTF-8 JSON and {reads} normally ({_ranked(readable)})", "")
    total = sum(unreadable.values())
    detail = (f"{total} stored {'value is' if total == 1 else 'values are'} not text and "
              f"{'reads' if total == 1 else 'read'} as empty ({_ranked(unreadable)})")
    if n_read:
        detail += f"; {n_read} more {holds} UTF-8 JSON and {reads} normally ({_ranked(readable)})"
    ordered = [k for k, _n in sorted(unreadable.items(), key=lambda kv: (-kv[1], kv[0]))]
    fix = ("These rows are read safely: a value that is not text reads as empty, so saves keep "
           "working. This check changes nothing. What a writer does with each column: "
           + _fates(ordered)
           + ". To see them: " + _show_query(db_path, ordered[0])
           + " (the same query with the other table and column names shows the rest)")
    return ("WARN", detail, fix)


__all__ = ["binary_json_value_kinds", "binary_json_values", "binary_json_values_finding"]
