# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The ``slm doctor`` row for the automatic data repairs of 4.2.8.

M065 (made-up dates) and M066 (wrongly archived memories) run once, on the
daemon's first 4.2.8 start, with no command. This row says what they did, from
their undo record (``storage.data_repair_undo``). Read-only.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from superlocalmemory.storage import data_repair_undo as undo_log

_M065 = "M065_fabricated_dates_are_removed"
_M066 = "M066_wrongly_archived_memories_are_restored"


def _ran(conn: sqlite3.Connection, name: str) -> bool:
    try:
        row = conn.execute("SELECT status FROM migration_log WHERE name = ?", (name,)).fetchone()
    except sqlite3.Error:
        return False
    return row is not None and row[0] == "complete"


def _undone(summary: dict) -> str:
    return f" (undone: {summary['undone']} change(s) put back)" if summary["undone"] else ""


def _dates_line(conn: sqlite3.Connection) -> str | None:
    if not _ran(conn, _M065):
        return None
    s = undo_log.summary(conn, _M065)
    if not s["facts"]:
        return "made-up dates: none found"
    return (f"made-up dates removed from {s['facts']} memories ({s['text_changed']} sentence(s) "
            f"rewritten, the rest lost only the date; {s['rows_removed']} date row(s) "
            f"removed){_undone(s)}")


def _archive_line(conn: sqlite3.Connection) -> str | None:
    if not _ran(conn, _M066):
        return None
    s = undo_log.summary(conn, _M066)
    if not s["facts"]:
        return "wrongly archived memories: none found"
    return f"{s['lifecycle_restored']} wrongly archived memories restored{_undone(s)}"


def data_repairs_finding(db_path: str | Path) -> tuple[str, str, str]:
    """``(status, detail, fix)``. Repairs are not failures, so the row is PASS."""
    path = Path(db_path)
    if not path.exists():
        return "PASS", "nothing to report (no store yet)", ""
    conn = sqlite3.connect(f"{path.absolute().as_uri()}?mode=ro", uri=True, timeout=2)
    try:
        lines = [line for line in (_dates_line(conn), _archive_line(conn)) if line]
    finally:
        conn.close()
    if not lines:
        return "PASS", "nothing to report (the 4.2.8 repairs have not run on this store)", ""
    if all(line.endswith("none found") for line in lines):
        return "PASS", f"nothing needed repair ({'; '.join(lines)})", ""
    return "PASS", "; ".join(lines), ""


__all__ = ["data_repairs_finding"]
