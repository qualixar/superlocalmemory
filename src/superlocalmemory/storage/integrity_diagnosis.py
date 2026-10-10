# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What SQLite's own file check says, in words a person can act on.

``slm doctor`` and ``slm restart`` ask SQLite to check the file. The answer is
one row per finding, and a connection from ``memory_read`` returns each row as a
``sqlite3.Row``: formatting the row itself prints ``<sqlite3.Row object at ...>``
instead of the finding (GitHub #204). The message text is read here, in one
place, and a finding that names a keyword index is told apart from damage in the
data pages. The keyword indexes (``atomic_facts_fts``, ``fact_expansion_fts``)
are derived from the stored memories, so ``slm db repair --apply`` rebuilds them
with nothing lost; recreating the whole database is never the answer to them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from superlocalmemory.storage.fts_residue import FTS_TABLES

#: Findings SQLite 3.44+ words two ways: the index disagrees with itself
#: ("malformed inverted index for FTS5 table main.<name>"), or a block of it
#: cannot be read ('fts5: corruption found reading blob N from table "<name>"').
_INDEX_FINDING = re.compile(
    r'(?:FTS5 table (?:\w+\.)?(?P<a>\w+))|(?:fts5:.*?from table "(?P<b>\w+)")',
    re.IGNORECASE | re.DOTALL)
#: How many findings one check reads. SQLite's own default is 100.
FINDING_LIMIT = 20
#: Findings shown in a one-line summary.
_SHOWN = 3
_RESTORE = ("Back up memory.db first, then restore a restore point "
            "(slm db restore-points, then slm db restore)")


@dataclass(frozen=True)
class Diagnosis:
    """The outcome of one SQLite file check."""

    ok: bool
    check: str
    messages: tuple[str, ...]
    damaged_indexes: tuple[str, ...]
    fix: str

    def summary(self) -> str:
        if self.ok:
            return "ok"
        text = "; ".join(self.messages[:_SHOWN])
        more = len(self.messages) - _SHOWN
        return f"{text} (+{more} more)" if more > 0 else text


def damaged_indexes_in(messages: Any) -> tuple[str, ...]:
    """The keyword indexes ``slm db repair`` can rebuild that these findings name."""
    names: list[str] = []
    for message in messages:
        found = _INDEX_FINDING.search(str(message))
        name = (found.group("a") or found.group("b")) if found else None
        if name in FTS_TABLES and name not in names:
            names.append(name)
    return tuple(names)


def _repair_command(root: Path | str | None) -> str:
    return f"slm db repair --apply --root {root if root else '<your SLM data folder>'}"


def _fix(damaged: tuple[str, ...], messages: tuple[str, ...], root: Path | str | None) -> str:
    only_indexes = bool(damaged) and all(damaged_indexes_in([m]) for m in messages)
    if not damaged:
        return f"{_RESTORE}; the findings above say which pages are damaged"
    targeted = (f"{_repair_command(root)}  (rebuilds {', '.join(damaged)} from your stored "
                "memories and turns off the SQLite setting that damaged it; nothing is lost)")
    return targeted if only_indexes else f"{targeted}. For the rest: {_RESTORE}"


def _rows(connection: Any, sql: str) -> list[Any]:
    result = connection.execute(sql)
    return list(result.fetchall() if hasattr(result, "fetchall") else result)


def check_database(conn: Any, *, deep: bool = False,
                   root: Path | str | None = None) -> Diagnosis:
    """Run SQLite's file check and read every finding as text.

    ``deep`` runs ``integrity_check`` (every page); otherwise ``quick_check``.
    """
    pragma = "integrity_check" if deep else "quick_check"
    messages = tuple(str(row[0]) for row in _rows(conn, f"PRAGMA {pragma}({FINDING_LIMIT})"))
    if messages == ("ok",):
        return Diagnosis(True, pragma, messages, (), "")
    damaged = damaged_indexes_in(messages)
    return Diagnosis(False, pragma, messages, damaged, _fix(damaged, messages, root))


def restart_report(db_path: Path | str, root: Path | str | None = None) -> tuple[str, str]:
    """Step 5 of ``slm restart``: ``("ok" | "fail", detail)``."""
    from superlocalmemory.storage.memory_write import memory_read

    with memory_read(db_path) as conn:
        result = check_database(conn, deep=True, root=root or Path(db_path).parent)
        facts = conn.execute("SELECT COUNT(*) FROM atomic_facts").fetchone()[0]
        entities = conn.execute("SELECT COUNT(*) FROM canonical_entities").fetchone()[0]
    detail = f"integrity={result.summary()}, {facts} facts, {entities} entities"
    if result.ok:
        return "ok", detail
    return "fail", f"{detail}. Fix: {result.fix}"


__all__ = ["Diagnosis", "check_database", "damaged_indexes_in", "restart_report"]
