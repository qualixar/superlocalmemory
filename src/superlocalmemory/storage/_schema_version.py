# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Schema-version singleton: read, write, and version-conflict guard.

Encapsulates the one-row ``schema_version`` table written to the learning DB
after each zero-failure migration run.  A stored version that exceeds the
runner's supported ceiling means a newer build wrote the database and this
installation must not operate on it.

Design:
  - ``_read_schema_version_from_db`` is strictly read-only (no writes).
  - ``_ensure_schema_version_table`` + ``_write_schema_version`` are only
    called by the runner after a confirmed zero-failure apply_all.
  - Missing table or unreadable DB → version 0 (legacy; safe to upgrade).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

#: Highest schema_version this runner can write.  Matches the trailing serial of
#: the latest migration (M050).  Increment when adding new migrations or
#: table-level breaking changes.
#:
#: This sat at 42 while M043, M044 and M045 shipped, so for three migrations the
#: ceiling did not move and nothing prevented an older build from opening a
#: newer store.  That was tolerable only because those three were additive: a
#: build that did not know about a new column or table simply never read it.
#:
#: M046 is not additive.  It rebuilds ``atomic_facts`` with a constraint that
#: rejects the value an older build classifies planned events as, so an older
#: writer against a migrated store fails its INSERT.  The ceiling is what turns
#: that from a lost memory into a refusal to start, which is why it moves here
#: and why it moves to the trailing serial rather than to 43.
#:
#: M049 is additive — a unique index on ``schema_version`` plus the removal of
#: the duplicate rows that index could not otherwise be created over — so an
#: older build could in principle read a store it has touched.  The ceiling
#: still moves, because the convention is "trailing serial, always": the cost of
#: moving it for an additive migration is an older build declining a store it
#: could have read, and the cost of NOT moving it for one that turns out not to
#: be additive is a silent bad write.  Those are not comparable, and judging
#: additivity per migration is exactly the judgement that let it fall three
#: behind.
#:
#: M052 is additive (five nullable columns and two tables); the ceiling moves to
#: it by the same convention. An older build that must run on the store goes
#: through prepare-for-downgrade, which may lower the stamp to M052's
#: ``DOWNGRADE_FLOOR``.
#:
#: M053 is additive (two new learning.db tables for the Answer Check history);
#: the ceiling moves to it by the same convention. Its ``DOWNGRADE_FLOOR`` is 51,
#: so going back to 4.1.18 stays possible exactly as M052 left it.
#:
#: M054 is additive (one new learning.db table for saved views); the ceiling
#: moves to it by the same convention. Its ``DOWNGRADE_FLOOR`` is 51 as well, so
#: going back to 4.1.20, 4.1.19 or 4.1.18 stays possible.
#:
#: M055 is additive (one learning.db column with a partial unique index and one
#: new learning.db table for learning attribution); the ceiling moves to it by
#: the same convention, with ``DOWNGRADE_FLOOR`` 51.
#: M056 adds recall indexes only; the ceiling follows the newest migration.
#: M059 (erasure visibility), M060 (memory description: two atomic_facts
#: columns and the enrichment ledger tables) and M061 (one new memory.db table
#: of hub baselines and its triggers) are additive, with ``DOWNGRADE_FLOOR`` 51.
#: M062 sets an existing flag column on rows the stub rule recognises (no schema
#: change); the ceiling follows it by the same convention, floor 51.
#: M063 stores blank temporal dates as NULL (data only), floor 51.
#: M064 removes index rows of memories that are gone (data only), floor 51.
#: M065 removes made-up dates (data + an undo table), floor 51.
#: M066 restores wrongly archived memories (data, same undo table), floor 51.
#: M067 adds one trigger that queues fact changes for the projections, floor 51.
SUPPORTED_SCHEMA_VERSION: int = 67


class SchemaVersionError(RuntimeError):
    """Raised when the DB's recorded schema_version exceeds the supported max.

    The caller must not attempt any write after catching this — the DB was
    produced by a newer installation and this build is too old for it.
    """


# NOTE: this singleton lives in its OWN table ``slm_schema_version`` rather than
# ``schema_version``.  The legacy multi-row migration-history table
# ``schema_version`` (columns version/applied_at/description, seeded by
# schema.py + the schema_v34x migrations) already exists on every real database
# — fresh installs and upgrades alike — so a ``CREATE TABLE IF NOT EXISTS
# schema_version`` here would no-op against that legacy shape and every
# ``id``-keyed read/write would fail ("no column named id").  A distinct table
# name decouples the version-ceiling guard from that legacy history table.
_SCHEMA_VERSION_DDL = """\
CREATE TABLE IF NOT EXISTS slm_schema_version (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    version     INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
INSERT OR IGNORE INTO slm_schema_version (id, version, updated_at)
VALUES (1, 0, datetime('now'));
"""


def read_schema_version(db_path: Path) -> int:
    """Read the stored schema_version without any write.  Returns 0 if absent."""
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT version FROM slm_schema_version WHERE id = 1"
            ).fetchone()
            return int(row[0]) if row is not None else 0
        except sqlite3.OperationalError:
            return 0  # slm_schema_version table does not exist (legacy DB)
        finally:
            conn.close()
    except sqlite3.Error:
        return 0  # cannot open DB; let the runner surface the open error


def ensure_schema_version_table(conn: sqlite3.Connection) -> None:
    """Create the schema_version table if absent and seed the singleton row."""
    conn.executescript(_SCHEMA_VERSION_DDL)


def write_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Overwrite the singleton version record."""
    conn.execute(
        "UPDATE slm_schema_version "
        "SET version = ?, updated_at = datetime('now') WHERE id = 1",
        (version,),
    )


def _detect_all_installs() -> list:
    """Delegate to core.install_detector; fail-open (returns [] on any error)."""
    try:
        from superlocalmemory.core.install_detector import (
            _detect_all_installs as _real,
        )
        return _real()
    except Exception:  # import error, timeout, etc.
        return []


def _format_install_block(installs: list) -> str:
    """Return a formatted multi-line string listing all detected installs."""
    if not installs:
        return ""

    _upgrade_hints = {
        "pipx": "  pipx upgrade superlocalmemory",
        "venv": "  pip install -U superlocalmemory  (in ~/.slm-venv)",
        "npm": "  npm install -g superlocalmemory@latest",
    }

    lines = ["\nAll detected installs:"]
    seen_types: set[str] = set()
    for entry in installs:
        install_type = entry.get("type", "unknown")
        version = entry.get("version", "unknown")
        path = entry.get("path", "")
        lines.append(f"  {install_type:<6} ({version}): {path}")
        seen_types.add(install_type)

    lines.append("\nUpgrade all installs to the latest version:")
    for install_type, cmd in _upgrade_hints.items():
        if install_type in seen_types:
            lines.append(cmd)

    return "\n".join(lines)


def check_version_or_raise(db_path: Path) -> None:
    """Raise SchemaVersionError if the DB's stored version exceeds supported max.

    When raising, the error message lists every SLM installation detected on
    this machine so the user knows which copies to upgrade.

    Non-mutating.  Must be called before any write at startup.
    """
    stored = read_schema_version(db_path)
    if stored > SUPPORTED_SCHEMA_VERSION:
        installs = _detect_all_installs()
        install_block = _format_install_block(installs)
        raise SchemaVersionError(
            f"Database schema_version={stored} exceeds the maximum "
            f"supported version={SUPPORTED_SCHEMA_VERSION}. "
            "This installation is too old for this database."
            + install_block
        )


__all__ = (
    "SUPPORTED_SCHEMA_VERSION",
    "SchemaVersionError",
    "read_schema_version",
    "ensure_schema_version_table",
    "write_schema_version",
    "check_version_or_raise",
)
