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

#: Highest schema_version this runner can write.  Matches the trailing serial
#: of the latest migration (M039).  Increment when adding new migrations or
#: table-level breaking changes.
SUPPORTED_SCHEMA_VERSION: int = 39


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


def check_version_or_raise(db_path: Path) -> None:
    """Raise SchemaVersionError if the DB's stored version exceeds supported max.

    Non-mutating.  Must be called before any write at startup.
    """
    stored = read_schema_version(db_path)
    if stored > SUPPORTED_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"Database schema_version={stored} exceeds the maximum "
            f"supported version={SUPPORTED_SCHEMA_VERSION}. "
            "This installation is too old for this database. "
            "Upgrade SLM to a version that supports schema_version "
            f"{stored} or higher."
        )


__all__ = (
    "SUPPORTED_SCHEMA_VERSION",
    "SchemaVersionError",
    "read_schema_version",
    "ensure_schema_version_table",
    "write_schema_version",
    "check_version_or_raise",
)
