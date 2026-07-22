# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — RBAC hardening (SEC-H-01, audit-02)

"""M026 — add FK on rbac_memberships → profiles ON DELETE CASCADE (memory.db).

SEC-H-01: ``rbac_memberships`` (created by M024) had no foreign key to
``profiles``. Deleting a profile therefore left its role grants behind; a
profile later recreated with the same ``profile_id`` silently re-inherited
them — a privilege-persistence bug.

The runtime path (``delete_profile_from_db``) already purges memberships
explicitly. This migration is the *defense-in-depth* half: a real FK with
``ON DELETE CASCADE`` so that ANY path deleting a ``profiles`` row — not just
that one helper — cannot leave orphaned grants.

SQLite cannot add a constraint in place, so the table is rebuilt (the
recommended safe-alter dance): rename → create canonical table with the FK →
copy rows → drop old → recreate the user index. Wrapped in BEGIN IMMEDIATE /
COMMIT for atomicity; the runner connects with ``foreign_keys = OFF`` so the
rename/copy/drop never trips a constraint action mid-rebuild.

Idempotent: skips when the FK is already present (fresh installs still get the
FK because this always rebuilds M024's FK-less table on first run). Tolerant of
a missing table (nothing to migrate).

Deferred because the FK target ``profiles`` is created at engine init
(``storage.schema``), after ``apply_all`` runs — same ordering constraint as
M021 / M023.

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import sqlite3

NAME = "M026_rbac_memberships_fk"
DB_TARGET = "memory"

# Documentation + drift-hash fingerprint. apply() below is the authoritative
# executor (a static DDL string cannot express the conditional rebuild).
DDL = (
    "-- rebuild rbac_memberships with "
    "FOREIGN KEY (profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE"
)

# Canonical rebuilt table. Column set is byte-identical to M024's, plus the FK.
# The index is created separately AFTER the old table is dropped: RENAME keeps
# the old index's name, so an inline/early CREATE INDEX would collide.
_NEW_TABLE = """
CREATE TABLE rbac_memberships (
    profile_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    added_at TEXT NOT NULL,
    added_by TEXT DEFAULT 'owner',
    PRIMARY KEY (profile_id, user_id),
    FOREIGN KEY (profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE
)
"""


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _has_profiles_fk(conn: sqlite3.Connection, table: str) -> bool:
    """True when ``table`` already declares a foreign key onto ``profiles``."""
    try:
        rows = conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
    except sqlite3.OperationalError:  # pragma: no cover — table absent
        return False
    # PRAGMA foreign_key_list row layout: (id, seq, table, from, to, ...).
    return any(r[2] == "profiles" for r in rows)


def apply(conn: sqlite3.Connection) -> None:
    """Rebuild rbac_memberships with the profiles FK.

    No-op when the table is absent (nothing to migrate) or the FK already
    exists (idempotent re-run). Existing grants are preserved verbatim.
    """
    if not _table_exists(conn, "rbac_memberships"):
        return
    if _has_profiles_fk(conn, "rbac_memberships"):
        return

    has_added_by = "added_by" in _cols(conn, "rbac_memberships")

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("ALTER TABLE rbac_memberships RENAME TO _rbac_memberships_old")
        # NB: single statement via execute(), not executescript() — the latter
        # force-commits the pending transaction, breaking our BEGIN/COMMIT.
        conn.execute(_NEW_TABLE)
        if has_added_by:
            conn.execute(
                "INSERT INTO rbac_memberships "
                "(profile_id, user_id, role, added_at, added_by) "
                "SELECT profile_id, user_id, role, added_at, added_by "
                "FROM _rbac_memberships_old"
            )
        else:  # exotic/older install missing the column — default it
            conn.execute(
                "INSERT INTO rbac_memberships "
                "(profile_id, user_id, role, added_at) "
                "SELECT profile_id, user_id, role, added_at "
                "FROM _rbac_memberships_old"
            )
        # Dropping the old table also drops its index, freeing the name.
        conn.execute("DROP TABLE _rbac_memberships_old")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rbac_memberships_user "
            "ON rbac_memberships(user_id)"
        )
        conn.execute("COMMIT")
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:  # pragma: no cover — best-effort
            pass
        raise


def verify(conn: sqlite3.Connection) -> bool:
    """Applied once rbac_memberships carries the profiles FK (or is absent)."""
    if not _table_exists(conn, "rbac_memberships"):
        return True  # nothing to migrate; fresh install rebuilds it on apply
    return _has_profiles_fk(conn, "rbac_memberships")
