# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory V3
"""M026 — rbac_memberships FK → profiles ON DELETE CASCADE (SEC-H-01).

M024 created rbac_memberships with no foreign key to profiles, so deleting a
profile left its role grants behind (privilege-persistence on id reuse). M026
rebuilds the table with FOREIGN KEY (profile_id) REFERENCES profiles(profile_id)
ON DELETE CASCADE, preserving existing grants.

The runner connects with isolation_level=None (autocommit) and foreign_keys=OFF;
these tests reproduce that contract because apply() drives its own
BEGIN IMMEDIATE / COMMIT and the rebuild must not fire constraint actions.
"""

from __future__ import annotations

import sqlite3

from superlocalmemory.storage.migrations import M026_rbac_memberships_fk as M


def _legacy_db(path) -> sqlite3.Connection:
    """A pre-M026 memory.db: profiles + M024's FK-less rbac_memberships."""
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.executescript(
        "CREATE TABLE profiles ("
        " profile_id TEXT PRIMARY KEY, name TEXT NOT NULL);"
        "CREATE TABLE rbac_memberships ("
        " profile_id TEXT NOT NULL, user_id TEXT NOT NULL, role TEXT NOT NULL,"
        " added_at TEXT NOT NULL, added_by TEXT DEFAULT 'owner',"
        " PRIMARY KEY (profile_id, user_id));"
        "CREATE INDEX idx_rbac_memberships_user ON rbac_memberships(user_id);"
        "INSERT INTO profiles (profile_id, name) VALUES ('corp','corp');"
        "INSERT INTO rbac_memberships "
        " (profile_id, user_id, role, added_at, added_by) "
        " VALUES ('corp','user1','admin','2026-01-01','owner');"
    )
    return conn


def _fk_rows(conn: sqlite3.Connection) -> list:
    return conn.execute("PRAGMA foreign_key_list(rbac_memberships)").fetchall()


def test_migration_adds_fk_and_preserves_rows(tmp_path):
    conn = _legacy_db(tmp_path / "m.db")
    assert M.verify(conn) is False  # no FK yet
    M.apply(conn)

    fks = _fk_rows(conn)
    assert len(fks) == 1, "exactly one FK expected"
    # row layout: (id, seq, table, from, to, on_update, on_delete, match)
    _, _, ref_table, from_col, to_col, _, on_delete, _ = fks[0]
    assert ref_table == "profiles"
    assert from_col == "profile_id" and to_col == "profile_id"
    assert on_delete == "CASCADE"

    row = conn.execute(
        "SELECT profile_id, user_id, role FROM rbac_memberships").fetchone()
    assert row == ("corp", "user1", "admin"), "grant must survive the rebuild"

    idx = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND tbl_name='rbac_memberships'").fetchall()
    assert ("idx_rbac_memberships_user",) in idx, "user index must be recreated"
    assert M.verify(conn) is True


def test_profile_delete_cascades_membership(tmp_path):
    conn = _legacy_db(tmp_path / "m.db")
    M.apply(conn)
    # With enforcement on, deleting the profile must cascade to its grants.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("DELETE FROM profiles WHERE profile_id = 'corp'")
    n = conn.execute(
        "SELECT COUNT(*) FROM rbac_memberships WHERE profile_id='corp'"
    ).fetchone()[0]
    assert n == 0, "profile delete must cascade-purge role grants (SEC-H-01)"


def test_idempotent_and_missing_table_safe(tmp_path):
    # Missing table → verify True, apply is a no-op (must not raise).
    empty = sqlite3.connect(str(tmp_path / "empty.db"), isolation_level=None)
    assert M.verify(empty) is True
    M.apply(empty)

    # Second apply on an already-migrated table is a no-op.
    conn = _legacy_db(tmp_path / "m.db")
    M.apply(conn)
    M.apply(conn)  # no-op — FK already present
    assert M.verify(conn) is True
    assert len(_fk_rows(conn)) == 1, "re-apply must not duplicate the FK"
