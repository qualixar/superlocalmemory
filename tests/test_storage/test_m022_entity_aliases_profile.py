# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory V3
"""M022 — profile_id on entity_aliases (I-7 clean part).

entity_aliases was keyed by entity_id alone; when the same entity_id appears in
two profiles, aliases bled across them. M022 adds profile_id (+ index) and
backfills each alias from its PARENT canonical entity's profile — not a blind
'default' dump.
"""

from __future__ import annotations

import sqlite3

from superlocalmemory.storage.migrations import M022_entity_aliases_profile as M


def _legacy(path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        "CREATE TABLE canonical_entities "
        "(entity_id TEXT, profile_id TEXT, canonical_name TEXT);"
        "INSERT INTO canonical_entities VALUES "
        "('e1','work','Alice'),('e2','home','Bob');"
        "CREATE TABLE entity_aliases ("
        " alias_id TEXT PRIMARY KEY, entity_id TEXT NOT NULL, alias TEXT NOT NULL,"
        " confidence REAL DEFAULT 1.0, source TEXT DEFAULT '');"
        "INSERT INTO entity_aliases (alias_id, entity_id, alias) VALUES "
        "('a1','e1','Ali'),('a2','e2','Bobby'),('a3','ghost','Orphan');"
    )
    conn.commit()
    return conn


def test_backfills_profile_from_parent_entity(tmp_path):
    conn = _legacy(tmp_path / "m.db")
    assert M.verify(conn) is False
    M.apply(conn)
    conn.commit()
    got = dict(conn.execute(
        "SELECT alias_id, profile_id FROM entity_aliases").fetchall())
    assert got == {"a1": "work", "a2": "home", "a3": "default"}, (
        "each alias must inherit its parent entity's profile; orphan → default"
    )
    assert M.verify(conn) is True


def test_index_created(tmp_path):
    conn = _legacy(tmp_path / "m.db")
    M.apply(conn)
    conn.commit()
    idx = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' "
        "AND name='idx_aliases_profile_entity'").fetchone()
    assert idx is not None


def test_fresh_install_gets_index(tmp_path):
    """Column already present (fresh install) but index missing → apply adds it."""
    conn = sqlite3.connect(str(tmp_path / "fresh.db"))
    conn.execute(
        "CREATE TABLE entity_aliases ("
        " alias_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL DEFAULT 'default',"
        " entity_id TEXT NOT NULL, alias TEXT NOT NULL,"
        " confidence REAL DEFAULT 1.0, source TEXT DEFAULT '')"
    )
    conn.commit()
    assert M.verify(conn) is False  # column present, index absent
    M.apply(conn)
    conn.commit()
    assert M.verify(conn) is True


def test_idempotent_and_missing_table_safe(tmp_path):
    empty = sqlite3.connect(str(tmp_path / "empty.db"))
    assert M.verify(empty) is True
    M.apply(empty)  # no raise
    conn = _legacy(tmp_path / "m.db")
    M.apply(conn); conn.commit()
    M.apply(conn); conn.commit()  # second apply no-op
    assert M.verify(conn) is True
