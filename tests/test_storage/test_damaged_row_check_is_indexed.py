# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""A save asks about one row by its id; it never scans the profile's whole table.

``typeof(col) = 'blob'`` cannot use an index, so running it per save would put a
full scan of ``memory_scenes`` or ``entity_profiles`` on every remember.
"""

from __future__ import annotations

import sqlite3

from superlocalmemory.storage.damaged_values import row_is_damaged


class _Db:
    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("CREATE TABLE memory_scenes (scene_id TEXT PRIMARY KEY, profile_id TEXT, fact_ids_json TEXT)")
        self.sql: list[str] = []

    def execute(self, sql, params=()):
        self.sql.append(sql)
        return self.conn.execute(sql, params).fetchall()


def test_row_is_damaged_reads_one_row_by_id():
    db = _Db()
    db.conn.execute("INSERT INTO memory_scenes VALUES ('s1', 'p', ?)", (b"ab\x81cd",))
    db.conn.execute("INSERT INTO memory_scenes VALUES ('s2', 'p', '[\"f\"]')")
    db.conn.execute("INSERT INTO memory_scenes VALUES ('s3', 'p', ?)", (b'["f"]',))
    assert row_is_damaged(db, "memory_scenes", "scene_id", "s1", "p", "fact_ids_json") is True
    assert row_is_damaged(db, "memory_scenes", "scene_id", "s2", "p", "fact_ids_json") is False
    assert row_is_damaged(db, "memory_scenes", "scene_id", "s3", "p", "fact_ids_json") is False
    assert row_is_damaged(db, "memory_scenes", "scene_id", "missing", "p", "fact_ids_json") is False
    assert row_is_damaged(db, "memory_scenes", "scene_id", "s1", "other", "fact_ids_json") is False
    assert all('"scene_id" = ?' in sql and "typeof" not in sql for sql in db.sql)


def test_a_store_without_the_column_is_not_damaged():
    db = _Db()
    assert row_is_damaged(db, "memory_scenes", "scene_id", "s1", "p", "no_such_column") is False


def test_save_paths_use_the_one_row_check():
    import inspect

    from superlocalmemory.encoding import observation_builder, scene_builder
    from superlocalmemory.storage import database

    for module in (observation_builder, scene_builder, database):
        source = inspect.getsource(module)
        assert "damaged_ids(" not in source, module.__name__
