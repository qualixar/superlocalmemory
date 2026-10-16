# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A Python error inside one migration's ``apply`` is recorded as that migration
failing; it does not stop the pass or the engine.

Before 4.2.8 ``_apply_single`` caught only ``sqlite3.Error``. A ``KeyError`` or
``ImportError`` in a data migration escaped ``apply_deferred``, and
``MemoryEngine`` (core/engine.py) turned it into "initialization stopped": one
bug in a one-time repair kept the whole memory from starting, and left the
migration logged ``in_progress`` inside an open transaction.
"""

from __future__ import annotations

import sqlite3

from superlocalmemory.storage.migration_runner import apply_deferred
from superlocalmemory.storage.migrations import M065_fabricated_dates_are_removed as M065
from superlocalmemory.storage.migrations import (
    M066_wrongly_archived_memories_are_restored as M066,
)
from tests.test_migrations._repair_store import forget_migration, migrated_store


def test_a_python_error_is_a_failed_migration_not_a_crash(tmp_path, monkeypatch) -> None:
    memory, learning = migrated_store(tmp_path)
    forget_migration(memory, M065.NAME)
    forget_migration(memory, M066.NAME)

    def broken(conn):
        conn.execute("BEGIN IMMEDIATE")
        raise KeyError("a bug, not a database error")

    monkeypatch.setattr(M065, "apply", broken)
    result = apply_deferred(learning, memory)

    assert M065.NAME in result["failed"], result
    assert M066.NAME in result["applied"], result  # the pass went on
    conn = sqlite3.connect(memory)
    status = conn.execute("SELECT status FROM migration_log WHERE name = ?",
                          (M065.NAME,)).fetchone()[0]
    assert status == "failed"
    conn.execute("BEGIN IMMEDIATE")  # no transaction was left open on the file
    conn.rollback()

    monkeypatch.undo()
    assert M065.NAME in apply_deferred(learning, memory)["applied"]  # retried next start
