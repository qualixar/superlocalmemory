# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""Archived facts must not be retrievable through any direct read path.

Archival is how an erasure/forget request hides a fact from recall. A read
path that ignores ``archive_status`` re-exposes a fact that policy has
retired, so every by-id and cross-profile read is covered here against a real
database.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, MemoryRecord


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "test.db")
    mgr.initialize(real_schema)
    # The archive column is introduced by a deferred migration; add it here so
    # the read paths run against a migrated-shape table.
    mgr.execute(
        "ALTER TABLE atomic_facts ADD COLUMN archive_status TEXT DEFAULT 'live'"
    )
    # A fact references a memory row (foreign key); seed one for the default
    # profile so facts can be stored.
    mgr.store_memory(MemoryRecord(memory_id="m", profile_id="default", content="seed"))
    return mgr


def _archive(db: DatabaseManager, fact_id: str) -> None:
    db.execute(
        "UPDATE atomic_facts SET archive_status = 'archived' WHERE fact_id = ?",
        (fact_id,),
    )


def test_get_facts_by_ids_excludes_archived(db: DatabaseManager) -> None:
    db.store_fact(AtomicFact(fact_id="f_live", memory_id="m", content="live"))
    db.store_fact(AtomicFact(fact_id="f_arch", memory_id="m", content="archived"))
    _archive(db, "f_arch")

    got = {f.fact_id for f in db.get_facts_by_ids(["f_live", "f_arch"], "default")}
    assert got == {"f_live"}


def test_get_external_visible_facts_excludes_archived(db: DatabaseManager) -> None:
    db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('other', 'Other')"
    )
    db.store_fact(
        AtomicFact(fact_id="g_live", memory_id="m", profile_id="other",
                   content="global live", scope="global")
    )
    db.store_fact(
        AtomicFact(fact_id="g_arch", memory_id="m", profile_id="other",
                   content="global archived", scope="global")
    )
    _archive(db, "g_arch")

    got = {
        f.fact_id
        for f in db.get_external_visible_facts("default", include_global=True)
    }
    assert "g_live" in got
    assert "g_arch" not in got
