# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""Keyword recall must not surface archived facts.

The full-text channel is the primary keyword recall path. It joins the facts
table for scoping, so it must also respect ``archive_status`` — otherwise a
retired fact reappears for any query that matches its text.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.retrieval.bm25_channel import BM25Channel
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, MemoryRecord


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "test.db")
    mgr.initialize(real_schema)
    mgr.execute(
        "ALTER TABLE atomic_facts ADD COLUMN archive_status TEXT DEFAULT 'live'"
    )
    mgr.store_memory(MemoryRecord(memory_id="m", profile_id="default", content="seed"))
    return mgr


def test_fts5_search_excludes_archived(db: DatabaseManager) -> None:
    db.store_fact(
        AtomicFact(fact_id="k_live", memory_id="m", content="quantum entanglement")
    )
    db.store_fact(
        AtomicFact(fact_id="k_arch", memory_id="m", content="quantum tunnelling")
    )
    db.execute(
        "UPDATE atomic_facts SET archive_status = 'archived' WHERE fact_id = ?",
        ("k_arch",),
    )

    channel = BM25Channel(db)
    hits = {fid for fid, _ in channel._fts5_search("quantum", "default", top_k=10)}

    assert "k_live" in hits
    assert "k_arch" not in hits
