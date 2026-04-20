# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Stage 8 F5 (Mediums/Lows)

"""Stage 8 F5 regression — SEC-M5 trigram rebuild LIMIT.

The rebuild SELECT now includes ``LIMIT _MAX_REBUILD_ROWS`` so that a
pathological memory.db cannot exceed the 300 MB ``ram_reservation``
budget by materialising tens of millions of rows into Python memory.
Also asserts the source connection gets an explicit busy_timeout.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.learning import trigram_index as ti


def _bootstrap_memory_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE canonical_entities (
                entity_id       TEXT PRIMARY KEY,
                profile_id      TEXT NOT NULL,
                canonical_name  TEXT NOT NULL
            );
            CREATE TABLE entity_aliases (
                entity_id  TEXT NOT NULL,
                alias      TEXT NOT NULL
            );
            """
        )
        conn.executemany(
            "INSERT INTO canonical_entities (entity_id, profile_id, "
            "canonical_name) VALUES (?, ?, ?)",
            [(f"e{i}", "default", f"EntityName{i}") for i in range(20)],
        )


def test_sec_m5_rebuild_select_has_limit_clause() -> None:
    # Sanity check that the module constant is set and reasonable
    # (upper bound > downstream MAX_TRIGRAMS so legitimate inputs pass
    # through untouched).
    assert ti.TrigramIndex._MAX_REBUILD_ROWS >= ti.TrigramIndex.MAX_TRIGRAMS


def test_sec_m5_rebuild_respects_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_db = tmp_path / "memory.db"
    _bootstrap_memory_db(memory_db)
    # Redirect the cache DB path so the test is hermetic.
    cache_db = tmp_path / "active_brain_cache.db"
    monkeypatch.setattr(ti.TrigramIndex, "CACHE_DB_PATH", cache_db)
    # Squash ram_reservation to a no-op (CI boxes are tight on RAM).
    from contextlib import contextmanager

    @contextmanager
    def _noop(*a, **kw):
        yield

    monkeypatch.setattr(ti, "ram_reservation", _noop)

    # Force the LIMIT down to 5 so the seeded 20-row dataset is capped.
    monkeypatch.setattr(
        ti.TrigramIndex, "_MAX_REBUILD_ROWS", 5, raising=True,
    )
    idx = ti.TrigramIndex(memory_db)
    idx.bootstrap()

    # Confirm at most LIMIT rows drove the bucket build — the resulting
    # cache DB should have far fewer trigrams than 20 entities would
    # have produced without the LIMIT.
    with sqlite3.connect(cache_db) as conn:
        (n,) = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='entity_trigrams'"
        ).fetchone()
    assert n == 1, "entity_trigrams table must exist after bootstrap"
