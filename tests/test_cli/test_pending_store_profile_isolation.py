# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory V3
"""Per-profile isolation of the legacy pending queue (I-3).

pending_memories had no profile_id, so the drain worker materialized a queued
memory under whatever profile was active at DRAIN time — not the profile that
was active when it was enqueued. `slm remember` on profile A then a switch to B
before the next drain tick silently moved the memory into B.

Fix: capture profile_id at enqueue; the drain claims only its own profile's
items via get_pending(profile_id=...). Other profiles' items wait until their
profile is active — never materialized under the wrong one.
"""

from __future__ import annotations

import sqlite3

from superlocalmemory.cli.pending_store import get_pending, store_pending


def test_store_pending_captures_profile(tmp_path):
    store_pending("A note", profile_id="work", base_dir=tmp_path)
    row = sqlite3.connect(tmp_path / "pending.db").execute(
        "SELECT profile_id FROM pending_memories").fetchone()
    assert row[0] == "work"


def test_get_pending_scopes_to_profile(tmp_path):
    store_pending("work1", profile_id="work", base_dir=tmp_path)
    store_pending("work2", profile_id="work", base_dir=tmp_path)
    store_pending("home1", profile_id="home", base_dir=tmp_path)

    work = get_pending(base_dir=tmp_path, profile_id="work")
    home = get_pending(base_dir=tmp_path, profile_id="home")
    assert len(work) == 2 and all(i["profile_id"] == "work" for i in work)
    assert len(home) == 1
    # A drain on 'home' NEVER sees 'work' items → no misroute.
    assert all(i["content"] != "work1" for i in home)


def test_get_pending_unscoped_returns_all(tmp_path):
    store_pending("a", profile_id="work", base_dir=tmp_path)
    store_pending("b", profile_id="home", base_dir=tmp_path)
    assert len(get_pending(base_dir=tmp_path)) == 2  # no profile_id → all


def test_default_profile_backfill_for_legacy_enqueue(tmp_path):
    """A row inserted with the OLD schema (no profile_id) reads back as 'default'."""
    # Simulate a pre-isolation pending.db.
    conn = sqlite3.connect(tmp_path / "pending.db")
    conn.execute(
        "CREATE TABLE pending_memories ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT NOT NULL,"
        " tags TEXT, metadata TEXT, created_at TEXT NOT NULL,"
        " status TEXT NOT NULL DEFAULT 'pending', error TEXT,"
        " retry_count INTEGER DEFAULT 0, next_retry_at REAL DEFAULT 0)"
    )
    conn.execute(
        "INSERT INTO pending_memories (content, created_at, status) "
        "VALUES ('legacy', '2026-01-01T00:00:00', 'pending')"
    )
    conn.commit()
    conn.close()

    # get_pending() opens via _get_db which self-migrates (adds profile_id).
    default_items = get_pending(base_dir=tmp_path, profile_id="default")
    assert len(default_items) == 1
    assert default_items[0]["content"] == "legacy"
    # And it is invisible to any other profile's drain.
    assert get_pending(base_dir=tmp_path, profile_id="other") == []
