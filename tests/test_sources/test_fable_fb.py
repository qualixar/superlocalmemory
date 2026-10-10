"""Fable audit, package FB: folder bookkeeping that must not lie about what is hidden or saved."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from superlocalmemory.sources import ingest
from tests.test_sources.conftest import ListDb


def _runtime(*rows):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("CREATE TABLE atomic_facts(fact_id, memory_id, lifecycle)")
    conn.executemany("INSERT INTO atomic_facts VALUES (?, ?, ?)", rows)
    return SimpleNamespace(_db=ListDb(conn))


def test_any_archived_sees_archived_memories_in_the_list_the_database_returns():
    """F-2: the database hands back a list; reading it as a cursor made this always False."""
    runtime = _runtime(("f1", "m1", "archived"), ("f2", "m2", "active"))
    assert ingest.any_archived(runtime, ["m1"]) is True
    assert ingest.any_archived(runtime, ["m2"]) is False
    assert ingest.any_archived(runtime, ["m1", "m2"]) is True
    assert ingest.any_archived(runtime, []) is False
