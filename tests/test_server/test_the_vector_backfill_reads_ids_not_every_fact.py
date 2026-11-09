# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The start-up vector repair finds what is missing without loading every fact.

It used to call ``get_all_facts`` (every row, every embedding decoded) on every
daemon start, even when nothing was missing: seconds of read and decode while
the first recalls waited, and ~1.2 GB of extra memory on a 22k-fact store.
"""

from __future__ import annotations

import inspect
import json
import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.server import vector_backfill
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.schema import create_all_tables

DIM = 4


def _store(tmp_path: Path) -> DatabaseManager:
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(path))
    create_all_tables(conn)
    # Arrives with a later migration on real stores; the predicate checks for it.
    conn.execute("ALTER TABLE atomic_facts ADD COLUMN archive_status TEXT DEFAULT 'live'")
    conn.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES ('m1','default','s')")
    conn.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('other','Other')")
    conn.execute("INSERT INTO memories (memory_id, profile_id, content) VALUES ('m2','other','s')")
    rows = [
        ("f-indexed", "default", [0.1, 0.2, 0.3, 0.4], 0, "live"),
        ("f-missing", "default", [0.5, 0.6, 0.7, 0.8], 0, "live"),
        ("f-no-vector", "default", None, 0, "live"),
        ("f-wrong-dim", "default", [0.1, 0.2], 0, "live"),
        ("f-withheld", "default", [0.9, 0.9, 0.9, 0.9], 1, "live"),
        ("f-archived", "default", [0.9, 0.8, 0.9, 0.9], 0, "archived"),
        ("f-other", "other", [0.4, 0.3, 0.2, 0.1], 0, "live"),
    ]
    for fid, pid, emb, quarantined, archive in rows:
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, embedding, "
            "quarantined, archive_status) VALUES (?,?,?,?,?,?,?)",
            (fid, "m1" if pid == "default" else "m2", pid, f"text {fid}",
             json.dumps(emb) if emb is not None else None, quarantined, archive),
        )
    conn.commit()
    conn.close()
    return DatabaseManager(path)


class _Index:
    def __init__(self, held: set[str]) -> None:
        self._held = held

    def indexed_fact_ids(self, profile_id: str) -> set[str]:
        return set(self._held)


def test_only_the_missing_visible_vector_is_returned(tmp_path: Path) -> None:
    db = _store(tmp_path)
    got = vector_backfill.missing_vectors(db, _Index({"f-indexed"}), "default", DIM)
    assert [(f, p) for f, p, _ in got] == [("f-missing", "default")]
    assert got[0][2] == pytest.approx([0.5, 0.6, 0.7, 0.8])


def test_it_never_loads_every_fact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = _store(tmp_path)
    loaded: list[str] = []
    monkeypatch.setattr(type(db), "get_all_facts",
                        lambda self, *a, **k: loaded.append("all") or [], raising=False)
    real_execute = type(db).execute

    def spy(self, sql, params=()):
        if "SELECT *" in sql.upper() and "ATOMIC_FACTS" in sql.upper():
            loaded.append("select-star")
        return real_execute(self, sql, params)

    monkeypatch.setattr(type(db), "execute", spy)
    got = vector_backfill.missing_vectors(db, _Index({"f-indexed", "f-missing"}), "default", DIM)
    assert got == []
    assert loaded == [], "the repair read every fact to find that nothing was missing"


def test_the_daemon_start_uses_it() -> None:
    from superlocalmemory.server import unified_daemon

    source = inspect.getsource(unified_daemon)
    body = source[source.index("def _backfill_vector_store"):source.index("def _self_heal")]
    assert "missing_vectors(" in body
    assert "get_all_facts(" not in body


def test_every_profile_is_repaired_not_only_default(tmp_path: Path) -> None:
    """Pre-existing: the repair asked the store for ``list_profiles()``, which the
    store does not have, so it always fell back to ``default`` alone."""
    db = _store(tmp_path)
    assert set(vector_backfill.profile_ids(db)) >= {"default", "other"}
    got = vector_backfill.missing_vectors(db, _Index(set()), "other", DIM)
    assert [f for f, _, _ in got] == ["f-other"]


def test_the_daemon_start_repairs_every_profile() -> None:
    from superlocalmemory.server import unified_daemon

    source = inspect.getsource(unified_daemon)
    body = source[source.index("def _backfill_vector_store"):source.index("def _self_heal")]
    assert "profile_ids(db)" in body
