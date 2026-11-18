# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M085: import runs. Registered, deferred, additive, repairable; erasing a memory erases its record."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.storage import _schema_version, profile_fold
from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS
from superlocalmemory.storage.migrations import M084_decision_log as M084
from superlocalmemory.storage.migrations import M085_import_runs as M085


def _run(conn, run_id="r1", profile="default", **extra) -> None:
    values = {"run_id": run_id, "profile_id": profile, "format": "mem0", "state": "done",
              "created_at": "2026-11-18T10:00:00+00:00", **extra}
    conn.execute(f"INSERT INTO import_runs ({', '.join(values)}) "  # noqa: S608
                 f"VALUES ({', '.join('?' * len(values))})", tuple(values.values()))


def _item(conn, run_id="r1", memory="m1", profile="default") -> None:
    conn.execute("INSERT INTO import_run_items VALUES (?, ?, ?, 'fp', 'now')",
                 (run_id, profile, memory))


def test_m085_is_deferred_after_m084_additive_and_repairable() -> None:
    assert _MODULES[M085.NAME] is M085
    catalogued = {m.name: m for m in DEFERRED_MIGRATIONS}
    assert M085.NAME in catalogued and catalogued[M085.NAME].dependencies == (M084.NAME,)
    assert M085.DOWNGRADE_FLOOR == 51 and M085.DB_TARGET == "memory"
    assert _schema_version.SUPPORTED_SCHEMA_VERSION >= 85
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE memories (memory_id TEXT PRIMARY KEY)")
    assert not M085.verify(conn)
    M085.apply(conn)
    M085.apply(conn)
    assert M085.verify(conn)
    conn.execute("DROP TRIGGER trg_import_items_memory_erased")
    assert not M085.verify(conn)
    M085.repair(conn)
    assert M085.verify(conn)


def test_erasing_a_memory_removes_the_runs_record_of_it_and_only_it() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE memories (memory_id TEXT PRIMARY KEY)")
    M085.apply(conn)
    _run(conn)
    for memory in ("m1", "m2"):
        conn.execute("INSERT INTO memories VALUES (?)", (memory,))
        _item(conn, memory=memory)
    conn.execute("DELETE FROM memories WHERE memory_id = 'm1'")
    assert [r[0] for r in conn.execute("SELECT memory_id FROM import_run_items")] == ["m2"]
    assert conn.execute("SELECT COUNT(*) FROM import_runs").fetchone()[0] == 1  # the run stays


def test_a_store_without_memories_builds_the_tables_without_the_erase_trigger() -> None:
    conn = sqlite3.connect(":memory:")
    M085.apply(conn)
    assert M085.verify(conn)
    assert not conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'trigger'").fetchone()


def test_the_checks_hold_invariants_only() -> None:
    conn = sqlite3.connect(":memory:")
    M085.apply(conn)
    _run(conn, format="anything-new", state="some_future_state")  # no list of formats or states
    for bad in ({"format": ""}, {"state": ""}, {"total_items": -1}, {"cancel_requested": 2},
                {"dispute_cap": -1}):
        with pytest.raises(sqlite3.IntegrityError):
            _run(conn, run_id="bad", **bad)
    with pytest.raises(sqlite3.IntegrityError):
        _run(conn)  # the run id is unique
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO import_run_items VALUES ('r1', 'default', 'm', '', 'now')")


def test_the_tables_hold_no_memory_text_and_the_fold_moves_both() -> None:
    conn = sqlite3.connect(":memory:")
    M085.apply(conn)
    columns = {t: {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}
               for t in ("import_runs", "import_run_items")}
    assert not ({"content", "text", "raw_content"} & columns["import_runs"] | columns["import_run_items"]
                & {"content", "text", "raw_content"})
    assert all("profile_id" in cols for cols in columns.values())  # erased with their profile
    assert profile_fold.DECISIONS["import_runs"][0] == profile_fold.MOVE
    assert profile_fold.DECISIONS["import_run_items"][0] == profile_fold.MOVE
    assert "import_run_items" in profile_fold.__doc__ and "import_runs" in profile_fold.__doc__


def test_a_real_store_gets_the_tables_and_the_trigger_from_the_runner(tmp_path) -> None:
    from tests.test_core.test_coverage_backfill import _db

    db = _db(tmp_path / "memory.db")
    names = {r[0] for r in db.execute("SELECT name FROM sqlite_master")}
    assert {"import_runs", "import_run_items", "trg_import_items_memory_erased"} <= names
    applied = {dict(r)["name"] for r in db.execute("SELECT name FROM migration_log "
                                                   "WHERE status = 'complete'")}
    assert M085.NAME in applied
