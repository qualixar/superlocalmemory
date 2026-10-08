# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The daemon reads its store once at start, so the first recalls find it in memory.

Measured on a copy of a 2 GB, 22k-fact store with a cold OS cache: the first
lookup of a popular entity took 0.5-3 s and the 50-newest query 5.8 s, because
each reads rows scattered across the file. Reading the whole file in order
took 354 ms; afterwards the same lookup took 11 ms. Nothing is changed: the
bytes are read and dropped.
"""

from __future__ import annotations

import inspect
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from superlocalmemory.storage import store_cache_warm


def _file(path: Path, size: int) -> Path:
    path.write_bytes(b"x" * size)
    return path


def test_the_store_and_its_journal_are_read_in_full(tmp_path: Path) -> None:
    db = _file(tmp_path / "memory.db", 3 * store_cache_warm.CHUNK + 17)
    wal = _file(tmp_path / "memory.db-wal", 1234)
    expected = db.stat().st_size + wal.stat().st_size
    assert store_cache_warm.warm([db, wal], max_bytes=10**9) == expected


def test_a_store_larger_than_the_cap_is_not_read(tmp_path: Path) -> None:
    big = _file(tmp_path / "memory.db", 4096)
    small = _file(tmp_path / "memory.db-wal", 100)
    assert store_cache_warm.warm([big, small], max_bytes=1000) == 100


def test_a_missing_journal_is_skipped(tmp_path: Path) -> None:
    db = _file(tmp_path / "memory.db", 500)
    assert store_cache_warm.warm([db, tmp_path / "memory.db-wal"], max_bytes=10**9) == 500


def test_the_cap_follows_this_computer_s_memory(tmp_path: Path, monkeypatch) -> None:
    db = _file(tmp_path / "memory.db", 2048)
    engine = SimpleNamespace(_db=SimpleNamespace(db_path=db))
    monkeypatch.setattr(store_cache_warm, "total_ram_gb", lambda: 16.0)
    assert store_cache_warm.warm_engine_store(engine) == 2048
    monkeypatch.setattr(store_cache_warm, "total_ram_gb", lambda: 0.0)  # unknown: do nothing
    assert store_cache_warm.warm_engine_store(engine) == 0
    assert store_cache_warm.warm_engine_store(SimpleNamespace(_db=None)) == 0


def test_reading_the_store_keeps_this_process_s_sqlite_locks(
    tmp_path: Path, monkeypatch,
) -> None:
    db = tmp_path / "memory.db"
    service = sqlite3.connect(db)
    service.execute("PRAGMA journal_mode=WAL")
    service.execute("CREATE TABLE t(x)")
    service.execute("INSERT INTO t VALUES (1)")
    service.commit()
    engine = SimpleNamespace(_db=SimpleNamespace(db_path=db))
    monkeypatch.setattr(store_cache_warm, "total_ram_gb", lambda: 16.0)

    assert store_cache_warm.warm_engine_store(engine) > 0

    # Another process (a hook, the MCP server) opens and closes the store. While
    # the service still holds its lock, that close must leave the -wal in place.
    subprocess.run(
        [sys.executable, "-c",
         "import sqlite3, sys; c = sqlite3.connect(sys.argv[1]); "
         "c.execute('SELECT * FROM t').fetchall(); c.close()", str(db)],
        check=True,
    )
    assert Path(f"{db}-wal").exists()
    assert service.execute("SELECT x FROM t").fetchall() == [(1,)]
    service.close()


def test_a_child_that_cannot_run_reads_nothing(tmp_path: Path, monkeypatch) -> None:
    db = _file(tmp_path / "memory.db", 500)
    monkeypatch.setattr(store_cache_warm.sys, "executable", str(tmp_path / "missing"))
    assert store_cache_warm.warm_in_child([db], max_bytes=10**9) == 0


def test_the_daemon_starts_it() -> None:
    from superlocalmemory.server import unified_daemon

    source = inspect.getsource(unified_daemon.lifespan)
    assert "store_cache_warm.start(engine)" in source
