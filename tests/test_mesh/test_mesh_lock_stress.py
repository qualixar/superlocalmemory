# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Heavy mesh traffic, long memory writes and a CLI process writing memory.db,
all at once: with mesh.db no write sees "database is locked".

The memory writer holds each transaction for 0.3-1.2 s (a backfill batch on a
busy store), through the canonical ``memory_write``. A separate OS process
writes ``tool_events`` the way ``slm ingest`` does. Four mesh threads register,
heartbeat, send, set state and take locks. Every "database is locked" is
counted, including the ones the broker's retry loop absorbed.

The ``legacy`` run puts the mesh tables back in memory.db (4.2.7) to show the
harness does produce the collision it guards against.
"""

from __future__ import annotations

import random
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from superlocalmemory.mesh import broker as broker_mod
from superlocalmemory.mesh.broker import MeshBroker
from superlocalmemory.mesh.mesh_store import create_schema
from superlocalmemory.storage.memory_write import memory_write

RUN_S = 6.0
_CLI = textwrap.dedent("""
    import sqlite3, sys, time
    db, until, locked, wrote = sys.argv[1], time.time() + float(sys.argv[2]), 0, 0
    while time.time() < until:
        conn = sqlite3.connect(db, timeout=10)
        try:
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("INSERT INTO tool_events (tool_name, created_at) VALUES ('cli', 'x')")
            conn.commit()
            wrote += 1
        except sqlite3.OperationalError as exc:
            locked += "locked" in str(exc)
        finally:
            conn.close()
        time.sleep(0.01)
    print(locked, wrote)
""")


def _memory_store(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE stress_rows (id INTEGER PRIMARY KEY, body TEXT)")
    conn.execute("CREATE TABLE tool_events (id INTEGER PRIMARY KEY, tool_name TEXT,"
                 " created_at TEXT)")
    conn.commit()
    conn.close()
    return path


class _Counter:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.locked = 0  # every "database is locked" a mesh call met, retried or not
        self.failed = 0  # mesh calls that gave up
        self.calls = 0

    def bump(self, field: str) -> None:
        with self.lock:
            setattr(self, field, getattr(self, field) + 1)


def _memory_writer(db: Path, stop: threading.Event, errors: list) -> None:
    rnd = random.Random(1)
    while not stop.is_set():
        try:
            with memory_write(db) as conn:
                conn.execute("INSERT INTO stress_rows (body) VALUES (?)", ("x" * 2048,))
                time.sleep(rnd.uniform(0.3, 1.2))  # a long backfill batch
        except sqlite3.OperationalError as exc:
            errors.append(str(exc))
        stop.wait(rnd.uniform(0.05, 0.15))  # the next batch is being prepared


def _mesh_worker(broker: MeshBroker, n: int, stop: threading.Event, c: _Counter) -> None:
    peer = broker.register_peer(f"s{n}")["peer_id"]
    i = 0
    while not stop.is_set():
        i += 1
        ops = (lambda: broker.heartbeat(peer), lambda: broker.register_peer(f"s{n}-{i % 5}"),
               lambda: broker.send_message(peer, "broadcast", f"m{i}"),
               lambda: broker.set_state(f"k{n}", str(i), peer),
               lambda: broker.update_summary(peer, f"sum {i}"))
        try:
            ops[i % len(ops)]()
            c.bump("calls")
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc):
                raise
            c.bump("failed")


def _run(memory_db: Path, broker: MeshBroker, monkeypatch) -> tuple[_Counter, list, str]:
    counter = _Counter()
    real = broker_mod.MeshBroker._is_transient_lock

    def counting(exc):
        hit = real(exc)
        if hit:
            counter.bump("locked")
        return hit

    monkeypatch.setattr(broker_mod.MeshBroker, "_is_transient_lock", staticmethod(counting))
    stop, mem_errors = threading.Event(), []
    cli = subprocess.Popen([sys.executable, "-c", _CLI, str(memory_db), str(RUN_S)],
                           stdout=subprocess.PIPE, text=True)
    threads = [threading.Thread(target=_memory_writer, args=(memory_db, stop, mem_errors))]
    threads += [threading.Thread(target=_mesh_worker, args=(broker, n, stop, counter))
                for n in range(4)]
    for t in threads:
        t.start()
    time.sleep(RUN_S)
    stop.set()
    for t in threads:
        t.join(30)
    out, _ = cli.communicate(timeout=60)
    return counter, mem_errors, out.strip()


def test_mesh_db_writes_never_meet_a_locked_memory_db(tmp_path: Path, monkeypatch) -> None:
    memory_db = _memory_store(tmp_path / "memory.db")
    broker = MeshBroker.for_store(memory_db)
    counter, mem_errors, cli = _run(memory_db, broker, monkeypatch)
    cli_locked, cli_wrote = (int(x) for x in cli.split())
    print(f"\nmesh.db: calls={counter.calls} locked={counter.locked} failed={counter.failed} "
          f"memory_errors={len(mem_errors)} cli_locked={cli_locked} cli_wrote={cli_wrote}")
    assert counter.calls > 50 and cli_wrote > 10
    assert (counter.locked, counter.failed, mem_errors, cli_locked) == (0, 0, [], 0)


@pytest.mark.slow
def test_legacy_layout_does_collide(tmp_path: Path, monkeypatch) -> None:
    """Non-vacuity: the same load with the mesh inside memory.db meets the lock."""
    memory_db = _memory_store(tmp_path / "memory.db")
    conn = sqlite3.connect(memory_db)
    create_schema(conn)
    conn.close()
    broker = MeshBroker(memory_db)
    counter, _mem_errors, _cli = _run(memory_db, broker, monkeypatch)
    print(f"\nlegacy memory.db: calls={counter.calls} locked={counter.locked} "
          f"failed={counter.failed}")
    assert counter.locked > 0
