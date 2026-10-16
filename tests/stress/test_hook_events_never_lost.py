# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Hook events under write load: none lost, none locked, the hook stays fast.

Real hook processes (``python -m superlocalmemory.cli hook post_tool_outcome``,
what every host runs on every tool call) fire while a memory writer holds
memory.db for 0.3-1.2 s per transaction (a backfill batch on a busy store) and
mesh threads register, heartbeat, send and set state. The "daemon" here is the
real route and the real writer behind a real HTTP server and ``daemon.json``.

Before 4.2.8 the hook wrote ``tool_events`` itself with a 50 ms busy timeout
and swallowed the failure: 29/145 and 63/88 events lost in the run that found
it (``test_the_old_direct_write_loses_events_under_the_same_load`` keeps that
measurable).
"""

from __future__ import annotations

import json
import os
import random
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest
from tests._portable import child_env_base
from tests.helpers.hook_event_db import bootstrap, query, seed_pending

import superlocalmemory
from superlocalmemory.server import hook_event_writer as hew
from superlocalmemory.storage.memory_write import memory_write

REPO_SRC = str(Path(superlocalmemory.__file__).resolve().parents[1])
#: Concurrent hook processes, scaled to the machine so the load (not the
#: runner's core count) is what the test measures. 8 on a developer Mac.
WORKERS = min(8, max(2, (os.cpu_count() or 2) // 2))
PER_WORKER = 20
#: The hook's own time (stdin to exit, measured in-process, perf log). The
#: bound itself is gated deterministically in test_hook_event_delivery.py (a
#: hung daemon, the hook budget). Here, under real load, the median must stay
#: under the 200 ms cap over the hook socket (POSIX, the default) and under
#: twice it over the HTTP fallback (no socket, e.g. Windows), whose cold
#: ``daemon_request`` imports compete for the GIL after the wait ends. Tails
#: are printed, not asserted: they measure the runner. Nothing is lost either
#: way: what is not acked in time is spooled.
HOOK_CEILING_MS = 200.0


def _memory_writer(db: Path, stop: threading.Event, errors: list) -> None:
    rnd = random.Random(1)
    while not stop.is_set():
        try:
            with memory_write(db) as conn:
                conn.execute("INSERT INTO stress_rows (body) VALUES (?)", ("x" * 2048,))
                time.sleep(rnd.uniform(0.3, 1.2))  # a long backfill batch
        except sqlite3.OperationalError as exc:
            errors.append(str(exc))
        stop.wait(rnd.uniform(0.05, 0.15))


def _mesh_worker(db: Path, n: int, stop: threading.Event, failed: list) -> None:
    from superlocalmemory.mesh.broker import MeshBroker

    broker = MeshBroker(db)
    peer = broker.register_peer(f"mesh-{n}")["peer_id"]
    i = 0
    while not stop.is_set():
        i += 1
        ops = (lambda: broker.heartbeat(peer),
               lambda: broker.send_message(peer, "broadcast", f"m{i}"),
               lambda: broker.set_state(f"k{n}", str(i), peer))
        try:
            ops[i % len(ops)]()
        except sqlite3.OperationalError as exc:
            failed.append(str(exc))
        stop.wait(0.01)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def _daemon(root: Path, *, hook_socket: bool):
    """The real route behind a real HTTP server named by daemon.json, and (as
    the daemon does on POSIX) the real hook socket. The writer is started by
    the test."""
    import uvicorn
    from fastapi import FastAPI

    from superlocalmemory.infra.daemon_identity import build_descriptor, write_descriptor
    from superlocalmemory.server.routes import hook_events

    port = _free_port()
    descriptor = build_descriptor(data_root=root, port=port, version="test",
                                  pid=os.getpid(), state="ready")
    app = FastAPI()
    app.state.daemon_descriptor = descriptor
    app.add_api_route("/health", lambda: {"status": "ok", **descriptor.public_health_fields()})
    app.include_router(hook_events.router)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started
    write_descriptor(descriptor, data_root=root)
    sock = None
    if hook_socket:
        from superlocalmemory.hooks.hook_daemon import HookDaemon

        sock = HookDaemon(queue_db_path=root / "recall_queue.db")
        sock.start()
    try:
        yield
    finally:
        if sock is not None:
            sock.stop()
        (root / "daemon.json").unlink(missing_ok=True)
        server.should_exit = True
        thread.join(10)


def _hook(env: dict, action: str, payload: dict) -> tuple[int, str, float]:
    started = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "superlocalmemory.cli", "hook", action],
                          input=json.dumps(payload), capture_output=True, text=True,
                          env=env, timeout=60)
    return proc.returncode, proc.stderr, (time.perf_counter() - started) * 1000


def _env(tmp_path: Path, root: Path) -> dict:
    (tmp_path / "home").mkdir(exist_ok=True)
    return {**os.environ, **child_env_base(tmp_path / "home"), "SLM_DATA_DIR": str(root),
            "PYTHONPATH": REPO_SRC, "SLM_TEST_ISOLATION": "1"}


def _fire(env: dict, marker_for: dict[int, str]) -> list[tuple[int, str, float]]:
    def worker(n: int) -> list:
        out = []
        for i in range(PER_WORKER):
            marker = marker_for.get(n, "") if i == PER_WORKER // 2 else ""
            out.append(_hook(env, "post_tool_outcome", {
                "session_id": f"stress-{n}", "tool_name": "Edit" if marker else "Bash",
                "tool_input": {"i": i}, "tool_response": f"ok {marker}", "cwd": "/p"}))
        return out

    with ThreadPoolExecutor(WORKERS) as pool:
        return [r for chunk in pool.map(worker, range(WORKERS)) for r in chunk]


def _perf(root: Path) -> list[dict]:
    path = root / "logs" / "hook-perf.log"
    return [json.loads(x) for x in path.read_text().splitlines() if x] if path.exists() else []


@pytest.fixture()
def store(tmp_path: Path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(root))
    (root / ".install_token").write_text("a" * 64, encoding="utf-8")
    import superlocalmemory.core.security_primitives as sec

    monkeypatch.setattr(sec, "_cached_install_token", None, raising=False)
    db = bootstrap(root / "memory.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE stress_rows (id INTEGER PRIMARY KEY, body TEXT)")
    conn.commit()
    conn.close()
    from superlocalmemory.storage.schema_v343 import apply_v343_schema, apply_v346_schema

    apply_v343_schema(str(db))  # the mesh tables, in memory.db as on this branch
    apply_v346_schema(str(db))
    return root, db


def _count_writer_failures(monkeypatch) -> list[str]:
    failures: list[str] = []
    real = hew.HookEventWriter.drain_once

    def counted(self):
        try:
            return real(self)
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
            raise

    monkeypatch.setattr(hew.HookEventWriter, "drain_once", counted)
    return failures


_needs_unix = pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="hook socket is POSIX")


@pytest.mark.parametrize("transport", [pytest.param("socket", marks=_needs_unix), "http"])
def test_hook_events_survive_heavy_write_load(store, tmp_path: Path, monkeypatch,
                                              transport: str) -> None:
    from superlocalmemory.core.recall_pipeline import _emit_marker

    root, db = store
    markers = {n: _emit_marker(f"fact-{n}") for n in range(WORKERS)}
    for n in range(WORKERS):
        seed_pending(db, outcome_id=f"o-{n}", session_id=f"stress-{n}",
                     fact_ids=[f"fact-{n}"], expires_in_ms=600_000)
    writer_failures = _count_writer_failures(monkeypatch)
    stop, mem_errors, mesh_failed = threading.Event(), [], []
    load = [threading.Thread(target=_memory_writer, args=(db, stop, mem_errors))]
    load += [threading.Thread(target=_mesh_worker, args=(db, n, stop, mesh_failed))
             for n in range(2)]
    with _daemon(root, hook_socket=transport == "socket"):
        hew.start_writer(db)
        try:
            for t in load:
                t.start()
            results = _fire(_env(tmp_path, root), markers)
        finally:
            stop.set()
            for t in load:
                t.join(30)
            assert hew.stop_writer(timeout_s=30.0)

    total = WORKERS * PER_WORKER
    rows = query(db, "SELECT metadata FROM tool_events WHERE session_id LIKE 'stress-%'")
    ids = {json.loads(r["metadata"])["event_id"] for r in rows}
    perf = [p for p in _perf(root) if p["hook"] == "post_tool_outcome"]
    outcomes = {p["outcome"] for p in perf}
    hook_ms = sorted(p["duration_ms"] for p in perf)
    wall_ms = sorted(r[2] for r in results)
    signals = {r["outcome_id"]: json.loads(r["signals_json"])
               for r in query(db, "SELECT outcome_id, signals_json FROM pending_outcomes")}
    print(f"\n[{transport}] hook events: fired={total} written={len(rows)} distinct={len(ids)} "
          f"outcomes={sorted(outcomes)} "
          f"via_daemon={sum(p['outcome'].startswith('daemon') for p in perf)} "
          f"writer_failures={len(writer_failures)} memory_errors={len(mem_errors)} "
          f"mesh_failed={len(mesh_failed)}\n"
          f"hook in-process ms p50={hook_ms[len(hook_ms)//2]:.1f} "
          f"p95={hook_ms[int(len(hook_ms)*.95)]:.1f} max={hook_ms[-1]:.1f}; "
          f"process wall ms p50={wall_ms[len(wall_ms)//2]:.0f} max={wall_ms[-1]:.0f}")
    assert all(rc == 0 and "slm-hook" not in err for rc, err, _ in results), results[:3]
    assert (len(rows), len(ids)) == (total, total)
    assert len(perf) == total and outcomes <= {"daemon", "spooled", "daemon_signal",
                                               "spooled_signal"}
    assert all(signals[f"o-{n}"].get("edit") is True for n in range(WORKERS))
    assert writer_failures == [] and mem_errors == [] and mesh_failed == []
    ceiling = HOOK_CEILING_MS if transport == "socket" else 2 * HOOK_CEILING_MS
    assert hook_ms[len(hook_ms) // 2] < ceiling
    assert (root / "hook_spool").is_dir() and not list((root / "hook_spool").glob("*.draining"))


def test_daemon_down_events_are_spooled_then_drained_exactly_once(store, tmp_path) -> None:
    root, db = store
    env = _env(tmp_path, root)
    results = [_hook(env, "post_tool_outcome", {"session_id": "down", "tool_name": "Read",
                                                "tool_response": f"r{i}"}) for i in range(12)]
    results.append(_hook(env, "stop_outcome", {"session_id": "down"}))
    assert all(rc == 0 for rc, _err, _ms in results)
    assert query(db, "SELECT COUNT(*) AS n FROM tool_events")[0]["n"] == 0
    assert {p["outcome"] for p in _perf(root)} == {"spooled", "finalize_spooled"}

    writer = hew.HookEventWriter(db, root=root / "hook_spool")
    first = writer.drain_once()
    hew.HookEventWriter(db, root=root / "hook_spool").drain_once()  # a restart
    assert first.applied == 13
    assert query(db, "SELECT COUNT(*) AS n FROM tool_events")[0]["n"] == 12


@pytest.mark.slow
def test_the_old_direct_write_loses_events_under_the_same_load(store) -> None:
    """Non-vacuity: the 4.2.7 hook write pattern under this harness's load."""
    _root, db = store
    stop, errors = threading.Event(), []
    writer = threading.Thread(target=_memory_writer, args=(db, stop, errors))
    writer.start()
    ok = lost = 0
    end = time.monotonic() + 6
    while time.monotonic() < end:
        conn = sqlite3.connect(str(db), timeout=0.05)
        try:
            conn.execute("PRAGMA busy_timeout=50")
            conn.execute("INSERT INTO tool_events (session_id, tool_name, created_at) "
                         "VALUES ('old', 'Bash', 'x')")
            conn.commit()
            ok += 1
        except sqlite3.OperationalError as exc:
            lost += "locked" in str(exc)
        finally:
            conn.close()
        time.sleep(0.02)
    stop.set()
    writer.join(30)
    print(f"\n4.2.7 direct write: ok={ok} lost_to_locked={lost}")
    assert lost > 0
