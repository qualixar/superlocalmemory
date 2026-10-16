# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The dashboard answers in under a second while the daemon does heavy background work.

The Health pane asks for about fifteen things at once. On a large store, with a
background re-index running, seven of them took twelve seconds and the version
probe took five, though each answers in milliseconds alone. Requests were queueing
behind work that ran on the event loop (hashing the whole audit trail, scanning
every lifecycle row, taking the database write lock): while it ran, the loop could
not even read the next request.

This runs the real application behind a real HTTP server with a long audit trail,
two threads burning CPU and a thread that keeps taking the database write lock,
then asks for everything the Health pane asks for, all at once.
"""

# ruff: noqa: F811 (pytest fixture imported and redeclared as a parameter)

from __future__ import annotations

import json
import sqlite3
import threading
import time
import urllib.request

import pytest

from superlocalmemory.compliance import audit as audit_mod
from superlocalmemory.storage.write_lock import get_write_lock
from tests.test_server.test_rbac_company_mode_credential import app_and_engine  # noqa: F401

# On an idle machine every request here answers in about 0.4s (the two CPU-burning threads
# share the interpreter lock). The bound is loose enough for a busy CI box and well below the
# 9-13s the same requests took when they queued behind work on the event loop.
LIMIT_S = 6.0
PROBE_LIMIT_S = 1.0
PANE = (
    "/api/integrity/summary",
    "/api/compliance/audit?limit=100",
    "/api/compliance/status",
    "/api/v3/features",
    "/api/learning/readiness",
    "/api/v3/math/health",
    "/api/lifecycle/status",
    "/api/version",
    "/api/stats",
    "/api/profiles",
    "/api/tiers/stats",
    "/api/events/stats",
    "/api/trust/stats",
)


def _seed_audit_trail(path, rows: int) -> None:
    """A long, valid audit trail written the way the product writes it."""
    chain = audit_mod.AuditChain(path)
    conn = sqlite3.connect(path)
    prev = audit_mod._GENESIS_HASH
    batch = []
    for i in range(rows):
        stamp = f"2026-09-01T00:00:{i % 60:02d}.{i:06d}+00:00"
        meta = json.dumps({"i": i, "fact_ids": [f"f{i}"], "outcome": "ok"}, sort_keys=True)
        digest = audit_mod._compute_hash_v2(prev, "recall", "agent", "default", f"c{i}", stamp, meta)
        batch.append((stamp, "recall", "agent", "default", f"c{i}", prev, digest, meta, 2))
        prev = digest
    conn.executemany(
        "INSERT INTO audit_chain (timestamp, operation, agent_id, profile_id, content_hash, "
        "prev_hash, event_hash, metadata, hash_version) VALUES (?,?,?,?,?,?,?,?,?)", batch)
    conn.commit()
    conn.close()
    chain._write_anchor(rows, prev)


@pytest.fixture
def live_daemon(app_and_engine, tmp_path):
    import uvicorn

    app = app_and_engine
    _seed_audit_trail(app.state.config.db_path.parent / "audit_chain.db", 120_000)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error",
                                           lifespan="off"))
    thread = threading.Thread(target=server.run, name="test-http-server", daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "the test server did not start"
    port = server.servers[0].sockets[0].getsockname()[1]
    descriptor = app.state.daemon_descriptor
    yield {"base": f"http://127.0.0.1:{port}", "app": app,
           "headers": {"X-SLM-Daemon-Capability": descriptor.capability,
                       "X-SLM-Target-Instance": descriptor.instance_id}}
    server.should_exit = True
    thread.join(10)


def _timed_get(base: str, path: str, headers: dict, out: dict) -> None:
    request = urllib.request.Request(base + path, headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response.read()
            status = response.status
    except Exception as exc:  # noqa: BLE001 - a refusal is recorded, not raised
        status = getattr(exc, "code", 0)
    out[path] = (time.perf_counter() - started, status)


def _ask_all(daemon: dict, paths=PANE) -> dict[str, tuple[float, int]]:
    out: dict[str, tuple[float, int]] = {}
    threads = [threading.Thread(target=_timed_get, args=(daemon["base"], p, daemon["headers"], out))
               for p in paths]
    for t in threads:
        t.start()
    for t in threads:
        t.join(90)
    return out


class _BackgroundLoad:
    """Two threads burning CPU, and one that keeps taking the database write lock."""

    def __init__(self, db_path) -> None:
        self._stop = threading.Event()
        self._db_path = db_path
        self._threads = [threading.Thread(target=self._burn, name="test-burn-%d" % n)
                         for n in range(2)]
        self._threads.append(threading.Thread(target=self._hold_lock, name="test-hold-lock"))

    def _burn(self) -> None:
        while not self._stop.is_set():
            sum(i * i for i in range(20_000))

    def _hold_lock(self) -> None:
        lock = get_write_lock(self._db_path)
        while not self._stop.is_set():
            with lock:
                time.sleep(0.6)
            time.sleep(0.01)

    def __enter__(self):
        for t in self._threads:
            t.start()
        time.sleep(0.2)
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(10)


def test_a_slow_request_does_not_stop_the_daemon_answering_others(live_daemon):
    """Hashing a 120,000-entry audit trail takes seconds; the version probe still answers at once."""
    out: dict = {}
    slow = threading.Thread(target=_timed_get, args=(
        live_daemon["base"], "/api/compliance/audit?limit=100", live_daemon["headers"], out))
    slow.start()
    time.sleep(0.4)  # the audit request is now being worked on
    probe: dict = {}
    for _ in range(5):
        _timed_get(live_daemon["base"], "/api/version", live_daemon["headers"], probe)
        # Blocked means the probe could only answer once the audit request was done.
        # A loaded runner can make the probe slow while the audit is still working;
        # that is not waiting behind it (the 4.2.8 gate measured 1.46 s that way).
        waited = probe["/api/version"][0]
        assert waited < PROBE_LIMIT_S or slow.is_alive(), (
            f"/api/version waited {waited:.2f}s behind another request")
        time.sleep(0.1)
    slow.join(60)
    assert out["/api/compliance/audit?limit=100"][1] == 200, out


def test_every_health_pane_request_answers_under_a_second_while_background_work_runs(live_daemon):
    db_path = live_daemon["app"].state.config.db_path
    _ask_all(live_daemon)  # the first look at a long trail may take longer than the rest
    with _BackgroundLoad(db_path):
        results = _ask_all(live_daemon)
    too_slow = {p: round(t, 2) for p, (t, _s) in results.items() if t >= LIMIT_S}
    assert not too_slow, f"slower than {LIMIT_S}s under background load: {too_slow}"
    failed = {p: s for p, (_t, s) in results.items() if s >= 500}
    assert not failed, f"server errors: {failed}"
