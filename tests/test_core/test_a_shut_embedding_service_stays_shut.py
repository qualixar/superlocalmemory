# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Once an embedding service is shut down it stays shut (4.1.20 audit C-1).

``MemoryEngine.close()`` shuts its embedder down. An embed already waiting on
the worker used to sit out the full response timeout (180 s by default), then
treat the closed pipe as a crashed worker and spawn a fresh 1 GB model process
nobody would ever stop -- and because that wait ran on an executor thread, the
interpreter could not exit until it ended (~184 s measured).

The workers here are tiny Python children standing in for the model process:
one that never answers and one that answers after an optional delay. No model,
no network.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from superlocalmemory.core import embeddings as emb_mod
from superlocalmemory.core.config import EmbeddingConfig
from superlocalmemory.core.embeddings import EmbeddingService
from tests.helpers import fake_embedding_worker as fake

REPO_SRC = str(Path(__file__).resolve().parents[2] / "src")

#: The response timeout every in-process test here runs under.
_PATCHED_RESPONSE_TIMEOUT_S = 60
#: Liveness bound, NOT a performance claim. An embed that shutdown fails to
#: wake is still waiting on its 60 s response timeout long after this; one that
#: is woken ends within a ``_CANCEL_POLL_SECONDS`` poll. The gap is the margin.
_WAKE_BOUND_S = _PATCHED_RESPONSE_TIMEOUT_S / 4


@pytest.fixture()
def spawner(monkeypatch):
    made: list[fake.Spawner] = []

    def install(*codes: str, on_spawn=None) -> fake.Spawner:
        sp = fake.install(monkeypatch, *codes, on_spawn=on_spawn)
        made.append(sp)
        return sp

    # Long enough that only a wake-up -- never the timeout -- can end a wait.
    monkeypatch.setattr(emb_mod, "_SUBPROCESS_RESPONSE_TIMEOUT", _PATCHED_RESPONSE_TIMEOUT_S)
    yield install
    for sp in made:
        sp.reap()


def _service() -> EmbeddingService:
    return EmbeddingService(EmbeddingConfig(dimension=4))


def _signal_when_waiting(monkeypatch) -> threading.Event:
    """Set once an embed has written its request and starts waiting on the worker."""
    waiting = threading.Event()
    real = EmbeddingService._readline_with_timeout

    def _readline(stream, timeout_seconds, *, cancelled=None):
        waiting.set()
        return real(stream, timeout_seconds, cancelled=cancelled)

    monkeypatch.setattr(EmbeddingService, "_readline_with_timeout", staticmethod(_readline))
    return waiting


def test_an_embed_in_flight_stops_at_shutdown_and_spawns_nothing(spawner, monkeypatch) -> None:
    sp = spawner()
    waiting = _signal_when_waiting(monkeypatch)
    svc = _service()
    out: dict = {}

    def _embed() -> None:
        out["v"] = svc.embed("in flight")

    th = threading.Thread(target=_embed, name="in-flight-embed", daemon=True)
    th.start()
    # The request is written and the embed is waiting on the silent worker.
    assert waiting.wait(timeout=_WAKE_BOUND_S), "the embed never reached the worker"
    assert len(sp.procs) == 1

    svc.shutdown(timeout=1.0)
    th.join(timeout=_WAKE_BOUND_S)

    assert not th.is_alive(), "the in-flight embed sat out its timeout after shutdown"
    assert out["v"] is None
    assert len(sp.procs) == 1, "a worker was respawned after shutdown"
    assert sp.alive() == []


def test_an_embed_after_shutdown_returns_at_once_and_spawns_nothing(spawner) -> None:
    sp = spawner()
    svc = _service()
    svc.shutdown(timeout=0.1)
    out: dict = {}

    def _embed_after_shutdown() -> None:
        out["one"] = svc.embed("after")
        out["batch"] = svc.embed_batch(["a", "b"])

    th = threading.Thread(target=_embed_after_shutdown, daemon=True)
    th.start()
    th.join(timeout=_WAKE_BOUND_S)

    assert not th.is_alive(), "an embed after shutdown waited on a worker"
    assert out == {"one": None, "batch": [None, None]}
    assert sp.procs == []
    assert svc.is_closed is True


def test_a_shutdown_during_the_spawn_checks_spawns_nothing(spawner, monkeypatch) -> None:
    sp = spawner()
    svc = _service()

    def _check_then_close() -> bool:
        svc._closed = True  # shutdown() started while the check ran
        return True

    monkeypatch.setattr(svc, "_check_memory_pressure", _check_then_close)
    assert svc.embed("racing") is None
    assert sp.procs == []


def test_a_shutdown_while_the_child_starts_leaves_no_orphan(spawner) -> None:
    holder: dict = {}
    sp = spawner(on_spawn=lambda: setattr(holder["svc"], "_closed", True))
    svc = holder["svc"] = _service()

    assert svc.embed("racing") is None
    assert len(sp.procs) == 1
    deadline = time.monotonic() + 5
    while sp.alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert sp.alive() == [], "the child spawned during shutdown was left running"
    assert svc._worker_proc is None


#: The shutdown poll, stretched so a read it delays would take this long.
_AMPLIFIED_POLL_S = 30.0
#: Real-time bound: a ready line is read at once. A read held back by even one
#: poll interval takes ``_AMPLIFIED_POLL_S``, six times this, so host load
#: cannot push a correct read past it nor pull a delayed one under it.
_READY_BOUND_S = _AMPLIFIED_POLL_S / 6


def test_a_ready_response_is_not_delayed_by_the_shutdown_poll(monkeypatch) -> None:
    monkeypatch.setattr(emb_mod, "_CANCEL_POLL_SECONDS", _AMPLIFIED_POLL_S)
    r, w = os.pipe()
    with os.fdopen(r, "r") as reader, os.fdopen(w, "w") as writer:
        writer.write('{"ok": true}\n')
        writer.flush()
        t0 = time.monotonic()
        line = EmbeddingService._readline_with_timeout(
            reader, _AMPLIFIED_POLL_S * 2, cancelled=lambda: False,
        )
    assert line == '{"ok": true}\n'
    assert time.monotonic() - t0 < _READY_BOUND_S


#: The child's response timeout: what an embed nobody wakes would wait out.
_CHILD_RESPONSE_TIMEOUT_S = 120
#: Liveness bound, NOT a performance claim: an exit held by the in-flight embed
#: lasts the whole 120 s response timeout, so half of it separates the two.
_EXIT_BOUND_S = _CHILD_RESPONSE_TIMEOUT_S / 2


def test_interpreter_exit_is_not_held_by_an_embed_in_flight(tmp_path) -> None:
    """No close() at all: the process must still exit promptly."""
    script = textwrap.dedent(
        f"""
        import concurrent.futures, subprocess, sys, threading
        from types import SimpleNamespace
        from superlocalmemory.core import embeddings as E
        from superlocalmemory.core.config import EmbeddingConfig

        def _popen(argv, *a, **k):
            return subprocess.Popen([sys.executable, "-c", {fake.SILENT_WORKER!r}], *a, **k)
        E.subprocess = SimpleNamespace(
            Popen=_popen, PIPE=subprocess.PIPE, DEVNULL=subprocess.DEVNULL)
        E.EmbeddingService._check_memory_pressure = staticmethod(lambda: True)
        waiting = threading.Event()
        _real_readline = E.EmbeddingService._readline_with_timeout
        def _readline(stream, timeout_seconds, *, cancelled=None):
            waiting.set()  # the request is written and the embed is waiting
            return _real_readline(stream, timeout_seconds, cancelled=cancelled)
        E.EmbeddingService._readline_with_timeout = staticmethod(_readline)
        svc = E.EmbeddingService(EmbeddingConfig(dimension=4))
        pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="slm-sg-embed")
        pool.submit(svc.embed, "in flight at exit")
        if not waiting.wait(30):
            sys.exit("the embed never reached the worker")
        print("exiting", flush=True)
        """
    )
    env = {
        **os.environ,
        "PYTHONPATH": REPO_SRC,
        "SLM_EMBED_RESPONSE_TIMEOUT": str(_CHILD_RESPONSE_TIMEOUT_S),
        "SLM_DATA_DIR": str(tmp_path),
        "HOME": str(tmp_path),
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True,
            text=True, timeout=_EXIT_BOUND_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"exit was held past {_EXIT_BOUND_S:.0f}s by the in-flight embed")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "exiting" in proc.stdout
    assert "did not respond" not in proc.stderr
