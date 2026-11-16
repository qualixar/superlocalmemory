# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""SLM gives memory back when the machine runs short (16 Nov 2026 incident).

A laptop hung with swap full while SLM held a 3.3 GB picture worker idle (it
released only after 30 minutes and ignored SIGTERM) and a 2 GB reranker, all
under SLM's own 40% budget. These tests hold the fix: one machine-wide pressure
signal; the health monitor releases loaded models under pressure, least harm
first; a released reranker is not reloaded while the machine is pressed; the
picture worker leaves after two idle minutes; and every worker exits at once on
SIGTERM, whatever threads its libraries started.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from superlocalmemory.core import health_monitor as hm
from superlocalmemory.infra import memory_pressure as mp

POSIX = sys.platform != "win32"


# -- the signal -----------------------------------------------------------------------------------

def _machine(monkeypatch, *, level, available, floor=2048.0):
    from superlocalmemory.core import embedding_memory

    monkeypatch.setattr(mp, "kernel_level", lambda: level)
    monkeypatch.setattr(embedding_memory, "available_mb", lambda: available)
    monkeypatch.setattr(embedding_memory, "load_floor_mb", lambda: floor)


@pytest.mark.real_memory_pressure
@pytest.mark.parametrize(("level", "available", "pressed"), [
    (mp.LEVEL_NORMAL, 8000.0, False),
    (mp.LEVEL_WARNING, 8000.0, True),     # the OS says so, whatever the free count
    (mp.LEVEL_CRITICAL, 8000.0, True),
    (mp.LEVEL_NORMAL, 1900.0, True),      # under the embedder's own 2 GB floor
    (None, 1900.0, True),                 # no kernel signal (Linux, Windows): the floor
    (None, None, False),                  # nothing readable: never stop SLM working
])
def test_the_machine_is_pressed_by_the_os_signal_or_the_floor(monkeypatch, level, available,
                                                              pressed) -> None:
    _machine(monkeypatch, level=level, available=available)
    got = mp.reading()
    assert got.pressed is pressed and (bool(got.reason) is pressed)


@pytest.mark.real_memory_pressure
def test_an_unreadable_machine_is_not_pressed(monkeypatch) -> None:
    def boom():
        raise RuntimeError("no sysctl")

    monkeypatch.setattr(mp, "reading", boom)
    assert mp.pressed() is False


@pytest.mark.real_memory_pressure
def test_the_kernel_level_is_read_on_macos() -> None:
    level = mp.kernel_level()
    if sys.platform == "darwin":
        assert level in (mp.LEVEL_NORMAL, mp.LEVEL_WARNING, mp.LEVEL_CRITICAL)
    else:
        assert level is None


# -- the monitor releases, least harm first -------------------------------------------------------

def _w(pid, ident, mb=1000.0):
    return {"pid": pid, "rss_mb": mb, "cmdline": f"python {ident}"}


PICTURE = _w(11, "/x/multimodal_worker.py", 3300)
RERANKER = _w(12, "-m superlocalmemory.core.reranker_worker", 2000)
EMBEDDER = _w(13, "-m superlocalmemory.core.embedding_worker", 900)
PRESSED = mp.Reading(True, 1900.0, 2048.0, mp.LEVEL_WARNING, "pressure")
CALM = mp.Reading(False, 8000.0, 2048.0, mp.LEVEL_NORMAL, "")


@pytest.fixture
def stopped(monkeypatch):
    got: list[int] = []
    monkeypatch.setattr(hm, "stop_process", lambda pid, *a, **k: got.append(pid) or True)
    return got


def _monitor():
    return hm.HealthMonitor.__new__(hm.HealthMonitor)


def test_a_calm_machine_releases_nothing(stopped) -> None:
    assert _monitor()._relieve_pressure([PICTURE, RERANKER, EMBEDDER], CALM) is None
    assert stopped == []


def test_pressure_releases_the_picture_worker_first(stopped) -> None:
    assert _monitor()._relieve_pressure([EMBEDDER, RERANKER, PICTURE], PRESSED) is PICTURE
    assert stopped == [PICTURE["pid"]]  # one per check: release is gradual


def test_then_the_reranker(stopped) -> None:
    assert _monitor()._relieve_pressure([EMBEDDER, RERANKER], PRESSED) is RERANKER
    assert stopped == [RERANKER["pid"]]


def test_the_embedder_is_never_released_by_pressure(stopped) -> None:
    assert _monitor()._relieve_pressure([EMBEDDER], PRESSED) is None and stopped == []


@pytest.mark.real_memory_pressure
def test_pressure_acts_whatever_slms_own_budget_says(stopped, monkeypatch) -> None:
    """The incident: SLM was under its 40% budget while the machine had no memory left."""
    monitor = _monitor()
    monitor._budget_mb = 10 ** 6
    monkeypatch.setattr(mp, "reading", lambda: PRESSED)
    assert monitor._relieve_pressure([PICTURE]) is PICTURE


def test_every_check_runs_the_pressure_release(monkeypatch) -> None:
    calls: list = []
    monitor = hm.HealthMonitor(global_rss_budget_mb=10 ** 6, enable_structured_logging=False)
    monkeypatch.setattr(monitor, "_relieve_pressure", lambda workers: calls.append(workers))
    monitor._check_once()
    assert len(calls) == 1


# -- stopping a worker that will not stop ---------------------------------------------------------

def _child(code: str) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-c", textwrap.dedent(code)],
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready"
    return proc


@pytest.mark.skipif(not POSIX, reason="SIGTERM semantics")
def test_a_worker_that_ignores_sigterm_is_killed_after_the_grace() -> None:
    proc = _child("""
        import signal, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("ready", flush=True)
        time.sleep(60)
    """)
    started = time.monotonic()
    assert hm.stop_process(proc.pid, grace_s=0.5)
    assert proc.wait(timeout=5) is not None and time.monotonic() - started < 3


# -- every worker exits at once on SIGTERM --------------------------------------------------------

@pytest.mark.skipif(not POSIX, reason="SIGTERM semantics")
@pytest.mark.parametrize("module", [
    "superlocalmemory.core.reranker_worker",
    "superlocalmemory.core.embedding_worker",
    "superlocalmemory.runtimes.multimodal_worker",
])
def test_sigterm_ends_a_worker_even_with_a_library_thread_running(module) -> None:
    """``sys.exit`` in the handler waited for non-daemon threads (as model libraries
    start): the 16 Nov picture worker outlived SIGTERM holding 3.3 GB."""
    proc = _child(f"""
        import threading, time, signal
        import {module} as worker
        if worker.__name__.endswith("multimodal_worker"):
            signal.signal(signal.SIGTERM, worker._exit_now)  # main() installs it
        threading.Thread(target=time.sleep, args=(60,)).start()  # non-daemon
        print("ready", flush=True)
        time.sleep(60)
    """)
    started = time.monotonic()
    os.kill(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert time.monotonic() - started < 2.0


# -- the picture worker lets go after two idle minutes --------------------------------------------

def test_the_picture_worker_is_released_two_minutes_after_its_last_request(monkeypatch) -> None:
    from superlocalmemory.runtimes import worker_client

    assert worker_client.DEFAULT_IDLE_S == 120.0
    monkeypatch.delenv("SLM_MEDIA_WORKER_IDLE_S", raising=False)
    assert worker_client._env_number("SLM_MEDIA_WORKER_IDLE_S", worker_client.DEFAULT_IDLE_S) \
        == 120.0


# -- a released reranker is not reloaded under pressure -------------------------------------------

from tests.test_retrieval.test_reranker_recycle import (  # noqa: E402
    _candidates,
    _warm,
    fake_worker,  # noqa: F401
)


def test_a_released_reranker_stays_released_while_the_machine_is_pressed(
        fake_worker, monkeypatch) -> None:  # noqa: F811
    from superlocalmemory.retrieval import reranker as mod

    rr = fake_worker()
    _warm(rr)
    first = list(fake_worker.spawned())
    pid = rr._worker_proc.pid
    pressed = {"on": True}
    monkeypatch.setattr(mod, "_memory_pressed", lambda: pressed["on"])
    os.kill(pid, signal.SIGKILL)  # the monitor released it
    rr._worker_proc.wait(timeout=5)
    for _ in range(3):
        _, applied, _status = rr.rerank_with_status("q", _candidates())
        assert applied is False  # recall keeps its fusion order
    time.sleep(0.3)
    assert fake_worker.spawned() == first, "a model was reloaded under pressure"
    pressed["on"] = False
    deadline = time.monotonic() + 15
    applied = False
    while time.monotonic() < deadline and not applied:
        _, applied, _ = rr.rerank_with_status("q", _candidates())
        time.sleep(0.1)
    assert applied, "full quality returns once memory recovers"
