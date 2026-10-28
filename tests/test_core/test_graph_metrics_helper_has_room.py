# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The graph-metrics helper gets a memory limit sized to the graph (#304).

On a store of 3.6 million edges the daemon's memory watchdog killed the helper
at 2.9 GB against the 2.5 GB limit every worker gets, so PageRank and the
communities were never recomputed. The helper is a ``spawn`` child whose command
line looks like any other, so the limit travels with its PID instead.
"""

from __future__ import annotations

import threading
import time

from superlocalmemory.core import background_process, graph_metrics_process as gmp


def _sleep_then_pid(seconds: float) -> int:
    import os

    time.sleep(seconds)
    return os.getpid()


def test_a_child_carries_its_limit_while_it_runs() -> None:
    out: dict[str, int] = {}
    runner = threading.Thread(target=lambda: out.setdefault("pid", background_process.run_in_child(
        _sleep_then_pid, 1.5, timeout=30, memory_limit_mb=6000)))
    runner.start()
    seen: dict[int, int] = {}
    deadline = time.monotonic() + 20
    while not seen and time.monotonic() < deadline:
        seen = background_process.memory_limits()
        time.sleep(0.05)
    runner.join(30)
    assert list(seen.values()) == [6000]
    assert list(seen) == [out["pid"]]
    assert background_process.memory_limits() == {}  # gone once it ended


def test_the_watchdog_reads_the_childs_own_limit() -> None:
    from superlocalmemory.server.unified_daemon import _worker_limit_mb

    background_process._MEMORY_LIMITS[424242] = 6000
    try:
        assert _worker_limit_mb(["python", "-c", "spawn_main"], 2500, pid=424242) == 6000
        assert _worker_limit_mb(["python", "-c", "spawn_main"], 2500, pid=1) == 2500
    finally:
        background_process._MEMORY_LIMITS.pop(424242, None)


def test_the_limit_follows_the_graph_within_the_budget(monkeypatch) -> None:
    monkeypatch.setattr(gmp, "_memory_budget_mb", lambda: 6400)
    assert gmp.helper_limit_mb(20_000) == 2500           # a small graph keeps the default
    big = gmp.helper_limit_mb(3_600_000)
    assert 2927 < big <= 6400                            # room above where it was killed
    assert gmp.helper_limit_mb(50_000_000) == 6400       # never past the daemon's budget
