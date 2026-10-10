"""The global memory budget: what it counts, what it may kill, and the picture worker's own cap."""

from __future__ import annotations

import os

import pytest

psutil = pytest.importorskip("psutil")

from superlocalmemory.core import health_monitor as hm  # noqa: E402
from superlocalmemory.infra import proc_memory  # noqa: E402

DAEMON_PID = os.getpid()
EMBEDDER = "python -m superlocalmemory.core.embedding_worker"
RERANKER = "python -m superlocalmemory.core.reranker_worker"
PICTURE = "/media/venv/bin/python -I /site/superlocalmemory/runtimes/multimodal_worker.py"
PICTURE_CAP = 4500


class FakeProc:
    def __init__(self, pid, cmdline="", children=()):
        self.pid, self._cmdline, self._children = pid, cmdline, list(children)

    def cmdline(self):
        return self._cmdline.split()

    def children(self, recursive=False):
        return self._children


@pytest.fixture
def world(monkeypatch):
    """Fake process set: ``world.set(daemon_mb, {pid: (cmdline, mb)})``; ``world.killed`` lists terminated pids."""
    class World:
        def __init__(self):
            self.killed: list[int] = []

        def set(self, daemon_mb, workers):
            kids = [FakeProc(pid, cmd) for pid, (cmd, _mb) in workers.items()]
            procs = {DAEMON_PID: FakeProc(DAEMON_PID, "slm daemon", kids), **{k.pid: k for k in kids}}
            mem = {DAEMON_PID: daemon_mb, **{pid: mb for pid, (_c, mb) in workers.items()}}
            monkeypatch.setattr(proc_memory, "process_memory_mb", lambda pid: float(mem.get(pid, 0.0)))
            monkeypatch.setattr(hm.psutil, "Process", lambda pid: _Terminable(procs[pid], self.killed))
            monkeypatch.delenv("SLM_MEDIA_WORKER_RSS_LIMIT_MB", raising=False)

    return World()


class _Terminable:
    def __init__(self, proc, killed):
        self._proc, self._killed = proc, killed
        self.pid = proc.pid

    def children(self, recursive=False):
        return self._proc.children(recursive)

    def cmdline(self):
        return self._proc.cmdline()

    def terminate(self):
        self._killed.append(self.pid)


def _monitor(budget_mb):
    return hm.HealthMonitor(global_rss_budget_mb=budget_mb, enable_structured_logging=False)


def test_picture_worker_under_its_own_cap_is_not_killed_by_the_global_budget(world):
    # daemon + embedder + reranker + picture = 6700 > 4000, but the picture worker (3700) is under its 4500 cap
    world.set(500, {11: (EMBEDDER, 1200), 12: (RERANKER, 1300), 13: (PICTURE, 3700)})
    _monitor(4000)._check_once()
    assert world.killed == []


def test_the_rest_over_budget_still_kills_a_non_embedder_but_never_the_picture_worker(world):
    world.set(900, {11: (EMBEDDER, 2500), 12: (RERANKER, 1300), 13: (PICTURE, 3700)})
    _monitor(4000)._check_once()  # without the picture worker: 4700 > 4000
    assert world.killed == [12]


def test_picture_worker_over_its_cap_is_counted_and_is_the_first_kill(world):
    world.set(500, {11: (EMBEDDER, 1200), 12: (RERANKER, 1300), 13: (PICTURE, PICTURE_CAP + 400)})
    _monitor(4000)._check_once()
    assert world.killed == [13]  # the largest non-embedder; the embedder is spared


def test_the_embedder_is_never_killed_before_other_workers(world):
    world.set(500, {11: (EMBEDDER, 3500), 12: (RERANKER, 400), 13: (PICTURE, PICTURE_CAP + 1)})
    _monitor(3000)._check_once()
    assert 11 not in world.killed and world.killed == [13]


def test_without_a_picture_worker_the_budget_works_as_before(world):
    world.set(500, {11: (EMBEDDER, 1200), 12: (RERANKER, 1300)})
    _monitor(4000)._check_once()
    assert world.killed == []
    world.set(500, {11: (EMBEDDER, 2500), 12: (RERANKER, 1300)})
    _monitor(4000)._check_once()
    assert world.killed == [12]


def test_a_picture_worker_with_no_cap_is_counted_like_any_other(world, monkeypatch):
    world.set(500, {11: (EMBEDDER, 1200), 13: (PICTURE, 3700)})
    monkeypatch.setenv("SLM_MEDIA_WORKER_RSS_LIMIT_MB", "0")
    _monitor(4000)._check_once()
    assert world.killed == [13]


def test_the_cap_override_moves_the_exemption(world, monkeypatch):
    world.set(500, {11: (EMBEDDER, 1200), 13: (PICTURE, 3700)})
    monkeypatch.setenv("SLM_MEDIA_WORKER_RSS_LIMIT_MB", "3000")  # now over its cap
    _monitor(4000)._check_once()
    assert world.killed == [13]


def test_the_memory_check_counts_the_daemon_and_children_on_the_shared_reader(world):
    world.set(700, {11: (EMBEDDER, 1800), 13: (PICTURE, 3700)})
    result = _monitor(2500)._check_memory_budget()
    assert result["status"] == "critical" and "6200MB" in result["detail"]
    assert "700MB" in _monitor(2500)._check_daemon_health()["detail"]
