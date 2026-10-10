# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A reranker that cannot load its model must back off and then stop.

Observed with an empty model cache and no network: every recall re-armed the
five-attempt warm-up, and a worker process that held no model stayed resident
between rounds.  Rounds are now separated by a growing cooldown, the idle
worker is released after a failed round, and after a few rounds automatic
retries stop (an explicit ``warmup_sync`` still forces a fresh round).

No subprocess and no model: the worker and its pipe are faked.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from superlocalmemory.retrieval import reranker as rr

# Bound at import time: the suite-wide fixture later replaces the module name.
Real = rr.CrossEncoderReranker

OFFLINE = "offline mode is enabled and the model is not in the local cache"


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def env(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(
        rr, "time", SimpleNamespace(monotonic=clock.monotonic, sleep=lambda s: None),
    )
    monkeypatch.setattr(rr, "_WARMUP_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(rr, "_WARMUP_RETRY_BACKOFF_S", 0.0)
    monkeypatch.setattr(rr, "_WARMUP_COOLDOWN_S", 300.0)
    monkeypatch.setattr(rr, "_WARMUP_MAX_ROUNDS", 4)
    state = SimpleNamespace(clock=clock, loads=0, ok=False)

    def ensure_worker(self):
        if self._worker_proc is None:
            self._worker_proc = object()

    def send(self, req, timeout=None, **kw):
        state.loads += 1
        return {"ok": True} if state.ok else {"ok": False, "error": OFFLINE}

    monkeypatch.setattr(Real, "_ensure_worker", ensure_worker)
    monkeypatch.setattr(Real, "_send_request", send)
    monkeypatch.setattr(Real, "_stop_process", staticmethod(lambda p, t: None))
    return state


def _make():
    r = Real()
    r._warmup_thread.join(5)
    return r


def _recall(r):
    cand = [(SimpleNamespace(fact_id="a", content="x"), 1.0)]
    r.rerank_with_status("q", cand)
    t = r._warmup_thread
    if t is not None:
        t.join(5)


def test_failed_round_releases_the_worker(env):
    r = _make()
    assert env.loads == 1
    assert r._worker_proc is None
    assert r._model_loaded is False
    assert r._warmup_rounds_failed == 1
    assert r._warmup_not_before == pytest.approx(1000.0 + 300.0)


def test_recall_inside_the_cooldown_starts_nothing(env):
    r = _make()
    env.clock.now += 299
    _recall(r)
    assert env.loads == 1
    assert r._start_background_warmup() is False


def test_recall_after_the_cooldown_starts_a_round_with_longer_cooldown(env):
    r = _make()
    env.clock.now += 301
    _recall(r)
    assert env.loads == 2
    assert r._warmup_rounds_failed == 2
    assert r._warmup_not_before == pytest.approx(env.clock.now + 600.0)


def test_after_max_rounds_it_stops_for_good_with_one_warning(env, caplog):
    with caplog.at_level(logging.WARNING, logger=rr.logger.name):
        r = _make()
        for _ in range(10):
            env.clock.now += 100000
            _recall(r)
    assert env.loads == 4
    assert r._warmup_stopped is True
    stops = [m.getMessage() for m in caplog.records if "stopped" in m.getMessage()]
    assert len(stops) == 1
    assert "slm warmup" in stops[0]


def test_warmup_sync_forces_a_new_round_after_stop(env):
    r = _make()
    for _ in range(5):
        env.clock.now += 100000
        _recall(r)
    assert r._warmup_stopped
    before = env.loads
    env.ok = True
    assert r.warmup_sync(timeout=5) is True
    assert env.loads == before + 1
    assert (r._warmup_rounds_failed, r._warmup_not_before, r._warmup_stopped) == (0, 0.0, False)


def test_success_resets_counters(env):
    r = _make()
    env.clock.now += 301
    env.ok = True
    _recall(r)
    assert r._model_loaded is True
    assert (r._warmup_rounds_failed, r._warmup_not_before, r._warmup_stopped) == (0, 0.0, False)


def test_start_reports_whether_it_started(env):
    r = _make()
    env.clock.now += 301
    assert r._start_background_warmup() is True
    r._warmup_thread.join(5)
    assert r._start_background_warmup(force=True) is True
    r._warmup_thread.join(5)
