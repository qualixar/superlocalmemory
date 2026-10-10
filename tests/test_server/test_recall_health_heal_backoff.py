# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The recall-health monitor must not respawn a worker for a heal that keeps failing.

With no embedding model on disk and no network, every tick re-ran the heal,
which respawned a large embedding worker that could only fail again.  Failed
heals now back off (300 s, doubling) and stop after four; any healthy tick or
successful heal resets the counters.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from superlocalmemory.server import recall_health as rh


class _Dead:
    _available = True
    is_warm = False

    def __init__(self, revives_after: int | None = None) -> None:
        self.embed_calls = 0
        self.revives_after = revives_after

    def embed(self, text):
        self.embed_calls += 1
        if self.revives_after is not None and self.embed_calls > self.revives_after:
            self.is_warm = True
            return [0.1] * 8
        return None


class _Engine:
    def __init__(self, embedder) -> None:
        self._embedder = embedder

    def recall(self, query, limit=3, fast=True):
        return SimpleNamespace(results=[])


@pytest.fixture
def clock(monkeypatch):
    now = SimpleNamespace(t=10_000.0)
    monkeypatch.setattr(
        rh, "time", SimpleNamespace(time=lambda: now.t, sleep=lambda s: None),
    )
    return now


def _tick(engine, state, caplog):
    with caplog.at_level(logging.DEBUG, logger=rh.logger.name):
        rh.run_health_tick(engine, state)


def test_first_tick_heals_ticks_inside_backoff_do_not(clock, caplog):
    emb, state = _Dead(), rh.RecallHealth()
    engine = _Engine(emb)
    _tick(engine, state, caplog)
    assert emb.embed_calls == 1
    assert state.heal_failures_in_row == 1
    assert state.next_heal_at == pytest.approx(clock.t + 300)
    clock.t += 299
    _tick(engine, state, caplog)
    assert emb.embed_calls == 1
    assert state.healthy is False
    assert "paused" in state.last_error or "stopped" in state.last_error
    clock.t += 2
    _tick(engine, state, caplog)
    assert emb.embed_calls == 2
    assert state.next_heal_at == pytest.approx(clock.t + 600)


def test_after_max_attempts_it_never_heals_again_and_warns_once(clock, caplog):
    emb, state = _Dead(), rh.RecallHealth()
    engine = _Engine(emb)
    for _ in range(20):
        clock.t += 100_000
        _tick(engine, state, caplog)
    assert emb.embed_calls == rh.MAX_HEAL_ATTEMPTS == 4
    assert state.heal_stopped is True
    stops = [r for r in caplog.records
             if r.levelno == logging.WARNING and "self-heal stopped" in r.getMessage()]
    assert len(stops) == 1
    assert "slm warmup" in stops[0].getMessage()


def test_recovered_embedder_resets_the_counters(clock, caplog):
    emb, state = _Dead(revives_after=1), rh.RecallHealth()
    engine = _Engine(emb)
    _tick(engine, state, caplog)
    assert state.heal_failures_in_row == 1
    clock.t += 301
    _tick(engine, state, caplog)
    assert state.healthy is True
    assert (state.heal_failures_in_row, state.next_heal_at, state.heal_stopped) == (0, 0.0, False)


def test_health_snapshot_reports_heal_state(clock, caplog, monkeypatch):
    emb, state = _Dead(), rh.RecallHealth()
    monkeypatch.setattr(rh, "_GLOBAL_STATE", state)
    _tick(_Engine(emb), state, caplog)
    snap = rh.get_recall_health()
    assert snap["embedder_heal_stopped"] is False
    assert snap["embedder_heal_retry_in_s"] == pytest.approx(300, abs=1)
    for _ in range(10):
        clock.t += 100_000
        _tick(_Engine(emb), state, caplog)
    snap = rh.get_recall_health()
    assert snap["embedder_heal_stopped"] is True


class _Reranker:
    _model_loaded = False
    _worker_loading = False

    def __init__(self, result) -> None:
        self.result = result

    def _start_background_warmup(self):
        return self.result


@pytest.mark.parametrize("result,counted", [(True, 1), (None, 1), (False, 0)])
def test_rearm_is_counted_only_when_a_warmup_actually_started(result, counted):
    engine = SimpleNamespace(
        _retrieval_engine=SimpleNamespace(_reranker=_Reranker(result)),
    )
    state = rh.RecallHealth()
    rh._watch_reranker(engine, state, log=logging.getLogger("t"))
    assert state.reranker_rearms == counted
