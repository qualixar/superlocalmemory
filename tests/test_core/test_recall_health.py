"""Tests for the runtime recall-health monitor (v3.6.8).

The monitor keeps the full recall path warm, detects a "warm-but-broken"
embedder at runtime (semantic channel silently returning 0 because the embedder
returns None while the boot ``_embedding_warm`` flag still claims True), and
self-heals it. These tests exercise the pure logic with fakes — no daemon, no
Ollama, no sqlite-vec required.
"""
from __future__ import annotations

import logging
import time

from superlocalmemory.server.recall_health import (
    DEFAULT_PROBE,
    RecallHealth,
    get_recall_health,
    run_health_tick,
    start_recall_health_monitor,
)

LOG = logging.getLogger("test.recall_health")


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class _Result:
    def __init__(self, semantic: float) -> None:
        self.channel_scores = {"semantic": semantic, "bm25": 5.0}


class _Resp:
    def __init__(self, results: list) -> None:
        self.results = results


class _Embedder:
    """Embedder whose ``embed`` returns the queued values in order."""

    def __init__(self, returns: list) -> None:
        self._returns = list(returns)
        self._available = True
        self.calls = 0

    def embed(self, text: str):
        self.calls += 1
        if self._returns:
            return self._returns.pop(0)
        return [0.1] * 768


class _Engine:
    def __init__(self, resp, embedder=None, raises: bool = False) -> None:
        self._resp = resp
        self._embedder = embedder
        self._raises = raises
        self.recall_calls = 0

    def recall(self, query: str, limit: int = 3, fast: bool = False):
        self.recall_calls += 1
        if self._raises:
            raise RuntimeError("daemon recall failed: timed out")
        return self._resp


# --------------------------------------------------------------------------
# Tier 2 (readiness) + Tier 3 (self-heal) logic
# --------------------------------------------------------------------------
def test_healthy_recall_marks_healthy_no_heal():
    emb = _Embedder([])  # heal must NOT be called
    eng = _Engine(_Resp([_Result(0.44), _Result(0.42)]), embedder=emb)
    st = RecallHealth()

    run_health_tick(eng, st, log=LOG)

    assert st.healthy is True
    assert emb.calls == 0
    assert st.last_semantic_score > 0
    assert eng.recall_calls == 1  # Tier 1 re-warm fired


def test_warm_but_broken_triggers_heal_and_recovers():
    # Rows returned but semantic==0 everywhere → embedder is returning None.
    # The heal re-exercises the embedder, which now returns a vector.
    emb = _Embedder([[0.1] * 768])
    eng = _Engine(_Resp([_Result(0.0), _Result(0.0)]), embedder=emb)
    st = RecallHealth()

    run_health_tick(eng, st, log=LOG)

    assert emb.calls == 1                 # embedder was re-exercised
    assert emb._available is None         # cached-availability flag reset
    assert st.total_heals == 1
    assert st.healthy is True
    assert st.consecutive_failures == 0


def test_heal_fails_when_embedder_still_returns_none():
    emb = _Embedder([None])               # embedder genuinely dead
    eng = _Engine(_Resp([_Result(0.0)]), embedder=emb)
    st = RecallHealth()

    run_health_tick(eng, st, log=LOG)

    assert st.healthy is False
    assert st.consecutive_failures == 1
    assert st.last_error


def test_recall_exception_is_caught_and_marks_unhealthy():
    eng = _Engine(None, raises=True)
    st = RecallHealth()

    # Must not raise — the loop has to survive a timed-out recall.
    run_health_tick(eng, st, log=LOG)

    assert st.healthy is False
    assert st.consecutive_failures == 1
    assert "raised" in st.last_error


def test_empty_corpus_is_not_treated_as_broken():
    # No results at all is NOT the warm-but-broken signature (could be an
    # empty/ filtered corpus). Do not heal, do not flap to unhealthy.
    emb = _Embedder([])
    eng = _Engine(_Resp([]), embedder=emb)
    st = RecallHealth()

    run_health_tick(eng, st, log=LOG)

    assert st.healthy is True
    assert emb.calls == 0


def test_recovery_after_failure_logs_and_clears():
    emb = _Embedder([])
    eng = _Engine(_Resp([_Result(0.5)]), embedder=emb)
    st = RecallHealth(healthy=False, consecutive_failures=3, last_error="prior")

    run_health_tick(eng, st, log=LOG)

    assert st.healthy is True
    assert st.consecutive_failures == 0
    assert st.last_error == ""


# --------------------------------------------------------------------------
# Thread lifecycle
# --------------------------------------------------------------------------
def test_monitor_thread_starts_and_stops():
    emb = _Embedder([])
    eng = _Engine(_Resp([_Result(0.5)]), embedder=emb)
    t, stop, state = start_recall_health_monitor(eng, interval_s=1, log=LOG)
    try:
        assert t.is_alive()
    finally:
        stop.set()
        t.join(timeout=3)
    assert not t.is_alive()


def test_get_recall_health_shape():
    h = get_recall_health()
    assert "recall_healthy" in h
    assert "consecutive_failures" in h
    assert "total_heals" in h
    assert "checks" in h
