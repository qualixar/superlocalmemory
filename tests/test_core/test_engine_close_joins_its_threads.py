# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``MemoryEngine.close()`` returns with its pool threads gone, or bounded (C-6).

``close()`` shut its pools down without waiting and returned while their
threads were still running -- still able to touch the database it was about to
close, and still alive when a later caller (or test) went on to something else.
Waiting forever is not the answer either: a worker wedged in a model call would
hold shutdown hostage. The join is bounded by one shared budget.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np

from superlocalmemory.core import thread_join
from superlocalmemory.core.engine import MemoryEngine

_POOL_PREFIXES = ("slm-sg-embed", "slm-recall-channel", "slm-query-embed")
#: How long the wedged worker below holds out if nothing releases it.
_WEDGE_SECONDS = 30.0
#: Real-time bound on close(): its own join budget plus slack for the rest of
#: close() on a loaded host. A close held hostage by the wedge lasts ~30 s, so
#: a few seconds of slack cannot hide one.
_CLOSE_SLACK_SECONDS = 4.0


def _warm_embedder(block: threading.Event | None = None) -> MagicMock:
    emb = MagicMock()

    def _embed(text: str) -> list[float]:
        if block is not None:
            block.wait(_WEDGE_SECONDS)
        rng = np.random.RandomState(abs(hash(text)) % 2**31)
        v = rng.randn(768).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()

    emb.embed.side_effect = _embed
    emb.is_available = True
    emb._available = True
    emb.is_warm = True
    emb._config = MagicMock(is_cloud=False, is_openai_compatible=False)
    emb.compute_fisher_params.return_value = ([0.0] * 768, [1.0] * 768)
    return emb


def _engine(config, embedder) -> MemoryEngine:
    engine = MemoryEngine(config)
    with patch("superlocalmemory.core.engine_wiring.init_embedder", return_value=embedder):
        engine.initialize()
    engine._embedder = embedder
    return engine


def _pool_threads(before: set[int]) -> list[str]:
    return sorted(
        t.name for t in threading.enumerate()
        if t.ident not in before and t.name.startswith(_POOL_PREFIXES) and t.is_alive()
    )


def test_close_returns_with_its_pool_threads_gone(mode_a_config) -> None:
    before = {t.ident for t in threading.enumerate()}
    engine = _engine(mode_a_config, _warm_embedder())
    engine._warm_guard_embed("an inline write-path embed")
    engine.store("Alice moved to Paris in 2021.")
    engine.recall("Where did Alice move?")
    assert _pool_threads(before), "the scenario must start pool threads"

    engine.close()

    assert _pool_threads(before) == [], "close() returned with pool threads alive"


def test_close_is_bounded_when_a_worker_is_wedged(mode_a_config, caplog) -> None:
    before = {t.ident for t in threading.enumerate()}
    block = threading.Event()
    engine = _engine(mode_a_config, _warm_embedder())
    try:
        engine._embedder = _warm_embedder(block)  # its shutdown wakes nothing
        engine._warm_guard_embed("this embed never finishes on its own")
        assert any(n.startswith("slm-sg-embed") for n in _pool_threads(before))

        t0 = time.monotonic()
        engine.close()
        elapsed = time.monotonic() - t0
    finally:
        block.set()

    assert elapsed < thread_join.CLOSE_JOIN_SECONDS + _CLOSE_SLACK_SECONDS, (
        f"close took {elapsed:.2f}s")
    assert any("still running" in r.getMessage() for r in caplog.records)
    deadline = time.monotonic() + 5
    while _pool_threads(before) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert _pool_threads(before) == []


def test_join_threads_never_joins_the_calling_thread() -> None:
    assert thread_join.join_threads([threading.current_thread()], budget_seconds=5) == []
