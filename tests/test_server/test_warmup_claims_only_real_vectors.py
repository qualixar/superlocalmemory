# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A warm-up that produced no vector must not claim the model is warm.

With an empty model cache and no network the embedder returns ``None`` without
raising.  The old code logged "warm and ready" anyway and flipped the readiness
flag, so the log and /health said the opposite of the truth.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from superlocalmemory.server import unified_daemon as ud


class _Embedder:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        return self.result


@pytest.fixture(autouse=True)
def _local_embedder(monkeypatch):
    monkeypatch.setattr(
        "superlocalmemory.core.engine._is_remote_embedder", lambda e: False,
    )


def _run(embedder, caplog):
    engine = SimpleNamespace(_embedder=embedder)
    with caplog.at_level(logging.INFO, logger=ud.logger.name):
        thread = ud._start_embedder_warmup(engine)
        assert thread is not None
        thread.join(5)
    assert not thread.is_alive()
    return caplog.records


@pytest.mark.parametrize("empty", [None, [], ()])
def test_no_vector_is_not_warm(caplog, empty):
    records = _run(_Embedder(empty), caplog)
    messages = [(r.levelno, r.getMessage()) for r in records]
    assert not any("warm and ready" in m for _, m in messages)
    warnings = [m for lvl, m in messages if lvl == logging.WARNING]
    assert len(warnings) == 1
    assert "Embedding model not loaded" in warnings[0]
    assert "slm warmup" in warnings[0]


def test_real_vector_is_warm(caplog):
    records = _run(_Embedder([0.1, 0.2]), caplog)
    assert any("warm and ready" in r.getMessage() for r in records)
    assert not any(r.levelno >= logging.WARNING for r in records)


def test_lifespan_helper_no_vector_is_a_failed_warmup():
    retrieval = SimpleNamespace(_embedder=_Embedder(None))
    ok, reason = ud._warm_embedder_once(SimpleNamespace(), retrieval)
    assert ok is False
    assert "did not load" in reason


def test_lifespan_helper_vector_is_success():
    retrieval = SimpleNamespace(_embedder=_Embedder([0.5]))
    assert ud._warm_embedder_once(SimpleNamespace(), retrieval) == (True, "")


def test_lifespan_helper_without_embedder_means_retry_later():
    engine = SimpleNamespace(_retrieval_engine=None)
    assert ud._warm_embedder_once(engine, None) == (False, "")
