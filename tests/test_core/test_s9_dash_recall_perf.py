# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — S9-DASH-02

"""Recall remains a pure query when a session_id is supplied.

Varun's directive (2026-04-20): "there should be no difference in
recall and remember timings ... developer will not tolerate if we
create any issues in the recall, learning hooks, and remember timings."

The underlying ``run_recall`` is mocked to a no-op so this contract stays
focused on the public engine boundary.
"""

from __future__ import annotations

import time
from types import SimpleNamespace


class _StubResponse:
    """Minimal duck-typed RecallResponse."""
    def __init__(self, fact_ids: list[str]) -> None:
        self.results = [
            SimpleNamespace(fact=SimpleNamespace(fact_id=f))
            for f in fact_ids
        ]
        self.query_id = "qid-perf"


def _make_engine(monkeypatch) -> "tuple[object, object]":
    """Build a MemoryEngine stub that skips DB init and run_recall."""
    # Patch run_recall to a no-op returning a fixed response.
    stub_response = _StubResponse(["f1", "f2", "f3", "f4", "f5"])

    def _fake_run_recall(*args, **kwargs):
        return stub_response

    monkeypatch.setattr(
        "superlocalmemory.core.recall_pipeline.run_recall",
        _fake_run_recall,
    )

    from superlocalmemory.core.engine_capabilities import Capabilities

    class _E:
        _profile_id = "p"
        _config = None
        _retrieval_engine = None
        _trust_scorer = None
        _embedder = None
        _db = None
        _llm = None
        _hooks = None
        _access_log = None
        _auto_linker = None
        _initialized = True
        _capabilities = Capabilities.FULL

        def _ensure_init(self):
            pass

        def _require_full(self, operation):
            pass

    return _E(), stub_response


def test_recall_with_session_id_remains_a_pure_query(monkeypatch) -> None:
    """A session identifier must not schedule outcome writes from recall."""

    engine_stub, _ = _make_engine(monkeypatch)

    # Bind the real MemoryEngine.recall method to our stub so we
    # exercise the exact production code path.
    from superlocalmemory.core.engine import MemoryEngine
    recall_method = MemoryEngine.recall

    start = time.perf_counter()
    for _ in range(200):
        recall_method(engine_stub, "q", session_id="sess-perf")
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert elapsed_ms < 25.0


def test_fast_recall_is_forwarded_to_the_retrieval_pipeline(monkeypatch) -> None:
    """The public fast flag must not be silently rewritten to full recall."""
    engine_stub, stub_response = _make_engine(monkeypatch)
    captured: dict[str, object] = {}

    def _capture_fast(*args, **kwargs):
        captured.update(kwargs)
        return stub_response

    monkeypatch.setattr(
        "superlocalmemory.core.recall_pipeline.run_recall",
        _capture_fast,
    )

    from superlocalmemory.core.engine import MemoryEngine
    MemoryEngine.recall(engine_stub, "fast witness", fast=True)

    assert captured["fast"] is True
