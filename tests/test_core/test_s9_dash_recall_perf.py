# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21 — S9-DASH-02

"""Perf regression: engine.recall() with session_id must not add
meaningful wall-time vs engine.recall() without session_id.

Varun's directive (2026-04-20): "there should be no difference in
recall and remember timings ... developer will not tolerate if we
create any issues in the recall, learning hooks, and remember timings."

We test the *overhead* of the outcome-queue enqueue path only — the
underlying ``run_recall`` is mocked to a no-op so the test is CI-stable
and isolates the code we added in this commit.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest


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
    from superlocalmemory.core import engine as engine_mod
    # Patch run_recall to a no-op returning a fixed response.
    stub_response = _StubResponse(["f1", "f2", "f3", "f4", "f5"])

    def _fake_run_recall(*args, **kwargs):
        return stub_response

    monkeypatch.setattr(
        "superlocalmemory.core.recall_pipeline.run_recall",
        _fake_run_recall,
    )

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

        def _ensure_init(self):
            pass

    return _E(), stub_response


def test_recall_with_session_id_does_not_regress(monkeypatch) -> None:
    """The delta between recall(..., session_id=None) and
    recall(..., session_id="s") over 200 iterations must stay under
    5 ms total on a commodity laptop — i.e. < 25 µs per call of
    overhead added by the outcome-queue enqueue.

    Budget rationale: enqueue is a single ``put_nowait`` + dataclass
    build. 25 µs is a 25× safety margin over the measured ~1 µs path.
    If this test fails, someone likely added I/O to the hot path.
    """
    from superlocalmemory.learning import outcome_queue
    outcome_queue._reset_for_testing()

    engine_stub, _ = _make_engine(monkeypatch)

    # Bind the real MemoryEngine.recall method to our stub so we
    # exercise the exact production code path.
    from superlocalmemory.core.engine import MemoryEngine
    recall_method = MemoryEngine.recall

    # Warm-up — first call pays import costs.
    for _ in range(5):
        recall_method(engine_stub, "q")
        recall_method(engine_stub, "q", session_id="s")

    iterations = 200
    t0 = time.perf_counter()
    for _ in range(iterations):
        recall_method(engine_stub, "q")
    baseline_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    for _ in range(iterations):
        recall_method(engine_stub, "q", session_id="sess-perf")
    with_sid_ms = (time.perf_counter() - t0) * 1000.0

    delta_ms = with_sid_ms - baseline_ms
    per_call_delta_us = delta_ms * 1000.0 / iterations

    # Surface the numbers even on pass for CI trend watching.
    print(
        f"[s9-dash-perf] baseline={baseline_ms:.2f}ms "
        f"with_sid={with_sid_ms:.2f}ms "
        f"delta={delta_ms:.2f}ms "
        f"per_call_overhead={per_call_delta_us:.1f}us"
    )

    # Budget: 5 ms delta for 200 calls = 25 us per call. Very generous.
    assert delta_ms < 5.0, (
        f"recall(session_id=) regressed: +{delta_ms:.2f}ms over "
        f"{iterations} calls (budget 5ms). Someone likely added I/O "
        f"to the outcome-queue enqueue path."
    )

    # And at least one enqueue landed (the producer is actually wired).
    counters = outcome_queue.get_counters()
    assert counters["recall_enqueued"] >= 1, \
        "engine.recall did not call enqueue_recall — producer is unwired"
    outcome_queue._reset_for_testing()
