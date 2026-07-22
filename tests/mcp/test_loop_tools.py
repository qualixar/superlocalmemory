# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""MCP bounded-loop tools — slm_loop_run / slm_loop_history / slm_loop_show.

Two layers:

* Gate-logic tests run against a FAKE engine whose ``recall`` is canned, so the
  gate → verdict → loop-decision path is fast and deterministic (no embedder).
  A query containing "MATCH" returns a confident hit (score 0.5); anything else
  returns nothing above the evidence floor.
* One persistence test runs against a REAL isolated engine (store-then-match)
  and reads the run back through ``slm_loop_history`` / ``slm_loop_show`` to
  prove laps are durably written to SLM memory.

Run: SLM_TEST_ISOLATION=1 pytest tests/mcp/test_loop_tools.py -q
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from superlocalmemory.mcp.tools_loops import register_loop_tools


class _Capture:
    """Duck-typed MCP server that captures each registered tool by name."""

    def __init__(self) -> None:
        self.fns: dict = {}

    def tool(self, *args, **kwargs):
        def deco(fn):
            self.fns[fn.__name__] = fn
            return fn
        return deco


# ── Fake engine: deterministic recall + no-op ledger store contract ──────────

@dataclass
class _FakeResult:
    score: float = 0.5


@dataclass
class _FakeResp:
    results: list = field(default_factory=list)
    no_confident_match: bool = True


class _FakeDB:
    def execute(self, sql, params=()):  # ledger reads → nothing persisted
        return []


class _FakeEngine:
    """Instant, embedder-free engine. Satisfies the recall gate + ledger store
    contract (store / db.execute / profile_id) used by engine_backed_ledger."""

    profile_id = "default"

    def __init__(self) -> None:
        self.db = _FakeDB()
        self.stored: list = []

    def recall(self, query, limit=3, fast=True, **kw):
        if "MATCH" in query:
            return _FakeResp(results=[_FakeResult(0.5)], no_confident_match=False)
        return _FakeResp(results=[], no_confident_match=True)

    def store(self, content, *, session_id=None, metadata=None):  # ledger writes
        self.stored.append((session_id, content))
        return "fake"


@pytest.fixture
def fake_tools():
    cap = _Capture()
    register_loop_tools(cap, _FakeEngine)  # get_engine = fresh fake per call
    return cap.fns


def _run(coro):
    return asyncio.run(coro)


# ── Gate-logic tests (fake engine) ───────────────────────────────────────────

def test_registers_three_tools(fake_tools):
    assert set(fake_tools) == {"slm_loop_run", "slm_loop_history", "slm_loop_show"}


def test_halts_on_max_iterations_when_gate_never_passes(fake_tools):
    out = _run(fake_tools["slm_loop_run"](
        name="watch-nothing",
        gate_query="condition never present",
        max_iterations=3,
        poll_interval_s=0.25,
    ))
    assert out["ok"] is True
    assert out["status"] == "HALT"
    assert out["reason"] == "max-iterations"
    assert out["laps"] == 3
    assert out["passed"] is False
    assert out["run_id"].startswith("watch-nothing-")


def test_converges_when_gate_recall_hits(fake_tools):
    out = _run(fake_tools["slm_loop_run"](
        name="wait-build",
        gate_query="MATCH build pipeline passed",
        max_iterations=5,
        poll_interval_s=0.25,
    ))
    assert out["ok"] is True
    assert out["status"] == "DONE"
    assert out["passed"] is True
    assert out["laps"] == 1  # gate passes on the first lap


def test_min_score_can_refuse_a_weak_hit(fake_tools):
    # The hit scores 0.5; an unreachable min-score keeps the gate closed.
    out = _run(fake_tools["slm_loop_run"](
        name="strict-gate",
        gate_query="MATCH but weak",
        gate_min_score=0.999,
        max_iterations=2,
        poll_interval_s=0.25,
    ))
    assert out["ok"] is True
    assert out["status"] == "HALT"
    assert out["passed"] is False


def test_token_budget_bounds_the_loop(fake_tools):
    # A watcher spends 0 tokens/lap, so a token budget alone can't halt it —
    # this asserts max_tokens is accepted and the loop still finishes bounded.
    out = _run(fake_tools["slm_loop_run"](
        name="tok",
        gate_query="never",
        max_iterations=2,
        max_tokens=5,
        poll_interval_s=0.25,
    ))
    assert out["ok"] is True
    assert out["status"] == "HALT"
    assert out["reason"] == "max-iterations"


def test_run_validates_inputs(fake_tools):
    assert _run(fake_tools["slm_loop_run"](name="", gate_query="x"))["ok"] is False
    assert _run(fake_tools["slm_loop_run"](name="n", gate_query=""))["ok"] is False


def test_reads_validate_inputs(fake_tools):
    assert _run(fake_tools["slm_loop_history"](name=""))["ok"] is False
    assert _run(fake_tools["slm_loop_show"](run_id=""))["ok"] is False


# ── Persistence test (real engine, store-then-match keeps recall fast) ───────

@pytest.fixture
def real_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_TEST_ISOLATION", "1")
    from superlocalmemory.loops.ledger import open_engine_store

    store = open_engine_store(tmp_path / "memory.db")
    engine = store._engine
    cap = _Capture()
    register_loop_tools(cap, lambda: engine)
    try:
        yield engine, cap.fns
    finally:
        store.close()


def test_ledger_persists_and_reads_back_on_real_engine(real_tools):
    engine, fns = real_tools
    # Storing first both seeds the gate condition AND warms the embedder, so the
    # recall gate matches on lap 1 (DONE) without a cold-start stall.
    engine.store(
        "The quarterly build pipeline passed all integration tests today.",
        session_id="s-build",
        metadata={"tags": ["build"], "importance": 5, "project_name": "t"},
    )
    run = _run(fns["slm_loop_run"](
        name="wait-real",
        gate_query="build pipeline passed integration tests",
        max_iterations=5,
        poll_interval_s=0.25,
    ))
    assert run["ok"] is True
    assert run["status"] == "DONE"
    rid = run["run_id"]

    hist = _run(fns["slm_loop_history"](name="wait-real"))
    assert hist["ok"] is True
    assert rid in [r["run_id"] for r in hist["runs"]]

    show = _run(fns["slm_loop_show"](run_id=rid))
    assert show["ok"] is True
    assert show["count"] >= 1
    assert show["laps"][-1]["decision"] == "done"
    # The agent's own claim is recorded (advisory) and False for a watcher.
    assert show["laps"][0]["agent_claimed_done"] is False
