"""A glance asked by an agent is a question; a hook past its clock still answers.

4.2.2 on the author's store (27,064 facts, full recall ~2.5 s, 13 Oct 2026):

1. An agent's ``recall(depth=0)`` -- the tool tells agents to start at depth 0 --
   was classed as a prompt hook and got the hook's three-second clock. It came
   back with 0 results and every channel "timeout".
2. When a hook's full recall overran, the keyword fallback only started after
   the run was abandoned, with ~0.2 s left on the HTTP request's own clock. It
   never finished: "no keyword fallback matches were ready". The prompt hook
   injected nothing.

The earlier tests ran without the HTTP ingress clock and never checked that the
fallback returned anything, so neither showed. These run under that clock.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from superlocalmemory.server import profile_runtime, recall_core, recall_fallback, recall_ingress
from superlocalmemory.server.recall_ingress import explicit_scope


def _bare(results):
    """Results without the W3-7b ``why`` receipt every answer now carries."""
    return [{k: v for k, v in r.items() if k != "why"} if isinstance(r, dict) else r
            for r in results]

SNAP = SimpleNamespace(profile_id="default", generation=1)
KEYWORD = {"results": [{"content": "SLM 4.2.2 released"}], "retrieval_mode": "degraded_lexical"}


@pytest.fixture(autouse=True)
def _runtime(monkeypatch):
    monkeypatch.setattr(profile_runtime, "get_profile_runtime",
                        lambda state: SimpleNamespace(snapshot=SNAP))


def _call(**kw):
    base = dict(query="4.2.2 release npm", limit=3, session_id="", agent_id="a",
                fast=True, profile_id="default")
    base.update(kw)
    return recall_core.RecallCall(**base)


def _under_ingress_clock(call, budget):
    """Run one recall the way the daemon does: inside the request's own clock."""
    async def run():
        token = recall_ingress._DEADLINE.set(asyncio.get_running_loop().time() + budget)
        start = time.monotonic()
        try:
            out = await recall_core.run_recall(object(), call, app_state=SimpleNamespace())
            # Timed inside the loop: asyncio.run later joins the abandoned
            # engine thread, which the daemon's long-lived loop never does.
            return {**out, "_elapsed": time.monotonic() - start}
        finally:
            recall_ingress._DEADLINE.reset(token)
    return asyncio.run(run())


def test_an_agents_glance_is_an_explicit_question():
    assert recall_core.is_explicit(_call(depth=0))
    assert explicit_scope({"type": "http", "path": "/recall", "query_string": b"q=x&depth=0"})


def test_the_prompt_hook_keeps_the_short_clock():
    # auto_recall_hook sends answer_check=skip together with depth=0.
    assert not recall_core.is_explicit(_call(depth=0, skip_answer_check=True))
    assert not explicit_scope({"type": "http", "path": "/recall",
                               "query_string": b"q=x&answer_check=skip&depth=0"})


def test_a_hook_whose_recall_overruns_still_gets_keyword_matches(monkeypatch):
    def slow_engine(*a):
        time.sleep(2.0)
        return object()

    def keyword(*a, **kw):
        time.sleep(0.3)  # the measured fallback on a 27k-fact store: 67-455 ms
        return dict(KEYWORD)

    monkeypatch.setattr(recall_core, "_engine_call", slow_engine)
    monkeypatch.setattr(recall_fallback, "recall_keyword_fallback", keyword)
    monkeypatch.setattr(recall_core, "effective_budget_s", lambda call: 1.0)

    out = _under_ingress_clock(_call(fast=False, skip_answer_check=True, depth=0), 1.0)
    assert out["_elapsed"] < 1.05  # the hook's clock still holds
    assert _bare(out["results"]) == KEYWORD["results"]
    assert out["answerability"] == "unjudged"


def test_a_hook_whose_recall_finishes_gets_the_full_answer(monkeypatch):
    def engine(*a):
        time.sleep(0.6)  # past the hedge point, inside the run budget
        return object()

    monkeypatch.setattr(recall_core, "_engine_call", engine)
    monkeypatch.setattr(recall_core, "_envelope", lambda *a: {"results": [{"content": "full"}]})
    monkeypatch.setattr(recall_fallback, "recall_keyword_fallback", lambda *a, **kw: dict(KEYWORD))
    monkeypatch.setattr(recall_core, "effective_budget_s", lambda call: 1.0)

    out = _under_ingress_clock(_call(fast=False, skip_answer_check=True), 1.0)
    assert _bare(out["results"]) == [{"content": "full"}]


def test_session_init_gets_the_full_clock_but_is_never_judged():
    # session_init is the first call of every session; on the author's store
    # its recall timed out on every channel under the hook's clock (4.2.2).
    call = _call(context_load=True, skip_answer_check=True)
    assert recall_core.is_explicit(call)
    assert explicit_scope({"type": "http", "path": "/recall",
                           "query_string": b"q=x&answer_check=context"})
    # A relayed context load still answers inside its relay's budget.
    assert not recall_core.is_explicit(_call(context_load=True, budget_s=2.5))


def test_a_slow_keyword_fallback_on_a_loaded_machine_still_answers(monkeypatch):
    # Measured on the author's store with the machine loaded: 1.3 s of a 3 s
    # budget. Started a second before the deadline it lost; it must win.
    def slow_engine(*a):
        time.sleep(2.0)
        return object()

    def keyword(*a, **kw):
        time.sleep(0.45)
        return dict(KEYWORD)

    monkeypatch.setattr(recall_core, "_engine_call", slow_engine)
    monkeypatch.setattr(recall_fallback, "recall_keyword_fallback", keyword)
    monkeypatch.setattr(recall_core, "effective_budget_s", lambda call: 1.0)
    out = _under_ingress_clock(_call(fast=False, skip_answer_check=True), 1.0)
    assert _bare(out["results"]) == KEYWORD["results"] and out["_elapsed"] < 1.05
