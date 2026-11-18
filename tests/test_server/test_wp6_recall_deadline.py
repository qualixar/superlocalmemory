"""Three seconds includes admission; generous legacy environment values cannot widen it.

These are the prompt-blocking recalls (hooks, context loads), which keep the
three-second clock and the keyword fallback. An explicit question may queue
instead (test_explicit_recall_queue.py).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from superlocalmemory.server import recall_core


def _bare(results):
    """Results without the W3-7b ``why`` receipt every answer now carries."""
    return [{k: v for k, v in r.items() if k != "why"} if isinstance(r, dict) else r
            for r in results]


def test_default_and_override_cannot_exceed_three_seconds(monkeypatch):
    monkeypatch.delenv("SLM_SEARCH_RECALL_TIMEOUT_S", raising=False)
    assert recall_core.recall_budget_s() <= 3
    monkeypatch.setenv("SLM_SEARCH_RECALL_TIMEOUT_S", "25")
    assert recall_core.recall_budget_s() <= 3


def test_admission_is_part_of_the_same_deadline(monkeypatch):
    from superlocalmemory.server import profile_runtime, recall_fallback

    async def run():
        gate = asyncio.Semaphore(0)
        monkeypatch.setattr(recall_core, "RECALL_SEMAPHORE", gate)
        monkeypatch.setattr(recall_core, "effective_budget_s", lambda call: 0.12)
        monkeypatch.setattr(
            profile_runtime,
            "get_profile_runtime",
            lambda state: SimpleNamespace(
                snapshot=SimpleNamespace(profile_id="default", generation=1)
            ),
        )
        monkeypatch.setattr(
            recall_fallback,
            "recall_keyword_fallback",
            lambda *a, **kw: {
                "answerability": "unjudged",
                "answer_check_status": "skipped",
                "results": [],
            },
        )
        monkeypatch.setattr(recall_core, "_envelope", lambda *a: {"answerability": "unjudged"})
        engine = SimpleNamespace(recall=Mock())
        call = recall_core.RecallCall(
            query="a fresh question",
            limit=3,
            session_id="",
            agent_id="test",
            profile_id="default",
            fast=False,
            skip_answer_check=True,
        )
        task = asyncio.create_task(
            recall_core.run_recall(engine, call, app_state=SimpleNamespace())
        )
        await asyncio.sleep(0.2)
        completed_inside_budget = task.done()
        gate.release()  # lets the old implementation leave instead of leaking a test task
        try:
            result = await task
            assert completed_inside_budget
            assert result["answerability"] == "unjudged"
            engine.recall.assert_not_called()
        finally:
            if not task.done():
                task.cancel()

    asyncio.run(run())


def test_slow_response_assembly_is_also_bounded(monkeypatch):
    import time

    from superlocalmemory.server import profile_runtime

    async def run():
        monkeypatch.setattr(recall_core, "effective_budget_s", lambda call: 0.1)
        monkeypatch.setattr(
            profile_runtime,
            "get_profile_runtime",
            lambda state: SimpleNamespace(
                snapshot=SimpleNamespace(profile_id="default", generation=1)
            ),
        )
        monkeypatch.setattr(recall_core, "_engine_call", lambda *a: object())

        def slow(*a):
            time.sleep(0.3)
            return {"answerability": "supported"}

        monkeypatch.setattr(recall_core, "_envelope", slow)
        call = recall_core.RecallCall("a question?", 3, "", "test", True, skip_answer_check=True)
        started = time.monotonic()
        result = await recall_core.run_recall(object(), call, app_state=SimpleNamespace())
        assert time.monotonic() - started < 0.2
        assert result["answerability"] == "unjudged"
        assert result["no_confident_match"] is False

    asyncio.run(run())


def test_executor_work_is_counted_and_transition_leased_until_it_finishes(monkeypatch):
    import threading
    from contextlib import contextmanager

    from superlocalmemory.core import recall_gate
    from superlocalmemory.server import profile_runtime

    entered, release = threading.Event(), threading.Event()
    active = []

    @contextmanager
    def lease():
        active.append(True)
        try:
            yield True
        finally:
            active.pop()

    runtime = SimpleNamespace(
        snapshot=SimpleNamespace(profile_id="default", generation=1), operation_nowait=lease
    )

    def slow(*args):
        entered.set()
        release.wait(1)
        return {}

    monkeypatch.setattr(profile_runtime, "get_profile_runtime", lambda state: runtime)
    monkeypatch.setattr(recall_core, "effective_budget_s", lambda call: 0.1)
    # The fallback's own floor is cut short here too: the subject is the lease
    # a still-running fallback keeps after the answer has gone.
    monkeypatch.setattr(recall_core, "RECALL_BUDGET_FLOOR_S", 0.1)
    monkeypatch.setattr(recall_core, "_engine_call", lambda *a: None)
    monkeypatch.setattr(recall_core, "_fallback", slow)
    engine = object()

    async def run():
        call = recall_core.RecallCall("a question?", 3, "", "test", True, skip_answer_check=True)
        baseline = recall_gate.in_flight()
        try:
            await recall_core.run_recall(engine, call, app_state=SimpleNamespace(engine=engine))
            assert entered.is_set()
            assert active
            assert recall_gate.in_flight() > baseline
        finally:
            release.set()

    asyncio.run(run())


def test_paused_background_work_does_not_refuse_foreground_recall(monkeypatch):
    from superlocalmemory.server import profile_runtime

    async def run():
        runtime = profile_runtime.ProfileRuntime('default')
        runtime._background_paused = True
        monkeypatch.setattr(profile_runtime, 'get_profile_runtime', lambda state: runtime)
        monkeypatch.setattr(recall_core, '_engine_call', lambda *a: object())
        monkeypatch.setattr(recall_core, '_envelope', lambda *a: {'worked': True})
        engine = object()
        result = await recall_core.run_recall(engine,
            recall_core.RecallCall('a question?', 3, '', 'test', True, skip_answer_check=True),
            app_state=SimpleNamespace(engine=engine))
        assert result == {'worked': True}
    asyncio.run(run())


def test_the_keyword_fallback_keeps_its_floor_after_a_short_budget(monkeypatch):
    """A budget shortened below the fallback's floor (SLM_SEARCH_RECALL_TIMEOUT_S,
    a relay) gives up on full recall at that budget, but the keyword fallback,
    with its source check, still gets the floor to answer."""
    import time

    from superlocalmemory.server import profile_runtime

    async def run():
        monkeypatch.setattr(recall_core, "effective_budget_s", lambda call: 0.05)
        monkeypatch.setattr(
            profile_runtime, "get_profile_runtime",
            lambda state: SimpleNamespace(
                snapshot=SimpleNamespace(profile_id="default", generation=1)))

        def slow_engine(*a):
            time.sleep(0.3)
            return object()

        def fallback(*a):
            time.sleep(0.1)  # longer than the budget, well inside the floor
            return {"results": [{"content": "checked source"}],
                    "degraded_reason": "recall_budget_exceeded"}

        monkeypatch.setattr(recall_core, "_engine_call", slow_engine)
        monkeypatch.setattr(recall_core, "_fallback", fallback)
        call = recall_core.RecallCall("a question?", 3, "", "test", True, skip_answer_check=True)
        started = time.monotonic()
        result = await recall_core.run_recall(object(), call, app_state=SimpleNamespace())
        assert _bare(result["results"]) == [{"content": "checked source"}]
        assert time.monotonic() - started < recall_core.RECALL_BUDGET_FLOOR_S

    asyncio.run(run())


def test_the_http_clock_leaves_the_fallback_its_floor(monkeypatch):
    from superlocalmemory.server import recall_ingress

    monkeypatch.setenv("SLM_SEARCH_RECALL_TIMEOUT_S", "0.01")
    seen = []

    async def app(scope, receive, send):
        seen.append(recall_ingress._DEADLINE.get() - asyncio.get_running_loop().time())

    asyncio.run(recall_ingress.serve(app, None, {"type": "http", "path": "/recall"},
                                     None, None))
    assert seen and seen[0] > recall_core.RECALL_BUDGET_FLOOR_S - 0.1
