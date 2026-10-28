"""An explicit question queues for the full pipeline; it is never a keyword guess.

Hooks and context loads keep the three-second clock (test_wp6_recall_deadline).
An agent's recall tool, the CLI and the dashboard wait for a slot, run the full
pipeline, and are told plainly to ask again when every slot stays taken.
Identical questions share one run, and an answer is kept a few seconds while
nothing is written.
"""

import asyncio
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from superlocalmemory.server import profile_runtime, recall_cache, recall_core, recall_fallback
from superlocalmemory.server.recall_ingress import explicit_scope
from superlocalmemory.storage import write_generation

SNAP = SimpleNamespace(profile_id="default", generation=1)


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(profile_runtime, "get_profile_runtime",
                        lambda state: SimpleNamespace(snapshot=SNAP))
    fallback = Mock(return_value={"results": [{"content": "keyword guess"}]})
    monkeypatch.setattr(recall_fallback, "recall_keyword_fallback", fallback)
    return fallback


def _call(**kw):
    base = dict(query="what am I working on?", limit=3, session_id="", agent_id="a",
                fast=True, profile_id="default")
    base.update(kw)
    return recall_core.RecallCall(**base)


def _answer(text="checked answer"):
    return {"ok": True, "answerability": "supported", "results": [{"content": text}]}


def test_the_classes():
    assert recall_core.is_explicit(_call())  # an agent's tool: fast and explicit
    assert recall_core.is_explicit(_call(fast=False))
    assert not recall_core.is_explicit(_call(skip_answer_check=True))  # a hook
    assert recall_core.is_explicit(_call(context_load=True))  # session_init waits (4.2.3)
    assert recall_core.is_explicit(_call(depth=0))  # an agent's glance (4.2.3)
    assert not recall_core.is_explicit(_call(budget_s=2.5))  # a relayed call


def test_ingress_reads_the_class_from_the_request():
    def scope(path, qs=b""):
        return {"type": "http", "path": path, "query_string": qs}

    assert explicit_scope(scope("/recall", b"q=x"))
    assert explicit_scope(scope("/api/v3/recall/trace"))
    assert not explicit_scope(scope("/recall", b"q=x&answer_check=skip&depth=0"))
    assert explicit_scope(scope("/recall", b"q=x&answer_check=context"))  # 4.2.3
    assert not explicit_scope(scope("/recall", b"q=x&budget_s=4"))


def test_a_full_house_says_busy_and_never_guesses(monkeypatch, runtime):
    monkeypatch.setattr(recall_core, "RECALL_SEMAPHORE", asyncio.Semaphore(0))
    monkeypatch.setattr(recall_core, "EXPLICIT_QUEUE_S", 0.1)
    engine_call = Mock()
    monkeypatch.setattr(recall_core, "_engine_call", engine_call)

    async def run():
        monkeypatch.setattr(recall_core, "RECALL_SEMAPHORE", asyncio.Semaphore(0))
        start = time.monotonic()
        out = await recall_core.run_recall(object(), _call(), app_state=SimpleNamespace())
        assert time.monotonic() - start < 1.0
        return out

    out = asyncio.run(run())
    assert out["code"] == "RECALL_BUSY" and out["retryable"] is True
    assert out["retry_after_s"] >= 1 and out["results"] == []
    assert "ask again" in out["message"].lower()
    runtime.assert_not_called()  # no keyword fallback
    engine_call.assert_not_called()


def test_a_queued_question_runs_when_a_slot_frees(monkeypatch, runtime):
    monkeypatch.setattr(recall_core, "_engine_call", lambda *a: object())
    monkeypatch.setattr(recall_core, "_envelope", lambda *a: _answer())
    # Longer than the three-second budget, shorter than the queue ceiling.
    monkeypatch.setattr(recall_core, "effective_budget_s", lambda call: 0.2)

    async def run():
        gate = asyncio.Semaphore(0)
        monkeypatch.setattr(recall_core, "RECALL_SEMAPHORE", gate)
        task = asyncio.create_task(
            recall_core.run_recall(object(), _call(), app_state=SimpleNamespace()))
        await asyncio.sleep(0.5)  # well past the run budget, still queued
        assert not task.done()
        gate.release()
        return await task

    out = asyncio.run(run())
    assert out["results"] == [{"content": "checked answer"}]
    runtime.assert_not_called()


def test_a_slow_run_is_waited_for_within_its_grace(monkeypatch, runtime):
    def slow(*a):
        time.sleep(0.4)
        return object()

    monkeypatch.setattr(recall_core, "_engine_call", slow)
    monkeypatch.setattr(recall_core, "_envelope", lambda *a: _answer())
    monkeypatch.setattr(recall_core, "effective_budget_s", lambda call: 0.1)
    out = asyncio.run(recall_core.run_recall(object(), _call(), app_state=SimpleNamespace()))
    assert out["results"] == [{"content": "checked answer"}]
    runtime.assert_not_called()


def test_a_hook_still_gets_the_bounded_fallback(monkeypatch, runtime):
    def slow(*a):
        time.sleep(0.5)
        return object()

    monkeypatch.setattr(recall_core, "_engine_call", slow)
    monkeypatch.setattr(recall_core, "effective_budget_s", lambda call: 0.1)
    monkeypatch.setattr(recall_core, "RECALL_BUDGET_FLOOR_S", 0.2)
    out = asyncio.run(recall_core.run_recall(
        object(), _call(fast=False, skip_answer_check=True), app_state=SimpleNamespace()))
    assert out["answerability"] == "unjudged"
    assert out.get("code") != "RECALL_BUSY"


def test_busy_is_http_503_with_retry_after():
    response = recall_core.as_http(recall_core.busy(_call(), SNAP, retry_after_s=4.2))
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"
    body = json.loads(response.body)
    assert body["detail"]["code"] == "RECALL_BUSY"
    assert recall_core.as_http({"ok": True}) == {"ok": True}


def test_identical_questions_in_flight_share_one_run(monkeypatch, runtime):
    runs = []

    def engine_call(*a):
        runs.append(1)
        time.sleep(0.2)
        return object()

    monkeypatch.setattr(recall_core, "_engine_call", engine_call)
    monkeypatch.setattr(recall_core, "_envelope", lambda *a: _answer())

    async def run():
        calls = [recall_core.run_recall(object(), _call(), app_state=SimpleNamespace())
                 for _ in range(5)]
        return await asyncio.gather(*calls)

    outs = asyncio.run(run())
    assert len(runs) == 1
    assert all(o["results"] == [{"content": "checked answer"}] for o in outs)
    outs[0]["results"].append("mutated")  # callers get their own copies
    assert outs[1]["results"] == [{"content": "checked answer"}]


def test_an_answer_is_kept_until_a_write(monkeypatch, runtime):
    runs = []

    def engine_call(*a):
        runs.append(1)
        return object()

    monkeypatch.setattr(recall_core, "_engine_call", engine_call)
    monkeypatch.setattr(recall_core, "_envelope", lambda *a: _answer(f"run {len(runs)}"))
    ask = lambda: asyncio.run(  # noqa: E731
        recall_core.run_recall(object(), _call(), app_state=SimpleNamespace()))

    first = ask()
    second = ask()
    assert len(runs) == 1 and second["cache"] == "hit"
    assert second["results"] == first["results"]
    write_generation.bump()  # anything written: the kept answer is gone
    third = ask()
    assert len(runs) == 2 and "cache" not in third
    assert third["results"] == [{"content": "run 2"}]


def test_degraded_answers_and_other_sessions_are_not_shared(monkeypatch, runtime):
    runs = []

    def engine_call(*a):
        runs.append(1)
        return object()

    monkeypatch.setattr(recall_core, "_engine_call", engine_call)
    envelope = {"ok": True, "answerability": "unjudged", "results": []}
    monkeypatch.setattr(recall_core, "_envelope", lambda *a: dict(envelope))
    ask = lambda call: asyncio.run(  # noqa: E731
        recall_core.run_recall(object(), call, app_state=SimpleNamespace()))
    ask(_call())
    ask(_call())
    assert len(runs) == 2  # an unchecked answer is asked again

    envelope.update(answerability="supported")
    ask(_call(session_id="s1", include_marker=True))
    ask(_call(session_id="s2", include_marker=True))
    assert len(runs) == 4  # markers belong to one session
    ask(_call(session_id="s1", include_marker=True))
    assert len(runs) == 4


def test_a_write_during_the_run_keeps_nothing(monkeypatch, runtime):
    def engine_call(*a):
        write_generation.bump()
        return object()

    monkeypatch.setattr(recall_core, "_engine_call", engine_call)
    monkeypatch.setattr(recall_core, "_envelope", lambda *a: _answer())
    asyncio.run(recall_core.run_recall(object(), _call(), app_state=SimpleNamespace()))
    out = asyncio.run(recall_core.run_recall(object(), _call(), app_state=SimpleNamespace()))
    assert "cache" not in out


def test_the_store_moves_the_generation_on_every_write(tmp_path):
    from superlocalmemory.storage.database import DatabaseManager

    db = DatabaseManager(tmp_path / "m.db")
    db.execute("CREATE TABLE t (x INTEGER)")
    before = write_generation.current()
    db.execute("INSERT INTO t VALUES (1)")
    assert write_generation.current() != before
    before = write_generation.current()
    db.execute("SELECT x FROM t")
    assert write_generation.current() == before
    with db.transaction():
        db.execute("INSERT INTO t VALUES (2)")
    assert write_generation.current() != before


def test_the_daemon_client_reports_busy_not_an_outage(monkeypatch):
    from superlocalmemory.cli import daemon as daemon_mod
    from superlocalmemory.cli.daemon_errors import DaemonServerError
    from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy

    def busy_request(*a, **kw):
        assert kw.get("preserve_server_error") is True
        raise DaemonServerError(503, "RECALL_BUSY", "ask again in 3 seconds")

    monkeypatch.setattr(daemon_mod, "daemon_request", busy_request)
    proxy = DaemonPoolProxy.__new__(DaemonPoolProxy)
    proxy._timeout, proxy._unavailable = 1.0, False
    out = proxy.recall("q")
    assert out == {"ok": False, "success": False, "code": "RECALL_BUSY",
                   "retryable": True, "error": "ask again in 3 seconds"}


def test_background_generation_waits_for_a_running_recall(monkeypatch):
    from superlocalmemory.core import recall_gate
    from superlocalmemory.llm import backbone

    monkeypatch.setattr(backbone, "LLM_YIELD_MAX_S", 2.0)
    recall_gate.begin_recall()
    done = threading.Event()

    def background():
        with recall_gate.background_work():
            backbone._yield_to_recall()
        done.set()

    worker = threading.Thread(target=background)
    worker.start()
    try:
        assert not done.wait(0.3)  # parked while the recall runs
    finally:
        recall_gate.end_recall()
    assert done.wait(1.0)
    worker.join(1)
    start = time.monotonic()
    backbone._yield_to_recall()  # foreground never waits
    assert time.monotonic() - start < 0.05


def test_a_started_run_keeps_its_slot_until_the_work_ends(monkeypatch, runtime):
    """Reporting busy must not free the slot while the engine thread still runs."""
    release = threading.Event()

    def stuck(*a):
        release.wait(5)
        return object()

    monkeypatch.setattr(recall_core, "_engine_call", stuck)
    monkeypatch.setattr(recall_core, "_envelope", lambda *a: _answer())
    monkeypatch.setattr(recall_core, "effective_budget_s", lambda call: 0.05)
    monkeypatch.setattr(recall_core, "EXPLICIT_RUN_GRACE_S", 0.05)
    monkeypatch.setattr(recall_core, "EXPLICIT_CEILING_S", 0.2)

    async def run():
        gate = asyncio.Semaphore(1)
        monkeypatch.setattr(recall_core, "RECALL_SEMAPHORE", gate)
        out = await recall_core.run_recall(object(), _call(), app_state=SimpleNamespace())
        held_after_busy = gate.locked()
        release.set()
        for _ in range(50):
            if not gate.locked():
                break
            await asyncio.sleep(0.02)
        return out, held_after_busy, gate.locked()

    out, held_after_busy, held_at_end = asyncio.run(run())
    assert out.get("code") != "RECALL_BUSY"  # #304: an overrun is not a full house
    assert held_after_busy is True     # the thread still runs: the slot stays taken
    assert held_at_end is False        # and is given back when it ends


def test_an_overrun_answers_with_labelled_keyword_matches_not_503(monkeypatch, runtime):
    """#304: a started run that cannot finish is not "busy".

    On a 37k-fact store every explicit recall overran and was told to ask again
    in 2 seconds, forever: retrying never helps an overrun, only a full house.
    The keyword fallback answers instead, labelled as unchecked keyword matches.
    """
    release = threading.Event()

    def stuck(*a):
        release.wait(5)
        return object()

    monkeypatch.setattr(recall_core, "_engine_call", stuck)
    monkeypatch.setattr(recall_core, "effective_budget_s", lambda call: 0.05)
    monkeypatch.setattr(recall_core, "EXPLICIT_RUN_GRACE_S", 0.05)
    monkeypatch.setattr(recall_core, "EXPLICIT_CEILING_S", 0.6)
    monkeypatch.setattr(recall_core, "RECALL_BUDGET_FLOOR_S", 0.3)
    async def run():
        start = time.monotonic()
        out = await recall_core.run_recall(object(), _call(), app_state=SimpleNamespace())
        return out, time.monotonic() - start  # not the loop's wait for the stuck thread

    try:
        out, elapsed = asyncio.run(run())
    finally:
        release.set()
    assert out["answerability"] == "unjudged"
    assert out["results"] == [{"content": "keyword guess"}]
    assert "keyword" in out["answer_check_note"].lower()
    runtime.assert_called_once()
    assert elapsed < 1.5  # answered near its own ceiling, not after the stuck run
