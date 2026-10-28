# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""The daemon's recall, after the request has been read and authorised.

Extracted from the ``GET /recall`` handler in ``server/unified_daemon.py``
(4.1.21) so that a saved view runs exactly the recall ``/recall`` runs — the
same thread-pool call, the same full-recall semaphore, the same budget and
keyword fallback when the budget is exceeded, the same serializer and the same
envelope. Two copies of this would drift; one is what makes "a view gives the
same answer as the question it stands for" true by construction.

What stays in the handler: reading and validating query parameters, the
permission check, the unknown-profile 404, the session id and the actor. What
lives here is everything from "the request is good" to "the response body".

``origin`` tags the recall for the Answer Check history (who asked:
a dashboard test, or a saved view run from the dashboard, the CLI or an agent).
It is entered on the executor thread, because a context variable set on the
event loop does not cross ``run_in_executor``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("superlocalmemory.server.recall_core")

# v3.4.53: Limit concurrent full (non-fast) recalls. Without this, N parallel
# full recalls spawn N threads → Ollama serialises, the reranker lock queues,
# and total wall time is N × one recall. Three gives parallelism without
# oversaturation. Shared by every caller of :func:`run_recall`.
RECALL_SEMAPHORE = asyncio.Semaphore(3)


def recall_budget_s() -> float:
    """The public three-second ceiling; an environment override may only shorten it."""
    from superlocalmemory.retrieval.answer_check_status import RECALL_CEILING_S

    try:
        value = float(os.environ.get("SLM_SEARCH_RECALL_TIMEOUT_S", ""))
        return min(value, RECALL_CEILING_S) if value > 0 else RECALL_CEILING_S
    except (TypeError, ValueError):
        return RECALL_CEILING_S


#: A call budget is never allowed below this: the keyword fallback still has to run.
RECALL_BUDGET_FLOOR_S = 2.0

#: A short-clock recall still running after this long (or 40% of its budget,
#: if sooner) starts the keyword fallback beside it. A small store answers
#: first and never pays for it; on 27k facts the fallback measured 0.07-0.5 s
#: idle and 1.3 s on a loaded machine, so it needs most of the budget.
HEDGE_AFTER_S = 1.0


def parse_budget_s(raw: Any) -> float | None:
    """A caller's recall budget in seconds, or ``None`` when absent or unusable.

    Only a finite number above zero counts; anything else is ignored so a
    malformed value can never turn into an error or an unbounded wait.
    """
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def sanitize_json_text(text: str) -> str:
    """Strip control characters that break JSON serialization.

    Facts ingested from agent conversations can contain raw control characters
    that survive database round-trips but fail JSON encoding of the response.
    They are replaced with spaces so the byte length is preserved and
    truncation stays predictable. Only the ASCII control range is touched.
    """
    if not text:
        return text
    if all(c >= " " or c in "\n\r\t" for c in text):
        return text
    return "".join(c if c >= " " or c in "\n\r\t" else " " for c in text)


@dataclass(frozen=True, slots=True)
class RecallCall:
    """One recall, fully resolved: every value already validated and normalised."""

    query: str
    limit: int
    session_id: str
    agent_id: str
    fast: bool
    profile_id: str = ""
    include_global: bool | None = None
    include_shared: bool | None = None
    window: str = ""
    as_of: str = ""
    known_as_of: str = ""
    valid_at: str = ""
    include_unknown: bool = False
    facets: Any = None
    skip_answer_check: bool = False
    no_reorder: bool = False
    full: bool = False
    include_source: bool = False
    include_marker: bool = False
    origin: str = ""
    #: A per-call budget in seconds. It can only SHORTEN ``recall_budget_s()``
    #: (see :func:`effective_budget_s`); ``None`` leaves the default untouched.
    budget_s: float | None = None
    #: How a remote caller came in (see ``retrieval/remote_view``); ``""`` is a local caller.
    caller_view: str = ""
    trace: bool = False
    #: 4.1.25: add each result's evidence features and the question's own
    #: (``query_evidence``) to the envelope. Recall trace sets it.
    include_evidence: bool = False
    #: 4.1.25: the weak matches of a recall that abstains, labelled ``weak``.
    show_weak: bool = False
    #: 4.1.25: up to three close memories of a real question, labelled related.
    include_related: bool = False
    #: 4.1.25: a context load, not a question: no answer-support gate.
    context_load: bool = False
    #: 4.1.25 layered recall (``core.recall_layers``): 0 glance, 1 recall,
    #: 2 context; ``max_tokens`` caps the results and is never exceeded.
    depth: int = 1
    max_tokens: int | None = None
    #: 4.1.25: the monotonic instant the daemon stops waiting for the engine;
    #: the answer check, which has its own budget, never runs past it.
    answer_deadline: float | None = None


def effective_budget_s(call: RecallCall) -> float:
    """The seconds this call may wait before the keyword fallback is served.

    The call's own budget (a relayed call must answer before its relay gives
    up) can only shorten the default, and never below the floor the fallback
    query needs. Without one this is exactly ``recall_budget_s()``.
    """
    default = recall_budget_s()
    if call.budget_s is None:
        return default
    return min(default, max(call.budget_s, RECALL_BUDGET_FLOOR_S))


def _visibility_for(engine: Any, call: RecallCall) -> Any:
    """The caller's visibility context; local callers use the engine's scope."""
    if not call.caller_view:
        return nullcontext()
    from superlocalmemory.retrieval import remote_view, visibility

    ctx = remote_view.context_for(
        call.caller_view, engine._db, call.profile_id or engine.profile_id
    )
    return visibility.use(ctx) if ctx is not None else nullcontext()


def _engine_call(engine: Any, call: RecallCall) -> Any:
    """``engine.recall`` with the call's arguments, on the executor thread."""
    from superlocalmemory.core import answer_check_history as history
    from superlocalmemory.core.answer_check_scope import context_load, skip_answer_check
    from superlocalmemory.retrieval.answer_check_status import caller_deadline

    with (
        skip_answer_check() if call.skip_answer_check else nullcontext(),
        context_load() if call.context_load else nullcontext(),
        history.origin(call.origin) if call.origin else nullcontext(),
        _visibility_for(engine, call),
        caller_deadline(call.answer_deadline),
    ):
        return engine.recall(
            call.query,
            limit=call.limit,
            session_id=call.session_id,
            agent_id=call.agent_id,
            fast=call.fast,
            profile_id=call.profile_id or None,
            include_global=call.include_global,
            include_shared=call.include_shared,
            window=call.window or None,
            as_of=call.as_of or None,
            known_as_of=call.known_as_of or None,
            valid_at=call.valid_at or None,
            include_unknown=call.include_unknown,
            # Only when given: an engine stand-in need not know these.
            **({"facets": call.facets} if call.facets is not None else {}),
            **({"answer_check": "no_reorder"} if call.no_reorder else {}),
            **({"show_weak": True} if call.show_weak else {}),
            **({"include_related": True} if call.include_related else {}),
        )


def _envelope(engine: Any, call: RecallCall, response: Any, snapshot: Any) -> dict:
    if call.trace:
        from superlocalmemory.server.routes.recall_trace import envelope
        return envelope(engine, call, response)
    from superlocalmemory.server.recall_serializer import (
        recall_response_metadata,
        serialize_recall_response,
    )

    memory_ids = list(
        {r.fact.memory_id for r in response.results[: call.limit] if r.fact.memory_id}
    )
    memory_map = (
        engine._db.get_memory_content_batch(
            memory_ids,
            call.profile_id or engine.profile_id,
            include_global=True,
            include_shared=True,
        )
        if memory_ids
        else {}
    )
    retrieval = getattr(engine._config, "retrieval", None)
    from superlocalmemory.core.kind_query import engine_display_min_confidence
    from superlocalmemory.retrieval.media_channel import memory_sources

    results, no_confident_match = serialize_recall_response(
        response,
        limit=call.limit,
        memory_map={k: sanitize_json_text(v) for k, v in memory_map.items()},
        source_map=memory_sources(engine._db, memory_ids),
        per_fact_max=getattr(retrieval, "recall_per_fact_max_chars", 2400),
        total_max=getattr(retrieval, "recall_total_max_chars", 12000),
        # Markers only on session-bearing recalls: a marker can only buy a
        # learning signal when a pending outcome exists to settle.
        include_marker=call.include_marker,
        full=call.full,
        include_source=call.include_source,
        include_evidence=call.include_evidence,
        # One kind threshold for the filter and the labels on every surface.
        display_min_confidence=engine_display_min_confidence(engine),
    )
    for item in results:
        item["content"] = sanitize_json_text(item.get("content", ""))
    results, layers = _layers(engine, call, results)
    return {
        "ok": True,
        # The profile that actually served this recall: the routed profile when
        # one was named, else the active one. profile_generation describes the
        # global switch state, which per-request routing never moves.
        "profile": call.profile_id or snapshot.profile_id,
        "profile_generation": snapshot.generation,
        "query": call.query,
        "query_type": response.query_type,
        "result_count": len(results),
        "retrieval_time_ms": round(response.retrieval_time_ms, 1),
        "channel_weights": {k: round(v, 3) for k, v in (response.channel_weights or {}).items()},
        "total_candidates": getattr(response, "total_candidates", 0),
        "results": results,
        "count": len(results),
        "no_confident_match": no_confident_match,
        **({"layers": layers} if layers else {}),
        **recall_response_metadata(response),
        **({"query_evidence": dict(getattr(response, "query_evidence", {}) or {})}
           if call.include_evidence else {}),
    }


def _layers(engine: Any, call: RecallCall, results: list[dict]) -> tuple[list[dict], dict]:
    """The results at the call's depth within its budget (``core.recall_layers``);
    depth 2 context goes through the fetch gate and the caller's visibility."""
    if call.depth == 1 and call.max_tokens is None:
        return results, {}
    from superlocalmemory.core import recall_layers

    pid = call.profile_id or engine.profile_id
    with _visibility_for(engine, call):
        return recall_layers.shape(
            results, call.depth, call.max_tokens,
            context_of=lambda item: recall_layers.context_for(
                engine._db, pid, str(item.get("fact_id", ""))))


#: An explicit question (an agent's recall tool, the CLI, the dashboard) may
#: wait this long for a free slot before it is told the daemon is busy. The run
#: itself keeps the normal budget once it starts. Hooks and context loads block
#: a prompt, so they keep the three-second ceiling and the keyword fallback.
EXPLICIT_QUEUE_S = 12.0
#: Extra time an explicit run that has started may take on a loaded machine
#: before it is reported busy. A run that has started is waited for (up to the
#: request's own ceiling) rather than abandoned: abandoning it freed its slot
#: while its thread kept the CPU, so every later question ran slower still.
EXPLICIT_RUN_GRACE_S = 4.0
#: What an explicit question keeps after its answer check, to assemble the reply.
ANSWER_ASSEMBLY_MARGIN_S = 0.3
#: Never longer than this from arrival to answer (the HTTP ingress clock).
EXPLICIT_CEILING_S = EXPLICIT_QUEUE_S + 3.0 + EXPLICIT_RUN_GRACE_S
_explicit_waiting = 0


def is_explicit(call: RecallCall) -> bool:
    """A question someone is waiting on, not a prompt-blocking context load.

    ``fast`` (skip the internal agentic round, the agent default) does not
    matter here: an agent's recall tool is fast and explicit.
    """
    # depth 0 is how much to show, and a context load (session_init, slm
    # context) only skips the answer check: someone waits on both, so both
    # get the full clock. Only the prompt hook (answer_check=skip, it blocks
    # typing) and a relayed budget keep the short one (4.2.3: on a 27k-fact
    # store both came back empty, every channel "timeout").
    if call.context_load:
        return call.budget_s is None
    return not (call.skip_answer_check or call.budget_s is not None)


def busy(call: RecallCall, snapshot: Any, *, retry_after_s: float) -> dict:
    """The honest answer when every slot is taken: ask again shortly.

    Never a keyword guess dressed as recall. ``detail`` carries the code in the
    shape the daemon client reads from a 503 body.
    """
    seconds = max(1, int(math.ceil(retry_after_s)))
    message = (f"SuperLocalMemory is answering other questions right now. "
               f"Ask again in about {seconds} seconds; nothing was lost.")
    return {
        "ok": False, "success": False, "code": "RECALL_BUSY", "retryable": True,
        "retry_after_s": seconds, "error": message, "message": message,
        "query": call.query,
        "profile": call.profile_id or getattr(snapshot, "profile_id", ""),
        "results": [], "count": 0, "result_count": 0,
        "detail": {"code": "RECALL_BUSY", "message": message},
    }


def _retry_after() -> float:
    # Three runs at a time, about a second each on a loaded machine.
    return 2.0 + _explicit_waiting / 3.0


def as_http(envelope: Any) -> Any:
    """A busy reply becomes HTTP 503 with ``Retry-After``; anything else is unchanged."""
    if isinstance(envelope, dict) and envelope.get("code") == "RECALL_BUSY":
        from starlette.responses import JSONResponse

        return JSONResponse(envelope, status_code=503, headers={
            "Retry-After": str(envelope.get("retry_after_s", 5))})
    return envelope


async def run_recall(engine: Any, call: RecallCall, *, app_state: Any) -> dict:
    """One deadline covers queue admission, worker execution and response assembly.

    Identical questions in flight share one run, and a complete answer is kept
    for a few seconds while nothing is written (``server.recall_cache``).
    """
    from superlocalmemory.server import recall_cache

    return await recall_cache.recall_once(
        call, call.profile_id or getattr(engine, "profile_id", ""),
        lambda: _run_recall(engine, call, app_state=app_state))


async def _run_recall(engine: Any, call: RecallCall, *, app_state: Any) -> dict:
    from superlocalmemory.server.profile_runtime import get_profile_runtime
    from superlocalmemory.server.recall_admission import foreground_admission
    from superlocalmemory.server.recall_deadline import timed_out

    explicit = is_explicit(call)
    budget = effective_budget_s(call)
    from superlocalmemory.server.recall_ingress import remaining_deadline
    public_deadline = remaining_deadline(budget)
    worker_deadline = public_deadline - min(0.20, budget * 0.25)
    queue_deadline = remaining_deadline(EXPLICIT_QUEUE_S) if explicit else worker_deadline
    runtime = get_profile_runtime(app_state)
    snapshot = runtime.snapshot
    async with foreground_admission(runtime, queue_deadline) as admitted:
        if not admitted:
            if explicit:
                return busy(call, snapshot, retry_after_s=_retry_after())
            return timed_out(call, snapshot)
        snapshot = runtime.snapshot
        return await _run_admitted(
            engine, call, app_state, runtime, snapshot, public_deadline, worker_deadline,
            queue_deadline if explicit else None)


async def _acquire_slot(call, loop, worker_deadline, queue_deadline):
    """Wait for a full-recall slot; an explicit question may queue longer."""
    global _explicit_waiting
    until = queue_deadline if queue_deadline is not None else worker_deadline
    if queue_deadline is not None:
        _explicit_waiting += 1
    try:
        await asyncio.wait_for(RECALL_SEMAPHORE.acquire(), max(0, until - loop.time()))
        return True
    except asyncio.TimeoutError:
        return False
    finally:
        if queue_deadline is not None:
            _explicit_waiting -= 1


async def _run_admitted(engine, call, app_state, runtime, snapshot,
                        public_deadline, worker_deadline, queue_deadline=None):
    from superlocalmemory.core.recall_gate import RecallHold
    from superlocalmemory.server.recall_deadline import submit, wait_until

    loop = asyncio.get_running_loop()
    explicit = queue_deadline is not None
    hold, acquired, future = RecallHold(), False, None
    try:
        # Explicit questions always take a slot: a dozen engine runs at once on
        # one machine each overrun their channels and all degrade together.
        if not call.fast or explicit:
            acquired = await _acquire_slot(call, loop, worker_deadline, queue_deadline)
            if not acquired:
                if explicit:
                    return busy(call, snapshot, retry_after_s=_retry_after())
                return await _bounded_fallback(
                    engine, call, app_state, public_deadline, snapshot, hold, runtime)
            if explicit:
                # The run gets the full budget from the moment it starts, not
                # what is left after queueing; the clock is the ingress clock.
                from superlocalmemory.server.recall_ingress import remaining_deadline
                budget = effective_budget_s(call)
                public_deadline = remaining_deadline(budget)
                worker_deadline = public_deadline - min(0.20, budget * 0.25)
        run_deadline = worker_deadline
        answer_until = worker_deadline
        if explicit:
            from superlocalmemory.server.recall_ingress import remaining_deadline
            # The grace, or whatever the request's own clock still allows
            # (less a moment to assemble the answer), whichever is longer.
            run_deadline = max(
                remaining_deadline(max(0.0, worker_deadline - loop.time()) + EXPLICIT_RUN_GRACE_S),
                remaining_deadline(EXPLICIT_CEILING_S) - 1.0)
            public_deadline = max(public_deadline, run_deadline + 0.15)
            # 4.2.4: the answer check shares the question's run clock. Held to
            # the three-second budget it was skipped whenever search took more
            # than ~3 s on a large store, though it costs 30-140 ms and the
            # question still had seconds left: "budget_exhausted" on nearly
            # every recall of a 27k-fact store.
            answer_until = run_deadline - ANSWER_ASSEMBLY_MARGIN_S
        # The engine thread reads monotonic time; the loop's clock may differ.
        call = dataclasses.replace(call, answer_deadline=time.monotonic() + max(
            0.0, answer_until - loop.time()))
        future = submit(loop, hold, runtime, app_state, engine, _engine_call, engine, call)
        hedge = None
        if explicit:
            # #304: a run still going when only the fallback's floor is left
            # starts the keyword fallback beside it, so an overrun answers with
            # labelled keyword matches. It used to say "busy, ask again in 2 s"
            # on an idle daemon; on a 37k-fact store every retry overran too.
            response = await wait_until(future, run_deadline - RECALL_BUDGET_FLOOR_S)
            if response is None:
                hedge = submit(loop, hold, runtime, app_state, engine,
                               _fallback, engine, call, app_state)
                response = await wait_until(future, run_deadline)
        else:
            # Delayed hedge: a short-clock run still going near its deadline
            # starts the keyword fallback beside it, so an overrun answers with
            # keyword matches instead of nothing (4.2.3: a 2.5 s recall on a big
            # store left the fallback ~0.2 s and the prompt hook injected none).
            left = max(0.0, worker_deadline - loop.time())
            response = await wait_until(
                future, loop.time() + min(HEDGE_AFTER_S, left * 0.4))
            if response is None:
                hedge = submit(loop, hold, runtime, app_state, engine,
                               _fallback, engine, call, app_state)
                response = await wait_until(future, worker_deadline)
        if response is None:
            # Busy is only the answer when no slot was free (above): asking
            # again helps a full house, never a run that overran.
            if explicit:
                logger.warning("Explicit recall overran its run budget; "
                               "serving unjudged keyword fallback")
            else:
                logger.warning("Recall deadline reached; serving unjudged keyword fallback")
            return await _bounded_fallback(
                    engine, call, app_state, public_deadline, snapshot, hold, runtime,
                    started=hedge)
        minimal = _minimal_envelope(call, response, snapshot)
        envelope = submit(loop, hold, runtime, app_state, engine, _envelope,
                          engine, call, response, snapshot)
        assembled = await wait_until(envelope, public_deadline)
        from superlocalmemory.server.recall_deadline import unjudged
        return assembled if assembled is not None else unjudged(minimal)
    finally:
        if acquired:
            _release_when_done(future)
        hold.leave()


def _release_when_done(future: Any) -> None:
    """Give the slot back when the engine work ends, not when the caller stops
    waiting: a slot is a share of the CPU, and a run still going still uses it."""
    if future is None or future.done():
        RECALL_SEMAPHORE.release()
        return
    future.add_done_callback(lambda _f: RECALL_SEMAPHORE.release())


def _minimal_envelope(call, response, snapshot):
    """Keep already-retrieved facts without any source hydration/database access."""
    results = [
        {"id": result.fact.memory_id or result.fact.fact_id,
         "fact_id": result.fact.fact_id, "memory_id": result.fact.memory_id,
         "content": sanitize_json_text(result.fact.content[:2400]), "score": result.score}
        for result in (getattr(response, "results", ()) or ())[:call.limit]
    ]
    return {
        "ok": True, "query": call.query,
        "profile": call.profile_id or getattr(snapshot, "profile_id", ""),
        "profile_generation": getattr(snapshot, "generation", 0),
        "results": results, "count": len(results), "result_count": len(results),
        "channel_status": dict(getattr(response, "channel_status", {}) or {}),
        "retrieval_mode": "partial_context",
        "degraded_reason": "recall_assembly_budget_exceeded",
    }


async def _bounded_fallback(engine, call, app_state, deadline, snapshot, hold, runtime,
                            started=None):
    from superlocalmemory.server.recall_deadline import submit, timed_out, unjudged, wait_until
    from superlocalmemory.server.recall_ingress import fallback_deadline

    future = started if started is not None else submit(
        asyncio.get_running_loop(), hold, runtime, app_state, engine,
        _fallback, engine, call, app_state)
    # The fallback keeps its floor even after a budget shortened below it.
    result = await wait_until(future, fallback_deadline(deadline, RECALL_BUDGET_FLOOR_S))
    return unjudged(result) if result is not None else timed_out(call, snapshot)


def _fallback(engine: Any, call: RecallCall, app_state: Any) -> dict:
    from superlocalmemory.server.profile_runtime import get_profile_runtime
    from superlocalmemory.server.recall_fallback import recall_keyword_fallback

    snapshot = get_profile_runtime(app_state).snapshot
    with _visibility_for(engine, call):
        return recall_keyword_fallback(
            engine,
            call.query,
            call.limit,
            profile_id=call.profile_id or None,
            profile=call.profile_id or snapshot.profile_id,
            profile_generation=snapshot.generation,
            facets=call.facets,
            show_weak=call.show_weak,
            include_related=call.include_related,
        )


__all__ = [
    "EXPLICIT_CEILING_S",
    "RECALL_BUDGET_FLOOR_S",
    "RECALL_SEMAPHORE",
    "RecallCall",
    "as_http",
    "busy",
    "effective_budget_s",
    "is_explicit",
    "parse_budget_s",
    "recall_budget_s",
    "run_recall",
    "sanitize_json_text",
]
