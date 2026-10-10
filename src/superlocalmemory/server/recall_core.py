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
import logging
import math
import os
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
    """Generous latency budget for a recall before the keyword fallback (v3.8.3).

    SLM's value is quality recall under heavy multi-agent load, so semantic
    recall is given ample time; the keyword fallback is a LAST-RESORT safety
    net for a genuine hang (e.g. a wedged embedder), not a speed cutoff. Tune
    with SLM_SEARCH_RECALL_TIMEOUT_S (shared with the dashboard search route).
    """
    try:
        v = float(os.environ.get("SLM_SEARCH_RECALL_TIMEOUT_S", ""))
        return v if v > 0 else 25.0
    except (TypeError, ValueError):
        return 25.0


#: A call budget is never allowed below this: the keyword fallback still has to run.
RECALL_BUDGET_FLOOR_S = 2.0


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
    """What this caller may not see, for the thread that runs the recall (nothing for a local caller)."""
    from superlocalmemory.retrieval import remote_view, visibility

    ctx = remote_view.context_for(call.caller_view, engine._db, call.profile_id or engine.profile_id)
    return visibility.use(ctx) if ctx is not None else nullcontext()


def _engine_call(engine: Any, call: RecallCall) -> Any:
    """``engine.recall`` with the call's arguments, on the executor thread."""
    from superlocalmemory.core import answer_check_history as history
    from superlocalmemory.core.answer_check_scope import skip_answer_check

    with skip_answer_check() if call.skip_answer_check else nullcontext(), \
            history.origin(call.origin) if call.origin else nullcontext(), \
            _visibility_for(engine, call):
        return engine.recall(
            call.query, limit=call.limit, session_id=call.session_id,
            agent_id=call.agent_id, fast=call.fast,
            profile_id=call.profile_id or None,
            include_global=call.include_global, include_shared=call.include_shared,
            window=call.window or None, as_of=call.as_of or None,
            known_as_of=call.known_as_of or None, valid_at=call.valid_at or None,
            include_unknown=call.include_unknown,
            # Only when given: an engine stand-in need not know these.
            **({"facets": call.facets} if call.facets is not None else {}),
            **({"answer_check": "no_reorder"} if call.no_reorder else {}),
        )


def _envelope(engine: Any, call: RecallCall, response: Any, snapshot: Any) -> dict:
    from superlocalmemory.server.recall_serializer import (
        recall_response_metadata,
        serialize_recall_response,
    )

    memory_ids = list({r.fact.memory_id for r in response.results[:call.limit]
                       if r.fact.memory_id})
    memory_map = (
        engine._db.get_memory_content_batch(
            memory_ids, call.profile_id or engine.profile_id,
            include_global=True, include_shared=True,
        ) if memory_ids else {}
    )
    retrieval = getattr(engine._config, "retrieval", None)
    from superlocalmemory.retrieval.media_channel import memory_sources
    from superlocalmemory.core.kind_query import engine_display_min_confidence

    results, no_confident_match = serialize_recall_response(
        response, limit=call.limit,
        memory_map={k: sanitize_json_text(v) for k, v in memory_map.items()},
        source_map=memory_sources(engine._db, memory_ids),
        per_fact_max=getattr(retrieval, "recall_per_fact_max_chars", 2400),
        total_max=getattr(retrieval, "recall_total_max_chars", 12000),
        # Markers only on session-bearing recalls: a marker can only buy a
        # learning signal when a pending outcome exists to settle.
        include_marker=call.include_marker,
        full=call.full, include_source=call.include_source,
        # One kind threshold for the filter and the labels on every surface.
        display_min_confidence=engine_display_min_confidence(engine),
    )
    for item in results:
        item["content"] = sanitize_json_text(item.get("content", ""))
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
        "channel_weights": {k: round(v, 3)
                            for k, v in (response.channel_weights or {}).items()},
        "total_candidates": getattr(response, "total_candidates", 0),
        "results": results,
        "count": len(results),
        "no_confident_match": no_confident_match,
        **recall_response_metadata(response),
    }


async def run_recall(engine: Any, call: RecallCall, *, app_state: Any) -> dict:
    """Run one recall and return its response body. Raises on an engine failure.

    The caller has already authorised the request and checked the profile.
    """
    from superlocalmemory.core.recall_gate import RecallHold
    from superlocalmemory.server.profile_runtime import get_profile_runtime
    from superlocalmemory.server.recall_fallback import recall_keyword_fallback

    # Marks a recall in flight so background work pauses. The mark is held
    # until the engine work this recall started has ended, on every path.
    hold = RecallHold()
    acquired = False
    try:
        # Deep recalls are gated; fast recalls keep their bounded channels and
        # skip remote agentic verification, so they do not need the semaphore.
        if not call.fast:
            await RECALL_SEMAPHORE.acquire()
            acquired = True
        # v3.8.3: bound the recall so callers never hang on a wedged embedder.
        # Poll the executor future (which cannot be cancelled) without blocking
        # the loop; only past the generous budget is the keyword answer served.
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, hold.run, _engine_call, engine, call)
        budget = effective_budget_s(call)
        deadline = loop.time() + budget
        while not future.done() and loop.time() < deadline:
            await asyncio.sleep(0.05)
        snapshot = get_profile_runtime(app_state).snapshot
        if not future.done():
            future.add_done_callback(lambda f: (f.cancelled() or f.exception()))
            logger.warning("recall: semantic recall exceeded %.0fs budget for %r — "
                           "serving keyword fallback", budget, call.query[:80])
            # The fallback honours the same facets the primary path was given.
            with _visibility_for(engine, call):
                return recall_keyword_fallback(
                    engine, call.query, call.limit, profile_id=call.profile_id or None,
                    profile=call.profile_id or snapshot.profile_id,
                    profile_generation=snapshot.generation, facets=call.facets,
                )
        # Reads memory text and serialises: on the executor, never the loop, which
        # every other request (and the next recall's answer) is waiting on.
        return await loop.run_in_executor(None, hold.run, _envelope, engine, call,
                                          future.result(), snapshot)
    finally:
        if acquired:
            RECALL_SEMAPHORE.release()
        hold.leave()


__all__ = ["RECALL_BUDGET_FLOOR_S", "RECALL_SEMAPHORE", "RecallCall",
           "effective_budget_s", "parse_budget_s", "recall_budget_s", "run_recall",
           "sanitize_json_text"]
