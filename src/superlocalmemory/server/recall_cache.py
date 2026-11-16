# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com
"""Short-lived recall answers and one shared run for identical questions.

Many agents ask through one daemon. Two rules keep that cheap without ever
serving an answer older than the memory it describes:

* **One run per question in flight.** Identical recalls that arrive while one
  is running wait for it and share its answer (single flight).
* **A finished answer is kept for a few seconds** and only while nothing was
  written. The key carries the store's write generation and the picture
  generation, both read when the recall starts; an answer is kept only if
  neither moved while it ran. Writes from another process are not seen here,
  so the lifetime stays short (``RECALL_CACHE_TTL_S``).

Only complete, checked answers are kept: never a busy reply, a warming-up
answer, a keyword fallback or a partial assembly. A caller's session is part
of the key when its results carry outcome markers, so one session's markers
and outcome ticket never reach another.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import json
import os
from typing import Any, Awaitable, Callable

from superlocalmemory.cache.lru import LruCache

RECALL_CACHE_TTL_S = 8.0
RECALL_CACHE_ENTRIES = 512

#: Per-call values that never change what is retrieved.
_NOT_IN_KEY = frozenset({"answer_deadline", "origin", "budget_s", "via"})

_cache = LruCache(max_size=RECALL_CACHE_ENTRIES, ttl_seconds=RECALL_CACHE_TTL_S,
                  thread_safe=True)
_in_flight: dict[str, asyncio.Future] = {}


def enabled() -> bool:
    return os.environ.get("SLM_RECALL_CACHE", "1").strip().lower() not in ("0", "off", "false")


def generations() -> tuple[int, int]:
    from superlocalmemory.media.generation import generation as media_generation
    from superlocalmemory.storage import write_generation

    return write_generation.current(), media_generation()


def key_for(call: Any, profile_id: str, gens: tuple[int, int]) -> str:
    fields = {f.name: getattr(call, f.name) for f in dataclasses.fields(call)
              if f.name not in _NOT_IN_KEY}
    if not getattr(call, "include_marker", False):
        fields.pop("session_id", None)
    text = json.dumps({"p": profile_id, "g": list(gens), "c": fields},
                      sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def keepable(envelope: Any) -> bool:
    """A complete, checked answer; anything degraded is asked again."""
    if not isinstance(envelope, dict) or envelope.get("ok") is not True:
        return False
    if envelope.get("code") or envelope.get("warming_up") or envelope.get("degraded_reason"):
        return False
    if envelope.get("answerability") == "unjudged":
        return False
    return envelope.get("retrieval_mode") not in ("degraded_lexical", "partial_context")


def _hit(envelope: dict) -> dict:
    served = copy.deepcopy(envelope)
    served["cache"] = "hit"
    return served


async def recall_once(call: Any, profile_id: str,
                      run: Callable[[], Awaitable[dict]]) -> dict:
    """``run()`` once per question; repeats share it or reuse its answer."""
    if not enabled():
        return await run()
    gens = generations()
    key = key_for(call, profile_id, gens)
    kept = _cache.get(key)
    if kept is not None:
        return _hit(kept)
    leader = _in_flight.get(key)
    if leader is not None:
        await asyncio.wait({leader})  # never raises the leader's own failure
        if not leader.cancelled() and leader.exception() is None:
            return copy.deepcopy(leader.result())
        return await run()
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    _in_flight[key] = future
    try:
        envelope = await run()
    except BaseException:
        _in_flight.pop(key, None)
        if not future.done():
            future.cancel()  # followers run the question themselves
        raise
    _in_flight.pop(key, None)
    if not future.done():
        future.set_result(envelope)
    if keepable(envelope) and generations() == gens:
        _cache.set(key, copy.deepcopy(envelope))
    return envelope


def clear() -> None:
    _cache.clear()
    _in_flight.clear()


def stats() -> dict:
    return {"ttl_s": RECALL_CACHE_TTL_S, "in_flight": len(_in_flight), **_cache.get_stats()}


__all__ = ["RECALL_CACHE_TTL_S", "clear", "enabled", "keepable", "key_for",
           "recall_once", "stats"]
