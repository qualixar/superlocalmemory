# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-05 §7

"""MCP proactive-context tool — ``prestage_context``.

LLD-05 §7. Exposes a single MCP tool that returns top-K redacted memories
for a given query. Guardrails:
  - Rate limit 30 calls / minute (token bucket; A11).
  - Every returned text passes through ``redact_secrets`` (A9).
  - JSON response size bound ≤ 16 KB.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Callable

from superlocalmemory.core.security_primitives import redact_secrets

logger = logging.getLogger(__name__)

MAX_CALLS_PER_MINUTE = 30
MAX_RESPONSE_BYTES = 16 * 1024  # 16 KB
WINDOW_SECONDS = 60.0


@dataclass
class _RateLimiter:
    """Simple fixed-window rate limiter. Thread-safe.

    Keyed by session id. Clock is injectable for deterministic tests.
    """
    max_calls: int = MAX_CALLS_PER_MINUTE
    window: float = WINDOW_SECONDS
    now_fn: Callable[[], float] = time.monotonic
    _buckets: dict[str, tuple[float, int]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def allow(self, key: str) -> bool:
        with self._lock:
            now = self.now_fn()
            start, count = self._buckets.get(key, (now, 0))
            if now - start >= self.window:
                # Reset window.
                self._buckets[key] = (now, 1)
                return True
            if count >= self.max_calls:
                return False
            self._buckets[key] = (start, count + 1)
            return True

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


_DEFAULT_LIMITER = _RateLimiter()


# RecallFn for the tool. Callers inject a real recall engine; tests inject fakes.
PrestageRecallFn = Callable[[str, int, str], list[dict]]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cap_memory(memory: dict, *, max_text_bytes: int = 2048) -> dict:
    """Ensure each memory is bounded and redacted."""
    text = memory.get("text", "")
    if not isinstance(text, str):
        text = str(text)
    text = redact_secrets(text)
    if len(text.encode("utf-8")) > max_text_bytes:
        text = text.encode("utf-8")[:max_text_bytes].decode("utf-8", "ignore")
    score = float(memory.get("score", 0.0) or 0.0)
    return {
        "id": str(memory.get("id", "")),
        "text": text,
        "score": score,
        "source": str(memory.get("source", "recall")),
    }


def prestage_context(
    query: str,
    *,
    limit: int = 5,
    profile_id: str = "default",
    session_id: str = "default",
    recall_fn: PrestageRecallFn,
    limiter: _RateLimiter | None = None,
) -> dict:
    """Proactive-context tool body.

    Pure function: takes an injected recall_fn + limiter. The MCP server
    wrapper in the session process wires the real engine and shares a
    single limiter instance.
    """
    _limiter = limiter or _DEFAULT_LIMITER
    if not _limiter.allow(session_id):
        return {
            "error": "rate_limit_exceeded",
            "memories": [],
            "generated_at": _iso_now(),
            "limit": limit,
            "truncated_count": 0,
        }

    if not isinstance(query, str) or not query.strip():
        return {
            "error": "empty_query",
            "memories": [],
            "generated_at": _iso_now(),
            "limit": limit,
            "truncated_count": 0,
        }
    limit = max(1, min(int(limit), 50))

    try:
        raw = recall_fn(query, limit, profile_id) or []
    except Exception as exc:
        logger.warning("prestage_context recall failed: %s", exc)
        return {
            "error": "recall_error",
            "memories": [],
            "generated_at": _iso_now(),
            "limit": limit,
            "truncated_count": 0,
        }

    capped = [_cap_memory(m) for m in raw if isinstance(m, dict)]
    capped = capped[:limit]

    # Enforce total response size cap (A11/16 KB).
    response = {
        "memories": capped,
        "generated_at": _iso_now(),
        "limit": limit,
        "truncated_count": 0,
    }
    encoded = json.dumps(response).encode("utf-8")
    truncated = 0
    while len(encoded) > MAX_RESPONSE_BYTES and response["memories"]:
        response["memories"].pop()
        truncated += 1
        response["truncated_count"] = truncated
        encoded = json.dumps(response).encode("utf-8")
    return response


def register_prestage_tool(server, recall_fn: PrestageRecallFn,
                           *, session_id_fn: Callable[[], str] | None = None
                           ) -> None:
    """Register the ``prestage_context`` tool on an MCP server."""
    limiter = _RateLimiter()

    @server.tool()
    async def prestage_context_tool(  # pragma: no cover — MCP wiring
        query: str,
        limit: int = 5,
        profile_id: str = "default",
    ) -> dict:
        """Proactively return top-K memories for a query."""
        session_id = session_id_fn() if session_id_fn else "default"
        return prestage_context(
            query, limit=limit, profile_id=profile_id,
            session_id=session_id, recall_fn=recall_fn, limiter=limiter,
        )


__all__ = (
    "MAX_CALLS_PER_MINUTE",
    "MAX_RESPONSE_BYTES",
    "prestage_context",
    "register_prestage_tool",
)
