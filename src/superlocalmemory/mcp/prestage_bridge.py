# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""``engine.recall`` -> the ``prestage_context`` recall function, for any tool host.

Moved out of ``mcp/server.py`` so ``slm tool`` registers ``prestage_context``
with the same bridge without importing the stdio server (whose import starts
its watchdog threads). ``server.py`` re-exports :func:`memories_from` as
``_memories_from`` and wraps :func:`prestage_recall` as ``_prestage_recall``.
"""

from __future__ import annotations

from typing import Any, Callable


def memories_from(results: Any) -> list[dict]:
    """``engine.recall`` output (a RecallResponse, or a plain list) as prestage rows."""
    items = getattr(results, "results", results) or []
    out: list[dict] = []
    for r in items:
        if isinstance(r, dict):
            out.append(
                {
                    "id": str(r.get("fact_id") or r.get("id") or ""),
                    "text": str(r.get("content") or r.get("text") or ""),
                    "score": float(r.get("score") or r.get("relevance_score") or 0.0),
                    "source": str(r.get("source") or "recall"),
                }
            )
            continue
        fact = getattr(r, "fact", r)
        out.append(
            {
                "id": str(getattr(fact, "fact_id", "") or ""),
                "text": str(getattr(fact, "content", "") or ""),
                "score": float(getattr(r, "score", 0.0) or 0.0),
                "source": "recall",
            }
        )
    return out


def prestage_recall(
    get_engine: Callable[[], Any],
    query: str,
    limit: int,
    profile_id: str,
    as_of: str | None = None,
) -> list[dict]:
    """Bridge engine.recall -> prestage_context recall_fn shape.

    Serves ``profile_id`` when one is named, else the engine's (active)
    profile. The profile travels as an argument of this one recall: the shared
    engine's profile is never assigned, so no other caller is moved.
    """
    engine = get_engine()
    target = (profile_id or "").strip() or None
    kwargs: dict[str, Any] = {"profile_id": target, "limit": limit}
    if as_of is not None:
        kwargs["as_of"] = as_of
    # A remote caller never gets what it may not see (pictures, pages, folder files).
    from superlocalmemory.mcp.remote_visibility import current_view
    from superlocalmemory.retrieval import remote_view, visibility

    ctx = remote_view.context_for(
        current_view(),
        getattr(engine, "_db", None),
        target or getattr(engine, "profile_id", "default"),
    )
    if ctx is None:
        from superlocalmemory.core.engine_capabilities import CapabilityError

        try:
            return memories_from(engine.recall(query, **kwargs))
        except CapabilityError:
            # 4.2.3: an MCP stdio process holds a light engine that may not
            # recall in-process, so every prestage_context said recall_error.
            # Recall through the worker pool, as session_init does. A
            # point-in-time view is not forwarded there: refuse, never serve now.
            if as_of is not None:
                raise
            from superlocalmemory.core.answer_check_scope import context_load
            from superlocalmemory.mcp._pool_adapter import pool_recall

            with context_load():
                return memories_from(pool_recall(
                    query, limit=limit, **({"profile_id": target} if target else {})))
    with visibility.use(ctx):
        return memories_from(engine.recall(query, **kwargs))


__all__ = ["memories_from", "prestage_recall"]
