# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""MCP-side adapters onto the pool (daemon HTTP or local subprocess).

The pool returns plain dicts with an ``ok`` flag. Hooks
(``AutoRecall`` / ``AutoCapture``) expect a ``RecallResponse``-shaped
object and a list of fact ids. These adapters bridge the two.

On ``{"ok": False, "error": "..."}`` the adapters raise
:class:`PoolError` instead of silently returning empty results — worker
death must be distinguishable from "no memories" on the user side.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoolFact:
    fact_id: str = ""
    content: str = ""
    memory_id: str = ""
    created_at: str = ""


@dataclass(frozen=True)
class PoolRecallItem:
    fact: PoolFact = field(default_factory=PoolFact)
    score: float = 0.0
    confidence: float = 0.0
    relevance_score: float = 0.0
    ranking_score: float | None = None
    memory_confidence: float = 0.0
    match_confidence: float | None = None
    confidence_basis: str = "unmeasured"
    rank_position: int = 0
    trust_score: float = 0.0
    channel_scores: dict[str, float] = field(default_factory=dict)
    evidence_chain: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class PoolRecallResponse:
    results: list[PoolRecallItem] = field(default_factory=list)
    query_type: str = ""
    retrieval_time_ms: float = 0.0
    channel_weights: dict[str, float] = field(default_factory=dict)
    total_candidates: int = 0
    score_contract_version: str = "2"
    calibration_status: str = "uncalibrated"
    calibration_id: str | None = None
    answer_confidence: float | None = None
    abstained: bool = False
    abstention_reason: str | None = None
    #: Every field of ``recall_response_metadata`` the daemon returned (and
    #: the empty-recall default for any it did not), read-only. The typed
    #: fields above are kept for existing callers; this one is complete.
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


class PoolError(RuntimeError):
    """Raised when the pool returns an error envelope.

    Callers that want the old silent-empty behaviour (e.g. hook paths
    that must never break the user's session) catch this and log at
    WARNING. Callers that want to surface the failure (e.g. dashboard
    resources) let it propagate.
    """


def _pool():
    """Lazy pool factory — prefers the daemon HTTP proxy.

    Split out so tests can monkey-patch the factory without touching
    the real ``WorkerPool.shared()`` singleton.
    """
    from superlocalmemory.mcp._daemon_proxy import choose_pool
    return choose_pool()


def _unwrap_error(raw: Any, op: str) -> None:
    """Raise PoolError if the pool returned an ``{"ok": False}`` envelope."""
    if isinstance(raw, dict) and raw.get("ok") is False:
        reason = raw.get("error") or "pool returned ok=False"
        raise PoolError(f"pool.{op} failed: {reason}")


def pool_recall(query: str, limit: int = 10, **kwargs: Any) -> PoolRecallResponse:
    """Call pool.recall and reshape its dict into a typed response.

    Raises :class:`PoolError` on worker death or any non-ok envelope.
    """
    # v3.6.15 multi-scope: forward the scope-visibility flags when the caller
    # set them. ``None`` (the default) is passed through so the daemon/engine
    # resolves the configured default — shared memory is opt-in, so omitting
    # them keeps recall scoped to this profile only.
    _recall_kwargs: dict[str, Any] = {
        "query": query,
        "limit": limit,
        "session_id": str(kwargs.get("session_id") or ""),
        # v3.8.2 client-driven agentic: pass ``fast`` through unchanged. None
        # (unset) flows to the daemon/engine which resolves the configured
        # client-driven-agentic default; an explicit bool always wins.
        "fast": kwargs.get("fast", None),
    }
    if "include_global" in kwargs:
        _recall_kwargs["include_global"] = kwargs["include_global"]
    if "include_shared" in kwargs:
        _recall_kwargs["include_shared"] = kwargs["include_shared"]
    if kwargs.get("window"):
        _recall_kwargs["window"] = kwargs["window"]
    # W3-1: who asked, for the injection receipt (the proxy's default is "mcp").
    if kwargs.get("via"):
        _recall_kwargs["via"] = kwargs["via"]
    if kwargs.get("drop_channel"):  # W3-3: forwarded only when a treatment set it
        _recall_kwargs["drop_channel"] = kwargs["drop_channel"]
    # #150: a project filter / preference, forwarded only when set. A named
    # profile is per-request routing (the active profile is not moved).
    for _name in ("project", "prefer_project", "profile_id"):
        if (kwargs.get(_name) or "").strip():
            _recall_kwargs[_name] = kwargs[_name].strip()
    # 4.1.22: forwarded only when set, same as the other facets. A
    # string or a list are both accepted here and downstream (the daemon
    # proxy / HTTP route parse either the same way).
    _tags = kwargs.get("tags")
    if isinstance(_tags, str) and _tags.strip():
        _recall_kwargs["tags"] = _tags.strip()
    elif isinstance(_tags, (list, tuple)) and _tags:
        _recall_kwargs["tags"] = list(_tags)
    if "tags" in _recall_kwargs and (kwargs.get("tags_match") or "").strip():
        _recall_kwargs["tags_match"] = kwargs["tags_match"].strip()
    if kwargs.get("project_strict") and "project" in _recall_kwargs:
        _recall_kwargs["project_strict"] = True
    # S-M2: a context load (session start, auto-injection) is not a question.
    # Only an explicit False is forwarded, so every other call is unchanged;
    # a caller inside ``skip_answer_check()`` is covered by the proxy itself.
    if kwargs.get("answer_check") is False:
        _recall_kwargs["answer_check"] = False
    if not _recall_kwargs.get("profile_id"):
        from superlocalmemory.server.profile_runtime import current_request_profile

        pinned = current_request_profile()
        if pinned:
            _recall_kwargs["profile_id"] = pinned
    raw = _pool().recall(**_recall_kwargs)
    _unwrap_error(raw, "recall")
    items = raw.get("results", []) if isinstance(raw, dict) else []
    results = [
        PoolRecallItem(
            fact=PoolFact(
                fact_id=item.get("fact_id", ""),
                content=item.get("content", ""),
                memory_id=item.get("memory_id", ""),
                created_at=item.get("created_at", "") or "",
            ),
            score=float(item.get("score") or 0.0),
            confidence=float(item.get("confidence", 0.0)),
            relevance_score=float(item.get("relevance_score", item.get("score")) or 0.0),
            ranking_score=(
                float(item["ranking_score"])
                if item.get("ranking_score") is not None else None
            ),
            memory_confidence=float(
                item.get("memory_confidence", item.get("confidence", 0.0))
            ),
            match_confidence=item.get("match_confidence"),
            confidence_basis=item.get("confidence_basis", "unmeasured"),
            rank_position=int(item.get("rank_position", 0)),
            trust_score=float(item.get("trust_score", 0.0)),
            channel_scores=item.get("channel_scores", {}) or {},
            evidence_chain=list(item.get("evidence_chain", []) or []),
        )
        for item in items
    ]
    return PoolRecallResponse(
        results=results,
        query_type=raw.get("query_type", "") if isinstance(raw, dict) else "",
        retrieval_time_ms=float(raw.get("retrieval_time_ms", 0.0))
        if isinstance(raw, dict) else 0.0,
        channel_weights=raw.get("channel_weights", {})
        if isinstance(raw, dict) else {},
        total_candidates=int(raw.get("total_candidates", 0))
        if isinstance(raw, dict) else 0,
        score_contract_version=str(raw.get("score_contract_version", "2"))
        if isinstance(raw, dict) else "2",
        calibration_status=str(raw.get("calibration_status", "uncalibrated"))
        if isinstance(raw, dict) else "uncalibrated",
        calibration_id=raw.get("calibration_id") if isinstance(raw, dict) else None,
        answer_confidence=raw.get("answer_confidence") if isinstance(raw, dict) else None,
        abstained=bool(raw.get("abstained", False)) if isinstance(raw, dict) else False,
        abstention_reason=raw.get("abstention_reason") if isinstance(raw, dict) else None,
        metadata=MappingProxyType(_forward_metadata(raw)),
    )


def _forward_metadata(raw: Any) -> dict[str, Any]:
    from superlocalmemory.mcp._recall_metadata import forward_recall_metadata

    return forward_recall_metadata(raw if isinstance(raw, dict) else {})


def pool_store(content: str, metadata: dict | None = None) -> list[str]:
    """Call pool.store and return fact id list (or pending tracker).

    v3.4.32: the daemon /remember endpoint is async by default — it
    returns ``pending_id`` and queues the write. We surface this to
    callers as ``["pending:<id>"]`` so they have a stable identifier
    without blocking the remember on the embedder worker.

    Legacy synchronous path (``?wait=true``) still returns real
    ``fact_ids``. Worker death raises :class:`PoolError`.
    """
    raw = _pool().store(content=content, metadata=metadata or {})
    _unwrap_error(raw, "store")
    if not isinstance(raw, dict):
        return []
    fact_ids = raw.get("fact_ids")
    if fact_ids:
        return list(fact_ids)
    pending_id = raw.get("pending_id")
    if pending_id is not None:
        return [f"pending:{pending_id}"]
    return []
