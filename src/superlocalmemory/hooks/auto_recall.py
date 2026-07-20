# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Auto-recall — inject relevant memories into AI context automatically."""

from __future__ import annotations

import logging
from typing import Any, Callable

from superlocalmemory.core.injection import InjectableMemory, render_context

logger = logging.getLogger(__name__)


class AutoRecall:
    """Automatically recalls relevant context for AI sessions.

    Called at session start or before each prompt to inject
    relevant memories without user intervention.

    Two ways to wire the recall side:

    - Pass ``engine=<MemoryEngine>`` (CLI/daemon path — historical shape).
    - Pass ``recall_fn=<callable>`` (MCP/LIGHT path — the callable adapts
      a worker-pool call to the ``RecallResponse``-shaped return value).
      When both are supplied, ``recall_fn`` wins.
    """

    def __init__(
        self,
        engine=None,
        config: dict | None = None,
        *,
        recall_fn: Callable[..., Any] | None = None,
    ):
        self._engine = engine
        self._recall_fn = recall_fn
        self._config = config or {}
        self._enabled = self._config.get("enabled", True)
        self._max_memories = self._config.get("max_memories_injected", 10)
        self._threshold = self._config.get("relevance_threshold", 0.3)

    def _recall(self, query: str, limit: int):
        if self._recall_fn is not None:
            return self._recall_fn(query, limit=limit)
        if self._engine is not None:
            return self._engine.recall(query, limit=limit)
        return None

    def get_session_context(self, project_path: str = "", query: str = "") -> str:
        """Get relevant context for a session or query.

        Returns a formatted string of relevant memories suitable
        for injection into an AI's system prompt.
        """
        if not self._enabled:
            return ""
        if self._recall_fn is None and self._engine is None:
            return ""

        try:
            # Build query from project path or explicit query
            search_query = query or f"project context {project_path}"
            response = self._recall(search_query, self._max_memories)

            if response is None or not response.results:
                return ""

            # Filter by relevance threshold
            relevant = [r for r in response.results if r.score >= self._threshold]

            if not relevant:
                return ""

            memories = [
                InjectableMemory(
                    content=r.fact.content,
                    score=float(r.score),
                    fact_id=str(r.fact.fact_id),
                    importance=float(getattr(r.fact, "importance", 0.0) or 0.0),
                    access_count=int(getattr(r.fact, "access_count", 0) or 0),
                    source_type="recall",
                )
                for r in relevant[:self._max_memories]
            ]
            return render_context(memories, mode="B", cfg=None, wrap=True)
        except Exception as exc:
            logger.warning("Auto-recall failed: %s", exc)
            return ""

    def get_query_context(self, query: str) -> list[dict]:
        """Get relevant memories for a specific query.

        Returns structured data (not formatted string) for MCP tools.
        """
        if not self._enabled:
            return []
        if self._recall_fn is None and self._engine is None:
            return []

        try:
            response = self._recall(query, self._max_memories)
            if response is None:
                return []
            results = []
            for r in response.results:
                if r.score >= self._threshold:
                    results.append({
                        "fact_id": r.fact.fact_id,
                        "content": r.fact.content[:300],
                        "score": round(r.score, 3),
                        "relevance_score": round(
                            getattr(r, "relevance_score", r.score) or 0.0, 3
                        ),
                        "ranking_score": getattr(r, "ranking_score", None),
                        "confidence": round(
                            getattr(
                                r, "memory_confidence",
                                getattr(r, "confidence", 0.0),
                            ) or 0.0, 3
                        ),
                        "memory_confidence": round(
                            getattr(
                                r, "memory_confidence",
                                getattr(r, "confidence", 0.0),
                            ) or 0.0, 3
                        ),
                        "rank_position": int(getattr(r, "rank_position", 0) or 0),
                    })
            return results
        except Exception as exc:
            logger.warning("Auto-recall query failed: %s", exc)
            return []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False
