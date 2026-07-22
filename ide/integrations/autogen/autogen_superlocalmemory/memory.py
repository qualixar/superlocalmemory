#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V3 — AutoGen Memory implementation.

Implements AutoGen's ``Memory`` ABC (``autogen-core``) backed by a local
SuperLocalMemory data root. Records stay in your local SLM data root; optional
SLM providers, connectors, backup, and downloads have separate network
behavior.

Note on AutoGen maintenance status
-----------------------------------
The ``autogen-agentchat`` package is in active maintenance from Microsoft and
the AutoGen community. If you are using Microsoft Agent Framework
(``agent-framework-core``), the shipped
``agent_framework_superlocalmemory`` adapter implements the
``HistoryProvider`` / ``ContextProvider`` protocol and may be a better fit.

Usage::

    from autogen_superlocalmemory import SuperLocalMemoryMemory

    mem = SuperLocalMemoryMemory()

    # In your AutoGen agent:
    # agent = AssistantAgent("assistant", memory=[mem], ...)
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

from autogen_core.memory import (
    Memory,
    MemoryContent,
    MemoryMimeType,
    MemoryQueryResult,
    UpdateContextResult,
)

logger = logging.getLogger("slm.adapters.autogen")


# ``SLM_INSTALL_DIR`` is legacy installer metadata only. It is deliberately
# never added to ``sys.path``; install the V3 runtime with
# ``python -m pip install superlocalmemory``.
def _data_root() -> Path:
    value = (
        os.environ.get("SLM_DATA_DIR")
        or os.environ.get("SL_MEMORY_PATH")
        or os.environ.get("SLM_HOME")
    )
    return Path(value).expanduser() if value else Path.home() / ".superlocalmemory"


def _extract_text(query: "str | MemoryContent") -> str:
    """Extract a query string from str or MemoryContent."""
    if isinstance(query, str):
        return query
    content = getattr(query, "content", None)
    if content is not None:
        return str(content)
    return ""


def _dict_to_memory_content(data: dict[str, Any]) -> MemoryContent:
    """Reconstruct a MemoryContent from a plain dict."""
    mime_str = data.get("mime_type", "text/plain")
    # Map common mime strings to MemoryMimeType enum values defensively.
    if mime_str in ("text/plain", "TEXT", "text"):
        mime = MemoryMimeType.TEXT
    else:
        try:
            mime = MemoryMimeType(mime_str)
        except ValueError:
            mime = MemoryMimeType.TEXT
    meta = data.get("metadata") or {}
    return MemoryContent(
        content=str(data.get("content", "")),
        mime_type=mime,
        metadata=meta if isinstance(meta, dict) else {},
    )


class SuperLocalMemoryMemory(Memory):
    """AutoGen Memory backed by SuperLocalMemory V3.

    Parameters
    ----------
    db_path : str or None
        Path to the SLM SQLite database. Defaults to
        ``~/.superlocalmemory/memory.db``.
    max_recall : int
        Maximum number of memories to inject via ``update_context``.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        max_recall: int = 10,
    ) -> None:
        from autogen_superlocalmemory._v3_autogen_store import V3AutogenStore

        store_path = Path(db_path) if db_path else _data_root() / "memory.db"
        self._store = V3AutogenStore(store_path)
        self._max_recall = max_recall

    # -- Memory ABC --------------------------------------------------------

    async def add(
        self,
        content: MemoryContent,
        cancellation_token: Any = None,
    ) -> None:
        """Persist a MemoryContent item."""
        content_str = str(getattr(content, "content", "") or "")
        mime = getattr(content, "mime_type", MemoryMimeType.TEXT)
        mime_str = getattr(mime, "value", str(mime))
        meta = dict(getattr(content, "metadata", None) or {})
        await asyncio.to_thread(
            self._store.add,
            content_str,
            mime_type=mime_str,
            metadata=meta,
        )

    async def query(
        self,
        query: "str | MemoryContent",
        cancellation_token: Any = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Semantic recall via SLM's text-search pipeline."""
        query_text = _extract_text(query)
        limit = int(kwargs.get("limit", self._max_recall))
        raw = await asyncio.to_thread(
            self._store.query_text, query_text, limit=limit
        )
        results = [_dict_to_memory_content(d) for d in raw]
        return MemoryQueryResult(results=results)

    async def update_context(self, model_context: Any) -> UpdateContextResult:
        """Inject recent memories as a SystemMessage into ``model_context``.

        Mirrors AutoGen's ``ListMemory.update_context``: retrieves the most
        recent memories, formats them as a system-level summary, and appends
        a ``SystemMessage`` to the context so the LLM is aware of prior state.
        """
        recent = await asyncio.to_thread(
            self._store.get_recent, limit=self._max_recall
        )
        if recent:
            lines = [f"- {r['content']}" for r in recent]
            summary = "Relevant memory from SuperLocalMemory:\n" + "\n".join(lines)
            try:
                from autogen_core.models import SystemMessage
                await model_context.add_message(SystemMessage(content=summary))
            except Exception as exc:
                # Context API varies across AutoGen versions; surface the
                # degradation rather than silently dropping memory injection.
                logger.warning("update_context injection skipped: %s", exc)

        # Build MemoryQueryResult from the recent memories.
        memory_contents = [_dict_to_memory_content(d) for d in recent]
        recall_result = MemoryQueryResult(results=memory_contents)
        return UpdateContextResult(memories=recall_result)

    async def clear(self) -> None:
        """Remove all memories added via this adapter."""
        await asyncio.to_thread(self._store.clear)

    async def close(self) -> None:
        """Close the underlying store."""
        await asyncio.to_thread(self._store.close)
