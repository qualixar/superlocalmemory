#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V3 - OpenAI Agents SDK SessionABC backend.

Implements the OpenAI Agents SDK ``SessionABC`` (conversation history store)
backed by a local SuperLocalMemory data root.  Records stay in your local SLM
data root (optional SLM providers, connectors, backup, and downloads have
separate network behavior) and are visible through every other SLM surface
(CLI, MCP, dashboard).

``SessionABC`` is a per-session conversation-history protocol; each
``SLMSession`` instance manages the ordered list of
``TResponseInputItem`` dicts for one ``session_id``.

Usage::

    from openai_agents_superlocalmemory import SLMSession

    session = SLMSession(session_id="user-42-conv-7")
    runner = Runner(session=session)
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Optional

from agents.memory import SessionABC
from agents.items import TResponseInputItem


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


class SLMSession(SessionABC):
    """OpenAI Agents SDK session backed by SuperLocalMemory V3.

    Parameters
    ----------
    session_id : str
        A unique identifier for this conversation session.
    db_path : str or None
        Path to the SLM SQLite database.  Defaults to
        ``~/.superlocalmemory/memory.db``.
    session_settings : dict or None
        Optional metadata dict stored on the instance and returned by the
        ``session_settings`` attribute required by ``SessionABC``.
    """

    def __init__(
        self,
        session_id: str,
        db_path: Optional[str] = None,
        session_settings: Optional[dict[str, Any]] = None,
    ) -> None:
        from openai_agents_superlocalmemory._v3_session_store import V3SessionStore

        self._session_id = session_id
        self._session_settings: dict[str, Any] = session_settings or {}
        store_path = Path(db_path) if db_path else _data_root() / "memory.db"
        self._store = V3SessionStore(store_path)

    # -- SessionABC required attributes ------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def session_settings(self) -> dict[str, Any]:
        return self._session_settings

    # -- SessionABC abstract methods ----------------------------------------

    async def get_items(
        self, limit: Optional[int] = None
    ) -> list[TResponseInputItem]:
        """Return stored items oldest-first, optionally the last *limit*."""
        items = await asyncio.to_thread(
            self._store.get_items, self._session_id, limit
        )
        # Items are stored and retrieved as plain dicts; TResponseInputItem is
        # a TypedDict alias so the cast is type-safe at runtime.
        return items  # type: ignore[return-value]

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """Append *items* to the end of this session's conversation history."""
        # Convert TypedDicts to plain dicts for the framework-free store core.
        raw: list[dict[str, Any]] = [dict(item) for item in items]
        await asyncio.to_thread(self._store.append_items, self._session_id, raw)

    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return the most-recent item, or ``None`` if empty."""
        item = await asyncio.to_thread(self._store.pop_item, self._session_id)
        return item  # type: ignore[return-value]

    async def clear_session(self) -> None:
        """Permanently remove all items in this session."""
        await asyncio.to_thread(self._store.clear_session, self._session_id)

    def close(self) -> None:
        self._store.close()
