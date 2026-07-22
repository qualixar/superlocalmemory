#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V3 - Google ADK BaseMemoryService backend.

Implements Google ADK's ``BaseMemoryService`` backed by a local SuperLocalMemory
data root.  Records stay in your local SLM data root (optional SLM providers,
connectors, backup, and downloads have separate network behavior) and are
visible through every other SLM surface (CLI, MCP, dashboard).

``BaseMemoryService`` declares two abstract async methods:

* ``add_session_to_memory(session)`` — persist all events in a session.
* ``search_memory(*, app_name, user_id, query)`` — semantic search.

This adapter maps both onto the framework-free :class:`V3ADKStore`.

Session fields are read defensively with ``getattr`` because ``Session``,
``Event``, and ``Content`` are pydantic models defined in google-adk that
cannot be imported in the framework-free store core.

Usage::

    from google_adk_superlocalmemory import SuperLocalMemoryService

    service = SuperLocalMemoryService()
    runner = Runner(agent=agent, app_name="my-app", memory_service=service)
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from google.adk.memory.base_memory_service import BaseMemoryService, SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry


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


def _extract_events(session: Any) -> tuple[str, str, str, list[dict[str, Any]]]:
    """Extract (app_name, user_id, session_id, event_dicts) from an ADK Session.

    Uses only ``getattr`` — never direct attribute access — so this function
    remains safe when called with any pydantic model version.
    """
    app_name = str(getattr(session, "app_name", "") or "")
    user_id = str(getattr(session, "user_id", "") or "")
    session_id = str(
        getattr(session, "id", None)
        or getattr(session, "session_id", None)
        or ""
    )
    raw_events = getattr(session, "events", None) or []
    event_dicts: list[dict[str, Any]] = []

    for event in raw_events:
        # Extract text from content.parts[].text
        content = getattr(event, "content", None)
        parts_text: list[str] = []
        if content is not None:
            raw_parts = getattr(content, "parts", None) or []
            for part in raw_parts:
                text = getattr(part, "text", None)
                if text:
                    parts_text.append(str(text))

        author = str(getattr(event, "author", None) or "unknown")

        # Timestamp may be float (Unix epoch) or datetime or absent.
        raw_ts = getattr(event, "timestamp", None)
        if isinstance(raw_ts, (int, float)):
            ts_float = float(raw_ts)
            ts_iso = datetime.fromtimestamp(ts_float, tz=timezone.utc).isoformat()
        elif isinstance(raw_ts, datetime):
            ts_float = raw_ts.timestamp()
            ts_iso = raw_ts.isoformat()
        else:
            ts_float = 0.0
            ts_iso = datetime.now(timezone.utc).isoformat()

        event_dicts.append(
            {
                "text": "\n".join(parts_text),
                "author": author,
                "timestamp_float": ts_float,
                "timestamp_iso": ts_iso,
            }
        )

    return app_name, user_id, session_id, event_dicts


def _build_memory_entry(envelope: dict[str, Any]) -> MemoryEntry:
    """Convert a stored envelope dict into a google.adk ``MemoryEntry``.

    Tries to construct a ``Content`` object with typed ``Part``s when the
    google-genai types are importable; falls back to a plain string otherwise
    so the wrapper degrades gracefully across ADK minor versions.
    """
    evt = envelope.get("event", {})
    text = evt.get("text", "")
    author = evt.get("author", "unknown")
    ts_float = float(evt.get("timestamp_float", 0.0))
    # ADK's MemoryEntry types timestamp as `datetime | None` — pass a datetime,
    # not the raw epoch float (pydantic will not coerce a bare float).
    from datetime import datetime as _dt, timezone as _tz
    ts = _dt.fromtimestamp(ts_float, tz=_tz.utc) if ts_float and ts_float > 0 else None

    try:
        from google.genai import types as genai_types

        content = genai_types.Content(
            role=author,
            parts=[genai_types.Part(text=text)],
        )
    except Exception:  # ADK version difference or import failure
        content = text  # type: ignore[assignment]  # graceful fallback

    try:
        return MemoryEntry(content=content, author=author, timestamp=ts)
    except TypeError:
        # Defensive: an ADK minor version that doesn't accept `timestamp` as a
        # constructor kwarg still yields a usable entry.
        return MemoryEntry(content=content, author=author)


class SuperLocalMemoryService(BaseMemoryService):
    """Google ADK memory service backed by SuperLocalMemory V3.

    Parameters
    ----------
    db_path : str or None
        Path to the SLM SQLite database. Defaults to
        ``~/.superlocalmemory/memory.db``.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        from google_adk_superlocalmemory._v3_adk_store import V3ADKStore

        store_path = Path(db_path) if db_path else _data_root() / "memory.db"
        self._store = V3ADKStore(store_path)

    # -- abstract method implementations ------------------------------------

    async def add_session_to_memory(self, session: Any) -> None:  # type: ignore[override]
        """Persist all events in *session* to the local SLM data root.

        Calling this method again for the same session replaces any previously
        stored events for that session (idempotent upsert).
        """
        app_name, user_id, session_id, event_dicts = _extract_events(session)
        await asyncio.to_thread(
            self._store.add_events,
            app_name,
            user_id,
            session_id,
            event_dicts,
        )

    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        """Search stored memories for *query* scoped to *app_name* / *user_id*.

        Uses SuperLocalMemory's semantic recall pipeline for ranking and
        post-filters results to the target app+user namespace.  Returns an
        empty ``SearchMemoryResponse`` when no relevant memories are found.
        """
        envelopes = await asyncio.to_thread(
            self._store.search,
            app_name,
            user_id,
            query,
            10,
        )
        memories = [_build_memory_entry(env) for env in envelopes]
        return SearchMemoryResponse(memories=memories)

    def close(self) -> None:
        self._store.close()
