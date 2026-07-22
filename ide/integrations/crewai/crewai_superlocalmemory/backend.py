#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V3 — CrewAI StorageBackend.

Implements the CrewAI v1.x ``StorageBackend`` ``@runtime_checkable`` Protocol
backed by a local SuperLocalMemory data root. Records stay in your local SLM
data root (optional SLM providers, connectors, backup, and downloads have
separate network behavior) and are visible through every other SLM surface
(CLI, MCP, dashboard).

CrewAI owns the embedder and supplies pre-computed embeddings to ``save`` and
``search``. This adapter persists them and ranks by cosine similarity in pure
Python (no numpy dependency).

Usage::

    from crewai_superlocalmemory import SuperLocalMemoryBackend

    backend = SuperLocalMemoryBackend()
    # CrewAI sets this on agent/task/crew memory configuration.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from crewai.memory.storage.backend import StorageBackend
from crewai.memory.types import MemoryRecord, ScopeInfo


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


def _parse_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return None


def _record_to_dict(record: MemoryRecord) -> dict[str, Any]:
    """Convert a CrewAI MemoryRecord to the plain dict the core uses."""
    emb = getattr(record, "embedding", None)
    return {
        "id": str(record.id),
        "content": str(getattr(record, "content", "") or ""),
        "embedding": list(emb) if emb is not None else [],
        "scope": str(getattr(record, "scope", "/") or "/"),
        "category": str(getattr(record, "category", "general") or "general"),
        "metadata": dict(getattr(record, "metadata", None) or {}),
    }


def _dict_to_record(data: dict[str, Any]) -> MemoryRecord:
    """Reconstruct a MemoryRecord from a plain dict."""
    kwargs: dict[str, Any] = {
        "id": data["id"],
        "content": data.get("content", ""),
        "scope": data.get("scope", "/"),
        "category": data.get("category", "general"),
        "metadata": data.get("metadata", {}),
    }
    emb = data.get("embedding")
    if emb is not None:
        kwargs["embedding"] = list(emb)
    created = _parse_dt(data.get("created_at"))
    updated = _parse_dt(data.get("updated_at"))
    if created is not None:
        kwargs["created_at"] = created
    if updated is not None:
        kwargs["updated_at"] = updated
    return MemoryRecord(**kwargs)


def _dict_to_scope_info(data: dict[str, Any]) -> ScopeInfo:
    """Reconstruct a ScopeInfo from the core's aggregate dict."""
    kwargs: dict[str, Any] = {
        "scope": data["scope"],
        "record_count": data.get("record_count", 0),
        "categories": data.get("categories", {}),
    }
    created = _parse_dt(data.get("created_at"))
    updated = _parse_dt(data.get("updated_at"))
    if created is not None:
        kwargs["created_at"] = created
    if updated is not None:
        kwargs["updated_at"] = updated
    try:
        return ScopeInfo(**kwargs)
    except TypeError:
        # Defensive: if ScopeInfo constructor differs across CrewAI versions,
        # try positional construction with required fields only.
        return ScopeInfo(scope=data["scope"])  # type: ignore[call-arg]


class SuperLocalMemoryBackend(StorageBackend):
    """CrewAI StorageBackend backed by SuperLocalMemory V3.

    Parameters
    ----------
    db_path : str or None
        Path to the SLM SQLite database. Defaults to
        ``~/.superlocalmemory/memory.db``.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        from crewai_superlocalmemory._v3_crewai_store import V3CrewAIStore

        store_path = Path(db_path) if db_path else _data_root() / "memory.db"
        self._store = V3CrewAIStore(store_path)

    # -- sync Protocol methods ---------------------------------------------

    def save(self, records: list[MemoryRecord]) -> None:
        for record in records:
            self._store.save(_record_to_dict(record))

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: Optional[str] = None,
        categories: Optional[list[str]] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        raw = self._store.search(
            query_embedding,
            scope_prefix=scope_prefix,
            categories=categories,
            metadata_filter=metadata_filter,
            limit=limit,
            min_score=min_score,
        )
        return [(_dict_to_record(d), score) for d, score in raw]

    def delete(
        self,
        record_id: Optional[str] = None,
        scope_prefix: Optional[str] = None,
    ) -> int:
        if record_id is not None:
            return self._store.delete_by_id(record_id)
        if scope_prefix is not None:
            return self._store.delete_by_scope(scope_prefix)
        return 0

    def update(self, record: MemoryRecord) -> None:
        self._store.update(_record_to_dict(record))

    def get_record(self, record_id: str) -> Optional[MemoryRecord]:
        data = self._store.get_record(record_id)
        return _dict_to_record(data) if data is not None else None

    def list_records(
        self,
        scope_prefix: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        raw = self._store.list_records(scope_prefix=scope_prefix, limit=limit, offset=offset)
        return [_dict_to_record(d) for d in raw]

    def get_scope_info(self, scope: str) -> ScopeInfo:
        data = self._store.get_scope_info(scope)
        return _dict_to_scope_info(data)

    def list_scopes(self, parent: str = "/") -> list[str]:
        return self._store.list_scopes(parent=parent)

    def list_categories(
        self, scope_prefix: Optional[str] = None
    ) -> dict[str, int]:
        return self._store.list_categories(scope_prefix=scope_prefix)

    def count(self, scope_prefix: Optional[str] = None) -> int:
        return self._store.count(scope_prefix=scope_prefix)

    def reset(self, scope_prefix: Optional[str] = None) -> None:
        self._store.reset(scope_prefix=scope_prefix)

    # -- async variants (delegate to sync via thread) ----------------------

    async def asave(self, records: list[MemoryRecord]) -> None:
        await asyncio.to_thread(self.save, records)

    async def asearch(
        self,
        query_embedding: list[float],
        scope_prefix: Optional[str] = None,
        categories: Optional[list[str]] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        return await asyncio.to_thread(
            self.search,
            query_embedding,
            scope_prefix,
            categories,
            metadata_filter,
            limit,
            min_score,
        )

    async def adelete(
        self,
        record_id: Optional[str] = None,
        scope_prefix: Optional[str] = None,
    ) -> int:
        return await asyncio.to_thread(self.delete, record_id, scope_prefix)

    def close(self) -> None:
        self._store.close()
