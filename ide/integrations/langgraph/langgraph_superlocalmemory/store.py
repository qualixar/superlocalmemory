#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V3 - LangGraph BaseStore backend.

Implements LangGraph's ``BaseStore`` (long-term memory store) backed by a local
SuperLocalMemory data root. Data stays in your local SLM data root; optional
SLM providers, connectors, backup, and downloads have separate network behavior.

``BaseStore`` declares only two abstract methods, ``batch`` and ``abatch``;
every convenience method (``get``/``put``/``search``/``delete``/
``list_namespaces`` and their async variants) is inherited and dispatches
through them. This adapter therefore implements just those two, translating
each op onto the framework-free :class:`V3KVStore`.

Usage::

    from langgraph_superlocalmemory import SuperLocalMemoryStore

    store = SuperLocalMemoryStore()
    store.put(("users", "1"), "profile", {"name": "Ada"})
    item = store.get(("users", "1"), "profile")
    hits = store.search(("users",), filter={"name": "Ada"})
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)


def _data_root() -> Path:
    value = (
        os.environ.get("SLM_DATA_DIR")
        or os.environ.get("SL_MEMORY_PATH")
        or os.environ.get("SLM_HOME")
    )
    return Path(value).expanduser() if value else Path.home() / ".superlocalmemory"


def _parse_dt(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


class SuperLocalMemoryStore(BaseStore):
    """LangGraph long-term memory store backed by SuperLocalMemory V3.

    Parameters
    ----------
    db_path : str or None
        Path to the SLM SQLite database. Defaults to
        ``~/.superlocalmemory/memory.db``.
    """

    supports_ttl = False

    def __init__(self, db_path: Optional[str] = None) -> None:
        from langgraph_superlocalmemory._v3_kv_store import V3KVStore

        store_path = Path(db_path) if db_path else _data_root() / "memory.db"
        self._store = V3KVStore(store_path)

    # -- the two abstract methods -----------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                env = self._store.get(op.namespace, op.key)
                results.append(self._to_item(env) if env else None)
            elif isinstance(op, PutOp):
                if op.value is None:
                    self._store.delete(op.namespace, op.key)
                else:
                    self._store.put(op.namespace, op.key, op.value)
                results.append(None)
            elif isinstance(op, SearchOp):
                envs = self._store.search(
                    op.namespace_prefix,
                    filter=op.filter,
                    limit=op.limit,
                    offset=op.offset,
                )
                results.append([self._to_search_item(e) for e in envs])
            elif isinstance(op, ListNamespacesOp):
                results.append(self._handle_list_namespaces(op))
            else:  # pragma: no cover - unknown op kind
                results.append(None)
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        # The engine is synchronous; run it off the event loop so async
        # graphs are never blocked.
        return await asyncio.to_thread(self.batch, list(ops))

    def close(self) -> None:
        self._store.close()

    # -- helpers -----------------------------------------------------------

    def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        prefix: tuple[str, ...] | None = None
        suffix: tuple[str, ...] | None = None
        for condition in op.match_conditions or ():
            # Literal paths only; wildcard ("*") segments are treated literally
            # and will simply not match — documented as a known limitation.
            path = tuple(str(p) for p in condition.path)
            if condition.match_type == "prefix":
                prefix = path
            elif condition.match_type == "suffix":
                suffix = path
        return self._store.list_namespaces(
            prefix=prefix,
            suffix=suffix,
            max_depth=op.max_depth,
            limit=op.limit,
            offset=op.offset,
        )

    def _to_item(self, env: dict) -> Item:
        return Item(
            value=env["value"],
            key=env["key"],
            namespace=tuple(env["namespace"]),
            created_at=_parse_dt(env["created_at"]),
            updated_at=_parse_dt(env["updated_at"]),
        )

    def _to_search_item(self, env: dict) -> SearchItem:
        return SearchItem(
            namespace=tuple(env["namespace"]),
            key=env["key"],
            value=env["value"],
            created_at=_parse_dt(env["created_at"]),
            updated_at=_parse_dt(env["updated_at"]),
            score=None,
        )
