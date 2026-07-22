"""Framework-free key-value store over SuperLocalMemory's V3 engine.

LangGraph's ``BaseStore`` is a namespaced key-value store: each item is keyed
by a ``(namespace tuple, key)`` pair and carries a JSON ``value`` plus
``created_at`` / ``updated_at`` timestamps. This module implements that data
model on top of ``MemoryEngine`` with no LangGraph import, so the storage
contract can be exercised and verified on its own.

Storage mapping
---------------
Each item is one SLM memory whose ``session_id`` deterministically encodes the
namespace and key, and whose content is a JSON envelope::

    {"namespace": [...], "key": "...", "value": {...},
     "created_at": "<iso>", "updated_at": "<iso>"}

``created_at`` is preserved across updates; ``updated_at`` is refreshed. Prefix
search and namespace listing read the envelopes back and filter on the real
namespace tuple (element-wise), so ``("users",)`` matches ``("users", "1")``
but never ``("users2",)``.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_CONTENT_BYTES = 1_000_000
_SID_PREFIX = "lg:"
_NS_SEP = "\x1f"   # unit separator between namespace elements
_KEY_SEP = "\x1e"  # record separator between the namespace path and the key


def _load_v3_types():
    try:
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.engine import MemoryEngine
        from superlocalmemory.storage.models import Mode
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(
            "SuperLocalMemory V3 is not installed in this Python environment. "
            "Install it with: python -m pip install superlocalmemory."
        ) from exc
    return SLMConfig, MemoryEngine, Mode


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ns_path(namespace: tuple[str, ...]) -> str:
    return _NS_SEP.join(namespace)


def _session_id(namespace: tuple[str, ...], key: str) -> str:
    return f"{_SID_PREFIX}{_ns_path(namespace)}{_KEY_SEP}{key}"


def _is_prefix(prefix: tuple[str, ...], namespace: tuple[str, ...]) -> bool:
    """True when *namespace* begins with *prefix*, compared element-wise."""
    return len(namespace) >= len(prefix) and namespace[: len(prefix)] == prefix


class V3KVStore:
    """Namespaced key-value storage backed by a SuperLocalMemory engine."""

    def __init__(self, db_path: str | Path) -> None:
        SLMConfig, MemoryEngine, Mode = _load_v3_types()
        path = Path(db_path).expanduser().resolve()
        config = SLMConfig.for_mode(Mode.A, base_dir=path.parent)
        config.db_path = path
        config.forgetting = replace(config.forgetting, enabled=False)
        config.retrieval.use_cross_encoder = False
        self._engine = MemoryEngine(config)
        self._engine.initialize()

    # -- writes ------------------------------------------------------------

    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        now = _now_iso()
        existing = self.get(namespace, key)
        created = existing["created_at"] if existing else now
        envelope = {
            "namespace": list(namespace),
            "key": key,
            "value": value,
            "created_at": created,
            "updated_at": now,
        }
        content = json.dumps(envelope, ensure_ascii=False)
        if len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
            raise ValueError(
                f"value exceeds maximum size of {MAX_CONTENT_BYTES} bytes"
            )
        self._delete_session(_session_id(namespace, key))
        self._engine.store(
            content,
            session_id=_session_id(namespace, key),
            metadata={
                "integration": "langgraph",
                "lg_namespace": _ns_path(namespace),
                "lg_key": key,
                "tags": ["langgraph"],
                "importance": 3,
                "project_name": "langgraph",
            },
        )

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        self._delete_session(_session_id(namespace, key))

    # -- reads -------------------------------------------------------------

    def get(self, namespace: tuple[str, ...], key: str) -> dict | None:
        rows = self._engine.db.execute(
            "SELECT content FROM memories WHERE profile_id=? AND session_id=? "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (self._engine.profile_id, _session_id(namespace, key)),
        )
        for row in rows:
            return self._parse(dict(row).get("content", ""))
        return None

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict]:
        matched = [
            env
            for env in self._all_envelopes()
            if _is_prefix(namespace_prefix, tuple(env["namespace"]))
            and _matches_filter(env["value"], filter)
        ]
        matched.sort(key=lambda e: (e.get("updated_at", ""), e.get("key", "")))
        return matched[offset : offset + limit]

    def list_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        seen: list[tuple[str, ...]] = []
        for env in self._all_envelopes():
            ns = tuple(env["namespace"])
            if prefix is not None and not _is_prefix(prefix, ns):
                continue
            if suffix is not None and ns[-len(suffix):] != suffix:
                continue
            if max_depth is not None:
                ns = ns[:max_depth]
            if ns not in seen:
                seen.append(ns)
        seen.sort()
        return seen[offset : offset + limit]

    def close(self) -> None:
        self._engine.close()

    # -- internals ---------------------------------------------------------

    def _all_envelopes(self) -> list[dict]:
        escaped = _SID_PREFIX.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self._engine.db.execute(
            "SELECT content FROM memories WHERE profile_id=? AND session_id LIKE ? ESCAPE '\\' "
            "ORDER BY created_at ASC, rowid ASC",
            (self._engine.profile_id, escaped + "%"),
        )
        out = []
        for row in rows:
            env = self._parse(dict(row).get("content", ""))
            if env is not None:
                out.append(env)
        return out

    def _delete_session(self, session_id: str) -> None:
        facts = self._engine.db.execute(
            "SELECT af.fact_id FROM atomic_facts af JOIN memories m "
            "ON af.memory_id = m.memory_id "
            "WHERE m.session_id=? AND m.profile_id=?",
            (session_id, self._engine.profile_id),
        )
        fact_ids = [str(dict(row)["fact_id"]) for row in facts]
        with self._engine.db.transaction():
            self._engine.db.execute(
                "DELETE FROM memories WHERE session_id=? AND profile_id=?",
                (session_id, self._engine.profile_id),
            )
        self._remove_external_indexes(fact_ids)

    @staticmethod
    def _parse(content: str) -> dict | None:
        try:
            data = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or "namespace" not in data or "key" not in data:
            return None
        data.setdefault("value", {})
        data.setdefault("created_at", "")
        data.setdefault("updated_at", data["created_at"])
        return data

    def _remove_external_indexes(self, fact_ids: list[str]) -> None:
        ann = getattr(self._engine, "_ann_index", None)
        vector = getattr(self._engine, "_vector_store", None)
        for fact_id in fact_ids:
            if ann is not None:
                try:
                    ann.remove(fact_id)
                except Exception:
                    pass
            if vector is not None and getattr(vector, "available", False):
                try:
                    vector.delete(fact_id)
                except Exception:
                    pass


def _matches_filter(value: dict[str, Any], filter: dict[str, Any] | None) -> bool:
    if not filter:
        return True
    if not isinstance(value, dict):
        return False
    return all(value.get(k) == v for k, v in filter.items())
