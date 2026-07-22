"""Framework-free AutoGen memory store over SuperLocalMemory's V3 engine.

AutoGen's Memory ABC is an async interface for conversational agents. Each
``add()`` call persists a MemoryContent; ``query()`` uses SLM's text-recall
pipeline to retrieve semantically relevant memories; ``clear()`` removes all
memories added via this adapter.

Storage mapping
---------------
Each MemoryContent is one SLM memory:
  session_id  = "autogen-mem:{uuid4()}"   — unique per add() call
  content     = JSON envelope with adapter/timestamp fields so even a short
                content string persists a queryable SLM row

``get_recent()`` performs a direct SQL scan of all ``autogen-mem:`` rows,
bounded by ``_MAX_SCAN``. ``query_text()`` calls ``engine.recall()`` with
``fast=True`` for low-latency semantic retrieval.

``clear()`` deletes only ``autogen-mem:`` rows, leaving all other SLM memories
(personal, shared, etc.) untouched.
"""

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
from __future__ import annotations

import json
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_CONTENT_BYTES = 1_000_000
_MAX_SCAN = 10_000       # cap rows pulled by a prefix scan (unbounded-query guard)
_MEM_PREFIX = "autogen-mem:"


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


def _like_escape(value: str) -> str:
    """Escape LIKE special characters so a prefix scan is literal."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class V3AutogenStore:
    """Append-only memory store backed by a SuperLocalMemory engine.

    Framework-free: no ``autogen_core`` import. Content is stored and
    retrieved as plain dicts with ``content``, ``mime_type``, ``metadata``,
    and ``score`` fields.
    """

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

    def add(
        self,
        content: str,
        *,
        mime_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist one memory item.

        Wraps content in a descriptive envelope so even a short string
        produces a queryable SLM row.
        """
        if len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
            raise ValueError(
                f"memory content exceeds maximum size of {MAX_CONTENT_BYTES} bytes"
            )
        mem_id = str(uuid.uuid4())
        envelope = {
            "adapter": "autogen memory",
            "memory_id": mem_id,
            "content": content,
            "mime_type": mime_type,
            "metadata": dict(metadata or {}),
            "created_at": _now_iso(),
        }
        self._engine.store(
            json.dumps(envelope, ensure_ascii=False),
            session_id=f"{_MEM_PREFIX}{mem_id}",
            metadata={
                "integration": "autogen",
                "autogen_memory_id": mem_id,
                "mime_type": mime_type,
                "tags": ["autogen"],
                "importance": 3,
                "project_name": "autogen",
            },
        )

    def clear(self) -> None:
        """Delete all memories added via this adapter (``autogen-mem:`` prefix)."""
        escaped = _like_escape(_MEM_PREFIX)
        rows = self._engine.db.execute(
            "SELECT session_id FROM memories WHERE profile_id=? AND session_id LIKE ? ESCAPE '\\' "
            "LIMIT ?",
            (self._engine.profile_id, escaped + "%", _MAX_SCAN),
        )
        session_ids = [dict(r)["session_id"] for r in rows]
        for sid in session_ids:
            self._delete_session(sid)

    # -- reads -------------------------------------------------------------

    def get_recent(self, *, limit: int = 10) -> list[dict]:
        """Return the most recently added memories in insertion order."""
        limit = min(int(limit), _MAX_SCAN)
        escaped = _like_escape(_MEM_PREFIX)
        rows = self._engine.db.execute(
            "SELECT content FROM memories WHERE profile_id=? AND session_id LIKE ? ESCAPE '\\' "
            "ORDER BY created_at ASC, rowid ASC LIMIT ?",
            (self._engine.profile_id, escaped + "%", limit),
        )
        out: list[dict] = []
        for row in rows:
            parsed = self._parse(dict(row).get("content", ""))
            if parsed is not None:
                out.append(parsed)
        return out

    def query_text(self, query: str, *, limit: int = 10) -> list[dict]:
        """Semantic recall via SLM's text-search pipeline (``fast=True``).

        Returns a list of ``{content, score, metadata, mime_type}`` dicts
        sourced from SLM's atomic-facts layer. Results may include memories
        from any session (not limited to autogen-mem: rows) — this is
        intentional; the full SLM knowledge base enriches AutoGen's context.
        """
        if not query.strip():
            return []
        try:
            resp = self._engine.recall(query, fast=True, limit=limit)
        except Exception:
            return []
        out: list[dict] = []
        for result in resp.results:
            out.append({
                "content": result.fact.content,
                "score": float(result.score),
                "metadata": {"fact_id": result.fact.fact_id},
                "mime_type": "text/plain",
            })
        return out

    def close(self) -> None:
        self._engine.close()

    # -- internals ---------------------------------------------------------

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

    @staticmethod
    def _parse(content: str) -> dict | None:
        """Parse a stored envelope; return the user-visible dict or None."""
        try:
            data = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or "content" not in data:
            return None
        return {
            "content": data.get("content", ""),
            "mime_type": data.get("mime_type", "text/plain"),
            "metadata": data.get("metadata", {}),
            "created_at": data.get("created_at", ""),
        }
