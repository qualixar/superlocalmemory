"""Framework-free conversation/message store over SuperLocalMemory's V3 engine.

The Microsoft Agent Framework persists a session's messages through a
``HistoryProvider`` and enriches each run through a ``ContextProvider``. Both
need durable per-session message storage. This module provides that on
``MemoryEngine`` with no ``agent_framework`` import, so the storage contract is
testable on its own.

Each message is one SLM memory sharing the session's ``session_id``
(``af:<session>``), stored as a descriptive JSON envelope so even a short
message persists a queryable row and round-trips losslessly.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_CONTENT_BYTES = 1_000_000
_SESSION_PREFIX = "af:"
_DEFAULT_SESSION = "default"


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


def _session_key(session_id: str | None) -> str:
    return f"{_SESSION_PREFIX}{session_id if session_id else _DEFAULT_SESSION}"


class V3MessageStore:
    """Per-session message storage backed by a SuperLocalMemory engine."""

    def __init__(self, db_path: str | Path) -> None:
        SLMConfig, MemoryEngine, Mode = _load_v3_types()
        path = Path(db_path).expanduser().resolve()
        config = SLMConfig.for_mode(Mode.A, base_dir=path.parent)
        config.db_path = path
        config.forgetting = replace(config.forgetting, enabled=False)
        config.retrieval.use_cross_encoder = False
        self._engine = MemoryEngine(config)
        self._engine.initialize()

    def append(self, session_id: str | None, message: dict[str, Any]) -> None:
        if len(json.dumps(message, ensure_ascii=False).encode("utf-8")) > MAX_CONTENT_BYTES:
            raise ValueError(
                f"message exceeds maximum size of {MAX_CONTENT_BYTES} bytes"
            )
        envelope = {
            "adapter": "microsoft-agent-framework message",
            "session_id": session_id if session_id else _DEFAULT_SESSION,
            "message": message,
            "created_at": _now_iso(),
        }
        self._engine.store(
            json.dumps(envelope, ensure_ascii=False),
            session_id=_session_key(session_id),
            metadata={
                "integration": "microsoft-agent-framework",
                "af_session": session_id if session_id else _DEFAULT_SESSION,
                "tags": ["agent-framework"],
                "importance": 3,
                "project_name": "agent-framework",
            },
        )

    def messages(self, session_id: str | None, *, limit: int = 10_000) -> list[dict]:
        rows = self._engine.db.execute(
            "SELECT content FROM memories WHERE profile_id=? AND session_id=? "
            "ORDER BY created_at ASC, rowid ASC LIMIT ?",
            (self._engine.profile_id, _session_key(session_id), int(limit)),
        )
        out = []
        for row in rows:
            msg = self._extract(dict(row).get("content", ""))
            if msg is not None:
                out.append(msg)
        return out

    def clear(self, session_id: str | None) -> None:
        self._delete_session(_session_key(session_id))

    def close(self) -> None:
        self._engine.close()

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _extract(content: str) -> dict | None:
        try:
            data = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or "message" not in data:
            return None
        msg = data["message"]
        return msg if isinstance(msg, dict) else None

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
