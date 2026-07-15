"""Small installed-package adapter over SuperLocalMemory's V3 engine contract."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

MAX_CONTENT_BYTES = 1_000_000


def _load_v3_types():
    """Import the installed SLM package without treating its data root as code."""
    try:
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.engine import MemoryEngine
        from superlocalmemory.storage.models import Mode
    except ImportError as exc:
        legacy_root = os.environ.get("SLM_INSTALL_DIR")
        legacy_note = (
            f" Legacy SLM_INSTALL_DIR={legacy_root!r} is not added to sys.path;"
            if legacy_root
            else ""
        )
        raise ImportError(
            "SuperLocalMemory V3 is not installed in this Python environment. "
            "Install it with: python -m pip install superlocalmemory."
            f"{legacy_note}"
        ) from exc
    return SLMConfig, MemoryEngine, Mode


class V3ChatStore:
    """Persist exact chat payloads through ``MemoryEngine.store``.

    Canonical ingestion creates the searchable facts. The parent ``memories``
    rows remain the exact, lossless chat-history source used for round trips.
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

    def add(self, content: str, *, session_id: str, metadata: dict[str, Any]) -> None:
        size = len(content.encode("utf-8"))
        if size > MAX_CONTENT_BYTES:
            raise ValueError(
                f"content exceeds maximum size of {MAX_CONTENT_BYTES} bytes"
            )
        fact_ids = self._engine.store(
            content,
            session_id=session_id,
            metadata=metadata,
        )
        if not fact_ids:
            raise ValueError("content was rejected by SuperLocalMemory ingestion")

    def list_session(self, session_id: str, *, limit: int = 10_000) -> list[dict]:
        rows = self._engine.db.execute(
            "SELECT rowid, memory_id, content, created_at, metadata_json "
            "FROM memories WHERE profile_id=? AND session_id=? "
            "ORDER BY created_at ASC, rowid ASC LIMIT ?",
            (self._engine.profile_id, session_id, int(limit)),
        )
        return [self._memory_dict(row) for row in rows]

    def list_prefix(self, session_prefix: str, *, limit: int = 10_000) -> list[dict]:
        escaped = session_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self._engine.db.execute(
            "SELECT rowid, memory_id, content, created_at, metadata_json "
            "FROM memories WHERE profile_id=? AND session_id LIKE ? ESCAPE '\\' "
            "ORDER BY created_at ASC, rowid ASC LIMIT ?",
            (self._engine.profile_id, escaped + "%", int(limit)),
        )
        return [self._memory_dict(row) for row in rows]

    def delete(self, memory_id: str) -> None:
        facts = self._engine.db.execute(
            "SELECT fact_id FROM atomic_facts WHERE memory_id=? AND profile_id=?",
            (memory_id, self._engine.profile_id),
        )
        fact_ids = [str(dict(row)["fact_id"]) for row in facts]
        with self._engine.db.transaction():
            self._engine.db.execute(
                "DELETE FROM memories WHERE memory_id=? AND profile_id=?",
                (memory_id, self._engine.profile_id),
            )
        self._remove_external_indexes(fact_ids)

    @staticmethod
    def _memory_dict(row: Any) -> dict:
        value = dict(row)
        try:
            metadata = json.loads(value.get("metadata_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        return {
            "id": value["memory_id"],
            "content": value["content"],
            "created_at": value["created_at"],
            "metadata": metadata,
            "tags": metadata.get("tags", []),
        }

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

    def close(self) -> None:
        self._engine.close()
