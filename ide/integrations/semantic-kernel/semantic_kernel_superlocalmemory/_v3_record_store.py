"""Framework-free record store over SuperLocalMemory's V3 engine.

Semantic Kernel's ``VectorStoreCollection`` persists *records* — dicts with a
key field, data fields, and optionally a vector — grouped into named
collections. This module implements that data model on ``MemoryEngine`` with no
Semantic Kernel import, so the storage contract is testable on its own.

Storage mapping
---------------
Each record is one SLM memory whose ``session_id`` is ``sk:<collection>:<key>``
and whose content is the JSON-encoded record dict. A per-collection marker
memory (``sk-coll:<collection>``) tracks explicit collection creation so
``collection_exists`` is meaningful even before any record is written.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_CONTENT_BYTES = 1_000_000
_REC_PREFIX = "sk:"
_COLL_PREFIX = "sk-coll:"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _record_sid(collection: str, key: str) -> str:
    return f"{_REC_PREFIX}{collection}\x1f{key}"


def _record_prefix(collection: str) -> str:
    return f"{_REC_PREFIX}{collection}\x1f"


def _coll_sid(collection: str) -> str:
    return f"{_COLL_PREFIX}{collection}"


class V3RecordStore:
    """Collection-scoped record storage backed by a SuperLocalMemory engine."""

    def __init__(self, db_path: str | Path) -> None:
        SLMConfig, MemoryEngine, Mode = _load_v3_types()
        path = Path(db_path).expanduser().resolve()
        config = SLMConfig.for_mode(Mode.A, base_dir=path.parent)
        config.db_path = path
        config.forgetting = replace(config.forgetting, enabled=False)
        config.retrieval.use_cross_encoder = False
        self._engine = MemoryEngine(config)
        self._engine.initialize()

    # -- collection lifecycle ---------------------------------------------

    def create_collection(self, collection: str) -> None:
        if self.collection_exists(collection):
            return
        marker = {
            "adapter": "semantic-kernel record store",
            "kind": "collection-marker",
            "collection": collection,
            "created_at": _now_iso(),
        }
        self._engine.store(
            json.dumps(marker, ensure_ascii=False),
            session_id=_coll_sid(collection),
            metadata={
                "integration": "semantic-kernel",
                "sk_collection": collection,
                "tags": ["semantic-kernel", "sk-collection-marker"],
                "importance": 2,
                "project_name": "semantic-kernel",
            },
        )

    def collection_exists(self, collection: str) -> bool:
        if self._session_rows(_coll_sid(collection)):
            return True
        # A collection with records but no marker still counts as existing.
        return bool(self._prefix_rows(_record_prefix(collection)))

    def delete_collection(self, collection: str) -> None:
        for key in self.list_keys(collection):
            self.delete(collection, key)
        self._delete_session(_coll_sid(collection))

    # -- record CRUD -------------------------------------------------------

    def upsert(self, collection: str, key: str, record: dict[str, Any]) -> str:
        # Guard on the caller-controlled record size, not the envelope.
        if len(json.dumps(record, ensure_ascii=False).encode("utf-8")) > MAX_CONTENT_BYTES:
            raise ValueError(
                f"record exceeds maximum size of {MAX_CONTENT_BYTES} bytes"
            )
        now = _now_iso()
        existing = self._get_envelope(collection, key)
        created = existing["created_at"] if existing else now
        # Wrap in a descriptive envelope: this preserves timestamps AND keeps
        # the stored content substantial enough that SLM ingestion always
        # persists a queryable row (tiny raw records otherwise extract no facts).
        envelope = {
            "adapter": "semantic-kernel record store",
            "collection": collection,
            "key": key,
            "record": record,
            "created_at": created,
            "updated_at": now,
        }
        self._delete_session(_record_sid(collection, key))
        self._engine.store(
            json.dumps(envelope, ensure_ascii=False),
            session_id=_record_sid(collection, key),
            metadata={
                "integration": "semantic-kernel",
                "sk_collection": collection,
                "sk_key": key,
                "tags": ["semantic-kernel"],
                "importance": 3,
                "project_name": "semantic-kernel",
            },
        )
        return key

    def get(self, collection: str, key: str) -> dict | None:
        env = self._get_envelope(collection, key)
        return env["record"] if env is not None else None

    def _get_envelope(self, collection: str, key: str) -> dict | None:
        rows = self._session_rows(_record_sid(collection, key))
        for row in rows:
            return self._parse(row.get("content", ""))
        return None

    def get_many(self, collection: str, keys: list[str]) -> list[dict]:
        out = []
        for key in keys:
            rec = self.get(collection, key)
            if rec is not None:
                out.append(rec)
        return out

    def list_records(self, collection: str) -> list[dict]:
        rows = self._prefix_rows(_record_prefix(collection))
        envelopes = [self._parse(r.get("content", "")) for r in rows]
        return [env["record"] for env in envelopes if env is not None]

    def list_keys(self, collection: str) -> list[str]:
        keys = []
        for row in self._prefix_rows(_record_prefix(collection)):
            sid = row.get("session_id", "")
            marker = _record_prefix(collection)
            if sid.startswith(marker):
                keys.append(sid[len(marker):])
        return keys

    def delete(self, collection: str, key: str) -> None:
        self._delete_session(_record_sid(collection, key))

    def close(self) -> None:
        self._engine.close()

    # -- internals ---------------------------------------------------------

    def _session_rows(self, session_id: str) -> list[dict]:
        rows = self._engine.db.execute(
            "SELECT session_id, content FROM memories "
            "WHERE profile_id=? AND session_id=? "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (self._engine.profile_id, session_id),
        )
        return [dict(r) for r in rows]

    def _prefix_rows(self, prefix: str) -> list[dict]:
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self._engine.db.execute(
            "SELECT session_id, content FROM memories "
            "WHERE profile_id=? AND session_id LIKE ? ESCAPE '\\' "
            "ORDER BY created_at ASC, rowid ASC",
            (self._engine.profile_id, escaped + "%"),
        )
        return [dict(r) for r in rows]

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
        """Parse a stored envelope. Returns the full envelope dict, or None."""
        try:
            data = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or "record" not in data:
            return None
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
