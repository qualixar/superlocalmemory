"""Framework-free CrewAI storage backend over SuperLocalMemory's V3 engine.

CrewAI's StorageBackend is a @runtime_checkable Protocol for hierarchically-
scoped memory records. Each record carries an id, content, a pre-computed
embedding, a hierarchical scope path (e.g. /project/alpha), a category, and
arbitrary metadata.

Storage mapping
---------------
Each record is one SLM memory:
  session_id  = "crewai-rec:{record_id}"
  content     = JSON envelope with adapter/timestamp fields so even a short
                content string persists a queryable row.

Scope-prefix filtering is Python-side after a bounded LIKE scan, using
hierarchical matching: "/project" matches "/project/alpha" but not
"/projectx".

Search ranking uses cosine similarity in pure Python (no numpy dependency).
CrewAI owns the embedder and passes pre-computed query embeddings; the store
only persists and retrieves them.
"""

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_CONTENT_BYTES = 1_000_000
_MAX_SCAN = 10_000       # cap rows pulled by a prefix scan (unbounded-query guard)
_REC_PREFIX = "crewai-rec:"


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


def _session_id(record_id: str) -> str:
    return f"{_REC_PREFIX}{record_id}"


def _like_escape(value: str) -> str:
    """Escape LIKE special characters so a prefix scan is literal."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _scope_matches_prefix(scope: str, prefix: str) -> bool:
    """Hierarchical prefix check: '/project' matches '/project' and
    '/project/alpha' but NOT '/projectx'."""
    if not prefix or prefix == "/":
        return True
    return scope == prefix or scope.startswith(prefix + "/")


def _cosine(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity; returns 0.0 for zero-length or zero-magnitude vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


class V3CrewAIStore:
    """Scope-aware record storage backed by a SuperLocalMemory engine.

    Framework-free: no ``crewai`` import. All records are plain dicts.
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

    def save(self, record: dict[str, Any]) -> None:
        """Insert or replace a record. ``record["id"]`` is the primary key."""
        record_id = str(record["id"])
        content_str = str(record.get("content", ""))
        # Guard on total envelope size, not just content.
        if len(content_str.encode("utf-8")) > MAX_CONTENT_BYTES:
            raise ValueError(
                f"record content exceeds maximum size of {MAX_CONTENT_BYTES} bytes"
            )
        now = _now_iso()
        existing = self._get_envelope(record_id)
        created = existing["created_at"] if existing else now
        # Descriptive envelope: ensures even a one-word content string
        # produces a queryable row in SLM ingestion.
        envelope = {
            "adapter": "crewai storage backend",
            "record_id": record_id,
            "content": content_str,
            "embedding": list(record.get("embedding") or []),
            "scope": str(record.get("scope", "/")),
            "category": str(record.get("category", "general")),
            "metadata": dict(record.get("metadata") or {}),
            "created_at": created,
            "updated_at": now,
        }
        sid = _session_id(record_id)
        self._delete_session(sid)
        self._engine.store(
            json.dumps(envelope, ensure_ascii=False),
            session_id=sid,
            metadata={
                "integration": "crewai",
                "crewai_record_id": record_id,
                "crewai_scope": envelope["scope"],
                "crewai_category": envelope["category"],
                "tags": ["crewai"],
                "importance": 3,
                "project_name": "crewai",
            },
        )

    def update(self, record: dict[str, Any]) -> None:
        """Replace an existing record; same as ``save``."""
        self.save(record)

    def delete_by_id(self, record_id: str) -> int:
        """Delete a single record by id. Returns 1 if found, 0 if not."""
        if self._get_envelope(record_id) is None:
            return 0
        self._delete_session(_session_id(record_id))
        return 1

    def delete_by_scope(self, scope_prefix: str) -> int:
        """Delete all records whose scope matches ``scope_prefix``. Returns count."""
        envelopes = self._all_envelopes()
        matched = [e for e in envelopes if _scope_matches_prefix(e["scope"], scope_prefix)]
        for env in matched:
            self._delete_session(_session_id(env["record_id"]))
        return len(matched)

    # -- reads -------------------------------------------------------------

    def get_record(self, record_id: str) -> dict | None:
        """Return the stored record dict, or None if not found."""
        env = self._get_envelope(record_id)
        if env is None:
            return None
        return self._envelope_to_record(env)

    def search(
        self,
        query_embedding: list[float],
        *,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[dict, float]]:
        """Rank stored records by cosine similarity to ``query_embedding``.

        Returns a list of ``(record_dict, score)`` tuples, sorted descending
        by score. Records with embeddings shorter than the query are still
        ranked (missing dimensions treated as zero). Zero-vector query returns
        score 0.0 for all candidates.
        """
        limit = max(1, int(limit))
        candidates = self._all_envelopes()

        if scope_prefix is not None:
            candidates = [e for e in candidates if _scope_matches_prefix(e["scope"], scope_prefix)]
        if categories:
            cat_set = set(categories)
            candidates = [e for e in candidates if e["category"] in cat_set]
        if metadata_filter:
            candidates = [
                e for e in candidates
                if all(e["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]

        scored: list[tuple[dict, float]] = []
        for env in candidates:
            score = _cosine(query_embedding, env.get("embedding") or [])
            if score >= min_score:
                scored.append((self._envelope_to_record(env), score))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:limit]

    def list_records(
        self,
        *,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict]:
        """List records, optionally filtered by scope prefix."""
        limit = min(int(limit), _MAX_SCAN)
        envelopes = self._all_envelopes()
        if scope_prefix is not None:
            envelopes = [e for e in envelopes if _scope_matches_prefix(e["scope"], scope_prefix)]
        page = envelopes[offset: offset + limit]
        return [self._envelope_to_record(e) for e in page]

    def count(self, *, scope_prefix: str | None = None) -> int:
        """Count records, optionally filtered by scope prefix."""
        envelopes = self._all_envelopes()
        if scope_prefix is None:
            return len(envelopes)
        return sum(1 for e in envelopes if _scope_matches_prefix(e["scope"], scope_prefix))

    def list_scopes(self, *, parent: str = "/") -> list[str]:
        """Return all distinct scopes that are children of ``parent``."""
        envelopes = self._all_envelopes()
        seen: set[str] = set()
        for env in envelopes:
            scope = env["scope"]
            if _scope_matches_prefix(scope, parent):
                seen.add(scope)
        return sorted(seen)

    def list_categories(self, *, scope_prefix: str | None = None) -> dict[str, int]:
        """Return {category: count} mapping, optionally filtered by scope."""
        envelopes = self._all_envelopes()
        if scope_prefix is not None:
            envelopes = [e for e in envelopes if _scope_matches_prefix(e["scope"], scope_prefix)]
        cats: dict[str, int] = {}
        for env in envelopes:
            cats[env["category"]] = cats.get(env["category"], 0) + 1
        return cats

    def get_scope_info(self, scope: str) -> dict:
        """Return aggregate info for an exact scope as a plain dict."""
        envelopes = [e for e in self._all_envelopes() if e["scope"] == scope]
        cats: dict[str, int] = {}
        for env in envelopes:
            cats[env["category"]] = cats.get(env["category"], 0) + 1
        timestamps = [e["created_at"] for e in envelopes if e.get("created_at")]
        return {
            "scope": scope,
            "record_count": len(envelopes),
            "categories": cats,
            "created_at": min(timestamps) if timestamps else None,
            "updated_at": max(timestamps) if timestamps else None,
        }

    def reset(self, *, scope_prefix: str | None = None) -> None:
        """Delete all records, or those matching ``scope_prefix``."""
        self.delete_by_scope(scope_prefix if scope_prefix is not None else "/")

    def close(self) -> None:
        self._engine.close()

    # -- internals ---------------------------------------------------------

    def _get_envelope(self, record_id: str) -> dict | None:
        rows = self._engine.db.execute(
            "SELECT content FROM memories WHERE profile_id=? AND session_id=? "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (self._engine.profile_id, _session_id(record_id)),
        )
        for row in rows:
            return self._parse(dict(row).get("content", ""))
        return None

    def _all_envelopes(self) -> list[dict]:
        escaped = _like_escape(_REC_PREFIX)
        rows = self._engine.db.execute(
            "SELECT content FROM memories WHERE profile_id=? AND session_id LIKE ? ESCAPE '\\' "
            "ORDER BY created_at ASC, rowid ASC LIMIT ?",
            (self._engine.profile_id, escaped + "%", _MAX_SCAN),
        )
        out: list[dict] = []
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
        try:
            data = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or "record_id" not in data:
            return None
        data.setdefault("embedding", [])
        data.setdefault("scope", "/")
        data.setdefault("category", "general")
        data.setdefault("metadata", {})
        data.setdefault("created_at", "")
        data.setdefault("updated_at", data["created_at"])
        return data

    @staticmethod
    def _envelope_to_record(env: dict) -> dict:
        """Convert an internal envelope to a plain record dict."""
        return {
            "id": env["record_id"],
            "content": env.get("content", ""),
            "embedding": env.get("embedding", []),
            "scope": env.get("scope", "/"),
            "category": env.get("category", "general"),
            "metadata": env.get("metadata", {}),
            "created_at": env.get("created_at", ""),
            "updated_at": env.get("updated_at", ""),
        }
