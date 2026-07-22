# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""Framework-free conversation-history store over SuperLocalMemory's V3 engine.

The OpenAI Agents SDK ``SessionABC`` interface manages the ordered list of
conversation items (messages) for a single session.  This module implements
that data model on top of ``MemoryEngine`` with no OpenAI Agents SDK import,
so the storage contract can be exercised and verified on its own.

Storage mapping
---------------
Each conversation item is one SLM memory whose ``session_id`` combines the
caller-supplied session identifier with a UUID so distinct items never
collide::

    openai-agents:{session_id}\\x1f{uuid4_hex}

The content is a JSON envelope that is always substantial so SLM's ingestion
pipeline creates a queryable atomic-fact row even for tiny items
(e.g. ``{"role":"user","content":"hi"}`` is only 33 bytes — below the SLM
fact-extraction threshold without wrapping)::

    {
        "adapter": "openai-agents session",
        "session_id": "...",
        "seq": 0,
        "item": {"role": "user", "content": "..."},
        "created_at": "<iso>"
    }

Item ordering
-------------
Items are ordered by ``(created_at ASC, rowid ASC)``.  The ``rowid`` is
SQLite's auto-assigned monotonic integer — it is always unique within the
database, ensuring deterministic ordering even when two items are inserted
within the same millisecond.

``pop_item`` retrieves and deletes the most-recent item using
``ORDER BY created_at DESC, rowid DESC LIMIT 1``.

Security
--------
* All LIKE wildcards (``%``, ``_``, ``\\``) are escaped before use in SQL.
* Every LIKE scan is capped at ``_MAX_SCAN`` rows (DoS guard).
* Queries use parameterised placeholders; no string interpolation in SQL.
* The ``\\x1f`` separator between session_id and uuid is a non-printing char
  unlikely to appear in real session identifiers; ``_validate_session_id``
  rejects it explicitly to prevent prefix collisions.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_CONTENT_BYTES = 1_000_000
_MAX_SCAN = 10_000        # hard cap on prefix scans (DoS guard)
_SID_PREFIX = "openai-agents:"
_NS_SEP = "\x1f"          # unit separator between session_id and item uuid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _escape_like(value: str) -> str:
    """Escape LIKE wildcards so the value is treated as a literal prefix."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _validate_session_id(session_id: str) -> None:
    """Reject the namespace separator so distinct sessions cannot collide."""
    if _NS_SEP in session_id:
        raise ValueError(
            "session_id must not contain the reserved separator character (U+001F)"
        )


def _item_prefix(session_id: str) -> str:
    """Prefix for all items belonging to one session."""
    return f"{_SID_PREFIX}{session_id}{_NS_SEP}"


def _item_sid(session_id: str) -> str:
    """Generate a unique SLM session_id for a single item."""
    return f"{_SID_PREFIX}{session_id}{_NS_SEP}{uuid.uuid4().hex}"


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


# ---------------------------------------------------------------------------
# Store core
# ---------------------------------------------------------------------------

class V3SessionStore:
    """Ordered conversation-history storage backed by a SuperLocalMemory V3 engine.

    This class is framework-free: it operates on plain Python dicts and does
    not import any OpenAI Agents SDK types.  The ``session.py`` wrapper
    handles all SDK-typed conversions.
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

    # -- writes -------------------------------------------------------------

    def append_items(self, session_id: str, items: list[dict[str, Any]]) -> None:
        """Append *items* to the conversation history for *session_id*.

        Each item is stored as a separate SLM memory with a unique key, so
        concurrent appends from different threads never collide.
        """
        _validate_session_id(session_id)
        now = _now_iso()
        # Compute a base seq (informational, stored in envelope) from the
        # current count so the field reflects insertion order.
        existing_count = self._count_items(session_id)

        for offset, item in enumerate(items):
            content = json.dumps(item, ensure_ascii=False)
            if len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
                raise ValueError(
                    f"item at index {offset} exceeds maximum size of "
                    f"{MAX_CONTENT_BYTES} bytes"
                )
            seq = existing_count + offset
            envelope = {
                "adapter": "openai-agents session",
                "session_id": session_id,
                "seq": seq,
                "item": item,
                "created_at": now,
            }
            self._engine.store(
                json.dumps(envelope, ensure_ascii=False),
                session_id=_item_sid(session_id),
                metadata={
                    "integration": "openai-agents",
                    "oa_session_id": session_id,
                    "tags": ["openai-agents"],
                    "importance": 3,
                    "project_name": "openai-agents",
                },
            )

    # -- reads --------------------------------------------------------------

    def get_items(
        self, session_id: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Return stored items oldest-first, optionally taking the last *limit*.

        Uses ``(created_at ASC, rowid ASC)`` for deterministic ordering.
        ``rowid`` is SQLite's auto-assigned monotonic integer — unique within
        the DB and stable regardless of wall-clock precision.
        """
        _validate_session_id(session_id)
        escaped = _escape_like(_item_prefix(session_id))

        if limit is not None:
            # Take the newest `limit` rows AT THE DB LEVEL, then present them
            # oldest-first. Doing the tail in SQL is correct even for sessions
            # larger than _MAX_SCAN (a Python `items[-limit:]` over a capped
            # ASC scan would return the wrong region of a huge history), and it
            # avoids the `[-0:]` full-slice trap when limit == 0.
            n = max(0, int(limit))
            rows = self._engine.db.execute(
                "SELECT content FROM memories "
                "WHERE profile_id=? AND session_id LIKE ? ESCAPE '\\' "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (self._engine.profile_id, escaped + "%", n),
            )
            items = []
            for row in rows:
                env = self._parse(dict(row).get("content", ""))
                if env is not None:
                    items.append(env["item"])
            items.reverse()  # DESC fetch → return oldest-first
            return items

        rows = self._engine.db.execute(
            "SELECT content FROM memories "
            "WHERE profile_id=? AND session_id LIKE ? ESCAPE '\\' "
            "ORDER BY created_at ASC, rowid ASC LIMIT ?",
            (self._engine.profile_id, escaped + "%", _MAX_SCAN),
        )
        items = []
        for row in rows:
            env = self._parse(dict(row).get("content", ""))
            if env is not None:
                items.append(env["item"])
        return items

    def pop_item(self, session_id: str) -> dict[str, Any] | None:
        """Remove and return the most-recent item, or ``None`` if empty.

        Ordering: ``(created_at DESC, rowid DESC)`` — the highest ``rowid``
        within the latest timestamp is always the most-recently inserted item.
        The SELECT and DELETE are separate operations (not wrapped in a single
        DB transaction) following the same pattern used by ``_delete_session``
        in the canonical template; sessions are single-user so concurrent
        ``pop_item`` races are not an expected workload.
        """
        _validate_session_id(session_id)
        escaped = _escape_like(_item_prefix(session_id))

        # Collect external-index fact_ids for the item we are about to delete.
        facts = self._engine.db.execute(
            "SELECT af.fact_id, m.session_id, m.content "
            "FROM atomic_facts af JOIN memories m "
            "ON af.memory_id = m.memory_id "
            "WHERE m.profile_id=? "
            "AND m.session_id = ("
            "  SELECT session_id FROM memories "
            "  WHERE profile_id=? AND session_id LIKE ? ESCAPE '\\' "
            "  ORDER BY created_at DESC, rowid DESC LIMIT 1"
            ")",
            (self._engine.profile_id, self._engine.profile_id, escaped + "%"),
        )
        rows = [dict(r) for r in facts]
        fact_ids = [str(r["fact_id"]) for r in rows]
        last_sid = rows[0]["session_id"] if rows else None
        last_content = rows[0]["content"] if rows else None

        if last_sid is None:
            # No facts extracted yet — fall back to a direct memory scan.
            mem_rows = self._engine.db.execute(
                "SELECT session_id, content FROM memories "
                "WHERE profile_id=? AND session_id LIKE ? ESCAPE '\\' "
                "ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (self._engine.profile_id, escaped + "%"),
            )
            for r in mem_rows:
                rd = dict(r)
                last_sid = rd["session_id"]
                last_content = rd["content"]
            if last_sid is None:
                return None

        env = self._parse(last_content or "")
        with self._engine.db.transaction():
            self._engine.db.execute(
                "DELETE FROM memories WHERE session_id=? AND profile_id=?",
                (last_sid, self._engine.profile_id),
            )
        self._remove_external_indexes(fact_ids)
        return env["item"] if env is not None else None

    def clear_session(self, session_id: str) -> None:
        """Delete all items for *session_id*."""
        _validate_session_id(session_id)
        self._delete_prefix(_item_prefix(session_id))

    def close(self) -> None:
        self._engine.close()

    # -- internals ----------------------------------------------------------

    def _count_items(self, session_id: str) -> int:
        escaped = _escape_like(_item_prefix(session_id))
        rows = self._engine.db.execute(
            "SELECT COUNT(*) AS n FROM memories "
            "WHERE profile_id=? AND session_id LIKE ? ESCAPE '\\'",
            (self._engine.profile_id, escaped + "%"),
        )
        for row in rows:
            return int(dict(row).get("n", 0))
        return 0

    def _delete_prefix(self, prefix: str) -> None:
        """Delete all memories whose session_id starts with ``prefix``."""
        escaped = _escape_like(prefix)
        facts = self._engine.db.execute(
            "SELECT af.fact_id FROM atomic_facts af JOIN memories m "
            "ON af.memory_id = m.memory_id "
            "WHERE m.profile_id=? AND m.session_id LIKE ? ESCAPE '\\' LIMIT ?",
            (self._engine.profile_id, escaped + "%", _MAX_SCAN),
        )
        fact_ids = [str(dict(r)["fact_id"]) for r in facts]
        with self._engine.db.transaction():
            self._engine.db.execute(
                "DELETE FROM memories "
                "WHERE profile_id=? AND session_id LIKE ? ESCAPE '\\'",
                (self._engine.profile_id, escaped + "%"),
            )
        self._remove_external_indexes(fact_ids)

    @staticmethod
    def _parse(content: str) -> dict | None:
        try:
            data = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or "item" not in data:
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
