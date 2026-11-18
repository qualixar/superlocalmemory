# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Mem0: what ``get_all`` (open source or platform) returns, saved as JSON.

Mem0 memories are extracted facts, not turns, so each item becomes one memory. The text is
``memory``; the time is ``created_at``; there is no speaker. The shape has changed between
versions, so every one of these is accepted: a bare array of items, ``{"results": [...]}`` (the
open-source SDK), the platform's paginated envelope ``{"count", "next", "previous", "results"}``
(several saved pages are merged), ``{"memories": [...]}``, and a single item. Older items
that call the text ``text`` or ``content`` are read too. An item with an ``expiration_date`` in
the past, or one a newer memory replaced (``replaced_by`` set), is skipped and counted. The
item ``id`` is the dedupe key; ``user_id``, ``agent_id``, ``session_id``, ``metadata`` and
``categories`` are ignored. Unknown fields never matter.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from superlocalmemory.portability.model import MEMORY, Item, Parsed, Tally, to_iso
from superlocalmemory.portability.readers import Doc

FORMAT = "mem0"
LABEL = "Mem0"
_TEXT_KEYS = ("memory", "text", "content")


def _items(obj: Any) -> list[dict]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict) and _text_of(x) is not None]
    if isinstance(obj, dict):
        for key in ("results", "memories", "items", "data"):
            if isinstance(obj.get(key), (list, dict)):
                inner = _items(obj[key])
                if inner:
                    return inner
        if _text_of(obj) is not None and ("id" in obj or "created_at" in obj or "memory" in obj):
            return [obj]
    return []


def _text_of(item: dict) -> str | None:
    for key in _TEXT_KEYS:
        if isinstance(item.get(key), str):
            return item[key]
    return None


def detect(docs: list[Doc]) -> bool:
    """True when a JSON file holds items with a ``memory`` field (``text`` and ``content`` alone
    are too common to name Mem0; they are read only when the format is chosen by name)."""
    return any(d.suffix == ".json" and not _is_other(d.json())
               and any("memory" in x for x in _items(d.json())) for d in docs)


def _is_other(obj: Any) -> bool:
    """A conversation export or an agent file is not Mem0, even if a field name overlaps."""
    first = obj[0] if isinstance(obj, list) and obj and isinstance(obj[0], dict) else obj
    return isinstance(first, dict) and any(
        k in first for k in ("mapping", "chat_messages", "agents", "blocks"))


def _expired(item: dict, now: datetime) -> bool:
    stamp = to_iso(item.get("expiration_date"))
    return bool(stamp) and datetime.fromisoformat(stamp) < now


def parse(docs: list[Doc]) -> Parsed:
    tally = Tally()
    items: list[Item] = []
    seen: set[str] = set()
    now = datetime.now(UTC)
    for doc in docs:
        if doc.suffix != ".json":
            continue
        for n, raw in enumerate(_items(doc.json())):
            text = (_text_of(raw) or "").strip()
            ref = str(raw.get("id") or f"{doc.name}#{n}")
            if ref in seen:
                tally.add("duplicate")
                continue
            seen.add(ref)
            if not text:
                tally.add("empty")
            elif _expired(raw, now):
                tally.add("expired")
            elif raw.get("replaced_by"):
                tally.add("replaced")
            else:
                items.append(Item(text, MEMORY, created_at=to_iso(raw.get("created_at")), ref=ref))
    return Parsed(FORMAT, tuple(items), 0, tally.result())


__all__ = ["FORMAT", "LABEL", "detect", "parse"]
