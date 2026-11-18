# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Claude's (claude.ai) data export: ``conversations.json``, and ``memories.json`` when it is there.

Shape (Anthropic's help page does not give one; this is from third-party descriptions of real
exports, see the facts file): ``conversations.json`` is an array of conversations with
``uuid``, ``name``, ``created_at``, ``updated_at`` and ``chat_messages``; each message has
``uuid``, ``sender`` (``human`` or ``assistant``), ``text``, ``content`` (typed blocks) and
``created_at`` (ISO 8601, UTC). A message's text is ``text``; when that is empty the ``text``
blocks of ``content`` are joined. ``thinking``, ``tool_use`` and ``tool_result`` blocks are never
imported. ``human`` is stored as the role ``user``.

Accepted variants: an array, a single conversation object, an object holding the array under
``conversations``, a conversation that calls its messages ``messages`` and its sender ``role``,
and a zip of the export. When ``memories.json`` is present (some exports have it) its
``conversations_memory`` text and every ``project_memories`` text become memory entries, the
same way Claude's pasted memory text does. ``projects.json`` and ``users.json`` are ignored.
"""

from __future__ import annotations

from typing import Any

from superlocalmemory.portability import constants as C
from superlocalmemory.portability.model import (
    MEMORY,
    TURN,
    Item,
    Parsed,
    Tally,
    split_long,
    to_iso,
)
from superlocalmemory.portability.readers import Doc

FORMAT = "claude"
LABEL = "Claude"
_ROLE = {"human": "user", "user": "user", "assistant": "assistant"}


def _messages(conv: dict) -> list | None:
    for key in ("chat_messages", "messages"):
        if isinstance(conv.get(key), list):
            return conv[key]
    return None


def _conversations(obj: Any) -> list[dict]:
    if isinstance(obj, dict):
        if isinstance(obj.get("conversations"), list):
            return _conversations(obj["conversations"])
        return [obj] if _messages(obj) is not None and (obj.get("uuid") or obj.get("name")) else []
    if isinstance(obj, list):
        return [c for c in obj if isinstance(c, dict) and _messages(c) is not None
                and ("uuid" in c or "name" in c or "created_at" in c)]
    return []


def _memory_docs(docs: list[Doc]) -> list[dict]:
    out = []
    for d in docs:
        if d.base == "memories.json" and isinstance(d.json(), list):
            out += [m for m in d.json() if isinstance(m, dict)
                    and ("conversations_memory" in m or "project_memories" in m)]
    return out


def detect(docs: list[Doc]) -> bool:
    if any(d.suffix == ".json" and _conversations(d.json()) for d in docs):
        return True
    return bool(_memory_docs(docs))


def _text(message: dict) -> tuple[str, str]:
    text = message.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip(), ""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip(), ""
    if isinstance(content, list):
        blocks = [b for b in content if isinstance(b, dict)]
        strings = [str(b.get("text")).strip() for b in blocks
                   if b.get("type") in (None, "text") and isinstance(b.get("text"), str)
                   and str(b.get("text")).strip()]
        if strings:
            return "\n\n".join(strings), ""
        if blocks and all(b.get("type") in ("tool_use", "tool_result", "thinking") for b in blocks):
            return "", "tool"
    return "", "empty"


def _turns(conv: dict, conv_id: str, conv_at: str, tally: Tally) -> list[Item]:
    out: list[Item] = []
    for n, message in enumerate(_messages(conv) or []):
        if not isinstance(message, dict):
            continue
        role = _ROLE.get(str(message.get("sender") or message.get("role") or "").lower(), "")
        if not role:
            tally.add("system" if str(message.get("sender") or message.get("role")
                                      or "").lower() == "system" else "tool")
            continue
        text, problem = _text(message)
        if not text:
            tally.add(problem or "empty")
            continue
        stamp = to_iso(message.get("created_at")) or conv_at
        ref = str(message.get("uuid") or f"{conv_id}#{n}")
        for k, piece in enumerate(split_long(text, C.TURN_LIMIT)):
            out.append(Item(piece, TURN, stamp, role, role, conv_id, conv_at,
                            ref if k == 0 else f"{ref}#{k}"))
    return out


def _memories(docs: list[Doc], tally: Tally) -> list[Item]:
    from superlocalmemory.portability.claude_memory import entries_of

    out: list[Item] = []
    for record in _memory_docs(docs):
        texts = [("conversations_memory", record.get("conversations_memory"))]
        projects = record.get("project_memories")
        if isinstance(projects, dict):
            texts += [(f"project:{k}", v) for k, v in projects.items()]
        for ref, value in texts:
            if isinstance(value, str):
                out += [Item(e, MEMORY, ref=f"{ref}#{i}") for i, e in
                        enumerate(entries_of(value), 1)]
            elif value is not None:
                tally.add("unsupported")
    return out


def parse(docs: list[Doc]) -> Parsed:
    tally = Tally()
    items: list[Item] = []
    seen: set[str] = set()
    found = 0
    for doc in docs:
        if doc.suffix != ".json":
            continue
        for conv in _conversations(doc.json()):
            conv_id = str(conv.get("uuid") or conv.get("id") or f"conv-{found}")
            if conv_id in seen:
                continue
            seen.add(conv_id)
            found += 1
            conv_at = to_iso(conv.get("created_at"))
            turns = _turns(conv, conv_id, conv_at, tally)
            if not turns:
                tally.add("empty_conversation")
            items += turns
    items += _memories(docs, tally)
    return Parsed(FORMAT, tuple(items), found, tally.result())


__all__ = ["FORMAT", "LABEL", "detect", "parse"]
