# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""ChatGPT's data export: ``conversations.json`` (or ``conversations-000.json``, ``-001``...).

Shape (secondary sources only; OpenAI's help pages could not be read, see the facts file):
a JSON array of conversations, each with ``id``, ``title``, ``create_time`` (seconds, a float),
``current_node`` and ``mapping``, a dictionary of nodes ``{id, message, parent, children}``. It
is a tree, not a transcript: an edited prompt or a regenerated answer stays in the tree as a
branch. The conversation as it was shown is the path from ``current_node`` up through
``parent`` to the root, so that is the only path imported; every message on another branch is
counted as skipped, never saved.

Accepted variants: one array, several numbered files (merged; a conversation id met twice is
kept once), a single conversation object, an object holding the array under ``conversations``,
and a zip of the export (the other files in it are ignored). Unknown fields are ignored.
A message becomes one turn: its text is the string parts of ``content.parts`` joined by a blank
line, its role is ``author.role``, its time is ``create_time`` (the conversation's when missing).
System, hidden and tool turns, code runs and empty messages are skipped.
"""

from __future__ import annotations

from typing import Any

from superlocalmemory.portability import constants as C
from superlocalmemory.portability.model import TURN, Item, Parsed, Tally, split_long, to_iso
from superlocalmemory.portability.readers import Doc

FORMAT = "chatgpt"
LABEL = "ChatGPT"
KEPT_ROLES = ("user", "assistant")
#: Content types whose text is a person-readable message; anything else is skipped as non-text.
_TEXT_TYPES = ("text", "multimodal_text")
_HIDDEN_TYPES = ("user_editable_context", "system_error", "model_editable_context")
MAX_DEPTH = 50_000  # a path longer than this is a loop or a broken file, not a conversation


def _conversations(obj: Any) -> list[dict]:
    if isinstance(obj, dict):
        if isinstance(obj.get("conversations"), list):
            return _conversations(obj["conversations"])
        return [obj] if isinstance(obj.get("mapping"), dict) else []
    if isinstance(obj, list):
        return [c for c in obj if isinstance(c, dict) and isinstance(c.get("mapping"), dict)]
    return []


def detect(docs: list[Doc]) -> bool:
    return any(d.suffix == ".json" and _conversations(d.json()) for d in docs)


def _path(mapping: dict, current: Any) -> list[str]:
    """Node ids from the root down to the node shown last (``current_node``)."""
    node = current if current in mapping else _fallback_leaf(mapping)
    path: list[str] = []
    seen: set[str] = set()
    while isinstance(node, str) and node in mapping and node not in seen and len(path) < MAX_DEPTH:
        seen.add(node)
        path.append(node)
        parent = mapping[node].get("parent") if isinstance(mapping[node], dict) else None
        node = parent
    path.reverse()
    return path


def _fallback_leaf(mapping: dict) -> str | None:
    """With no usable ``current_node``: the leaf whose message is newest."""
    best, best_time = None, -1.0
    for node_id, node in mapping.items():
        if not isinstance(node, dict) or node.get("children"):
            continue
        message = node.get("message")
        stamp = message.get("create_time") if isinstance(message, dict) else None
        value = float(stamp) if isinstance(stamp, (int, float)) else 0.0
        if best is None or value >= best_time:
            best, best_time = node_id, value
    return best


def _text(message: dict) -> tuple[str, str]:
    """``(text, problem)``: the person-readable text of a message, or why it has none."""
    content = message.get("content")
    if not isinstance(content, dict):
        return "", "empty"
    kind = str(content.get("content_type") or "text")
    if kind in _HIDDEN_TYPES:
        return "", "system"
    if kind not in _TEXT_TYPES:
        return "", "tool"
    parts = content.get("parts")
    if not isinstance(parts, list):
        return "", "empty"
    strings = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
    if strings:
        return "\n\n".join(strings), ""
    return "", "unsupported" if any(not isinstance(p, str) for p in parts) else "empty"


def _turn(node_id: str, node: dict, conv_id: str, conv_at: str, tally: Tally) -> Item | None:
    message = node.get("message") if isinstance(node, dict) else None
    if not isinstance(message, dict):
        return None  # the root, or an empty node: not a message
    meta = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
    author = message.get("author") if isinstance(message.get("author"), dict) else {}
    role = str(author.get("role") or "").lower()
    if role == "system" or meta.get("is_visually_hidden_from_conversation"):
        tally.add("system")
        return None
    if role not in KEPT_ROLES:
        tally.add("tool")
        return None
    text, problem = _text(message)
    if not text:
        tally.add(problem or "empty")
        return None
    return Item(text=text, kind=TURN, role=role, speaker=role,
                created_at=to_iso(message.get("create_time")) or conv_at, session_key=conv_id,
                session_at=conv_at, ref=str(message.get("id") or node_id))


def _messages_off_path(mapping: dict, on_path: set[str]) -> int:
    count = 0
    for node_id, node in mapping.items():
        message = node.get("message") if isinstance(node, dict) else None
        if node_id not in on_path and isinstance(message, dict):
            author = message.get("author") if isinstance(message.get("author"), dict) else {}
            if str(author.get("role") or "").lower() in KEPT_ROLES and _text(message)[0]:
                count += 1
    return count


def parse(docs: list[Doc]) -> Parsed:
    tally = Tally()
    items: list[Item] = []
    seen: set[str] = set()
    conversations = 0  # found, empty ones included
    for doc in docs:
        if doc.suffix != ".json":
            continue
        for conv in _conversations(doc.json()):
            conv_id = str(conv.get("id") or conv.get("conversation_id") or "")
            if conv_id and conv_id in seen:
                continue  # the same conversation in two numbered files
            seen.add(conv_id)
            conversations += 1
            mapping = conv["mapping"]
            conv_at = to_iso(conv.get("create_time"))
            path = _path(mapping, conv.get("current_node"))
            before = len(items)
            for node_id in path:
                turn = _turn(node_id, mapping[node_id], conv_id or f"conv-{conversations}",
                             conv_at, tally)
                if turn is None:
                    continue
                for n, piece in enumerate(split_long(turn.text, C.TURN_LIMIT)):
                    items.append(turn if n == 0 and piece == turn.text else
                                 Item(piece, TURN, turn.created_at, turn.role, turn.speaker,
                                      turn.session_key, turn.session_at, f"{turn.ref}#{n}"))
            tally.add("other_branch", _messages_off_path(mapping, set(path)))
            if len(items) == before:
                tally.add("empty_conversation")
    return Parsed(FORMAT, tuple(items), conversations, tally.result())


__all__ = ["FORMAT", "LABEL", "detect", "parse"]
