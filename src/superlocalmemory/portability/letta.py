# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Letta's agent file (``.af``): a JSON file with an agent's memory blocks and message history.

Shape (from the ``letta-ai/agent-file`` repository): ``agents[]`` (each with ``messages[]``
and ``memory_blocks``), ``blocks[]`` (``label``, ``value``, ``description``, ``limit``,
``read_only``), and more (``tools``, ``sources``, ``groups``) that are ignored. A block is
memory-like, so each non-empty block ``value`` becomes one memory, once even when the same block
is listed both in ``blocks`` and in an agent's ``memory_blocks``. A message becomes one turn
when its role is ``user`` or ``assistant``; its text is the ``text`` of its ``content`` items
(or ``content`` when that is a string). System, tool and reasoning messages are skipped.
A message with no ``role`` is read from its ``type`` (``user_message``, ``assistant_message``).

Letta's archival memory (the long-term passages) is not part of an ``.af`` file, so it cannot be
imported; the preview says so every time. Letta Cloud and Letta Code no longer write ``.af``
files, but the files people already have still import. Accepted: ``.af`` or ``.json`` with this
shape, a bare array of blocks, and a zip holding either.
"""

from __future__ import annotations

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

FORMAT = "letta"
LABEL = "Letta"
ARCHIVAL_NOTE = ("A Letta agent file does not include archival memory (its long-term "
                 "passages), so that part cannot be imported.")
_ROLE_OF_TYPE = {"user_message": "user", "assistant_message": "assistant"}
_SKIP_ROLES = {"system": "system", "tool": "tool"}


def _files(docs: list[Doc]) -> list[dict]:
    out = []
    for d in docs:
        if d.suffix not in (".af", ".json"):
            continue
        obj = d.json()
        if isinstance(obj, dict) and (isinstance(obj.get("agents"), list)
                                      or isinstance(obj.get("blocks"), list)):
            out.append(obj)
        elif isinstance(obj, list) and obj and all(
                isinstance(b, dict) and "label" in b and "value" in b for b in obj):
            out.append({"blocks": obj})
    return out


def detect(docs: list[Doc]) -> bool:
    return bool(_files(docs))


def _content(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n\n".join(str(c.get("text")).strip() for c in content if isinstance(c, dict)
                           and c.get("type") in (None, "text") and isinstance(c.get("text"), str)
                           and str(c.get("text")).strip())
    return ""


def _blocks(af: dict) -> list[dict]:
    found = [b for b in af.get("blocks") or [] if isinstance(b, dict)]
    for agent in af.get("agents") or []:
        if isinstance(agent, dict):
            found += [b for b in agent.get("memory_blocks") or [] if isinstance(b, dict)]
    return found


def _turns(agent: dict, name: str, fallback_at: str, tally: Tally) -> list[Item]:
    messages = [m for m in agent.get("messages") or [] if isinstance(m, dict)]
    first = next((to_iso(m.get("created_at")) for m in messages if m.get("created_at")), "")
    session_at = first or fallback_at
    out: list[Item] = []
    for n, message in enumerate(messages):
        role = str(message.get("role") or _ROLE_OF_TYPE.get(str(message.get("type")), "")).lower()
        if role in _SKIP_ROLES:
            tally.add(_SKIP_ROLES[role])
            continue
        if role not in ("user", "assistant"):
            tally.add("tool")
            continue
        text = _content(message)
        if not text:
            tally.add("empty")
            continue
        ref = str(message.get("id") or f"{name}#{n}")
        for k, piece in enumerate(split_long(text, C.TURN_LIMIT)):
            out.append(Item(piece, TURN, to_iso(message.get("created_at")) or session_at, role,
                            role, name, session_at, ref if k == 0 else f"{ref}#{k}"))
    return out


def parse(docs: list[Doc]) -> Parsed:
    tally = Tally()
    items: list[Item] = []
    blocks_seen: set[tuple[str, str]] = set()
    conversations = 0
    for af in _files(docs):
        at = to_iso(af.get("created_at"))
        for block in _blocks(af):
            value = block.get("value")
            label = str(block.get("label") or "block")
            text = value.strip() if isinstance(value, str) else ""
            if not text:
                tally.add("empty")
            elif (label, text) in blocks_seen:
                tally.add("duplicate")
            else:
                blocks_seen.add((label, text))
                items.append(Item(text, MEMORY, created_at=at, ref=f"block:{label}"))
        for index, agent in enumerate(a for a in af.get("agents") or [] if isinstance(a, dict)):
            name = str(agent.get("id") or agent.get("name") or f"agent-{index}")
            if not agent.get("messages"):
                continue
            conversations += 1
            turns = _turns(agent, name, at, tally)
            items += turns
            if not turns:
                tally.add("empty_conversation")
    return Parsed(FORMAT, tuple(items), conversations, tally.result(), (ARCHIVAL_NOTE,))


__all__ = ["ARCHIVAL_NOTE", "FORMAT", "LABEL", "detect", "parse"]
