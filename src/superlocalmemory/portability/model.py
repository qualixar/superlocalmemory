# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What an importer produces: items to remember, and an honest count of what it left out.

An :class:`Item` is one thing to save: a memory-like entry (a Mem0 memory, a Letta block, a
Markdown section) or one turn of a conversation. A :class:`Parsed` is everything one file
yielded: the items, how many conversations they came from, a count for every kind of thing that
was skipped and why, and any plain-words notes a person should read before importing. Nothing
here touches a database. Items are immutable.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

MEMORY, TURN = "memory", "turn"

#: Plain words for every reason something is skipped; the preview and the dashboard print these.
SKIP_WORDS: Mapping[str, str] = {
    "empty": "empty messages or entries",
    "empty_conversation": "empty conversations",
    "system": "system and hidden instruction messages",
    "tool": "tool, code and plug-in turns",
    "other_branch": "replies you edited or regenerated away",
    "expired": "memories that had already expired",
    "replaced": "memories that a newer one replaced",
    "index": "index or listing files",
    "not_okf": "files with no type line (not OKF concepts)",
    "unsupported": "content that is not text (pictures, files)",
    "duplicate": "copies of something already in this file",
    "already_there": "things already in your memory (left exactly as they are)",
    "rejected": "pieces too short or too thin to remember",
    "unreadable": "files that could not be read",
    "too_big": "files over the size limit",
}


def to_iso(value: Any) -> str:
    """A time as ISO 8601 UTC, or ``""`` when it is missing or not a time.

    Seconds since 1970 (a number, as ChatGPT writes ``create_time``) and ISO strings (as Claude
    and Mem0 write them) are both read. Milliseconds are recognised by size.
    """
    if value is None or isinstance(value, bool):
        return ""
    try:
        if isinstance(value, (int, float)):
            seconds = float(value) / (1000.0 if abs(float(value)) > 1e11 else 1.0)
            return datetime.fromtimestamp(seconds, UTC).isoformat(timespec="seconds")
        text = str(value).strip()
        if not text:
            return ""
        if re.fullmatch(r"-?\d+(\.\d+)?", text):
            return to_iso(float(text))
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC).isoformat(timespec="seconds")
    except (ValueError, OverflowError, OSError):
        return ""


@dataclass(frozen=True)
class Item:
    text: str
    kind: str
    created_at: str = ""
    role: str = ""
    speaker: str = ""
    #: The conversation a turn belongs to (an id inside the file); ``""`` for a memory-like item.
    session_key: str = ""
    #: The conversation's own time, used as the session date of every turn in it.
    session_at: str = ""
    #: Where it sat in the file (a message id, an item id, a section path): the dedupe key.
    ref: str = ""

    def fingerprint(self, fmt: str) -> str:
        """Stable for the same item of the same format: the same file imported twice matches."""
        raw = "\x1f".join((fmt, self.kind, self.session_key, self.ref, self.role,
                           self.text))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


@dataclass(frozen=True)
class Parsed:
    format: str
    items: tuple[Item, ...]
    #: Conversations found in the file, empty ones included.
    conversations: int = 0
    skipped: Mapping[str, int] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def dates(self) -> tuple[str, str]:
        stamps = sorted(t for t in (i.created_at or i.session_at for i in self.items) if t)
        return (stamps[0], stamps[-1]) if stamps else ("", "")

    @property
    def skipped_total(self) -> int:
        return sum(self.skipped.values())


class Tally:
    """Counts what an importer skipped, by reason (a reason is a key of :data:`SKIP_WORDS`)."""

    def __init__(self) -> None:
        self._c: Counter[str] = Counter()

    def add(self, reason: str, n: int = 1) -> None:
        if n > 0:
            self._c[reason] += n

    def result(self) -> dict[str, int]:
        return dict(self._c)


class ImportFormatError(ValueError):
    """A file does not look like the format asked for; the message is plain words."""


def split_long(text: str, limit: int) -> list[str]:
    """``text`` as one piece, or cut at paragraph breaks into pieces of at most ``limit``."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    pieces: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        while len(para) > limit:  # one paragraph longer than the limit: cut on a space
            cut = para.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            if current:
                pieces.append(current)
                current = ""
            pieces.append(para[:cut].strip())
            para = para[cut:].strip()
        if len(current) + len(para) + 2 > limit and current:
            pieces.append(current)
            current = ""
        current = f"{current}\n\n{para}" if current else para
    if current:
        pieces.append(current)
    return pieces


__all__ = ["ImportFormatError", "Item", "MEMORY", "Parsed", "SKIP_WORDS", "TURN", "Tally",
           "split_long", "to_iso"]
