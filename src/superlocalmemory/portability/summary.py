# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The plain-words description of what a file holds, shared by the command line and the dashboard.

Most people who import are not technical: "We found 214 conversations from ChatGPT, March 2024 to
October 2026. 12 empty ones will be skipped." They read this sentence, not a table. It is built
here, once, from counts only (no memory text), so every surface says the same thing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from superlocalmemory.portability.model import MEMORY, SKIP_WORDS, Parsed
from superlocalmemory.portability.registry import BY_FORMAT


def _month(stamp: str) -> str:
    try:
        return datetime.fromisoformat(stamp).strftime("%B %Y")
    except ValueError:
        return ""


def date_words(first: str, last: str) -> str:
    """``March 2024 to October 2026``, ``March 2024``, or ``""`` when the file has no dates."""
    a, b = _month(first), _month(last)
    if not a:
        return ""
    return a if a == b else f"{a} to {b}"


def _plural(n: int, one: str, many: str) -> str:
    return f"{n:,} {one if n == 1 else many}"


def describe(parsed: Parsed, left_out: dict[str, int]) -> dict[str, Any]:
    """Counts and sentences for a preview. ``left_out`` is what the plan found already stored."""
    label = BY_FORMAT[parsed.format].LABEL
    turns = [i for i in parsed.items if i.kind != MEMORY]
    memories = [i for i in parsed.items if i.kind == MEMORY]
    first, last = parsed.dates
    when = date_words(first, last)
    empty_conversations = parsed.skipped.get("empty_conversation", 0)
    parts: list[str] = []
    if parsed.conversations:
        parts.append(_plural(parsed.conversations, "conversation", "conversations"))
    if memories:
        parts.append(_plural(len(memories), "memory", "memories"))
    found = " and ".join(parts) or "nothing"
    sentences = [f"We found {found} from {label}{', ' + when if when else ''}."]
    if empty_conversations:
        sentences.append(f"{empty_conversations:,} empty one{'' if empty_conversations == 1 else 's'} "
                         "will be skipped.")
    if turns:
        sentences.append(f"{_plural(len(turns), 'message', 'messages')} will be read in.")
    merged = dict(parsed.skipped)
    for key, n in left_out.items():
        merged[key] = merged.get(key, 0) + n
    merged.pop("empty_conversation", None)
    also = [f"{n:,} {SKIP_WORDS.get(k, k)}" for k, n in sorted(merged.items(), key=lambda kv: -kv[1])
            if n]
    if also:
        sentences.append("Also left out: " + ", ".join(also) + ".")
    sentences.extend(parsed.notes)
    return {
        "format": parsed.format, "label": label, "conversations": parsed.conversations,
        "empty_conversations": empty_conversations, "messages": len(turns),
        "memories": len(memories), "items": len(parsed.items), "first_at": first or None,
        "last_at": last or None, "dates": when,
        "skipped": [{"reason": k, "count": n, "words": SKIP_WORDS.get(k, k)}
                    for k, n in sorted(merged.items(), key=lambda kv: -kv[1]) if n],
        "notes": list(parsed.notes), "summary": " ".join(sentences),
    }


__all__ = ["date_words", "describe"]
