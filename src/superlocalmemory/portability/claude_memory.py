# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Claude's memory, as text you copied out of it (Settings, or "write out your memories of me").

Anthropic defines no file format for this (its help page says to paste the text into a local
file), so any plain text works: each bullet line is one memory and each paragraph is one memory.
Headings are not memories. A leading date like ``[2026-01-15]`` or ``2026-01-15:`` on an entry
is read as when it was said and removed from the text. A ``.txt`` file is read this way by
default; a Markdown file is read by the Markdown importer unless the format is named.
"""

from __future__ import annotations

import re

from superlocalmemory.portability.markup import entries
from superlocalmemory.portability.model import MEMORY, Item, Parsed, Tally, to_iso
from superlocalmemory.portability.readers import Doc

FORMAT = "claude-memory"
LABEL = "Claude memory text"
_DATE = re.compile(r"^\s*[\[(]?(\d{4}-\d{2}-\d{2})[\])]?\s*[:\-–—]?\s+(.+)$", re.DOTALL)


def entries_of(text: str) -> list[str]:
    return entries(text)


def detect(docs: list[Doc]) -> bool:
    return bool(docs) and all(d.suffix == ".txt" for d in docs)


def parse(docs: list[Doc]) -> Parsed:
    tally = Tally()
    items: list[Item] = []
    for doc in docs:
        if doc.suffix not in (".txt", ".md", ".markdown"):
            continue
        for n, entry in enumerate(entries_of(doc.text()), 1):
            when = ""
            match = _DATE.match(entry)
            if match:
                when, entry = to_iso(match.group(1)), match.group(2).strip()
            if not entry:
                tally.add("empty")
                continue
            items.append(Item(entry, MEMORY, created_at=when, ref=f"{doc.name}#{n}"))
    return Parsed(FORMAT, tuple(items), 0, tally.result())


__all__ = ["FORMAT", "LABEL", "detect", "entries_of", "parse"]
