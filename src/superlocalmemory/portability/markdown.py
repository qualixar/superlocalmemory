# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Plain Markdown: ``AGENTS.md``, ``CLAUDE.md``, ``MEMORY.md`` with its topic files, any ``.md``.

These are instruction and memory files agents keep (``AGENTS.md`` is "just standard Markdown";
Claude Code's auto memory is a ``MEMORY.md`` index plus topic files). Each file is cut at its
headings; each section becomes one memory whose first line is its heading path ("Rules > Tests"),
so a section read alone still says where it sat. A section longer than the section limit is cut
at paragraph breaks. Front matter, if there is any, is removed (it is not text to remember).
A file that is only a list of links (an index such as ``MEMORY.md``) is skipped, because the
files it points to are imported themselves. A file's time is left unset: nothing in Markdown
says when it was written.

Accepted: a single file, a folder (every ``.md`` and ``.markdown`` in it) or a zip.
"""

from __future__ import annotations

from superlocalmemory.portability import constants as C
from superlocalmemory.portability.markup import is_index, sections, split_front
from superlocalmemory.portability.model import MEMORY, Item, Parsed, Tally
from superlocalmemory.portability.readers import Doc

FORMAT = "markdown"
LABEL = "Markdown"
_MD = (".md", ".markdown")


def detect(docs: list[Doc]) -> bool:
    return any(d.suffix in _MD for d in docs)


def parse(docs: list[Doc]) -> Parsed:
    tally = Tally()
    items: list[Item] = []
    for doc in docs:
        if doc.suffix not in _MD:
            continue
        _meta, body = split_front(doc.text())
        if is_index(body):
            tally.add("index")
            continue
        found = sections(body, limit=C.SECTION_LIMIT)
        if not found:
            tally.add("empty")
        for n, section in enumerate(found, 1):
            items.append(Item(section.text(), MEMORY, ref=f"{doc.name}#{n}"))
    return Parsed(FORMAT, tuple(items), 0, tally.result())


__all__ = ["FORMAT", "LABEL", "detect", "parse"]
