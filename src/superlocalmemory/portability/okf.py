# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""OKF, the Open Knowledge Format: a folder of Markdown concept files with YAML front matter.

What it is (Google Cloud's open specification, version 0.2, repository
``GoogleCloudPlatform/open-knowledge-format``): knowledge as a tree of UTF-8 Markdown files;
each file is one concept and starts with a front matter block between ``---`` lines. Only
``type`` is required; ``title``, ``description``, ``resource``, ``tags``, ``status`` and the
provenance and trust fields are optional and are ignored here. ``index.md`` (a directory
listing) and ``log.md`` (a dated history) are reserved names and are skipped. OKF has no
speaker, role or conversation fields and no standard created time, so this is a document
import: each section of a concept file becomes one memory (the concept's title is the first
line of a section that has no heading of its own), and a file's time is left unset.

A ``.md`` file without a ``type`` line is not an OKF concept and is skipped (the Markdown
importer reads it). Accepted: a folder, a zip or a single file.
"""

from __future__ import annotations

from superlocalmemory.portability import constants as C
from superlocalmemory.portability.markup import is_index, sections, split_front
from superlocalmemory.portability.model import MEMORY, Item, Parsed, Tally
from superlocalmemory.portability.readers import Doc

FORMAT = "okf"
LABEL = "OKF (Open Knowledge Format)"
_RESERVED = ("index.md", "log.md")
_MD = (".md", ".markdown")


def _has_type(doc: Doc) -> bool:
    meta, _body = split_front(doc.text())
    return isinstance(meta.get("type"), str) and bool(str(meta["type"]).strip())


def detect(docs: list[Doc]) -> bool:
    md = [d for d in docs if d.suffix in _MD and d.base not in _RESERVED]
    return bool(md) and sum(_has_type(d) for d in md) * 2 >= len(md) and any(
        _has_type(d) for d in md)


def parse(docs: list[Doc]) -> Parsed:
    tally = Tally()
    items: list[Item] = []
    for doc in docs:
        if doc.suffix not in _MD:
            continue
        if doc.base in _RESERVED:
            tally.add("index")
            continue
        meta, body = split_front(doc.text())
        if not (isinstance(meta.get("type"), str) and str(meta["type"]).strip()):
            tally.add("not_okf")
            continue
        if is_index(body):
            tally.add("index")
            continue
        title = str(meta.get("title") or "").strip()
        found = sections(body, limit=C.SECTION_LIMIT)
        if not found:
            tally.add("empty")
        for n, section in enumerate(found, 1):
            text = section.text()
            if title and not section.path:
                text = f"{title}\n{text}"
            items.append(Item(text, MEMORY, ref=f"{doc.name}#{n}"))
    return Parsed(FORMAT, tuple(items), 0, tally.result())


__all__ = ["FORMAT", "LABEL", "detect", "parse"]
