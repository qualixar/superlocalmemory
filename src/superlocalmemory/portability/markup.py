# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Markdown helpers shared by the document importers: front matter and sections.

Front matter is the small YAML block between two ``---`` lines at the top of a file. Only a
safe subset is read (``key: value``, inline lists, block lists, quotes); nothing is evaluated,
no tags or anchors are followed, and a block that does not parse is left as plain text. A file
is cut into sections at its headings (a ``#`` inside a code fence is not a heading); each
section keeps its heading path as its first line, so a section read alone still says where it
sat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from superlocalmemory.portability.model import split_long

_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
_KEY = re.compile(r"^([A-Za-z_][\w .-]*):[ \t]*(.*)$")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_LINK_LINE = re.compile(r"^\s*(?:[-*+]\s+)?\[[^\]]*\]\([^)]*\)(?:\s*[-—:].*)?$")
MAX_FRONT_LINES = 60


def _scalar(raw: str) -> object:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        return [_scalar(p) for p in text[1:-1].split(",") if p.strip()]
    return text


def split_front(text: str) -> tuple[dict[str, object], str]:
    """``(front matter, body)``. No front matter, or one that does not parse: ``({}, text)``."""
    text = text.lstrip("﻿")
    lines = text.split("\n")
    if not lines or lines[0].rstrip() != "---":
        return {}, text
    end = next((i for i, ln in enumerate(lines[1:MAX_FRONT_LINES + 1], 1)
                if ln.rstrip() in ("---", "...")), None)
    if end is None:
        return {}, text
    meta: dict[str, object] = {}
    key = None
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if key and (m := _BULLET.match(line)) and isinstance(meta.get(key), list):
            meta[key].append(_scalar(m.group(1)))  # type: ignore[union-attr]
            continue
        match = _KEY.match(line)
        if not match or line.startswith((" ", "\t")):
            if key and line.startswith((" ", "\t")):
                continue  # a nested value we do not read
            return {}, text  # not the safe subset: leave the block as text
        key, value = match.group(1).strip().lower(), match.group(2)
        meta[key] = [] if value.strip() == "" else _scalar(value)
    return meta, "\n".join(lines[end + 1:]).lstrip("\n")


@dataclass(frozen=True)
class Section:
    path: str  # "Rules > Tests", or "" before the first heading
    body: str

    def text(self) -> str:
        return f"{self.path}\n{self.body}".strip() if self.path else self.body.strip()


def sections(body: str, *, limit: int) -> list[Section]:
    """The body cut at headings, long sections cut again at paragraph breaks (never empty)."""
    stack: list[tuple[int, str]] = []
    path, lines, found = "", [], []
    fenced = False

    def close() -> None:
        text = "\n".join(lines).strip()
        if text:
            for piece in split_long(text, limit):
                found.append(Section(path, piece))

    for line in body.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
        match = None if fenced else _HEADING.match(line)
        if match:
            close()
            lines = []
            level = len(match.group(1))
            stack = [(lv, t) for lv, t in stack if lv < level]
            stack.append((level, match.group(2).strip()))
            path = " > ".join(t for _lv, t in stack)
        else:
            lines.append(line)
    close()
    return found


def is_index(body: str) -> bool:
    """A listing: every non-empty line is a link (or a heading), so it holds no memory itself."""
    lines = [ln for ln in body.splitlines() if ln.strip() and not _HEADING.match(ln)]
    return bool(lines) and all(_LINK_LINE.match(ln) for ln in lines)


def _is_label(line: str, following: str | None) -> bool:
    """A short plain line that introduces a list ("Work context") is a heading, not a memory."""
    return (following is not None and bool(_BULLET.match(following)) and len(line.strip()) <= 60
            and not _BULLET.match(line) and not line.startswith((" ", "\t"))
            and line.rstrip()[-1:] not in ".!?;,")


def entries(text: str) -> list[str]:
    """Plain text as entries: each bullet is one, each paragraph is one (a heading is not)."""
    out: list[str] = []
    block: list[str] = []

    def flush() -> None:
        if block:
            out.append(" ".join(s.strip() for s in block).strip())
            block.clear()

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.strip() or _HEADING.match(line):
            flush()
            continue
        bullet = _BULLET.match(line)
        if bullet:
            flush()
            block.append(bullet.group(1))
        elif not block and _is_label(line, lines[i + 1] if i + 1 < len(lines) else None):
            continue
        else:
            block.append(line)  # a wrapped bullet line, or the next line of a paragraph
    flush()
    return [e for e in out if e]


__all__ = ["MAX_FRONT_LINES", "Section", "entries", "is_index", "sections", "split_front"]
