# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Which importer reads a file: detection, the explicit choice, and the refusal.

Importers are tried most specific first. A file that matches none is refused with a message that
names every format this release knows, so a person can tell at once whether their file is the
wrong one or the format is missing. Nothing here writes anything.
"""

from __future__ import annotations

from types import ModuleType

from superlocalmemory.portability import (
    chatgpt,
    claude,
    claude_memory,
    letta,
    markdown,
    mem0,
    okf,
)
from superlocalmemory.portability.model import ImportFormatError, Parsed
from superlocalmemory.portability.readers import Doc

#: Most specific first: ChatGPT and Claude conversations, then Letta, Mem0, OKF, Markdown, text.
IMPORTERS: tuple[ModuleType, ...] = (chatgpt, claude, letta, mem0, okf, markdown, claude_memory)
BY_FORMAT = {m.FORMAT: m for m in IMPORTERS}
_ALIASES = {"chatgpt": "chatgpt", "openai": "chatgpt", "claude": "claude", "claude-memory":
            "claude-memory", "claude_memory": "claude-memory", "mem0": "mem0", "letta": "letta",
            "af": "letta", "okf": "okf", "markdown": "markdown", "md": "markdown"}


def known_formats() -> str:
    return ", ".join(m.LABEL for m in IMPORTERS)


def refusal() -> str:
    return ("That file is not in a format this version of SuperLocalMemory can import. "
            f"It knows: {known_formats()}. "
            "Pick the format by name if you are sure of it.")


def resolve(name: str | None) -> ModuleType | None:
    """The importer for a format name given by a person; ``ImportFormatError`` if unknown."""
    if not name:
        return None
    key = _ALIASES.get(str(name).strip().lower())
    if key is None:
        raise ImportFormatError(f"'{name}' is not a format I know. It knows: {known_formats()}.")
    return BY_FORMAT[key]


def detect(docs: list[Doc]) -> ModuleType | None:
    return next((m for m in IMPORTERS if m.detect(docs)), None)


def parse_any(docs: list[Doc], fmt: str | None = None) -> Parsed:
    """Read ``docs`` with the named importer, or the first that recognises them."""
    importer = resolve(fmt) or detect(docs)
    if importer is None:
        raise ImportFormatError(refusal())
    parsed = importer.parse(docs)
    if not parsed.items:
        word = "conversations" if parsed.conversations else "entries"
        raise ImportFormatError(
            f"That looks like {importer.LABEL}, but there are no {word} in it to import "
            "(everything in it was empty or of a kind that is skipped).")
    return parsed


__all__ = ["BY_FORMAT", "IMPORTERS", "detect", "known_formats", "parse_any", "refusal", "resolve"]
