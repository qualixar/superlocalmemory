# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""What the scanner skips: fixed rules first, then ``.gitignore`` and ``.slmignore``.

The matcher is a subset of gitignore: ``#`` comments, ``!`` negation, a trailing ``/``
for folders, a leading ``/`` (or any ``/`` inside the pattern) to anchor, ``*``, ``?``,
``[...]`` classes and ``**``. The last matching pattern wins, and a file inside an
ignored folder stays ignored. ``.slmignore`` is read after ``.gitignore`` in the same
folder, so it can override it. The fixed rules are checked first and cannot be negated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TYPES: tuple[str, ...] = (
    ".md", ".markdown", ".txt", ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".canvas")
MAX_FILES = 50_000
_TEXT_TYPES = frozenset({".md", ".markdown", ".txt", ".canvas"})
_SIZE_CAPS = {"text": 5 * 1024 * 1024, "pdf": 50 * 1024 * 1024, "image": 25 * 1024 * 1024}
_EXCLUDED_DIRS = frozenset({"node_modules", "venv", ".venv", "__pycache__", "site-packages"})
_SECRET_NAME = re.compile(r"(?i)^(?:.*\.(?:pem|key|kdbx|p12|pfx|db|sqlite)|id_.*)$")
_TEMP_NAME = re.compile(r"(?i)^(?:.*\.(?:tmp|swp|crdownload|part)|~\$.*|\.#.*)$")
_IGNORE_FILES = (".gitignore", ".slmignore")
_MAX_IGNORE_BYTES = 256 * 1024


def kind_of(name: str) -> str:
    """``text``, ``pdf`` or ``image`` for a file name (by extension)."""
    ext = Path(name).suffix.lower()
    if ext in _TEXT_TYPES:
        return "text"
    return "pdf" if ext == ".pdf" else "image"


def _glob_to_regex(pattern: str) -> str:
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern[i:i + 2] == "**":
                after = pattern[i + 2:i + 3]
                before_ok = i == 0 or pattern[i - 1] == "/"
                if before_ok and after == "/":
                    out.append("(?:.*/)?")
                    i += 3
                    continue
                if before_ok and after == "":
                    out.append(".*")
                    i += 2
                    continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = pattern.find("]", i + 2 if pattern[i + 1:i + 2] in "!^" else i + 1)
            if j == -1:
                out.append(r"\[")
            else:
                body = pattern[i + 1:j]
                neg = body[:1] in ("!", "^")
                body = body[1:] if neg else body
                body = body.replace("\\", "\\\\")
                out.append("[" + ("^" if neg else "") + body + "]")
                i = j
        elif c == "\\" and i + 1 < n:
            i += 1
            out.append(re.escape(pattern[i]))
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


@dataclass(frozen=True)
class _Pattern:
    regex: re.Pattern[str]
    negate: bool
    dir_only: bool


def _parse_line(line: str) -> _Pattern | None:
    line = line.rstrip("\r\n")
    stripped = line.rstrip(" ")
    if stripped.endswith("\\") and len(stripped) < len(line):
        stripped = line[:len(stripped) + 1]
    line = stripped
    if not line or line.startswith("#"):
        return None
    negate = line.startswith("!")
    if negate:
        line = line[1:]
    dir_only = line.endswith("/") and not line.endswith("\\/")
    line = line.rstrip("/") if dir_only else line
    if not line:
        return None
    anchored = "/" in line
    line = line.lstrip("/")
    body = _glob_to_regex(line)
    prefix = "" if anchored else "(?:.*/)?"
    try:
        return _Pattern(re.compile(prefix + body + r"\Z"), negate, dir_only)
    except re.error:
        return None


class GitIgnore:
    """One ignore file's patterns, matched against paths relative to its folder."""

    def __init__(self, patterns: list[_Pattern]) -> None:
        self._patterns = patterns

    @classmethod
    def parse(cls, text: str) -> "GitIgnore":
        return cls([p for p in map(_parse_line, text.splitlines()) if p])

    def verdict(self, relpath: str, is_dir: bool) -> bool | None:
        """True (ignored), False (re-included) or None (no pattern matched)."""
        result: bool | None = None
        for pat in self._patterns:
            if pat.dir_only and not is_dir:
                continue
            if pat.regex.match(relpath):
                result = not pat.negate
        return result

    def ignored(self, relpath: str, is_dir: bool = False) -> bool:
        parts = relpath.split("/")
        for k in range(1, len(parts)):
            if self.verdict("/".join(parts[:k]), True):
                return True
        return bool(self.verdict(relpath, is_dir))


class IgnoreRules:
    """The skip decision for every path under one source root."""

    def __init__(self, root: Path, include_types: tuple[str, ...] = DEFAULT_TYPES) -> None:
        self._root = Path(root)
        self._types = frozenset(t.lower() for t in include_types)
        self._layers: list[tuple[str, GitIgnore]] = []

    def enter_dir(self, rel_dir: str) -> None:
        """Read the ``.gitignore`` then ``.slmignore`` of a folder about to be scanned."""
        patterns: list[_Pattern] = []
        for name in _IGNORE_FILES:
            try:
                path = self._root / rel_dir / name if rel_dir else self._root / name
                if path.is_symlink() or path.stat().st_size > _MAX_IGNORE_BYTES:
                    continue
                patterns += GitIgnore.parse(path.read_text(encoding="utf-8"))._patterns
            except (OSError, ValueError):
                continue
        if patterns:
            self._layers = [x for x in self._layers if x[0] != rel_dir]
            self._layers.append((rel_dir, GitIgnore(patterns)))

    def _from_files(self, relpath: str, is_dir: bool) -> bool:
        result = False
        for base, layer in self._layers:
            if base:
                if not relpath.startswith(base + "/"):
                    continue
                inner = relpath[len(base) + 1:]
            else:
                inner = relpath
            verdict = layer.verdict(inner, is_dir)
            if verdict is not None:
                result = verdict
        return result

    def skip_reason(self, relpath: str, *, is_dir: bool, size: int = 0) -> str | None:
        """Why a path is skipped, or ``None`` to scan it. The first matching rule wins."""
        name = relpath.rsplit("/", 1)[-1]
        if name.startswith("."):
            return "dotfile"
        if is_dir:
            if name in _EXCLUDED_DIRS:
                return "excluded_dir"
            return "ignored" if self._from_files(relpath, True) else None
        if _SECRET_NAME.match(name):
            return "secret_name"
        if self._from_files(relpath, False):
            return "ignored"
        if Path(name).suffix.lower() not in self._types:
            return "type_not_allowed"
        if size > _SIZE_CAPS[kind_of(name)]:
            return "too_large"
        return "temp_file" if _TEMP_NAME.match(name) else None


__all__ = ["DEFAULT_TYPES", "GitIgnore", "IgnoreRules", "MAX_FILES", "kind_of"]
