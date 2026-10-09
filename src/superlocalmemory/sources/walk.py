# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Walk a source folder read-only: what is there, what is skipped and why, what is cloud-only."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from superlocalmemory.sources.ignore import MAX_FILES, IgnoreRules
from superlocalmemory.sources.roots import is_inside

#: Windows: recall-on-data-access, recall-on-open, offline.
_WINDOWS_CLOUD_ATTRS = 0x400000 | 0x40000 | 0x1000
#: macOS ``SF_DATALESS``. Value taken from the system headers; confirm on a Mac before relying on it.
SF_DATALESS = 0x40000000


def is_placeholder(st: Any) -> bool:
    """True when the file's content is not on this machine (a cloud placeholder)."""
    attrs = getattr(st, "st_file_attributes", None)
    if attrs is not None and attrs & _WINDOWS_CLOUD_ATTRS:
        return True
    flags = getattr(st, "st_flags", None)
    return flags is not None and bool(flags & SF_DATALESS)


@dataclass(frozen=True)
class Entry:
    relpath: str
    size: int
    mtime_ns: int
    file_id: str
    placeholder: bool = False

    def signature(self) -> tuple[int, int, str]:
        return (self.size, self.mtime_ns, self.file_id)


@dataclass
class WalkResult:
    entries: list[Entry] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    unreadable: list[str] = field(default_factory=list)
    capped: bool = False

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def under_unreadable(self, relpath: str) -> bool:
        return any(relpath == d or relpath.startswith(d + "/") or d == "" for d in self.unreadable)


def stat_entry(root: Path, relpath: str) -> Entry | None:
    """A fresh look at one file, or None when it is gone."""
    try:
        st = os.stat(root / relpath)
    except OSError:
        return None
    return Entry(relpath, st.st_size, st.st_mtime_ns, f"{st.st_dev}:{st.st_ino}", is_placeholder(st))


def _classify(root: Path, dirpath: str, item: os.DirEntry, rules: IgnoreRules,
              result: WalkResult) -> tuple[str, Entry | None]:
    """``("dir", None)``, ``("file", Entry)`` or ``("skip", None)`` for one directory item."""
    rel = f"{dirpath}/{item.name}" if dirpath else item.name
    try:
        symlink = item.is_symlink()
        if symlink and not is_inside(root, item.path):
            result.skip("symlink_escape")
            return "skip", None
        is_dir = item.is_dir(follow_symlinks=False)
        if symlink:
            result.skip("symlink_dir" if item.is_dir() else "symlink_file")
            return "skip", None
        st = item.stat()
        if not st.st_ino:  # Windows listings carry no file id; ask the file itself
            st = os.stat(item.path)
    except OSError:
        result.skip("unreadable")
        return "skip", None
    if not is_dir and not item.is_file():
        result.skip("not_a_file")
        return "skip", None
    reason = rules.skip_reason(rel, is_dir=is_dir, size=st.st_size)
    if reason:
        result.skip(reason)
        return "skip", None
    if is_dir:
        return "dir", None
    return "file", Entry(rel, st.st_size, st.st_mtime_ns, f"{st.st_dev}:{st.st_ino}", is_placeholder(st))


def walk_tree(root: Path, rules: IgnoreRules) -> WalkResult:
    """Every candidate file under ``root``. Raises OSError when the root itself cannot be listed."""
    result = WalkResult()
    stack = [""]
    while stack:
        dirpath = stack.pop()
        rules.enter_dir(dirpath)
        try:
            with os.scandir(root / dirpath if dirpath else root) as it:
                items = sorted(it, key=lambda e: e.name)
        except OSError:
            if not dirpath:
                raise
            result.unreadable.append(dirpath)
            continue
        subdirs = []
        for item in items:
            kind, entry = _classify(root, dirpath, item, rules, result)
            if kind == "dir":
                subdirs.append(f"{dirpath}/{item.name}" if dirpath else item.name)
            elif entry is not None:
                if len(result.entries) >= MAX_FILES:
                    result.capped = True
                    return result
                result.entries.append(entry)
        stack.extend(reversed(subdirs))
    return result


__all__ = ["Entry", "MAX_FILES", "SF_DATALESS", "WalkResult", "is_placeholder", "stat_entry", "walk_tree"]
