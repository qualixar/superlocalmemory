# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Stage 8 SB-5

"""Path-level opt-out via ``.slmignore``.

A repository-scoped escape hatch: drop a ``.slmignore`` at any
ancestor of a workspace and SLM will skip hooks / recall / remember
for any path inside that ancestor. Roughly the shape of ``.gitignore``
but with simpler matching — one path pattern per line, ``#`` comments,
whitespace stripped.

Matching rules (intentionally minimal):

- Lines starting with ``#`` or empty ⇒ ignored.
- A line like ``node_modules`` matches any path segment named
  ``node_modules`` anywhere in the resolved absolute path.
- A line starting with ``/`` is an absolute-prefix match
  (``/Users/me/secret`` matches everything under that dir).
- Glob chars ``*`` / ``?`` are treated literally — keep it boring,
  avoid re-implementing ``.gitignore``'s subtleties.

Cache: the parsed ignore list is memoised per ignore-file path + mtime.
Look-up cost at the hook hot path is O(depth × patterns) and
patterns ≤ 50 in practice.
"""

from __future__ import annotations

import os
from pathlib import Path

_FILENAME = ".slmignore"
_CACHE: dict[tuple[str, float], tuple[str, ...]] = {}
_CACHE_CAP = 64


def _load_patterns(ignore_path: Path) -> tuple[str, ...]:
    """Parse one ``.slmignore`` file, memoised by path + mtime."""
    try:
        stat = ignore_path.stat()
    except OSError:
        return ()
    key = (str(ignore_path), stat.st_mtime)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    try:
        raw = ignore_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ()
    patterns: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped)
    result = tuple(patterns)
    if len(_CACHE) >= _CACHE_CAP:
        _CACHE.clear()
    _CACHE[key] = result
    return result


def _iter_ancestor_ignores(target: Path) -> list[tuple[Path, tuple[str, ...]]]:
    """Walk from ``target`` up to the filesystem root collecting ignore files."""
    hits: list[tuple[Path, tuple[str, ...]]] = []
    seen_dirs: set[Path] = set()
    probe = target if target.is_dir() else target.parent
    while probe not in seen_dirs:
        seen_dirs.add(probe)
        candidate = probe / _FILENAME
        if candidate.is_file():
            patterns = _load_patterns(candidate)
            if patterns:
                hits.append((candidate.parent, patterns))
        if probe.parent == probe:
            break
        probe = probe.parent
    return hits


def path_is_ignored(target: str | Path) -> bool:
    """Return True iff any ancestor ``.slmignore`` ignores the given path.

    Absolute paths are resolved (symlinks preserved — we match on name,
    not realpath, so a symlink into an ignored dir still matches).
    Non-existent targets are allowed; we walk the theoretical ancestry.
    """
    p = Path(target)
    try:
        abs_path = p.resolve(strict=False)
    except OSError:
        abs_path = p
    segments = set(abs_path.parts)
    for ignore_dir, patterns in _iter_ancestor_ignores(abs_path):
        for pat in patterns:
            if pat.startswith("/"):
                # Absolute prefix: the ignore file's directory provides the
                # anchor for relative-looking absolute patterns.
                candidate = Path(pat)
                try:
                    abs_path.relative_to(candidate)
                    return True
                except ValueError:
                    continue
            else:
                # Match any path segment.
                if pat in segments:
                    return True
                # Also honour a per-ignore-dir relative path.
                rel = abs_path.relative_to(ignore_dir) if (
                    ignore_dir in abs_path.parents or ignore_dir == abs_path
                ) else None
                if rel is not None and pat in set(rel.parts):
                    return True
    return False


def clear_cache() -> None:
    """Test-only helper: drop the memoised pattern cache."""
    _CACHE.clear()


__all__ = ("path_is_ignored", "clear_cache")
