# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Housekeeping for the image store.

Three kinds of leftovers are looked for:

* image rows whose memory no longer exists (the row, vectors, file and cached text go);
* files in the originals folder that no row points at (the file goes);
* memories that say they came from an image but have no image row (reported only: the
  words are the person's, so they are never deleted here).

A dry run reports and changes nothing. Files younger than ten minutes are left alone,
because a save may be between writing the file and writing its row.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from superlocalmemory.media import files, media_db_exists, open_media_store

logger = logging.getLogger(__name__)

YOUNG_FILE_S = 600.0
_SKIP_DIRS = ("tmp",)
#: Only a content-addressed original (``ab/ab<62 hex>.ext``) can be a stray. Everything else
#: under media/ (uploads.db and its sidecars, anything a later release adds) is never collected.
_ORIGINAL = re.compile(r"([0-9a-f]{2})/\1[0-9a-f]{62}\.[a-z0-9]{1,8}")


@dataclass
class GcReport:
    dry_run: bool = True
    rows_without_memory: list[str] = field(default_factory=list)
    files_without_row: list[str] = field(default_factory=list)
    memories_without_row: list[str] = field(default_factory=list)
    files_skipped_young: int = 0
    rows_removed: int = 0
    files_removed: int = 0


def _memory_conn(root: Path) -> sqlite3.Connection | None:
    path = root / "memory.db"
    if not path.is_file():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _existing(conn: sqlite3.Connection, profile_id: str, memory_ids: list[str]) -> set[str]:
    found: set[str] = set()
    for i in range(0, len(memory_ids), 400):
        chunk = memory_ids[i:i + 400]
        rows = conn.execute(
            f"SELECT memory_id FROM memories WHERE profile_id = ? AND memory_id IN ({','.join('?' * len(chunk))})",
            [profile_id, *chunk]).fetchall()
        found.update(r[0] for r in rows)
    return found


def _media_memories(conn: sqlite3.Connection, profile_id: str) -> list[tuple[str, str]]:
    """(memory_id, media_id) of the profile's memories saved from an image."""
    out: list[tuple[str, str]] = []
    rows = conn.execute("SELECT memory_id, metadata_json FROM memories WHERE profile_id = ? "
                        "AND metadata_json LIKE '%\"media\"%'", (profile_id,))
    for memory_id, raw in rows:
        try:
            source = (json.loads(raw or "{}") or {}).get("_slm_source") or {}
        except (TypeError, ValueError):
            continue
        if isinstance(source, dict) and source.get("type") == "media":
            out.append((memory_id, str(source.get("media_id") or "")))
    return out


def _stray_files(store, root: Path, now: float) -> tuple[list[Path], int]:
    known = store.known_relpaths()
    base = files.media_root(root)
    strays: list[Path] = []
    young = 0
    if not base.is_dir():
        return strays, young
    for path in sorted(base.rglob("*")):
        rel = path.relative_to(base)
        if rel.parts[0] in _SKIP_DIRS or path.is_symlink() or not path.is_file():
            continue
        if not _ORIGINAL.fullmatch(rel.as_posix()) or rel.as_posix() in known:
            continue
        if now - path.stat().st_mtime < YOUNG_FILE_S:
            young += 1
            continue
        strays.append(path)
    return strays, young


def gc(profile_id: str, dry_run: bool = True, *, data_root: str | Path | None = None) -> GcReport:
    """Find (and, unless ``dry_run``, fix) image leftovers of one profile."""
    report = GcReport(dry_run=dry_run)
    if data_root is None:
        from superlocalmemory.infra.data_root import canonical_data_root

        data_root = canonical_data_root()
    root = Path(data_root)
    if not media_db_exists(root):
        return report
    store = open_media_store(data_root=root)
    if store is None:
        return report
    conn = _memory_conn(root)
    try:
        if conn is not None:
            _memory_side(store, conn, profile_id, report, root)
        strays, report.files_skipped_young = _stray_files(store, root, time.time())
        report.files_without_row = [p.name for p in strays]
        if not dry_run:
            for path in strays:
                if files.remove_original(root, path.relative_to(files.media_root(root)).as_posix()):
                    report.files_removed += 1
    finally:
        if conn is not None:
            conn.close()
        store.close()
    return report


def _memory_side(store, conn: sqlite3.Connection, profile_id: str, report: GcReport, root: Path) -> None:
    from superlocalmemory.media.erasure import erase_items

    every = store.anchors(profile_id)
    anchored = {m: a for m, a in every.items() if a}
    alive = _existing(conn, profile_id, sorted(set(anchored.values())))
    orphans = sorted(m for m, a in anchored.items() if a not in alive)
    report.rows_without_memory = orphans
    report.memories_without_row = sorted(m for m, media_id in _media_memories(conn, profile_id)
                                         if media_id not in every)
    if orphans and not report.dry_run:
        out = erase_items(store, root, orphans)
        report.rows_removed = out["items"]
        report.files_removed += out["files"]
