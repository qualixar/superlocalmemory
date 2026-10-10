# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Picture and document moves that a profile delete could not finish.

A profile delete commits the memory move first; the images and documents
(media.db) follow. When that second step fails, the delete is not undone: the
move is written here and finished later, at daemon start and before any
profile is created. Until then the name stays reserved so a new profile cannot
inherit the old one's pictures.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

_log = logging.getLogger("superlocalmemory.pending_media_moves")
_LOCK = threading.Lock()
_FILE = "pending_media_moves.json"
_TABLES = ("media_items", "documents", "jobs", "sources")


def _path(data_root: Path) -> Path:
    return Path(data_root) / _FILE


def _read(data_root: Path) -> list[dict]:
    try:
        raw = json.loads(_path(data_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [m for m in raw if isinstance(m, dict) and m.get("from") and m.get("to")] \
        if isinstance(raw, list) else []


def _write(data_root: Path, moves: list[dict]) -> None:
    target = _path(data_root)
    if not moves:
        target.unlink(missing_ok=True)
        return
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(moves), encoding="utf-8")
    os.replace(tmp, target)


def record(data_root: Path, from_profile: str, to_profile: str) -> None:
    """Remember that ``from_profile``'s pictures still have to move to ``to_profile``."""
    with _LOCK:
        moves = [m for m in _read(data_root) if m["from"] != from_profile]
        moves.append({"from": from_profile, "to": to_profile})
        _write(data_root, moves)


def pending(data_root: Path) -> list[dict]:
    with _LOCK:
        return _read(data_root)


def is_pending(data_root: Path, profile: str) -> bool:
    return any(m["from"] == profile for m in pending(data_root))


def retry(data_root: Path) -> list[dict]:
    """Finish every recorded move that can be finished; returns those still pending."""
    from superlocalmemory.storage.profile_fold_sidecars import move_media

    with _LOCK:
        left: list[dict] = []
        for move in _read(data_root):
            try:
                move_media(Path(data_root), move["from"], move["to"])
            except Exception as exc:  # noqa: BLE001 -- stays recorded for the next try
                _log.warning("picture move %s -> %s still pending: %s", move["from"], move["to"], exc)
                left.append(move)
        _write(data_root, left)
        return left


def owns_media_rows(data_root: Path, profile: str) -> bool:
    """True when media.db still holds a row filed under ``profile``."""
    from superlocalmemory.media import open_media_store

    store = open_media_store(data_root=Path(data_root))
    if store is None:
        return False
    try:
        conn = store._read()
        return any(conn.execute(f"SELECT 1 FROM {t} WHERE profile_id = ? LIMIT 1", (profile,)).fetchone()
                   for t in _TABLES)
    finally:
        store.close()


def creation_blocker(data_root: Path, profile: str) -> str:
    """Why a profile named ``profile`` cannot be created yet ("" when it can).

    Pending moves are retried first, so a transient failure does not block for long.
    """
    retry(data_root)
    if is_pending(data_root, profile) or owns_media_rows(data_root, profile):
        return (f"Pictures and documents of a deleted profile named '{profile}' have not finished "
                "moving yet. Try again shortly, or pick another name.")
    return ""


__all__ = ["creation_blocker", "is_pending", "owns_media_rows", "pending", "record", "retry"]
