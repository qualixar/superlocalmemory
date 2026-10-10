# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Erasing a memory also erases the image saved with it.

``MediaErasureOwner`` sits beside the keyword, time and vector owners in the
erasure service, so the erasure receipt names the image store like any other
store. It acts at once (no grace period): the row, the vectors, the thumbnail,
the original file (unless another row still uses the same file) and the cached
text read from it. When ``media.db`` does not exist nothing is opened or created.

Nothing here logs memory text or file locations.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Iterable

from superlocalmemory.core.transactions.owners import (
    OperationContext, OwnerErasureProof, OwnerHealth, OwnerResult,
)
from superlocalmemory.media import files, media_db_exists, open_media_store

logger = logging.getLogger(__name__)

OWNER_NAME = "media"
_CHUNK = 400


def _checksum(*parts: str) -> str:
    return hashlib.sha256("\0".join([OWNER_NAME, *parts]).encode("utf-8")).hexdigest()


def _gone(root: Path, relpath: str | None) -> bool:
    if not relpath:
        return True
    target = files.media_root(root) / relpath
    return not (target.exists() or target.is_symlink())


def erase_items(store: Any, root: Path, media_ids: Iterable[str]) -> dict[str, Any]:
    """Remove the items, then every file and cached result nothing else uses.

    Returns ``{"items", "files", "cache_entries", "residue"}``; ``residue`` names what could
    not be removed (item ids and short content hashes, never paths).
    """
    removed = store.erase_items(list(media_ids))
    out: dict[str, Any] = {"items": len(removed), "files": 0, "cache_entries": 0, "residue": []}
    seen: set[tuple[Any, Any]] = set()
    for item in removed:
        sha, rel = item["stored_sha256"], item["original_relpath"]
        if (sha, rel) in seen or store.file_in_use(sha, rel):
            continue
        seen.add((sha, rel))
        if rel and not _gone(root, rel):
            files.remove_original(root, rel)
            if _gone(root, rel):
                out["files"] += 1
            else:
                out["residue"].append(f"file:{(sha or '')[:12]}")
        if sha:
            try:
                from superlocalmemory.cache.factory import invalidate_content

                out["cache_entries"] += invalidate_content(sha, root)
            except Exception as exc:  # noqa: BLE001 - reported, the erasure is not complete
                logger.warning("cached text could not be cleared (%s)", type(exc).__name__)
                out["residue"].append(f"cache:{sha[:12]}")
    return out


def erase_for_memories(root: str | Path, profile_id: str, memory_ids: Iterable[str]) -> dict[str, Any]:
    """Erase the images anchored to these memories of one profile. No media.db: nothing happens."""
    root = Path(root)
    out: dict[str, Any] = {"items": 0, "files": 0, "cache_entries": 0, "residue": []}
    store = open_media_store(data_root=root)
    if store is None:
        return out
    try:
        ids = store.item_ids_for_anchors(profile_id, list(memory_ids))
        return erase_items(store, root, ids) if ids else out
    finally:
        store.close()


def erase_profile(root: str | Path, profile_id: str) -> dict[str, Any]:
    """Erase every image row, vector, file and cached text of one profile."""
    root = Path(root)
    out: dict[str, Any] = {"items": 0, "files": 0, "cache_entries": 0, "residue": []}
    store = open_media_store(data_root=root)
    if store is None:
        return out
    try:
        out = erase_items(store, root, store.all_item_ids(profile_id))
        store.delete_profile_rows(profile_id)
        return out
    finally:
        store.close()


def erase_profile_into(counts: dict, root: str | Path | None, profile_id: str) -> None:
    """The profile wipe's hook: record the outcome in ``counts``; a failure blocks completeness."""
    if root is None:
        return
    try:
        out = erase_profile(root, profile_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("image erase failed for a profile (%s)", type(exc).__name__)
        counts["media_failed"] = 1
        return
    if out["items"]:
        counts["media_items"] = out["items"]
    if out["residue"]:
        counts["media_failed"] = 1


class MediaErasureOwner:
    """The image store's part of an erasure; same shape as the other owners."""

    name = OWNER_NAME

    def __init__(self, db: Any, *, data_root: str | Path | None = None) -> None:
        self._db = db
        self._root = Path(data_root) if data_root is not None else None
        self._memories: dict[str, set[str]] = {}
        self._left: dict[str, list[tuple[str, str]]] = {}

    def _data_root(self) -> Path | None:
        if self._root is not None:
            return self._root
        path = getattr(self._db, "db_path", None)
        return Path(path).parent if path else None

    def _active_root(self) -> Path | None:
        root = self._data_root()
        return root if root is not None and media_db_exists(root) else None

    def _memory_ids(self, context: OperationContext) -> set[str]:
        """Memories of the erased facts. Facts still exist while owners run; later calls reuse the answer."""
        found = self._memories.setdefault(context.operation_id, set())
        ids = list(context.fact_ids)
        for i in range(0, len(ids), _CHUNK):
            chunk = ids[i:i + _CHUNK]
            rows = self._db.execute(
                "SELECT DISTINCT memory_id FROM atomic_facts WHERE profile_id = ? "
                f"AND fact_id IN ({','.join('?' * len(chunk))})", (context.profile_id, *chunk))
            for row in rows:
                value = dict(row).get("memory_id") if hasattr(row, "keys") else row[0]
                if value:
                    found.add(str(value))
        return found

    def erase(self, context: OperationContext) -> OwnerErasureProof:
        root = self._active_root()
        if root is None:
            return OwnerErasureProof(OWNER_NAME, True, _checksum("none"))
        residue: list[str] = []
        detail: dict[str, Any] = {}
        try:
            memories = self._memory_ids(context)
            store = open_media_store(data_root=root)
            if store is not None:
                try:
                    ids = store.item_ids_for_anchors(context.profile_id, sorted(memories))
                    rows = store.items_by_id(ids)
                    out = erase_items(store, root, ids)
                finally:
                    store.close()
                residue = list(out["residue"])
                self._left[context.operation_id] = [
                    (r["stored_sha256"] or "", r["original_relpath"] or "") for r in rows]
                detail["items"] = out["items"]
        except Exception as exc:  # noqa: BLE001 - fail closed: the receipt says it is not erased
            logger.warning("image erase failed (%s)", type(exc).__name__)
            residue.append("media:error")
        if residue:
            detail["residue"] = sorted(residue)
        return OwnerErasureProof(OWNER_NAME, not residue, _checksum("erase", *sorted(residue)), detail)

    def prove_erased(self, context: OperationContext) -> OwnerErasureProof:
        root = self._active_root()
        if root is None:
            return OwnerErasureProof(OWNER_NAME, True, _checksum("none"))
        residue: list[str] = []
        try:
            store = open_media_store(data_root=root)
            if store is not None:
                try:
                    ids = store.item_ids_for_anchors(context.profile_id, sorted(self._memory_ids(context)))
                    residue.extend(f"media:{i}" for i in ids)
                    for sha, rel in self._left.get(context.operation_id, []):
                        if not store.file_in_use(sha, rel) and not _gone(root, rel):
                            residue.append(f"file:{sha[:12]}")
                finally:
                    store.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("image erase proof failed (%s)", type(exc).__name__)
            residue.append("media:error")
        detail = {"residue": sorted(residue)} if residue else {}
        return OwnerErasureProof(OWNER_NAME, not residue, _checksum("prove", *sorted(residue)), detail)

    def apply(self, context: OperationContext) -> OwnerResult:
        return OwnerResult(OWNER_NAME, True, _checksum("apply"))

    verify = apply
    compensate = apply

    def health(self) -> OwnerHealth:
        return OwnerHealth(OWNER_NAME, True)


def scrub_snapshot(db_path: str | Path, profile_id: str) -> dict[str, int]:
    """Remove one profile from a copy of media.db and its originals from the copy's media folder.

    The folder is ``<dir of the copy>/media``; a file goes unless a remaining row still uses it.
    Returns ``{"media_items": rows removed, "media_files": files removed}``.
    """
    from superlocalmemory.media.store import MediaStore

    db_path = Path(db_path)
    store = MediaStore(db_path)
    try:
        rows = store._read().execute(
            "SELECT stored_sha256, original_relpath FROM media_items WHERE profile_id = ?",
            (profile_id,)).fetchall()
        store.delete_profile_rows(profile_id)
        removed_files = 0
        for sha, rel in {(r[0], r[1]) for r in rows}:
            if not store.file_in_use(sha, rel) and rel and files.remove_original(db_path.parent, rel):
                removed_files += 1
        store._w.execute("PRAGMA secure_delete=ON")
        store._w.execute("VACUUM")
        store._w.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    finally:
        store.close()
    return {"media_items": len(rows), "media_files": removed_files}
