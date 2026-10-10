# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The picture and PDF originals in a local backup.

``media.db`` holds the library's records and thumbnails, but the originals live as
files under ``<data root>/media/<aa>/<address>.<ext>``. A backup of the database alone
cannot rebuild the library. So each local backup also keeps a mirror of those files in
``<backup dir>/media-originals/`` and a restore of ``media.db`` copies missing files back.

Why a mirror and not an archive per backup: originals are content addressed and never
change, so one copy of each is enough, and keeping ten timestamped archives of a library
that can reach 2 GB would cost ten times the space. A file leaves the mirror only when it
is gone from the media folder *and* ``media.db`` no longer lists it, so erasing a picture
also erases its backup copy, while a lost media folder never empties the mirror.

Cloud backup never reads this folder: pictures stay on this computer.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MIRROR_DIR = "media-originals"
_SKIP = "tmp"  # scratch files of saves in progress


@dataclass(frozen=True)
class MirrorReport:
    files: int = 0
    copied: int = 0
    removed: int = 0
    bytes: int = 0


def _human(size: int) -> str:
    if size >= 1024 ** 3:
        return f"{size / 1024 ** 3:.1f} GB"
    if size >= 1024 ** 2:
        return f"{size / 1024 ** 2:.1f} MB"
    return f"{max(1, size // 1024)} KB"


def _plain_files(base: Path) -> dict[str, Path]:
    """Every regular, non-link file under ``base`` (scratch folder skipped), by posix relative path."""
    found: dict[str, Path] = {}
    if not base.is_dir():
        return found
    for path in base.rglob("*"):
        rel = path.relative_to(base)
        if rel.parts[0] == _SKIP or path.is_symlink() or not path.is_file():
            continue
        found[rel.as_posix()] = path
    return found


def _copy_private(src: Path, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(dest.parent, 0o700)
    tmp = dest.with_name(dest.name + ".part")
    try:
        shutil.copyfile(src, tmp)
        os.chmod(tmp, 0o600)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest.stat().st_size


def _listed(slm_dir: Path) -> set[str] | None:
    """What media.db lists, or None when it cannot be read (then nothing is removed)."""
    from superlocalmemory.media import open_media_store

    try:
        store = open_media_store(data_root=slm_dir)
    except Exception as exc:  # noqa: BLE001 - an unreadable library must not cost a backup copy
        logger.warning("media originals: library unreadable, nothing pruned (%s)", type(exc).__name__)
        return None
    if store is None:
        return None
    try:
        return store.known_relpaths()
    finally:
        store.close()


def _prune(mirror_files: dict[str, Path], present: dict[str, Path], listed: set[str] | None) -> int:
    if listed is None:
        return 0
    removed = 0
    for rel, path in mirror_files.items():
        if rel in present or rel in listed:
            continue
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def sync_originals(slm_dir: Path, backup_dir: Path) -> MirrorReport:
    """Bring the mirror up to date; copies only what is new. Skips entirely with no library or no files."""
    slm_dir, backup_dir = Path(slm_dir), Path(backup_dir)
    present = _plain_files(slm_dir / "media") if (slm_dir / "media.db").exists() else {}
    mirror = backup_dir / MIRROR_DIR
    if not present and not mirror.is_dir():
        return MirrorReport()
    copied = 0
    for rel, src in present.items():
        dest = mirror / rel
        if dest.is_file() and dest.stat().st_size == src.stat().st_size:
            continue
        _copy_private(src, dest)
        copied += 1
    kept = _plain_files(mirror)
    removed = _prune(kept, present, _listed(slm_dir)) if mirror.is_dir() else 0
    total = sum(p.stat().st_size for p in _plain_files(mirror).values()) if mirror.is_dir() else 0
    report = MirrorReport(files=len(present), copied=copied, removed=removed, bytes=total)
    logger.info("media originals: %d files, %s in the backup (%d new, %d removed)",
                report.files, _human(total), copied, removed)
    return report


def restore_originals(slm_dir: Path, backup_dir: Path) -> int:
    """Copy back the originals the media folder lacks. Never overwrites; returns how many were put back."""
    slm_dir, backup_dir = Path(slm_dir), Path(backup_dir)
    media = slm_dir / "media"
    restored = 0
    for rel, src in _plain_files(backup_dir / MIRROR_DIR).items():
        dest = media / rel
        if ".." in Path(rel).parts or dest.exists():
            continue
        _copy_private(src, dest)
        restored += 1
    if restored:
        logger.info("media originals: %d files put back", restored)
    return restored
