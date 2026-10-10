# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Where original images live on disk: ``<data_root>/media/<address[:2]>/<address>.<ext>``.

The address is the hash of the profile id, a NUL byte and the stripped file that
is kept, so the same picture saved twice in one profile is one file while two
profiles never share a file: erasing one profile removes its bytes whatever
another profile saved. Paths handed in from outside are checked to stay inside
the media folder; nothing here follows a link out of it.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from pathlib import Path

_SHA = re.compile(r"[0-9a-f]{64}")
_EXT = re.compile(r"[a-z0-9]{1,8}")
_STALE_S = 3600.0
_swept: set[str] = set()
_sweep_lock = threading.Lock()


def media_root(data_root: str | Path) -> Path:
    return Path(data_root) / "media"


def original_relpath(stored_sha256: str, ext: str) -> str:
    """The address relative to the media folder; refuses anything that is not a hash and a plain extension."""
    if not _SHA.fullmatch(stored_sha256 or "") or not _EXT.fullmatch(ext or ""):
        raise ValueError("invalid content address")
    return f"{stored_sha256[:2]}/{stored_sha256}.{ext}"


def original_path(data_root: str | Path, stored_sha256: str, ext: str) -> Path:
    return media_root(data_root) / original_relpath(stored_sha256, ext)


def tmp_dir(data_root: str | Path) -> Path:
    """Private scratch folder; files older than an hour are removed the first time it is used."""
    path = media_root(data_root) / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    with _sweep_lock:
        first = str(path) not in _swept
        _swept.add(str(path))
    if first:
        _sweep(path)
    return path


def _sweep(path: Path) -> None:
    cutoff = time.time() - _STALE_S
    for entry in path.iterdir():
        try:
            if entry.lstat().st_mtime < cutoff:
                _remove_tree(entry)
        except OSError:
            continue


def _remove_tree(entry: Path) -> None:
    if entry.is_dir() and not entry.is_symlink():
        for child in entry.iterdir():
            _remove_tree(child)
        entry.rmdir()
    else:
        entry.unlink()


def content_address(profile_id: str, path: str | Path) -> str:
    """The address of this profile's copy of the file: sha256(profile id, NUL, bytes)."""
    digest = hashlib.sha256(profile_id.encode("utf-8") + b"\0")
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def planned_path(data_root: str | Path, profile_id: str, tmp_file: str | Path, ext: str) -> Path:
    """Where ``place_original`` would put this profile's copy of the scratch file."""
    return media_root(data_root) / original_relpath(content_address(profile_id, tmp_file), ext)


def place_original(data_root: str | Path, tmp_file: str | Path, profile_id: str, ext: str) -> str:
    """Move a scratch file to its profile's address (atomic, owner-only); returns the relative path.

    A file already at the address is kept when it really is the same content;
    a different one is never overwritten.
    """
    return place_original_noting_new(data_root, tmp_file, profile_id, ext)[0]


def place_original_noting_new(data_root: str | Path, tmp_file: str | Path, profile_id: str,
                              ext: str) -> tuple[str, bool]:
    """:func:`place_original`, also saying whether the file was new, with the content read once."""
    address = content_address(profile_id, tmp_file)
    rel = original_relpath(address, ext)
    dest = media_root(data_root) / rel
    src = Path(tmp_file)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or content_address(profile_id, dest) != address:
            raise FileExistsError("a different file already has this address")
        src.unlink(missing_ok=True)
        return rel, False
    os.chmod(src, 0o600)
    os.replace(src, dest)
    return rel, True


def remove_original(data_root: str | Path, relpath: str) -> bool:
    """Delete one original. False (and nothing deleted) for a path outside the media folder or a link."""
    if not relpath or os.path.isabs(relpath) or ".." in Path(relpath).parts:
        return False
    root = media_root(data_root)
    target = root / relpath
    current = root
    for part in Path(relpath).parts:
        current = current / part
        if current.is_symlink():
            return False
    if not target.is_file():
        return False
    target.unlink()
    return True
