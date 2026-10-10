# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Folders that can never be a source, and the check that a file stays inside its root."""

from __future__ import annotations

import os
import re
from pathlib import Path, PureWindowsPath
from typing import Mapping

_CREDENTIAL_DIRS = (".ssh", ".aws", ".gnupg")
_NETWORK_FS = re.compile(r"^(?:nfs\d*|cifs|smb\w*|fuse\.sshfs|afs|9p|ncpfs)$")
#: Folders inside ``~/Library`` that hold documents a user may keep a vault in.
_LIBRARY_DOC_DIRS = frozenset({"Mobile Documents", "CloudStorage"})


class RootRefused(ValueError):
    """The folder cannot be a source. ``code`` is stable for callers and tests."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def is_inside(root: str | Path, candidate: str | Path) -> bool:
    """True when ``candidate`` resolves to ``root`` or a path below it (symlinks followed)."""
    base = os.path.realpath(root)
    target = os.path.realpath(candidate)
    return target == base or target.startswith(base.rstrip(os.sep) + os.sep)


def _read_mounts() -> str:
    try:
        return Path("/proc/mounts").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _unescape(field: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


def network_mount_type(path: str, mounts: str) -> str | None:
    """The network file system holding ``path`` per a ``/proc/mounts`` listing, if any."""
    best_len, best_type = -1, None
    for line in mounts.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        point = _unescape(fields[1]).rstrip("/") or "/"
        inside = point == "/" or path == point or path.startswith(point + "/")
        if inside and len(point) > best_len:
            best_len, best_type = len(point), fields[2]
    return best_type if best_type and _NETWORK_FS.match(best_type) else None


def _refuse_windows(raw: str) -> None:
    win = PureWindowsPath(raw)
    if raw.startswith("\\\\") or raw.startswith("//"):
        raise RootRefused("network_share", "Network shares cannot be used as a source.")
    if win.anchor and win == PureWindowsPath(win.anchor):
        raise RootRefused("filesystem_root", "A whole drive cannot be a source; pick a folder.")


def _system_dirs(home: Path, environ: Mapping[str, str]) -> list[tuple[Path, bool]]:
    dirs = [(home / "Library", True)]
    if environ.get("APPDATA"):
        dirs.append((Path(environ["APPDATA"]), True))
    return dirs


def _is_system_folder(real: Path, home: Path, environ: Mapping[str, str]) -> bool:
    for folder, _ in _system_dirs(home, environ):
        base = Path(os.path.realpath(folder))
        if real == base:
            return True
        if base in real.parents:
            first = real.relative_to(base).parts[0]
            if folder.name == "Library" and first in _LIBRARY_DOC_DIRS:
                continue
            return True
    return False


def check_root(path: str | Path, *, home: Path | None = None,
               environ: Mapping[str, str] | None = None, mounts: str | None = None,
               windows: bool = False) -> Path:
    """The resolved folder when it may be a source; raises :class:`RootRefused` otherwise.

    The checks run on the path after ``realpath``, so a link to the home folder is
    refused as the home folder. A root that is itself a link to somewhere outside
    its own parent folder is refused too.
    """
    raw = str(path)
    if windows:
        _refuse_windows(raw)
    environ = os.environ if environ is None else environ
    given = Path(os.path.abspath(raw))
    real = Path(os.path.realpath(given))
    if not real.is_dir():
        raise RootRefused("not_a_folder", "That path is not a folder that exists.")
    home_real = Path(os.path.realpath(home if home is not None else Path.home()))
    if real == home_real:
        raise RootRefused("home_directory", "Your whole home folder cannot be a source; pick a subfolder.")
    if real == Path(real.anchor) or real.parent == real:
        raise RootRefused("filesystem_root", "A whole drive or the filesystem root cannot be a source.")
    if given.is_symlink() and not is_inside(given.parent, real):
        raise RootRefused("symlink_escape", "That folder is a link to somewhere outside its own location.")
    if _is_system_folder(real, home_real, environ):
        raise RootRefused("system_folder", "System and application data folders cannot be a source.")
    for name in _CREDENTIAL_DIRS:
        if (real / name).exists():
            raise RootRefused("holds_credentials", f"That folder holds {name}, which keeps credentials.")
    mount_type = network_mount_type(str(real), _read_mounts() if mounts is None else mounts)
    if mount_type:
        raise RootRefused("network_share", "Network shares cannot be used as a source in this version.")
    return real


__all__ = ["RootRefused", "check_root", "is_inside", "network_mount_type"]
