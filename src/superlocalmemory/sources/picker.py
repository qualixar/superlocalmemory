# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Choosing a folder on the computer SuperLocalMemory runs on.

``pick_folder`` opens the computer's own folder dialog and returns the path chosen, or ``None`` when
the person cancelled. ``suggestions`` lists folders worth offering with one click. Neither reads a
file; both only return folder paths. The commands are fixed argument lists: no text from a caller
ever reaches them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

from superlocalmemory.infra.data_root import overlaps_data_root

PICKER_TIMEOUT_S = 120
MAX_DEPTH = 3
MAX_DIRS_VISITED = 200
MAX_VAULTS = 20
_SKIP = frozenset({"library", "node_modules", "appdata", "applications", "site-packages"})

_MAC = ["osascript", "-e",
        'POSIX path of (choose folder with prompt "Choose a folder for SuperLocalMemory")']
_WINDOWS = ["powershell", "-NoProfile", "-STA", "-Command",
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$d.Description = 'Choose a folder for SuperLocalMemory'; "
            "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.SelectedPath }"]
_ZENITY = ["zenity", "--file-selection", "--directory", "--title=Choose a folder for SuperLocalMemory"]
_KDIALOG = ["kdialog", "--getexistingdirectory", "."]

#: ``runner(argv, timeout_s) -> (exit_code, stdout)``; tests replace it.
Runner = Callable[[list[str], int], tuple[int, str]]


class PickerUnavailable(Exception):
    """No folder dialog can be opened on this computer."""


class PickerBusy(Exception):
    """A folder dialog is already open."""


_lock = threading.Lock()


def _run(argv: list[str], timeout_s: int) -> tuple[int, str]:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s,  # noqa: S603
                              shell=False, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return 1, ""
    return done.returncode, done.stdout


def _commands() -> list[list[str]]:
    """The dialog commands to try on this computer, best first (empty when there is none)."""
    if sys.platform == "darwin":
        return [list(_MAC)] if shutil.which("osascript") else []
    if sys.platform.startswith("win"):
        return [list(_WINDOWS)] if shutil.which("powershell") else []
    found = []
    if shutil.which("zenity"):
        found.append(list(_ZENITY))
    if shutil.which("kdialog"):
        found.append(list(_KDIALOG))
    return found


def _clean(out: str) -> str | None:
    path = out.strip()
    if len(path) > 1 and path.endswith(("/", "\\")) and not path.endswith((":\\", ":/")):
        path = path[:-1]
    return path or None


def pick_folder(runner: Runner | None = None) -> str | None:
    """Open the folder dialog and wait (at most two minutes). The chosen path, or ``None`` if cancelled."""
    commands = _commands()
    if not commands:
        raise PickerUnavailable("No folder dialog is available on this computer.")
    if not _lock.acquire(blocking=False):
        raise PickerBusy("A folder dialog is already open.")
    try:
        run = runner or _run
        for argv in commands:
            try:
                code, out = run(argv, PICKER_TIMEOUT_S)
            except OSError:
                continue
            return _clean(out) if code == 0 else None
        raise PickerUnavailable("No folder dialog could be opened on this computer.")
    finally:
        _lock.release()


def _children(path: str) -> list[os.DirEntry]:
    try:
        with os.scandir(path) as it:
            return [e for e in it if e.is_dir(follow_symlinks=False)]
    except OSError:
        return []


def _find_vaults(home: Path) -> list[str]:
    """Folders under ``home`` that hold ``.obsidian``: breadth first, bounded, never into hidden folders."""
    vaults: list[str] = []
    queue: list[tuple[str, int]] = [(str(home), 0)]
    visited = 0
    while queue and visited < MAX_DIRS_VISITED and len(vaults) < MAX_VAULTS:
        path, depth = queue.pop(0)
        visited += 1
        kids = _children(path)
        names = {e.name for e in kids}
        if ".obsidian" in names:
            vaults.append(path)
        if depth >= MAX_DEPTH:
            continue
        for e in sorted(kids, key=lambda k: k.name):
            if not e.name.startswith(".") and e.name.lower() not in _SKIP:
                queue.append((e.path, depth + 1))
    return sorted(vaults)


def suggestions() -> list[dict[str, str]]:
    """Folders to offer with one click: Obsidian vaults under the home folder, then Documents and Desktop."""
    home = Path.home()
    found = [{"path": p, "kind": "obsidian", "name": os.path.basename(p) or p} for p in _find_vaults(home)]
    for name, kind in (("Documents", "documents"), ("Desktop", "desktop")):
        path = home / name
        if path.is_dir() and not path.is_symlink():
            found.append({"path": str(path), "kind": kind, "name": name})
    seen: set[str] = set()
    out = []
    for item in found:
        if item["path"] in seen or overlaps_data_root(item["path"]):
            continue
        seen.add(item["path"])
        out.append(item)
    return out
