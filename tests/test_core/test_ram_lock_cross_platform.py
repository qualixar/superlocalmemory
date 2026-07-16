# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Platform contract for the RAM-reservation file lock."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

from superlocalmemory.core import ram_lock


def test_ram_lock_uses_windows_stdlib_byte_range_lock_when_fcntl_is_absent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Windows must import and reserve without the POSIX-only ``fcntl``."""
    calls: list[tuple[int, int, int]] = []
    fake_msvcrt = types.SimpleNamespace(
        LK_NBLCK=1,
        LK_UNLCK=2,
        locking=lambda fd, mode, length: calls.append((fd, mode, length)),
    )
    monkeypatch.setattr(ram_lock, "_fcntl", None)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)

    path = tmp_path / "ram_lock.sem"
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        ram_lock._acquire_lock(fd)
        ram_lock._release_lock(fd)
    finally:
        os.close(fd)

    assert [mode for _, mode, _ in calls] == [
        fake_msvcrt.LK_NBLCK,
        fake_msvcrt.LK_UNLCK,
    ]
    assert path.read_bytes() == b"\0"
