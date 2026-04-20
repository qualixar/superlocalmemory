# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Stage 8 F5 (Mediums/Lows)

"""Stage 8 F5 regression — ram_lock parent-dir 0700.

SEC-M6: the ``ram_lock.sem`` audit marker contains ``{pid}:{name}``.
The lockfile itself is 0600, but the PARENT dir was 0755 on first
creation. Tighten to 0700 so other UIDs cannot enumerate which SLM
subsystem holds the lock.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest


if sys.platform == "win32":  # pragma: no cover
    pytest.skip("SEC-M6 targets POSIX only", allow_module_level=True)


def test_sec_m6_ram_lock_parent_dir_is_0700(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superlocalmemory.core import ram_lock

    # Redirect the lockfile into a fresh subdir so we can assert the
    # chmod happened on first acquisition.
    parent = tmp_path / "slm-home"
    lock = parent / "ram_lock.sem"
    monkeypatch.setattr(ram_lock, "RAM_LOCK_PATH", lock)

    with ram_lock.ram_reservation("stage8-m6", required_mb=1):
        assert parent.is_dir()
        mode = stat.S_IMODE(parent.stat().st_mode)
        assert mode == 0o700, oct(mode)
    # After release, dir perm is stable.
    mode = stat.S_IMODE(parent.stat().st_mode)
    assert mode == 0o700, oct(mode)
