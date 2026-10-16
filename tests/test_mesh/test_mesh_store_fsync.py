# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""mesh.db is flushed through a handle Windows can fsync.

Windows' fsync refuses a read-only handle (EBADF); the 4.2.8 Windows CI run
failed every mesh.db build on it. The flags are checked here so a Mac or Linux
run catches the regression too.
"""

from __future__ import annotations

import os
from pathlib import Path

from superlocalmemory.mesh import mesh_store


def test_the_file_is_opened_for_writing_before_fsync(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "mesh.db"
    target.write_bytes(b"x")
    seen: list[int] = []
    real_open = os.open

    def spy(path, flags, *args):
        seen.append(flags)
        return real_open(path, flags, *args)

    monkeypatch.setattr(mesh_store.os, "open", spy)
    mesh_store._fsync(target)
    assert seen and seen[0] & os.O_RDWR == os.O_RDWR
    assert target.read_bytes() == b"x"  # flushing never changes the file
