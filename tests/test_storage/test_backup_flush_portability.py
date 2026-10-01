"""Windows durability needs a write-capable staging descriptor, not ignored errors."""

from __future__ import annotations

import errno
import os
import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage import backup


def _source(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE proof(value TEXT)")
        conn.execute("INSERT INTO proof VALUES ('synthetic snapshot')")


def test_snapshot_flush_uses_write_capable_descriptor(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "snapshot.db"
    _source(source)
    original = source.read_bytes()
    real_open, real_fsync = os.open, os.fsync
    flags_by_fd: dict[int, int] = {}
    flushed: list[int] = []

    def opened(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        flags_by_fd[fd] = flags
        return fd

    def windows_flush(fd):
        if flags_by_fd.get(fd, 0) & os.O_RDWR != os.O_RDWR:
            raise OSError(errno.EBADF, "Windows requires write access to flush")
        flushed.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(backup.os, "open", opened)
    monkeypatch.setattr(backup.os, "fsync", windows_flush)
    backup._backup_via_sqlite_api(source, target)
    assert flushed
    assert source.read_bytes() == original
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT value FROM proof").fetchone()[0] == "synthetic snapshot"
    assert not target.with_name(target.name + ".partial").exists()


def test_flush_failure_cannot_publish_snapshot(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "snapshot.db"
    _source(source)
    original = source.read_bytes()

    def failed(fd):
        raise OSError(errno.EIO, "synthetic durable flush failure")

    monkeypatch.setattr(backup.os, "fsync", failed)
    with pytest.raises(OSError):
        backup._backup_via_sqlite_api(source, target)
    assert not target.exists()
    assert source.read_bytes() == original
