"""Windows-like conditions: no inode numbers from directory listings, and binary-mode opens."""

from __future__ import annotations

import hashlib
import os
import types

from superlocalmemory.sources import safe_read, walk


class _NoInode:
    """A directory entry whose stat() has no inode or device, as Windows listings do."""

    def __init__(self, real: os.DirEntry) -> None:
        self._real = real

    def stat(self, *, follow_symlinks: bool = True):
        st = self._real.stat(follow_symlinks=follow_symlinks)
        return types.SimpleNamespace(st_size=st.st_size, st_mtime_ns=st.st_mtime_ns, st_mode=st.st_mode,
                                     st_ino=0, st_dev=0)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_files_are_read_when_the_listing_has_no_inode(env, monkeypatch):
    real_scandir = os.scandir

    class Wrapped:
        def __init__(self, path):
            self._it = real_scandir(path)

        def __enter__(self):
            return [_NoInode(e) for e in self._it.__enter__()]

        def __exit__(self, *a):
            return self._it.__exit__(*a)

    monkeypatch.setattr(walk.os, "scandir", Wrapped)
    env.write("a.md", "note a")
    sid = env.add_and_confirm()
    stats = env.scan(sid)
    assert stats.errors == 0 and len(env.runtime.saved) == 1
    assert env.files(sid)["a.md"]["file_id"] not in (None, "0:0")


def test_open_flags_include_binary_mode_when_the_platform_has_it(monkeypatch):
    monkeypatch.setattr(os, "O_BINARY", 0x10000000, raising=False)
    assert safe_read._flags() & 0x10000000


def test_binary_bytes_hash_exactly(tmp_path):
    data = b"a\r\nb\x1a\r\nc\x00\r\n" * 10
    path = tmp_path / "f.bin"
    path.write_bytes(data)
    assert hashlib.sha256(safe_read.read_bounded(path, 10_000)).hexdigest() == hashlib.sha256(data).hexdigest()
