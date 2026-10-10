"""Where original images live on disk and how they get there."""

from __future__ import annotations

import hashlib
import os
import stat
import time

import pytest

from superlocalmemory.media import files


def _tmp_file(root, data=b"abc"):
    d = files.tmp_dir(root)
    p = d / "x.bin"
    p.write_bytes(data)
    return p, hashlib.sha256(data).hexdigest()


def test_content_address_layout(tmp_path):
    sha = "ab" + "c" * 62
    assert files.original_relpath(sha, "png") == f"ab/{sha}.png"
    assert files.original_path(tmp_path, sha, "png") == tmp_path / "media" / "ab" / f"{sha}.png"
    for bad_sha, bad_ext in (("zz", "png"), (sha, "../x"), (sha, ""), (sha, "p/g")):
        with pytest.raises(ValueError):
            files.original_relpath(bad_sha, bad_ext)


def test_tmp_dir_is_private_and_clears_stale_files_once(tmp_path):
    d = tmp_path / "media" / "tmp"
    d.mkdir(parents=True)
    old, fresh = d / "old.bin", d / "fresh.bin"
    old.write_bytes(b"1")
    fresh.write_bytes(b"2")
    past = time.time() - 7200
    os.utime(old, (past, past))
    got = files.tmp_dir(tmp_path)
    assert got == d and stat.S_IMODE(d.stat().st_mode) == 0o700
    assert not old.exists() and fresh.exists()


def test_place_original_is_atomic_private_and_idempotent(tmp_path):
    src, sha = _tmp_file(tmp_path)
    rel = files.place_original(tmp_path, src, sha, "png")
    dest = tmp_path / "media" / rel
    assert dest.read_bytes() == b"abc" and stat.S_IMODE(dest.stat().st_mode) == 0o600
    assert not src.exists()
    again, _ = _tmp_file(tmp_path)
    assert files.place_original(tmp_path, again, sha, "png") == rel


def test_place_original_never_overwrites_a_different_file(tmp_path):
    src, sha = _tmp_file(tmp_path)
    dest = files.original_path(tmp_path, sha, "png")
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"something else")
    with pytest.raises(FileExistsError):
        files.place_original(tmp_path, src, sha, "png")
    assert dest.read_bytes() == b"something else"


def test_remove_original_stays_inside_the_media_folder(tmp_path):
    src, sha = _tmp_file(tmp_path)
    rel = files.place_original(tmp_path, src, sha, "png")
    outside = tmp_path / "keep.txt"
    outside.write_text("x")
    assert files.remove_original(tmp_path, "../keep.txt") is False
    assert files.remove_original(tmp_path, str(outside)) is False
    assert outside.exists()
    link = tmp_path / "media" / "ab"
    link.mkdir(exist_ok=True)
    (link / ("l" * 64 + ".png")).symlink_to(outside)
    assert files.remove_original(tmp_path, "ab/" + "l" * 64 + ".png") is False
    assert outside.exists()
    assert files.remove_original(tmp_path, rel) is True
    assert not (tmp_path / "media" / rel).exists()
    assert files.remove_original(tmp_path, rel) is False
