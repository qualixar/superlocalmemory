"""Fable audit, package FB: backups hold content-addressed originals and never the upload-link database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.infra import backup_media
from superlocalmemory.media import files, open_media_store
from tests.helpers.env_capabilities import NO_VECTOR_SEARCH_REASON, vector_search_available

SHA = "ab" + "c" * 62
SIDECARS = ("uploads.db", "uploads.db-wal", "uploads.db-shm")


def _library(tmp_path: Path) -> tuple[Path, Path, str]:
    """A data folder with one real original and the upload-link database beside it (live, with sidecars)."""
    root, backup = tmp_path / "slm", tmp_path / "backups"
    media = root / "media"
    (media / SHA[:2]).mkdir(parents=True)
    rel = f"{SHA[:2]}/{SHA}.png"
    (media / rel).write_bytes(b"PIXELS")
    for name in SIDECARS:
        (media / name).write_bytes(b"live upload links " + name.encode())
    (media / "tmp").mkdir()
    (media / "tmp" / "half-written").write_bytes(b"x")
    (media / "notes.txt").write_bytes(b"not an original")
    sqlite3.connect(root / "media.db").close()
    return root, backup, rel


def test_is_original_accepts_only_a_content_address():
    assert files.is_original(f"{SHA[:2]}/{SHA}.png") and files.is_original(f"{SHA[:2]}/{SHA}.pdf")
    for rel in ("uploads.db", "uploads.db-wal", "uploads.db-shm", "tmp/x", "notes.txt", f"zz/{SHA}.png",
                f"{SHA[:2]}/{SHA}", f"de/{SHA}.png", f"{SHA[:2]}/{SHA}.png/extra", f"../{SHA[:2]}/{SHA}.png"):
        assert not files.is_original(rel), rel


def test_a_backup_mirrors_the_originals_and_nothing_else(tmp_path):
    root, backup, rel = _library(tmp_path)
    report = backup_media.sync_originals(root, backup)
    mirror = backup / backup_media.MIRROR_DIR
    found = sorted(p.relative_to(mirror).as_posix() for p in mirror.rglob("*") if p.is_file())
    assert found == [rel] and report.files == 1 and report.copied == 1


def test_a_restore_never_writes_the_upload_link_database(tmp_path):
    """Older backups may already hold a copy of uploads.db; putting it back would revive expired links."""
    root, backup, rel = _library(tmp_path)
    mirror = backup / backup_media.MIRROR_DIR
    (mirror / SHA[:2]).mkdir(parents=True)
    (mirror / rel).write_bytes(b"PIXELS")
    for name in (*SIDECARS, "notes.txt"):
        (mirror / name).write_bytes(b"old copy")
    for path in (root / "media").iterdir():
        if path.is_file():
            path.unlink()
    (root / "media" / rel).unlink(missing_ok=True)
    assert backup_media.restore_originals(root, backup) == 1
    assert (root / "media" / rel).read_bytes() == b"PIXELS"
    assert [p.name for p in (root / "media").iterdir() if p.is_file()] == []


@pytest.mark.skipif(not vector_search_available(), reason=NO_VECTOR_SEARCH_REASON)
def test_a_copy_of_the_upload_database_left_in_the_mirror_by_an_older_backup_is_pruned(tmp_path):
    root, backup, rel = _library(tmp_path)
    (root / "media.db").unlink()
    store = open_media_store(create=True, data_root=root)
    try:
        store.insert_item(profile_id="p1", kind="image", source_sha256=SHA, stored_sha256=SHA, mime="image/png",
                          bytes=6, origin="tool", original_relpath=rel)
    finally:
        store.close()
    mirror = backup / backup_media.MIRROR_DIR
    mirror.mkdir(parents=True)
    for name in SIDECARS:
        (mirror / name).write_bytes(b"old copy")
    backup_media.sync_originals(root, backup)
    assert sorted(p.relative_to(mirror).as_posix() for p in mirror.rglob("*") if p.is_file()) == [rel]


# -- CX1 residual: a picture saved while its memory was queued is linked to it without a manual gc ----

import asyncio  # noqa: E402

from superlocalmemory.media.gc import fill_anchors  # noqa: E402
from tests.test_media._erase_support import add_image, make_root  # noqa: E402


def _queued_picture_library(tmp_path):
    """One picture still without an anchor (its memory exists now), plus an orphan row and a stray file."""
    root, db, store = make_root(tmp_path)
    add_image(root, store, media_id="d" * 32, memory_id=None, data=b"queued")
    db.add_memory("m7", media_id="d" * 32)
    add_image(root, store, media_id="e" * 32, memory_id="deleted", data=b"orphan")  # rows without a memory
    stray = files.media_root(root) / "ab" / ("ab" + "0" * 62 + ".png")
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"x")
    import os
    import time

    os.utime(stray, (time.time() - 3600, time.time() - 3600))
    return root, store, stray


def test_fill_anchors_links_the_queued_pictures_of_every_profile_and_deletes_nothing(tmp_path):
    root, store, stray = _queued_picture_library(tmp_path)
    assert fill_anchors(data_root=root) == 1
    assert store.anchors("p1")["d" * 32] == "m7"
    assert "e" * 32 in store.anchors("p1") and stray.exists()  # gc's deletions are never done here
    assert fill_anchors(data_root=root) == 0  # nothing left to link: a no-op
    store.close()


def test_fill_anchors_without_a_library_is_a_no_op(tmp_path):
    assert fill_anchors(data_root=tmp_path / "nothing") == 0


def test_the_housekeeping_pass_links_queued_pictures_and_survives_a_failure(tmp_path):
    from superlocalmemory.server import media_housekeeping as hk

    root, store, _ = _queued_picture_library(tmp_path)
    calls = {"n": 0}

    def fill(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk busy")
        return fill_anchors(**kw)

    async def drive():
        task = asyncio.create_task(hk.run(root, fill=fill, interval_s=0.01, first_delay_s=0.0))
        await asyncio.sleep(0.3)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(drive())
    assert calls["n"] >= 2 and store.anchors("p1")["d" * 32] == "m7"
    store.close()


def test_the_daemon_starts_and_stops_the_media_housekeeping_loop():
    from superlocalmemory.server import media_housekeeping as hk

    assert 0 < hk.INTERVAL_S <= 3600
    source = (Path(hk.__file__).parent / "unified_daemon.py").read_text()
    assert "media_housekeeping" in source and source.count("_media_housekeeping_task") >= 3
