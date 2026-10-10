"""Local backups keep the picture and PDF originals, and a restore puts them back (audit F3)."""

from __future__ import annotations

import logging
import os
import sqlite3
import stat
from pathlib import Path

import pytest

from superlocalmemory.infra import backup_media
from superlocalmemory.infra.backup import BackupManager
from superlocalmemory.infra.cloud_backup import _find_latest_backup_set
from superlocalmemory.media import open_media_store
from tests.helpers.env_capabilities import NO_VECTOR_SEARCH_REASON, vector_search_available

pytestmark = pytest.mark.skipif(not vector_search_available(), reason=NO_VECTOR_SEARCH_REASON)

SHA = "ab" + "c" * 62
SHA2 = "de" + "f" * 62


def _memory_db(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "memory.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (x)")
    conn.commit()
    conn.close()
    return path


def _picture(root: Path, store, sha: str, data: bytes = b"PIXELS", *, row: bool = True) -> str:
    rel = f"{sha[:2]}/{sha}.png"
    path = root / "media" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if row:
        store.insert_item(profile_id="p1", kind="image", source_sha256=sha, stored_sha256=sha,
                          mime="image/png", bytes=len(data), origin="tool", original_relpath=rel)
    return rel


@pytest.fixture()
def lib(tmp_path):
    root = tmp_path / "slm"
    db = _memory_db(root)
    store = open_media_store(create=True, data_root=root)
    yield root, db, store
    store.close()


def _mirror(root: Path) -> Path:
    return root / "backups" / backup_media.MIRROR_DIR


def test_a_backup_keeps_a_copy_of_every_original(lib):
    root, db, store = lib
    rel = _picture(root, store, SHA, b"one")
    rel2 = _picture(root, store, SHA2, b"two")
    BackupManager(db, base_dir=root).create_backup()
    assert (_mirror(root) / rel).read_bytes() == b"one" and (_mirror(root) / rel2).read_bytes() == b"two"
    assert stat.S_IMODE((_mirror(root) / rel).stat().st_mode) == 0o600


def test_scratch_files_are_not_backed_up(lib):
    root, db, store = lib
    _picture(root, store, SHA)
    (root / "media" / "tmp").mkdir(exist_ok=True)
    (root / "media" / "tmp" / "half-written").write_bytes(b"x")
    BackupManager(db, base_dir=root).create_backup()
    assert not (_mirror(root) / "tmp").exists()


def test_without_images_nothing_is_added(tmp_path):
    root = tmp_path / "slm"
    db = _memory_db(root)
    BackupManager(db, base_dir=root).create_backup()
    assert not _mirror(root).exists()


def test_with_a_library_but_no_files_nothing_is_added(lib):
    root, db, _ = lib
    BackupManager(db, base_dir=root).create_backup()
    assert not _mirror(root).exists()


def test_a_second_backup_copies_only_what_is_new(lib, caplog):
    root, db, store = lib
    _picture(root, store, SHA)
    mgr = BackupManager(db, base_dir=root)
    mgr.create_backup()
    first = (_mirror(root) / f"{SHA[:2]}/{SHA}.png").stat().st_mtime_ns
    _picture(root, store, SHA2)
    with caplog.at_level(logging.INFO):
        mgr.create_backup()
    assert (_mirror(root) / f"{SHA[:2]}/{SHA}.png").stat().st_mtime_ns == first
    assert (_mirror(root) / f"{SHA2[:2]}/{SHA2}.png").exists()
    assert any("1 new" in r.getMessage() and "2 files" in r.getMessage() for r in caplog.records)


def test_the_size_is_logged(lib, caplog):
    root, db, store = lib
    _picture(root, store, SHA, b"x" * 2048)
    with caplog.at_level(logging.INFO):
        BackupManager(db, base_dir=root).create_backup()
    assert any("media originals" in r.getMessage() and "KB" in r.getMessage() for r in caplog.records)


def test_an_original_that_was_erased_leaves_the_mirror(lib):
    root, db, store = lib
    rel = _picture(root, store, SHA)
    mgr = BackupManager(db, base_dir=root)
    mgr.create_backup()
    (root / "media" / rel).unlink()
    with sqlite3.connect(root / "media.db") as conn:
        conn.execute("DELETE FROM media_items")
    mgr.create_backup()
    assert not (_mirror(root) / rel).exists()


def test_a_missing_file_that_the_library_still_lists_is_kept_in_the_mirror(lib):
    """If the media folder is lost, the next backup must not erase the only good copy."""
    root, db, store = lib
    rel = _picture(root, store, SHA, b"precious")
    mgr = BackupManager(db, base_dir=root)
    mgr.create_backup()
    (root / "media" / rel).unlink()
    mgr.create_backup()
    assert (_mirror(root) / rel).read_bytes() == b"precious"


def test_a_failing_copy_does_not_fail_the_backup(lib, monkeypatch):
    root, db, store = lib
    _picture(root, store, SHA)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(backup_media, "sync_originals", boom)
    assert BackupManager(db, base_dir=root).create_backup().startswith("memory-")


def test_restoring_the_library_puts_the_originals_back(lib):
    root, db, store = lib
    rel = _picture(root, store, SHA, b"one")
    mgr = BackupManager(db, base_dir=root)
    name = mgr.create_backup()
    media_backup = next(p.name for p in (root / "backups").glob("media-*.db"))
    store.close()
    (root / "media" / rel).unlink()
    assert mgr.restore_backup(media_backup) is True
    assert (root / "media" / rel).read_bytes() == b"one"
    assert name


def test_a_restore_never_overwrites_a_file_that_is_already_there(lib):
    root, db, store = lib
    rel = _picture(root, store, SHA, b"one")
    mgr = BackupManager(db, base_dir=root)
    mgr.create_backup()
    media_backup = next(p.name for p in (root / "backups").glob("media-*.db"))
    (root / "media" / rel).write_bytes(b"newer")
    assert mgr.restore_backup(media_backup) is True
    assert (root / "media" / rel).read_bytes() == b"newer"


def test_restoring_another_store_leaves_the_media_folder_alone(lib):
    root, db, store = lib
    rel = _picture(root, store, SHA)
    mgr = BackupManager(db, base_dir=root)
    name = mgr.create_backup()
    (root / "media" / rel).unlink()
    assert mgr.restore_backup(name) is True
    assert not (root / "media" / rel).exists()


def test_restore_ignores_anything_in_the_mirror_that_is_not_a_plain_file(lib, tmp_path):
    root, db, store = lib
    mirror = _mirror(root)
    (mirror / "ab").mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    os.symlink(outside, mirror / "ab" / "link.png")
    assert backup_media.restore_originals(root, root / "backups") == 0
    assert not (root / "media" / "ab" / "link.png").exists()


def test_the_cloud_set_never_lists_the_mirror(lib):
    root, db, store = lib
    _picture(root, store, SHA)
    BackupManager(db, base_dir=root).create_backup(label="cloud-sync")
    names = [p.name for p in _find_latest_backup_set(root / "backups")]
    assert names and not any(n.startswith("media") or n == backup_media.MIRROR_DIR for n in names)
