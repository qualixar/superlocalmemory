"""A coherent backup set keeps the picture and PDF originals too, like every other local backup.

``BackupManager`` already mirrors the originals beside its per-file copies; a ``BackupCoordinator``
set copied the databases only, so a set could not rebuild the library.
"""

from __future__ import annotations

import sqlite3
import stat
from pathlib import Path

import pytest

from superlocalmemory.infra import backup_media
from superlocalmemory.infra.backup import MANAGED_DATABASES, BackupCoordinator, BackupManager
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


def _picture(root: Path, store, sha: str, data: bytes = b"PIXELS") -> str:
    rel = f"{sha[:2]}/{sha}.png"
    path = root / "media" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    store.insert_item(profile_id="p1", kind="image", source_sha256=sha, stored_sha256=sha,
                      mime="image/png", bytes=len(data), origin="tool", original_relpath=rel)
    return rel


@pytest.fixture()
def lib(tmp_path):
    root = tmp_path / "slm"
    _memory_db(root)
    store = open_media_store(create=True, data_root=root)
    yield root, store
    store.close()


def _coordinator(root: Path, databases=MANAGED_DATABASES) -> BackupCoordinator:
    return BackupCoordinator(databases, root, root / "backups")


def _mirror(root: Path) -> Path:
    return root / "backups" / backup_media.MIRROR_DIR


def test_a_backup_set_keeps_a_copy_of_every_original(lib):
    root, store = lib
    rel, rel2 = _picture(root, store, SHA, b"one"), _picture(root, store, SHA2, b"two")
    _coordinator(root).create_backup_set()
    assert (_mirror(root) / rel).read_bytes() == b"one" and (_mirror(root) / rel2).read_bytes() == b"two"
    assert stat.S_IMODE((_mirror(root) / rel).stat().st_mode) == 0o600


def test_the_set_and_the_per_file_backup_share_one_mirror(lib):
    root, store = lib
    rel = _picture(root, store, SHA, b"one")
    BackupManager(root / "memory.db", base_dir=root).create_backup()
    first = (_mirror(root) / rel).stat().st_mtime_ns
    _coordinator(root).create_backup_set()
    assert (_mirror(root) / rel).stat().st_mtime_ns == first, "the second backup copied it again"
    assert [p.name for p in _mirror(root).rglob("*") if p.is_file()] == [f"{SHA}.png"]


def test_a_set_without_pictures_adds_no_mirror(tmp_path):
    root = tmp_path / "slm"
    _memory_db(root)
    _coordinator(root).create_backup_set()
    assert not _mirror(root).exists()


def test_the_mirror_does_not_look_like_a_backup_set(lib):
    root, store = lib
    _picture(root, store, SHA)
    manifest = _coordinator(root).create_backup_set()
    sets = [p.name for p in (root / "backups").iterdir() if (p / "manifest.json").exists()]
    assert sets == [f"backup_{manifest.set_id}"]


def test_a_failing_copy_does_not_fail_the_set(lib, monkeypatch):
    root, store = lib
    _picture(root, store, SHA)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(backup_media, "sync_originals", boom)
    manifest = _coordinator(root).create_backup_set()
    assert manifest.verified and (root / "backups" / f"backup_{manifest.set_id}" / "manifest.json").exists()


def test_restoring_a_set_puts_the_originals_back(lib):
    root, store = lib
    rel = _picture(root, store, SHA, b"one")
    coordinator = _coordinator(root)
    manifest = coordinator.create_backup_set()
    store.close()
    (root / "media" / rel).unlink()
    coordinator.restore_from_manifest(manifest)
    assert (root / "media" / rel).read_bytes() == b"one"


def test_restoring_a_set_never_overwrites_a_file_that_is_already_there(lib):
    root, store = lib
    rel = _picture(root, store, SHA, b"one")
    coordinator = _coordinator(root)
    manifest = coordinator.create_backup_set()
    store.close()
    (root / "media" / rel).write_bytes(b"newer")
    coordinator.restore_from_manifest(manifest)
    assert (root / "media" / rel).read_bytes() == b"newer"


def test_a_set_without_the_library_leaves_the_media_folder_alone(lib):
    root, store = lib
    rel = _picture(root, store, SHA, b"one")
    _coordinator(root).create_backup_set()  # the mirror now holds the picture
    only_memory = _coordinator(root, ("memory.db",))
    manifest = only_memory.create_backup_set()
    (root / "media" / rel).unlink()
    only_memory.restore_from_manifest(manifest)
    assert not (root / "media" / rel).exists()
