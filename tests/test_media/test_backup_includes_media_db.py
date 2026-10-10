"""media.db is backed up locally and never leaves the machine through cloud sync."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.infra.backup import MANAGED_DATABASES, BackupManager
from superlocalmemory.infra.cloud_backup import _find_latest_backup_set
from superlocalmemory.media import open_media_store
from tests.helpers.env_capabilities import NO_VECTOR_SEARCH_REASON, vector_search_available


def _memory_db(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "memory.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (x)")
    conn.commit()
    conn.close()
    return path


def test_media_db_is_managed():
    assert "media.db" in MANAGED_DATABASES


@pytest.mark.skipif(not vector_search_available(), reason=NO_VECTOR_SEARCH_REASON)
def test_backup_includes_media_db_when_present(tmp_path):
    root = tmp_path / "slm"
    db = _memory_db(root)
    open_media_store(create=True, data_root=root).close()
    mgr = BackupManager(db, base_dir=root)
    mgr.create_backup()
    names = [p.name for p in (root / "backups").glob("*.db")]
    assert any(n.startswith("media-") for n in names), names


def test_backup_without_media_db_adds_nothing(tmp_path):
    root = tmp_path / "slm"
    db = _memory_db(root)
    BackupManager(db, base_dir=root).create_backup()
    names = [p.name for p in (root / "backups").glob("*.db")]
    assert names and not any(n.startswith("media-") for n in names)
    assert not (root / "media.db").exists()


def test_cloud_sync_set_never_lists_media(tmp_path):
    backups = tmp_path
    for stem in ("memory", "learning", "media"):
        (backups / f"{stem}-20260101-000000.db").write_bytes(b"x")
    names = [p.name for p in _find_latest_backup_set(backups)]
    assert "memory-20260101-000000.db" in names
    assert not any(n.startswith("media-") for n in names)
