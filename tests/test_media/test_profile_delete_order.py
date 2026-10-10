"""Deleting a profile moves its pictures only once its memories have moved."""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.media import open_media_store
from superlocalmemory.server.routes import helpers
from superlocalmemory.storage import migration_runner as mr
from superlocalmemory.storage import profile_fold, schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import MemoryRecord


@pytest.fixture
def root(tmp_path: Path, monkeypatch) -> Path:
    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(real_schema)
    for pid in ("alice", "default"):
        mgr.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)", (pid, pid))
    mgr.store_memory(MemoryRecord(memory_id="m1", profile_id="alice", content="Alice info"))
    mr.apply_all(tmp_path / "learning.db", tmp_path / "memory.db")
    store = open_media_store(create=True, data_root=tmp_path)
    store.insert_item(profile_id="alice", kind="image", source_sha256="a" * 64, mime="image/png", bytes=1,
                      origin="tool", anchor_memory_id="m1")
    store.close()
    monkeypatch.setattr(helpers, "DB_PATH", tmp_path / "memory.db")
    return tmp_path


def _owners(root: Path) -> list[str]:
    store = open_media_store(data_root=root)
    try:
        return [r[0] for r in store._read().execute("SELECT profile_id FROM media_items")]
    finally:
        store.close()


def test_a_failed_memory_move_leaves_the_pictures_with_their_profile(root, monkeypatch):
    def broken(*_a, **_k):
        raise profile_fold.ProfileFoldError("injected")

    monkeypatch.setattr(profile_fold, "fold_profile", broken)
    with pytest.raises(profile_fold.ProfileFoldError):
        helpers.delete_profile_from_db("alice")
    assert _owners(root) == ["alice"]


def test_pictures_move_once_the_memories_have(root):
    helpers.delete_profile_from_db("alice")
    assert _owners(root) == ["default"]
