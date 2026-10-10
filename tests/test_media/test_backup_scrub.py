"""A backup snapshot must not keep an erased image."""

from __future__ import annotations

import shutil
import sqlite3

from superlocalmemory.infra.backup_obligations import (
    erase_profile_from_snapshot,
    scan_backup_snapshots_for_profile,
)
from superlocalmemory.media import files

from tests.test_media._erase_support import add_image, make_root


def _snapshot(root, store, name="media-20260101-000000.db"):
    backups = root / "backups"
    backups.mkdir(exist_ok=True)
    store._w.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    shutil.copy(root / "media.db", backups / name)
    return backups, backups / name


def test_the_obligation_scan_sees_a_media_snapshot(tmp_path):
    root, db, store = make_root(tmp_path)
    add_image(root, store, media_id="a" * 32, memory_id="m1")
    backups, snap = _snapshot(root, store)
    store.close()
    hits = scan_backup_snapshots_for_profile(backups, "p1")
    assert [h[0] for h in hits] == [str(snap)]
    assert scan_backup_snapshots_for_profile(backups, "other") == []


def test_the_obligation_scan_sees_media_in_a_backup_set(tmp_path):
    root, db, store = make_root(tmp_path)
    add_image(root, store, media_id="a" * 32, memory_id="m1")
    backups, snap = _snapshot(root, store)
    store.close()
    setdir = backups / "backup_0001"
    setdir.mkdir()
    (setdir / "manifest.json").write_text("{}")
    shutil.move(snap, setdir / "media.db")
    assert [h[0] for h in scan_backup_snapshots_for_profile(backups, "p1")] == [str(setdir)]


def test_the_scrub_removes_rows_vectors_and_the_erased_original(tmp_path):
    root, db, store = make_root(tmp_path)
    _, rel1 = add_image(root, store, media_id="a" * 32, profile="p1", memory_id="m1", data=b"one")
    _, rel2 = add_image(root, store, media_id="b" * 32, profile="p2", memory_id="m2", data=b"two")
    backups, snap = _snapshot(root, store)
    store.close()
    setdir = backups / "backup_0001"
    (setdir / "media").mkdir(parents=True)
    for rel in (rel1, rel2):  # the snapshot holds a copy of the originals folder
        dest = setdir / "media" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"copy")
    shutil.move(snap, setdir / "media.db")
    out = erase_profile_from_snapshot(setdir / "media.db", "p1")
    assert out.get("media_items") == 1
    assert not (setdir / "media" / rel1).exists()
    assert (setdir / "media" / rel2).exists()
    conn = sqlite3.connect(setdir / "media.db")
    assert [r[0] for r in conn.execute("SELECT media_id FROM media_items")] == ["b" * 32]
    assert conn.execute("SELECT COUNT(*) FROM media_vector_rows").fetchone()[0] == 1
    conn.close()
