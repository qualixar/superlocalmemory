# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""mesh.db (4.2.8: the mesh's own SQLite file) is in every backup, restore and export path.

Real temp directories and real SQLite files. The mesh.db here is built by the
product's own ``prepare_mesh_db`` and kept open in WAL mode with committed rows
that exist only in its ``-wal``, which is how the daemon holds it.
"""

from __future__ import annotations

import sqlite3
import tarfile
from pathlib import Path

import pytest

from superlocalmemory.infra.backup import (
    MANAGED_DATABASES,
    BackupCoordinator,
    BackupManager,
)
from superlocalmemory.infra.cloud_backup import _find_latest_backup_set
from superlocalmemory.mesh import mesh_store


def _memory_db(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "memory.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO atomic_facts VALUES ('f1', 'kept')")
    conn.commit()
    conn.close()
    return path


def _mesh(root: Path) -> Path:
    return mesh_store.prepare_mesh_db(root / "memory.db")


def _add_peer(path: Path, peer: str, *, keep_open: bool = False):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute(
        "INSERT INTO mesh_peers (peer_id, session_id, registered_at, last_heartbeat,"
        " peer_key) VALUES (?, ?, datetime('now'), datetime('now'), 'k')", (peer, peer))
    conn.commit()
    if keep_open:
        return conn
    conn.close()
    return None


def _peers(path: Path) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return [r[0] for r in conn.execute("SELECT peer_id FROM mesh_peers ORDER BY peer_id")]
    finally:
        conn.close()


@pytest.fixture
def root(tmp_path):
    base = tmp_path / "slm"
    _memory_db(base)
    _mesh(base)
    return base


def _coord(base: Path) -> BackupCoordinator:
    return BackupCoordinator(MANAGED_DATABASES, base, base / "backups")


def test_mesh_db_is_managed():
    assert "mesh.db" in MANAGED_DATABASES


def test_backup_set_captures_uncheckpointed_mesh_rows(root):
    held = _add_peer(root / "mesh.db", "p-wal", keep_open=True)
    try:
        assert (root / "mesh.db-wal").stat().st_size > 0  # the row lives only in the -wal
        manifest = _coord(root).create_backup_set()
    finally:
        held.close()
    entry = next(e for e in manifest.stores if e.store_name == "mesh.db")
    assert _peers(Path(entry.file_path)) == ["p-wal"]


def test_restore_puts_mesh_db_back(root):
    coord = _coord(root)
    _add_peer(root / "mesh.db", "p-before")
    manifest = coord.create_backup_set()
    _add_peer(root / "mesh.db", "p-after")
    held = _add_peer(root / "mesh.db", "p-after-wal", keep_open=True)
    try:
        coord.restore_from_manifest(manifest)
    finally:
        held.close()
    assert _peers(root / "mesh.db") == ["p-before"]


def test_old_backup_without_mesh_db_still_restores_and_mesh_is_rebuilt(tmp_path):
    base = tmp_path / "slm"
    _memory_db(base)  # a 4.2.7 store: no mesh.db at all
    coord = _coord(base)
    manifest = coord.create_backup_set()
    assert "mesh.db" not in {e.store_name for e in manifest.stores}

    conn = sqlite3.connect(base / "memory.db")
    conn.execute("UPDATE atomic_facts SET content = 'changed'")
    conn.commit()
    conn.close()
    coord.restore_from_manifest(manifest)  # must not raise

    conn = sqlite3.connect(base / "memory.db")
    assert conn.execute("SELECT content FROM atomic_facts").fetchone()[0] == "kept"
    conn.close()
    path = mesh_store.prepare_mesh_db(base / "memory.db")  # what the daemon does at start
    assert mesh_store.inspect(path)["status"] == "ok"


def test_old_backup_leaves_a_live_mesh_db_usable(root):
    coord = _coord(root)
    old = coord.create_backup_set()
    stores = tuple(e for e in old.stores if e.store_name != "mesh.db")
    old = type(old)(old.set_id, old.epoch, stores, coord._compute_manifest_hash(stores), True)
    _add_peer(root / "mesh.db", "p-live")
    coord.restore_from_manifest(old)
    assert mesh_store.inspect(root / "mesh.db")["status"] == "ok"
    assert _peers(root / "mesh.db") == ["p-live"]


def test_backup_manager_backs_up_and_restores_mesh_db(root):
    mgr = BackupManager(root / "memory.db", base_dir=root)
    _add_peer(root / "mesh.db", "p-one")
    mgr.create_backup()
    names = sorted(p.name for p in (root / "backups").glob("mesh-*.db"))
    assert len(names) == 1
    _add_peer(root / "mesh.db", "p-two")
    assert mgr.restore_backup(names[0]) is True
    assert _peers(root / "mesh.db") == ["p-one"]


def test_retention_covers_mesh_backups(root):
    mgr = BackupManager(root / "memory.db", base_dir=root)
    mgr.config["max_backups"] = 1
    for stamp in ("a", "b", "c"):
        mgr.create_backup(label=stamp)
    assert len(list((root / "backups").glob("mesh-*.db"))) == 1


def test_cloud_sync_never_uploads_mesh_backups(tmp_path):
    for stem in ("memory", "mesh"):
        (tmp_path / f"{stem}-20260101-000000.db").write_bytes(b"x")
    names = [p.name for p in _find_latest_backup_set(tmp_path)]
    assert names == ["memory-20260101-000000.db"]


def test_export_archive_carries_mesh_db(root):
    from superlocalmemory.infra.backup_archive import build_export

    out = root / "export.tar.gz"
    manifest = build_export(root, out)
    assert "mesh.db" in {s["name"] for s in manifest["stores"]}
    with tarfile.open(out) as tar:
        assert any(m.name.endswith("/set/mesh.db") for m in tar)


def test_recovery_point_covers_mesh_db(root):
    from superlocalmemory.server import backup_bundle

    _add_peer(root / "mesh.db", "p-point")
    manifest = backup_bundle.create_point(root)
    assert "mesh.db" in manifest["stores"]
    _add_peer(root / "mesh.db", "p-later")
    backup_bundle.restore_point(root, manifest["id"], settings=False)
    assert _peers(root / "mesh.db") == ["p-point"]


def _try_write(path: Path, sql: str) -> str:
    conn = sqlite3.connect(path, timeout=0.2)
    try:
        conn.execute(sql)
        conn.commit()
        return "ok"
    except sqlite3.OperationalError as exc:
        return str(exc)
    finally:
        conn.close()


def test_mesh_write_during_a_backup_set_succeeds(root, monkeypatch):
    """The memory.db fence never reaches mesh.db: heartbeats keep landing while a
    backup set copies the memory stores. mesh.db is copied by SQLite's online
    backup API on its own, and still lands in the set."""
    real = BackupCoordinator._sqlite_backup
    outcome: dict[str, str] = {}

    def copying(self, src: Path, dest: Path) -> None:
        if src.name == "memory.db":
            outcome["mesh"] = _try_write(
                root / "mesh.db", "INSERT INTO mesh_nonces VALUES ('during', 1.0)")
            outcome["memory"] = _try_write(
                root / "memory.db", "INSERT INTO atomic_facts VALUES ('f2', 'x')")
        real(self, src, dest)

    monkeypatch.setattr(BackupCoordinator, "_sqlite_backup", copying)
    manifest = _coord(root).create_backup_set()
    assert outcome["mesh"] == "ok"
    assert "locked" in outcome["memory"]  # memory.db is still fenced
    entry = next(e for e in manifest.stores if e.store_name == "mesh.db")
    conn = sqlite3.connect(entry.file_path)
    try:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    finally:
        conn.close()
