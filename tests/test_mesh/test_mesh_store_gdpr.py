# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""mesh.db is outside memory.db, so erasure, export and profile deletion must
reach it on their own: a profile's mesh rows and their children go, another
profile's stay."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from superlocalmemory.compliance.erasure_sidecars import erase_profile_sidecars
from superlocalmemory.compliance.export_encoding import ExportOptions
from superlocalmemory.compliance.gdpr_export import sidecar_tables
from superlocalmemory.mesh.broker import MeshBroker
from superlocalmemory.mesh.mesh_erasure import erase_profile


def _broker(root: Path) -> MeshBroker:
    return MeshBroker.for_store(root / "memory.db")


def _seed(broker: MeshBroker, pid: str) -> str:
    peer = broker.register_peer(f"sess-{pid}", profile_id=pid)["peer_id"]
    broker.send_message(peer, "broadcast", f"note of {pid}", profile_id=pid)
    broker.set_state("flag", f"value-{pid}", peer, profile_id=pid)
    broker.set_muted(peer, True, profile_id=pid)
    return peer


def _count(db: Path, sql: str, *args) -> int:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql, args).fetchone()[0]
    finally:
        conn.close()


def test_erase_profile_removes_rows_and_children(tmp_path: Path) -> None:
    broker = _broker(tmp_path)
    gone, kept = _seed(broker, "wipe"), _seed(broker, "keep")
    mesh = tmp_path / "mesh.db"
    assert erase_profile(mesh, "wipe") > 0
    for table in ("mesh_peers", "mesh_messages", "mesh_state", "mesh_events"):
        assert _count(mesh, f"SELECT COUNT(*) FROM {table} WHERE profile_id='wipe'") == 0
        assert _count(mesh, f"SELECT COUNT(*) FROM {table} WHERE profile_id='keep'") > 0
    assert _count(mesh, "SELECT COUNT(*) FROM mesh_peer_profiles WHERE peer_id=?", gone) == 0
    assert _count(mesh, "SELECT COUNT(*) FROM mesh_peer_profiles WHERE peer_id=?", kept) == 1
    assert _count(mesh, "SELECT COUNT(*) FROM mesh_message_envelopes WHERE message_id NOT IN"
                        " (SELECT id FROM mesh_messages)") == 0


def test_erase_profile_without_mesh_db_is_zero(tmp_path: Path) -> None:
    assert erase_profile(tmp_path / "mesh.db", "anyone") == 0
    assert not (tmp_path / "mesh.db").exists()


def test_gdpr_sidecar_erasure_reaches_mesh_db(tmp_path: Path) -> None:
    broker = _broker(tmp_path)
    _seed(broker, "wipe")
    counts: dict = {}
    erase_profile_sidecars(tmp_path, None, "wipe", counts)
    assert counts.get("mesh") and "mesh_failed" not in counts
    mesh = tmp_path / "mesh.db"
    assert _count(mesh, "SELECT COUNT(*) FROM mesh_messages WHERE profile_id='wipe'") == 0
    assert b"note of wipe" not in mesh.read_bytes()  # free pages rewritten too


def test_export_includes_the_profiles_mesh_rows(tmp_path: Path) -> None:
    broker = _broker(tmp_path)
    _seed(broker, "me")
    _seed(broker, "other")
    from superlocalmemory.compliance.gdpr_export import ProfileExport

    out = sidecar_tables(tmp_path / "mesh.db", "me", ExportOptions())
    assert out and {r["content"] for r in out["mesh_messages"]} == {"note of me"}
    assert "mesh" in {key for key, _kind, _payload in _sections(ProfileExport, tmp_path)}


def _sections(cls, root: Path):
    class _Db:
        db_path = root / "memory.db"

    sqlite3.connect(root / "memory.db").close()
    return list(cls(_Db(), "me", [], data_root=root, options=ExportOptions()).sections())
