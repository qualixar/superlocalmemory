# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""mesh.db: the mesh's own file, its one-time migration out of memory.db,
recovery when it is lost or corrupt, and the retention of mesh activity."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from superlocalmemory.mesh import broker_cleanup, mesh_store
from superlocalmemory.mesh.broker import MeshBroker
from superlocalmemory.mesh.broker_security import apply_security_schema
from superlocalmemory.mesh.node_identity import get_node_id
from superlocalmemory.mesh.outbox_remote import RemoteOutbox
from superlocalmemory.storage.migrations import M023_mesh_profile_isolation as m023
from superlocalmemory.storage.schema_v343 import apply_v343_schema, apply_v346_schema

NOW = datetime.now(timezone.utc)
OLD = (NOW - timedelta(days=10)).isoformat()
RECENT = (NOW - timedelta(days=1)).isoformat()


def _legacy_store(path: Path) -> Path:
    """A memory.db as 4.2.7 left it: memory tables plus every mesh table."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO atomic_facts VALUES ('f1', 'kept')")
    conn.commit()
    conn.close()
    apply_v343_schema(str(path))
    apply_v346_schema(str(path))
    conn = sqlite3.connect(path)
    m023.apply(conn)
    apply_security_schema(conn)
    conn.commit()
    conn.close()
    get_node_id(str(path))
    RemoteOutbox(str(path))
    return path


def _seed(path: Path) -> None:
    conn = sqlite3.connect(path)
    peers = [("p-old", OLD), ("p-new", NOW.isoformat()), ("web-old", OLD)]
    for peer_id, seen in peers:
        conn.execute(
            "INSERT INTO mesh_peers (peer_id, session_id, registered_at, last_heartbeat,"
            " peer_key) VALUES (?, ?, ?, ?, 'k-' || ?)", (peer_id, peer_id, seen, seen, peer_id))
    conn.execute("INSERT INTO mesh_peer_profiles (peer_id, kind, updated_at, muted)"
                 " VALUES ('web-old', 'web', ?, 0)", (OLD,))
    conn.execute("INSERT INTO mesh_peer_profiles (peer_id, kind, updated_at, muted,"
                 " retired_at) VALUES ('p-gone', 'local', ?, 1, ?)", (OLD, OLD))
    for mid, at in ((1, OLD), (2, RECENT)):
        conn.execute("INSERT INTO mesh_messages (id, from_peer, to_peer, content, created_at)"
                     " VALUES (?, 'a', 'b', 'hi', ?)", (mid, at))
        conn.execute("INSERT INTO mesh_reads VALUES (?, 'b', ?)", (mid, at))
        conn.execute("INSERT INTO mesh_message_envelopes (message_id, from_kind)"
                     " VALUES (?, 'local')", (mid,))
        conn.execute("INSERT INTO mesh_sent_ops VALUES (?, ?, ?)", (f"op{mid}", mid, at))
    conn.execute("INSERT INTO mesh_messages (id, from_peer, to_peer, content, created_at)"
                 " VALUES (40, 'a', 'b', 'x', ?)", (RECENT,))
    conn.execute("DELETE FROM mesh_messages WHERE id = 40")  # sequence now 40
    for key, at in (("old", OLD), ("new", RECENT)):
        conn.execute("INSERT INTO mesh_state (key, value, set_by, updated_at)"
                     " VALUES (?, 'v', 'p', ?)", (key, at))
        conn.execute("INSERT INTO mesh_events (event_type, emitted_by, created_at)"
                     " VALUES ('e', 'p', ?)", (at,))
    future = (NOW + timedelta(hours=2)).isoformat()
    conn.execute("INSERT INTO mesh_locks (file_path, locked_by, locked_at, expires_at)"
                 " VALUES ('/a', 'p', ?, ?)", (RECENT, future))
    conn.execute("INSERT INTO mesh_locks (file_path, locked_by, locked_at, expires_at)"
                 " VALUES ('/b', 'p', ?, ?)", (OLD, OLD))
    conn.execute("INSERT INTO mesh_nonces VALUES ('n1', 1.0)")
    conn.commit()
    conn.close()


@pytest.fixture()
def legacy(tmp_path: Path) -> Path:
    store = _legacy_store(tmp_path / "memory.db")
    _seed(store)
    return store


def _tables(path: Path, prefix: str = "mesh_") -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
            (prefix + "%",))}
    finally:
        conn.close()


def _col(path: Path, sql: str) -> list:
    conn = sqlite3.connect(path)
    try:
        return [r[0] for r in conn.execute(sql)]
    finally:
        conn.close()


def test_path_is_sibling_of_the_memory_store(tmp_path: Path) -> None:
    assert mesh_store.mesh_db_path(tmp_path / "x" / "memory.db") == tmp_path / "x" / "mesh.db"


def test_fresh_store_gets_full_schema_in_wal(tmp_path: Path) -> None:
    mesh = mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    assert mesh == tmp_path / "mesh.db"
    assert not (tmp_path / "memory.db").exists()  # never creates the memory store
    assert set(mesh_store.MESH_TABLES) <= _tables(mesh)
    assert _col(mesh, "PRAGMA journal_mode") == ["wal"]
    assert mesh_store.inspect(mesh)["status"] == "ok"


def test_migration_copies_config_and_recent_activity_only(legacy: Path) -> None:
    mesh = mesh_store.prepare_mesh_db(legacy)
    assert sorted(_col(mesh, "SELECT peer_id FROM mesh_peers")) == ["p-new", "web-old"]
    assert sorted(_col(mesh, "SELECT peer_id FROM mesh_peer_profiles")) == ["p-gone", "web-old"]
    assert _col(mesh, "SELECT id FROM mesh_messages") == [2]
    assert _col(mesh, "SELECT message_id FROM mesh_reads") == [2]
    assert _col(mesh, "SELECT message_id FROM mesh_message_envelopes") == [2]
    assert _col(mesh, "SELECT operation_id FROM mesh_sent_ops") == ["op2"]
    assert _col(mesh, "SELECT key FROM mesh_state") == ["new"]
    assert len(_col(mesh, "SELECT id FROM mesh_events")) == 1
    assert _col(mesh, "SELECT file_path FROM mesh_locks") == ["/a"]
    assert _col(mesh, "SELECT nonce FROM mesh_nonces") == ["n1"]
    assert _col(mesh, "SELECT peer_key FROM mesh_peers WHERE peer_id='web-old'") == ["k-web-old"]
    assert len(_col(mesh, "SELECT node_id FROM mesh_node_identity")) == 1
    # A deleted message id is never handed out again.
    assert _col(mesh, "SELECT seq FROM sqlite_sequence WHERE name='mesh_messages'") == [40]


def test_migration_drops_legacy_tables_and_keeps_memory(legacy: Path) -> None:
    mesh_store.prepare_mesh_db(legacy)
    assert _tables(legacy) == set()
    assert _col(legacy, "SELECT content FROM atomic_facts") == ["kept"]


def test_node_identity_survives_migration(legacy: Path) -> None:
    before = get_node_id(str(legacy))
    mesh = mesh_store.prepare_mesh_db(legacy)
    assert get_node_id(str(mesh)) == before


def test_rerun_is_a_no_op(legacy: Path) -> None:
    mesh = mesh_store.prepare_mesh_db(legacy)
    marker = mesh_store.inspect(mesh)["migrated_at"]
    conn = sqlite3.connect(mesh)
    conn.execute("INSERT INTO mesh_state (key, value, set_by, updated_at)"
                 " VALUES ('live', 'v', 'p', ?)", (NOW.isoformat(),))
    conn.commit()
    conn.close()
    assert mesh_store.prepare_mesh_db(legacy) == mesh
    assert mesh_store.inspect(mesh)["migrated_at"] == marker
    assert "live" in _col(mesh, "SELECT key FROM mesh_state")


def test_interrupted_build_is_retried_cleanly(legacy: Path) -> None:
    partial = legacy.with_name("mesh.db" + mesh_store.BUILD_SUFFIX)
    partial.write_bytes(b"half a database")  # the copy died before the rename
    mesh = mesh_store.prepare_mesh_db(legacy)
    assert not partial.exists()
    assert sorted(_col(mesh, "SELECT key FROM mesh_state")) == ["new"]
    assert _tables(legacy) == set()


def test_crash_between_build_and_drop_merges_config_then_drops(legacy: Path) -> None:
    mesh = mesh_store.prepare_mesh_db(legacy, drop_legacy=False)
    assert "mesh_peers" in _tables(legacy)
    conn = sqlite3.connect(legacy)
    conn.execute("INSERT INTO mesh_peer_profiles (peer_id, kind, updated_at)"
                 " VALUES ('late', 'web', ?)", (RECENT,))
    conn.commit()
    conn.close()
    mesh_store.prepare_mesh_db(legacy)
    assert "late" in _col(mesh, "SELECT peer_id FROM mesh_peer_profiles")
    assert _tables(legacy) == set()


def test_corrupt_mesh_db_is_quarantined_and_rebuilt(legacy: Path, caplog) -> None:
    mesh = mesh_store.prepare_mesh_db(legacy, drop_legacy=False)
    mesh.write_bytes(b"not a database at all" * 100)
    with caplog.at_level(logging.WARNING, logger="superlocalmemory.mesh"):
        assert mesh_store.prepare_mesh_db(legacy) == mesh
    assert mesh_store.inspect(mesh)["status"] == "ok"
    assert _col(mesh, "SELECT key FROM mesh_state") == ["new"]
    assert list(mesh.parent.glob("mesh.db.quarantine-*"))
    assert any("mesh.db" in r.getMessage() for r in caplog.records)


def test_lost_mesh_db_after_drop_comes_back_empty(legacy: Path) -> None:
    mesh = mesh_store.prepare_mesh_db(legacy)
    for suffix in ("", "-wal", "-shm"):
        Path(str(mesh) + suffix).unlink(missing_ok=True)
    assert mesh_store.prepare_mesh_db(legacy) == mesh
    assert mesh_store.inspect(mesh)["status"] == "ok"
    assert _col(mesh, "SELECT COUNT(*) FROM mesh_peers") == [0]


def test_drop_runs_once(legacy: Path, monkeypatch) -> None:
    calls: list[int] = []
    real = mesh_store._drop_tables

    def counting(conn, names):
        calls.append(len(names))
        return real(conn, names)

    monkeypatch.setattr(mesh_store, "_drop_tables", counting)
    mesh_store.prepare_mesh_db(legacy)
    mesh_store.prepare_mesh_db(legacy)
    assert len(calls) == 1 and calls[0] >= 10


def test_retention_deletes_old_activity_and_keeps_config(tmp_path: Path) -> None:
    mesh = mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    _seed(mesh)
    conn = sqlite3.connect(mesh)
    conn.row_factory = sqlite3.Row
    broker_cleanup.run_cleanup(conn, retention_days=7)
    conn.close()
    assert sorted(_col(mesh, "SELECT peer_id FROM mesh_peers")) == ["p-new", "web-old"]
    assert _col(mesh, "SELECT id FROM mesh_messages") == [2]
    assert _col(mesh, "SELECT key FROM mesh_state") == ["new"]
    assert _col(mesh, "SELECT operation_id FROM mesh_sent_ops") == ["op2"]
    assert _col(mesh, "SELECT file_path FROM mesh_locks") == ["/a"]
    assert len(_col(mesh, "SELECT id FROM mesh_events")) == 1
    assert sorted(_col(mesh, "SELECT peer_id FROM mesh_peer_profiles")) == ["p-gone", "web-old"]
    assert _col(mesh, "SELECT nonce FROM mesh_nonces") == ["n1"]


def test_retention_window_is_configurable(tmp_path: Path) -> None:
    mesh = mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    _seed(mesh)
    conn = sqlite3.connect(mesh)
    broker_cleanup.run_cleanup(conn, retention_days=30)
    conn.close()
    assert sorted(_col(mesh, "SELECT key FROM mesh_state")) == ["new", "old"]


def test_prepare_applies_retention_at_start(tmp_path: Path) -> None:
    mesh = mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    _seed(mesh)
    mesh_store.prepare_mesh_db(tmp_path / "memory.db", retention_days=7)
    assert _col(mesh, "SELECT key FROM mesh_state") == ["new"]


def test_broker_recreates_a_mesh_db_deleted_under_it(tmp_path: Path) -> None:
    broker = MeshBroker.for_store(tmp_path / "memory.db")
    assert broker.register_peer("s1")["peer_id"]
    mesh = tmp_path / "mesh.db"
    for suffix in ("", "-wal", "-shm"):
        Path(str(mesh) + suffix).unlink(missing_ok=True)
    assert broker.register_peer("s2")["peer_id"]
    assert broker.heartbeat(broker.list_peers()[0]["peer_id"])["ok"] in (True, False)
    assert mesh_store.inspect(mesh)["status"] == "ok"


def test_broker_for_store_uses_mesh_db(tmp_path: Path) -> None:
    broker = MeshBroker.for_store(tmp_path / "memory.db", retention_days=3)
    assert broker._db_path == str(tmp_path / "mesh.db")
    assert broker._retention_days == 3


@pytest.mark.parametrize("raw, expected", [
    ({"mesh": {"retention_days": 3}}, 3),
    ({}, 7),
    ({"mesh": {"retention_days": "x"}}, 7),
    ({"mesh": {"retention_days": 0}}, 7),
])
def test_retention_days_config(tmp_path: Path, monkeypatch, raw: dict, expected: int) -> None:
    import json

    from superlocalmemory.core.config import SLMConfig

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps(raw))
    config = SLMConfig.load(tmp_path / "config.json")
    assert config.mesh_retention_days == expected
    config.save(tmp_path / "config.json")
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["mesh"]["retention_days"] == expected


def test_daemon_builds_the_broker_on_the_configured_store() -> None:
    """mesh.db follows config.db_path (SLM_DATA_DIR, a configured store root)."""
    import inspect as _inspect

    from superlocalmemory.server import unified_daemon

    source = _inspect.getsource(unified_daemon)
    assert "MeshBroker.for_store(\n                db_path," in source
    assert "MeshBroker(str(db_path))" not in source
