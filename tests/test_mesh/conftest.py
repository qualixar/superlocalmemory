"""Shared fixtures for the mesh unit tests."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.mesh.broker import MeshBroker


def init_mesh_schema(db_path: str) -> None:
    from superlocalmemory.storage.schema_v343 import (
        _MESH_DDL, _MESH_V346_ALTERS, _MESH_V346_DDL,
    )
    conn = sqlite3.connect(db_path)
    conn.executescript(_MESH_DDL)
    for alter_sql in _MESH_V346_ALTERS:
        try:
            conn.execute(alter_sql)
        except sqlite3.OperationalError:
            pass
    conn.executescript(_MESH_V346_DDL)
    conn.commit()
    conn.close()


@pytest.fixture()
def broker(tmp_path, monkeypatch) -> MeshBroker:
    monkeypatch.delenv("SLM_MESH_HOST", raising=False)
    monkeypatch.delenv("SLM_MESH_SHARED_SECRET", raising=False)
    db = str(tmp_path / "mesh.db")
    init_mesh_schema(db)
    return MeshBroker(db)


def make_peer(broker: MeshBroker, name: str, agent: str = "claude_code") -> str:
    return broker.register_peer(name, agent_type=agent)["peer_id"]


def rows(broker: MeshBroker, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
    conn = broker._conn()
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()
