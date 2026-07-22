# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — per-profile isolation (I-6 / C2)

"""Mesh coordination bus is per-tenant (profile_id).

The mesh broker is a shared coordination bus. Before M023 every mesh table was
global, so in a multi-tenant deployment one tenant (profile) could see another
tenant's peers, receive its broadcast/direct messages, read/overwrite its shared
state by key collision, and be blocked by its file locks. These tests pin the
tenant boundary: two profiles operating the same broker never cross over.

Also verifies M023 migrates a legacy (pre-profile) mesh DB, backfilling
'default' and adopting the composite primary keys for state/locks.
"""

from __future__ import annotations

import sqlite3

import pytest

A = "acme"
B = "globex"


def _build_current_mesh_db(path) -> None:
    """Current schema: born with profile_id (see schema_v343._MESH_DDL)."""
    from superlocalmemory.storage.schema_v343 import (
        _MESH_DDL, _MESH_V346_ALTERS, _MESH_V346_DDL,
    )
    conn = sqlite3.connect(str(path))
    conn.executescript(_MESH_DDL)
    for alter_sql in _MESH_V346_ALTERS:
        try:
            conn.execute(alter_sql)
        except sqlite3.OperationalError:
            pass
    conn.executescript(_MESH_V346_DDL)
    conn.commit()
    conn.close()


@pytest.fixture
def broker(tmp_path):
    from superlocalmemory.mesh.broker import MeshBroker

    db_path = tmp_path / "mesh_iso.db"
    _build_current_mesh_db(db_path)
    return MeshBroker(str(db_path))


# --------------------------------------------------------------------------
# Peers
# --------------------------------------------------------------------------

def test_peers_isolated_by_profile(broker):
    broker.register_peer("s-a", profile_id=A)
    broker.register_peer("s-b", profile_id=B)

    peers_a = broker.list_peers(A)
    peers_b = broker.list_peers(B)

    assert [p["session_id"] for p in peers_a] == ["s-a"]
    assert [p["session_id"] for p in peers_b] == ["s-b"]
    # Status count is also tenant-scoped.
    assert broker.get_status(A)["peer_count"] == 1
    assert broker.get_status(B)["peer_count"] == 1


def test_same_session_id_is_distinct_peer_per_profile(broker):
    ra = broker.register_peer("shared-session", profile_id=A)
    rb = broker.register_peer("shared-session", profile_id=B)
    # Same session string in two tenants must NOT be merged into one peer.
    assert ra["peer_id"] != rb["peer_id"]


def test_cannot_heartbeat_another_profiles_peer(broker):
    r = broker.register_peer("s-a", profile_id=A)
    # Heartbeat from tenant B must not find tenant A's peer.
    assert broker.heartbeat(r["peer_id"], profile_id=B)["ok"] is False
    assert broker.heartbeat(r["peer_id"], profile_id=A)["ok"] is True


def test_cannot_deregister_across_profile(broker):
    r = broker.register_peer("s-a", profile_id=A)
    assert broker.deregister_peer(r["peer_id"], profile_id=B)["ok"] is False
    assert len(broker.list_peers(A)) == 1


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------

def test_direct_message_recipient_must_be_same_profile(broker):
    ra = broker.register_peer("s-a", profile_id=A)
    # Active tenant B sending to a peer that only exists in A → not found.
    res = broker.send_message("sender-b", ra["peer_id"], "hi", profile_id=B)
    assert res["ok"] is False and "not found" in res["error"]
    # Same tenant works.
    res_ok = broker.send_message("sender-a", ra["peer_id"], "hi", profile_id=A)
    assert res_ok["ok"] is True


def test_broadcast_does_not_cross_profiles(broker):
    peer_a = broker.register_peer("s-a", profile_id=A)["peer_id"]
    peer_b = broker.register_peer("s-b", profile_id=B)["peer_id"]

    broker.send_message("other-a", "broadcast", "team-A-only", profile_id=A)

    inbox_a = broker.get_inbox(peer_a, profile_id=A)
    inbox_b = broker.get_inbox(peer_b, profile_id=B)
    assert any(m["content"] == "team-A-only" for m in inbox_a)
    assert all(m["content"] != "team-A-only" for m in inbox_b)


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

def test_state_key_coexists_per_profile(broker):
    broker.set_state("release", "1.0", "a", profile_id=A)
    broker.set_state("release", "9.9", "b", profile_id=B)

    assert broker.get_state_key("release", profile_id=A)["value"] == "1.0"
    assert broker.get_state_key("release", profile_id=B)["value"] == "9.9"
    # get_state (all) is scoped too.
    assert set(broker.get_state(A)) == {"release"}
    assert broker.get_state(A)["release"]["value"] == "1.0"


# --------------------------------------------------------------------------
# Locks
# --------------------------------------------------------------------------

def test_lock_on_same_path_does_not_block_other_profile(broker):
    path = "/repo/app.py"
    a = broker.lock_action(path, "peer-a", "acquire", profile_id=A)
    assert a["ok"] is True
    # Tenant B locking the SAME path must succeed — locks are per tenant.
    b = broker.lock_action(path, "peer-b", "acquire", profile_id=B)
    assert b["ok"] is True

    assert broker.lock_action(path, "x", "query", profile_id=A)["by"] == "peer-a"
    assert broker.lock_action(path, "x", "query", profile_id=B)["by"] == "peer-b"


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------

def test_events_isolated_by_profile(broker):
    broker.register_peer("s-a", profile_id=A)
    broker.register_peer("s-b", profile_id=B)
    ev_a = broker.get_events(profile_id=A)
    ev_b = broker.get_events(profile_id=B)
    # Each tenant only sees its own peer_registered event.
    assert len(ev_a) == 1
    assert len(ev_b) == 1


# --------------------------------------------------------------------------
# M023 migration of a legacy (pre-profile) DB
# --------------------------------------------------------------------------

def test_m023_migrates_legacy_mesh_db(tmp_path):
    """A DB with the OLD (global) mesh shape gets profile_id backfilled to
    'default' and composite PKs for state/locks, without losing data."""
    from superlocalmemory.storage.migrations import (
        M023_mesh_profile_isolation as m023,
    )

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    # Old global shapes (pre-M023).
    conn.executescript(
        """
        CREATE TABLE mesh_peers (peer_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
            summary TEXT DEFAULT '', status TEXT DEFAULT 'active',
            host TEXT DEFAULT '127.0.0.1', port INTEGER DEFAULT 0,
            registered_at TEXT NOT NULL, last_heartbeat TEXT NOT NULL);
        CREATE TABLE mesh_messages (id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_peer TEXT NOT NULL, to_peer TEXT NOT NULL, msg_type TEXT DEFAULT 'text',
            content TEXT NOT NULL, read INTEGER DEFAULT 0, created_at TEXT NOT NULL);
        CREATE TABLE mesh_state (key TEXT PRIMARY KEY, value TEXT NOT NULL,
            set_by TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE mesh_locks (file_path TEXT PRIMARY KEY, locked_by TEXT NOT NULL,
            locked_at TEXT NOT NULL, expires_at TEXT NOT NULL DEFAULT '9999-12-31T23:59:59Z');
        CREATE TABLE mesh_events (id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL, payload TEXT DEFAULT '{}',
            emitted_by TEXT NOT NULL, created_at TEXT NOT NULL);
        """
    )
    conn.execute("INSERT INTO mesh_peers (peer_id, session_id, registered_at, last_heartbeat) "
                 "VALUES ('p1','s1','t','t')")
    conn.execute("INSERT INTO mesh_state (key, value, set_by, updated_at) "
                 "VALUES ('k','v','p1','t')")
    conn.execute("INSERT INTO mesh_locks (file_path, locked_by, locked_at) "
                 "VALUES ('/f','p1','t')")
    conn.commit()

    assert m023.verify(conn) is False  # not migrated yet
    m023.apply(conn)
    assert m023.verify(conn) is True

    # Columns added + backfilled to 'default'.
    assert conn.execute("SELECT profile_id FROM mesh_peers").fetchone()[0] == "default"
    assert conn.execute("SELECT profile_id FROM mesh_state").fetchone()[0] == "default"
    assert conn.execute("SELECT profile_id FROM mesh_locks").fetchone()[0] == "default"

    # Composite PK: the same key/path can now exist under a second profile.
    conn.execute("INSERT INTO mesh_state (profile_id, key, value, set_by, updated_at) "
                 "VALUES ('other','k','v2','p2','t')")
    conn.execute("INSERT INTO mesh_locks (profile_id, file_path, locked_by, locked_at) "
                 "VALUES ('other','/f','p2','t')")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM mesh_state WHERE key='k'").fetchone()[0] == 2

    # Idempotent: second apply is a no-op.
    m023.apply(conn)
    assert m023.verify(conn) is True
    conn.close()
