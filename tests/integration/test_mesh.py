# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the Elastic License 2.0 - see LICENSE file

"""Integration tests for SLM Mesh broker (Phase C)."""

from __future__ import annotations

import sqlite3
import pytest
from pathlib import Path


@pytest.fixture
def mesh_db(tmp_path):
    """Create a temp DB with mesh tables (v3.4.3 base + v3.4.6 enhancements)."""
    db_path = tmp_path / "mesh_test.db"
    conn = sqlite3.connect(str(db_path))
    from superlocalmemory.storage.schema_v343 import _MESH_DDL, _MESH_V346_DDL, _MESH_V346_ALTERS
    conn.executescript(_MESH_DDL)
    for alter_sql in _MESH_V346_ALTERS:
        try:
            conn.execute(alter_sql)
        except sqlite3.OperationalError:
            pass
    conn.executescript(_MESH_V346_DDL)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def broker(mesh_db):
    from superlocalmemory.mesh.broker import MeshBroker
    return MeshBroker(str(mesh_db))


class TestMeshPeers:
    def test_register_peer(self, broker):
        result = broker.register_peer("session-1", summary="testing")
        assert result["ok"] is True
        assert "peer_id" in result

    def test_register_idempotent(self, broker):
        r1 = broker.register_peer("session-1", summary="v1")
        r2 = broker.register_peer("session-1", summary="v2")
        assert r1["peer_id"] == r2["peer_id"]  # Same session = same peer

    def test_deregister_peer(self, broker):
        r = broker.register_peer("session-1")
        peer_id = r["peer_id"]
        dr = broker.deregister_peer(peer_id)
        assert dr["ok"] is True
        peers = broker.list_peers()
        assert len(peers) == 0

    def test_deregister_nonexistent(self, broker):
        result = broker.deregister_peer("fake-id")
        assert result["ok"] is False

    def test_heartbeat(self, broker):
        r = broker.register_peer("session-1")
        hb = broker.heartbeat(r["peer_id"])
        assert hb["ok"] is True

    def test_heartbeat_nonexistent(self, broker):
        result = broker.heartbeat("fake-id")
        assert result["ok"] is False

    def test_update_summary(self, broker):
        r = broker.register_peer("session-1", summary="old")
        broker.update_summary(r["peer_id"], "new summary")
        peers = broker.list_peers()
        assert peers[0]["summary"] == "new summary"

    def test_list_peers(self, broker):
        broker.register_peer("s1", summary="agent 1")
        broker.register_peer("s2", summary="agent 2")
        peers = broker.list_peers()
        assert len(peers) == 2


class TestMeshMessages:
    def test_send_and_inbox(self, broker):
        r1 = broker.register_peer("s1")
        r2 = broker.register_peer("s2")
        send_result = broker.send_message(r1["peer_id"], r2["peer_id"], "hello")
        assert send_result["ok"] is True
        assert "id" in send_result

        inbox = broker.get_inbox(r2["peer_id"])
        assert len(inbox) == 1
        assert inbox[0]["content"] == "hello"

    def test_send_to_nonexistent(self, broker):
        r1 = broker.register_peer("s1")
        result = broker.send_message(r1["peer_id"], "fake", "hello")
        assert result["ok"] is False

    def test_mark_read(self, broker):
        r1 = broker.register_peer("s1")
        r2 = broker.register_peer("s2")
        broker.send_message(r1["peer_id"], r2["peer_id"], "hello")
        inbox = broker.get_inbox(r2["peer_id"])
        msg_id = inbox[0]["id"]
        broker.mark_read(r2["peer_id"], [msg_id])
        inbox2 = broker.get_inbox(r2["peer_id"])
        assert inbox2[0]["read"] == 1


class TestMeshState:
    def test_set_and_get(self, broker):
        broker.set_state("project", "slm", "agent-1")
        result = broker.get_state_key("project")
        assert result["value"] == "slm"
        assert result["set_by"] == "agent-1"

    def test_get_all(self, broker):
        broker.set_state("k1", "v1", "a1")
        broker.set_state("k2", "v2", "a2")
        state = broker.get_state()
        assert "k1" in state
        assert "k2" in state

    def test_get_nonexistent_key(self, broker):
        result = broker.get_state_key("nope")
        assert result is None

    def test_overwrite(self, broker):
        broker.set_state("k", "v1", "a1")
        broker.set_state("k", "v2", "a2")
        result = broker.get_state_key("k")
        assert result["value"] == "v2"


class TestMeshLocks:
    def test_acquire_and_query(self, broker):
        r1 = broker.register_peer("s1")
        result = broker.lock_action("test.py", r1["peer_id"], "acquire")
        assert result["ok"] is True

        query = broker.lock_action("test.py", "other-peer", "query")
        assert query["locked"] is True
        assert query["by"] == r1["peer_id"]

    def test_lock_conflict(self, broker):
        r1 = broker.register_peer("s1")
        r2 = broker.register_peer("s2")
        broker.lock_action("test.py", r1["peer_id"], "acquire")
        conflict = broker.lock_action("test.py", r2["peer_id"], "acquire")
        assert conflict["locked"] is True
        assert conflict["by"] == r1["peer_id"]

    def test_release(self, broker):
        r1 = broker.register_peer("s1")
        broker.lock_action("test.py", r1["peer_id"], "acquire")
        broker.lock_action("test.py", r1["peer_id"], "release")
        query = broker.lock_action("test.py", "anyone", "query")
        assert query["locked"] is False


class TestMeshEvents:
    def test_events_logged_on_register(self, broker):
        broker.register_peer("s1")
        events = broker.get_events()
        assert len(events) >= 1
        assert events[0]["event_type"] == "peer_registered"

    def test_events_logged_on_send(self, broker):
        r1 = broker.register_peer("s1")
        r2 = broker.register_peer("s2")
        broker.send_message(r1["peer_id"], r2["peer_id"], "hi")
        events = broker.get_events()
        types = [e["event_type"] for e in events]
        assert "message_sent" in types


class TestMeshStatus:
    def test_status(self, broker):
        broker.register_peer("s1")
        status = broker.get_status()
        assert status["broker_up"] is True
        assert status["peer_count"] == 1
        assert "uptime_s" in status


# ========================= v3.4.6 Connected Brain Tests =========================


class TestMeshBroadcast:
    """Tests for broadcast messaging (v3.4.6)."""

    def test_send_broadcast(self, broker):
        r1 = broker.register_peer("s1")
        result = broker.send_message(r1["peer_id"], "broadcast", "hello everyone")
        assert result["ok"] is True
        assert result["target_type"] == "broadcast"
        assert result["expires_at"] is not None

    def test_broadcast_in_inbox(self, broker):
        r1 = broker.register_peer("s1")
        r2 = broker.register_peer("s2")
        broker.send_message(r1["peer_id"], "broadcast", "global announcement")
        inbox = broker.get_inbox(r2["peer_id"])
        assert len(inbox) == 1
        assert inbox[0]["content"] == "global announcement"
        assert inbox[0]["target_type"] == "broadcast"

    def test_broadcast_not_to_sender(self, broker):
        r1 = broker.register_peer("s1")
        broker.send_message(r1["peer_id"], "broadcast", "my own message")
        inbox = broker.get_inbox(r1["peer_id"])
        assert len(inbox) == 0  # Sender doesn't see own broadcast

    def test_broadcast_mark_read(self, broker):
        r1 = broker.register_peer("s1")
        r2 = broker.register_peer("s2")
        broker.send_message(r1["peer_id"], "broadcast", "read me")
        inbox = broker.get_inbox(r2["peer_id"])
        assert inbox[0]["read"] == 0
        broker.mark_read(r2["peer_id"], [inbox[0]["id"]])
        inbox2 = broker.get_inbox(r2["peer_id"])
        assert inbox2[0]["read"] == 1  # Now marked read via mesh_reads


class TestMeshProjectMessages:
    """Tests for project-based messaging (v3.4.6)."""

    def test_send_to_project(self, broker):
        r1 = broker.register_peer("s1", project_path="/projects/qos")
        result = broker.send_message(r1["peer_id"], "project:/projects/qos", "qos update")
        assert result["ok"] is True
        assert result["target_type"] == "project"

    def test_project_msg_in_inbox(self, broker):
        r1 = broker.register_peer("s1", project_path="/projects/qos")
        r2 = broker.register_peer("s2", project_path="/projects/qos")
        broker.send_message(r1["peer_id"], "project:/projects/qos", "skill evolution plan ready")
        inbox = broker.get_inbox(r2["peer_id"], project_path="/projects/qos")
        assert len(inbox) == 1
        assert inbox[0]["content"] == "skill evolution plan ready"

    def test_project_msg_not_for_other_project(self, broker):
        r1 = broker.register_peer("s1", project_path="/projects/qos")
        r2 = broker.register_peer("s2", project_path="/projects/slm")
        broker.send_message(r1["peer_id"], "project:/projects/qos", "qos only")
        inbox = broker.get_inbox(r2["peer_id"], project_path="/projects/slm")
        assert len(inbox) == 0  # Different project, no message


class TestMeshOfflineQueue:
    """Tests for offline message delivery (v3.4.6)."""

    def test_offline_broadcast_pending(self, broker):
        # s1 sends broadcast BEFORE s2 registers
        r1 = broker.register_peer("s1")
        broker.send_message(r1["peer_id"], "broadcast", "sent while you were away")
        # s2 registers AFTER the broadcast
        r2 = broker.register_peer("s2")
        assert r2["pending_messages"] > 0
        # s2 should see the message in inbox
        inbox = broker.get_inbox(r2["peer_id"])
        assert len(inbox) == 1
        assert inbox[0]["content"] == "sent while you were away"

    def test_offline_project_pending(self, broker):
        r1 = broker.register_peer("s1", project_path="/projects/qos")
        broker.send_message(r1["peer_id"], "project:/projects/qos", "new plan available")
        # s2 registers in the same project later
        r2 = broker.register_peer("s2", project_path="/projects/qos")
        pending = broker.get_pending(r2["peer_id"], "/projects/qos")
        assert len(pending) == 1


class TestMeshGuards:
    """Tests for anti-bloat guards (v3.4.6)."""

    def test_message_size_cap(self, broker):
        r1 = broker.register_peer("s1")
        r2 = broker.register_peer("s2")
        huge_msg = "x" * 5000  # Over 4KB
        result = broker.send_message(r1["peer_id"], r2["peer_id"], huge_msg)
        assert result["ok"] is False
        assert "too large" in result["error"]

    def test_queue_cap_evicts_oldest(self, broker):
        from superlocalmemory.mesh.broker import MAX_QUEUED_PER_TARGET
        r1 = broker.register_peer("s1")
        # Send MAX+5 broadcast messages
        for i in range(MAX_QUEUED_PER_TARGET + 5):
            broker.send_message(r1["peer_id"], "broadcast", f"msg-{i}")
        # Should be capped at MAX
        r2 = broker.register_peer("s2")
        inbox = broker.get_inbox(r2["peer_id"])
        assert len(inbox) <= MAX_QUEUED_PER_TARGET

    def test_register_with_project_and_agent_type(self, broker):
        r = broker.register_peer("s1", project_path="/projects/qos", agent_type="claude_code")
        assert r["ok"] is True
        peers = broker.list_peers()
        assert peers[0]["project_path"] == "/projects/qos"
        assert peers[0]["agent_type"] == "claude_code"
