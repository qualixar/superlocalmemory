# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for RemoteSyncClient — multi-machine mesh coordination."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture
def mesh_db(tmp_path):
    """Create a temp DB with mesh tables."""
    db_path = tmp_path / "mesh_test.db"
    conn = sqlite3.connect(str(db_path))
    from superlocalmemory.storage.schema_v343 import (
        _MESH_DDL,
        _MESH_V346_ALTERS,
        _MESH_V346_DDL,
    )

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


@pytest.fixture
def sync_client(broker):
    from superlocalmemory.mesh.remote_sync import RemoteSyncClient

    return RemoteSyncClient(broker)


class TestRemoteSyncClientInit:
    """Tests for RemoteSyncClient initialization."""

    def test_init_no_peer_url(self, broker):
        """When SLM_MESH_PEER_URL not set, client initializes but is inactive."""
        from superlocalmemory.mesh.remote_sync import RemoteSyncClient

        with patch.dict("os.environ", {}, clear=False):
            client = RemoteSyncClient(broker)
            assert client._peer_url is None
            assert client._shared_secret is None

    def test_init_with_peer_url(self, broker):
        """When SLM_MESH_PEER_URL set, client reads it."""
        from superlocalmemory.mesh.remote_sync import RemoteSyncClient

        with patch.dict(
            "os.environ",
            {
                "SLM_MESH_PEER_URL": "http://192.168.1.100:8765",
                "SLM_MESH_SHARED_SECRET": "secret123",
            },
        ):
            client = RemoteSyncClient(broker)
            assert client._peer_url == "http://192.168.1.100:8765"
            assert client._shared_secret == "secret123"

    def test_init_discovery_enabled_by_default(self, broker):
        """Discovery is on by default."""
        from superlocalmemory.mesh.remote_sync import RemoteSyncClient

        with patch.dict(
            "os.environ",
            {"SLM_MESH_DISCOVERY": "on"},
            clear=False,
        ):
            client = RemoteSyncClient(broker)
            assert client._discovery_enabled is True

    def test_init_discovery_can_be_disabled(self, broker):
        """Discovery can be turned off."""
        from superlocalmemory.mesh.remote_sync import RemoteSyncClient

        with patch.dict(
            "os.environ",
            {"SLM_MESH_DISCOVERY": "off"},
            clear=False,
        ):
            client = RemoteSyncClient(broker)
            assert client._discovery_enabled is False


class TestRemoteSyncPeerSync:
    """Tests for syncing peers from remote."""

    def test_sync_peers_populates_remote_peers(self, broker, sync_client):
        """sync_peers_from_remote fetches /mesh/peers and calls add_remote_peer."""
        sync_client._peer_url = "http://mock:8765"
        sync_client._shared_secret = "secret"

        mock_response = {
            "peers": [
                {
                    "peer_id": "remote-peer-1",
                    "session_id": "s1",
                    "summary": "agent on M4",
                    "status": "active",
                },
                {
                    "peer_id": "remote-peer-2",
                    "session_id": "s2",
                    "summary": "agent 2 on M4",
                    "status": "active",
                },
            ]
        }

        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_client:
            mock_get = MagicMock()
            mock_get.json.return_value = mock_response
            mock_get.raise_for_status.return_value = None
            mock_client.return_value.__enter__.return_value.get.return_value = mock_get

            sync_client._sync_peers_from_remote()

            # Verify add_remote_peer was called for each peer
            assert "remote-peer-1" in broker._remote_peers
            assert "remote-peer-2" in broker._remote_peers
            assert broker._remote_peers["remote-peer-1"]["summary"] == "agent on M4"

    def test_sync_peers_removes_stale_peers(self, broker, sync_client):
        """When a peer disappears from remote, it's removed locally."""
        sync_client._peer_url = "http://mock:8765"
        sync_client._shared_secret = "secret"

        # Pre-populate with a stale peer
        broker.add_remote_peer("stale-peer", {"summary": "was here"})
        sync_client._last_peers = {"stale-peer": {}}

        mock_response = {"peers": []}  # Remote has no peers now

        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_client:
            mock_get = MagicMock()
            mock_get.json.return_value = mock_response
            mock_get.raise_for_status.return_value = None
            mock_client.return_value.__enter__.return_value.get.return_value = mock_get

            sync_client._sync_peers_from_remote()

            # Stale peer should be removed
            assert "stale-peer" not in broker._remote_peers

    def test_sync_peers_error_handling_http(self, broker, sync_client):
        """HTTP errors during sync are logged but don't crash."""
        sync_client._peer_url = "http://mock:8765"
        sync_client._shared_secret = "secret"

        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.get.side_effect = (
                Exception("Network error")
            )

            # Should not raise
            sync_client._sync_peers_from_remote()

            # Remote peers should still be empty (no update happened)
            assert len(broker._remote_peers) == 0


class TestRemoteSyncSendMessage:
    """Tests for proxying messages to remote."""

    def test_send_to_remote_proxies_request(self, broker, sync_client):
        """send_to_remote POSTs to /mesh/send on remote."""
        sync_client._peer_url = "http://mock:8765"
        sync_client._shared_secret = "secret"

        message_data = {
            "from_peer": "local-peer-1",
            "content": "hello remote",
            "type": "text",
        }

        mock_response = {"ok": True, "id": 42}

        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_client:
            mock_post = MagicMock()
            mock_post.json.return_value = mock_response
            mock_post.raise_for_status.return_value = None
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_post
            )

            result = sync_client.send_to_remote("remote-peer-1", message_data)

            # Verify POST was made to correct endpoint
            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            assert "http://mock:8765/mesh/send" in str(call_args)
            assert result == mock_response

    def test_send_to_remote_includes_auth_header(self, broker, sync_client):
        """send_to_remote includes Authorization header."""
        sync_client._peer_url = "http://mock:8765"
        sync_client._shared_secret = "secret123"

        message_data = {
            "from_peer": "local-peer-1",
            "content": "test",
            "type": "text",
        }

        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_client:
            mock_post = MagicMock()
            mock_post.json.return_value = {"ok": True}
            mock_post.raise_for_status.return_value = None
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_post
            )

            sync_client.send_to_remote("remote-peer-1", message_data)

            # Check headers were passed
            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            assert "headers" in call_args.kwargs
            headers = call_args.kwargs["headers"]
            assert headers["Authorization"] == "Bearer secret123"

    def test_send_to_remote_error_handling(self, broker, sync_client):
        """When send fails, returns error dict instead of crashing."""
        sync_client._peer_url = "http://mock:8765"
        sync_client._shared_secret = "secret"

        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.side_effect = (
                Exception("Connection refused")
            )

            result = sync_client.send_to_remote("remote-peer-1", {"content": "test"})

            assert result["ok"] is False
            assert "error" in result

    def test_send_to_remote_no_peer_url(self, broker, sync_client):
        """When no peer URL configured, send_to_remote returns error."""
        sync_client._peer_url = None

        result = sync_client.send_to_remote("remote-peer-1", {"content": "test"})

        assert result["ok"] is False
        assert "no remote peer URL" in result["error"]


class TestMdnsSecretLeak:
    """M05: the shared secret must never leak to an unverified mDNS peer."""

    def _post_headers(self, sync_client) -> dict:
        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_client:
            mock_post = MagicMock()
            mock_post.json.return_value = {"ok": True}
            mock_post.raise_for_status.return_value = None
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_post
            )
            sync_client.send_to_remote("remote-peer-1", {"content": "hi"})
            call = mock_client.return_value.__enter__.return_value.post.call_args
            return call.kwargs.get("headers", {})

    def test_secret_withheld_from_discovered_peer_by_default(self, broker):
        """A peer adopted purely via mDNS is UNTRUSTED — no secret is sent."""
        from superlocalmemory.mesh.remote_sync import RemoteSyncClient

        with patch.dict("os.environ", {"SLM_MESH_SHARED_SECRET": "topsecret"},
                        clear=True):
            client = RemoteSyncClient(broker)
        # Simulate mDNS discovery adopting an attacker-advertised peer.
        client._update_peer_url("10.0.0.66", 8765)
        assert client._peer_url == "http://10.0.0.66:8765"
        assert client._peer_url_trusted is False
        headers = self._post_headers(client)
        assert "Authorization" not in headers, "secret leaked to discovered peer"

    def test_secret_sent_to_discovered_peer_when_operator_opts_in(self, broker):
        from superlocalmemory.mesh.remote_sync import RemoteSyncClient

        with patch.dict("os.environ",
                        {"SLM_MESH_SHARED_SECRET": "topsecret",
                         "SLM_MESH_TRUST_DISCOVERED": "on"}, clear=True):
            client = RemoteSyncClient(broker)
        client._update_peer_url("10.0.0.66", 8765)
        assert client._peer_url_trusted is True
        headers = self._post_headers(client)
        assert headers.get("Authorization") == "Bearer topsecret"

    def test_discovery_never_overrides_configured_peer(self, broker):
        from superlocalmemory.mesh.remote_sync import RemoteSyncClient

        with patch.dict("os.environ",
                        {"SLM_MESH_SHARED_SECRET": "topsecret",
                         "SLM_MESH_PEER_URL": "http://192.168.1.5:8765"},
                        clear=True):
            client = RemoteSyncClient(broker)
        # Attacker mDNS advertisement must not hijack the configured peer.
        client._update_peer_url("10.0.0.66", 8765)
        assert client._peer_url == "http://192.168.1.5:8765"
        assert client._peer_url_trusted is True
        headers = self._post_headers(client)
        assert headers.get("Authorization") == "Bearer topsecret"

    def test_configured_peer_is_trusted(self, broker):
        from superlocalmemory.mesh.remote_sync import RemoteSyncClient

        with patch.dict("os.environ",
                        {"SLM_MESH_SHARED_SECRET": "topsecret",
                         "SLM_MESH_PEER_URL": "http://192.168.1.5:8765"},
                        clear=True):
            client = RemoteSyncClient(broker)
        headers = self._post_headers(client)
        assert headers.get("Authorization") == "Bearer topsecret"


class TestBrokerRemoteIntegration:
    """Tests for broker + remote sync integration."""

    def test_send_message_to_remote_peer(self, broker):
        """broker.send_message proxies to remote when target is a remote peer."""
        from superlocalmemory.mesh.remote_sync import RemoteSyncClient

        # Create local peer
        r1 = broker.register_peer("s1", summary="local agent")

        # Register a "remote" peer (manually added to _remote_peers)
        broker.add_remote_peer("remote-peer-1", {"summary": "agent on M4"})

        # Mock sync client
        mock_sync_client = MagicMock(spec=RemoteSyncClient)
        mock_sync_client.send_to_remote.return_value = {"ok": True, "id": 99}
        broker._sync_client = mock_sync_client

        # Send to remote peer
        result = broker.send_message(r1["peer_id"], "remote-peer-1", "hello remote")

        # Should have called sync_client.send_to_remote
        assert result["ok"] is True
        mock_sync_client.send_to_remote.assert_called_once()

    def test_send_message_to_local_peer_unchanged(self, broker):
        """Sending to local peers still works (no proxy)."""
        from superlocalmemory.mesh.remote_sync import RemoteSyncClient

        r1 = broker.register_peer("s1")
        r2 = broker.register_peer("s2")

        mock_sync_client = MagicMock(spec=RemoteSyncClient)
        broker._sync_client = mock_sync_client

        result = broker.send_message(r1["peer_id"], r2["peer_id"], "local message")

        assert result["ok"] is True
        # Should NOT have called sync_client for local peer
        mock_sync_client.send_to_remote.assert_not_called()
