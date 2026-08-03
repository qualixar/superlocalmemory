# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
#
# Integration tests for SLM V4 sub-wave 3b: durable remote outbox (3b-1)
# and TLS-pinned transport (3b-3).
#
# xdist-UNSAFE: these tests share a SQLite file and must run SERIALLY.
# Run with: pytest tests/integration/test_mesh_transport_p6b.py -p no:cacheprovider -x

from __future__ import annotations

import json
import sqlite3
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mesh_db(tmp_path):
    """Create a minimal mesh DB with the broker schema applied."""
    db_path = tmp_path / "mesh_transport_test.db"
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


def _row_count(db_path, peer_url=None):
    """Return row count in mesh_outbox_remote (all or scoped to peer_url).

    Returns 0 if the table does not yet exist (lazy-init path where the
    outbox was never activated, e.g. BC success test).
    """
    conn = sqlite3.connect(str(db_path))
    try:
        if peer_url:
            row = conn.execute(
                "SELECT COUNT(*) FROM mesh_outbox_remote WHERE peer_url=?",
                (peer_url,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM mesh_outbox_remote"
            ).fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        # Table does not exist — outbox never activated (correct for BC tests)
        return 0
    finally:
        conn.close()


def _first_row(db_path):
    """Fetch the first row of mesh_outbox_remote, or None."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT * FROM mesh_outbox_remote ORDER BY id ASC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()


def _make_broker_mock(db_path):
    """Return a lightweight broker mock with _db_path wired."""
    broker = MagicMock()
    broker._db_path = str(db_path)
    broker._remote_peers = {}
    broker._remote_peers_lock = __import__("threading").RLock()
    return broker


def _make_client(broker, env=None):
    """Return a RemoteSyncClient with env patched."""
    from superlocalmemory.mesh.remote_sync import RemoteSyncClient
    env_overrides = env or {}
    with patch.dict("os.environ", env_overrides, clear=False):
        return RemoteSyncClient(broker)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mesh_db(tmp_path):
    return _make_mesh_db(tmp_path)


@pytest.fixture
def broker(mesh_db):
    return _make_broker_mock(mesh_db)


@pytest.fixture
def sync_client(broker, mesh_db):
    """RemoteSyncClient with a peer URL and shared secret configured."""
    from superlocalmemory.mesh.remote_sync import RemoteSyncClient
    with patch.dict(
        "os.environ",
        {
            "SLM_MESH_PEER_URL": "http://192.168.1.5:8765",
            "SLM_MESH_SHARED_SECRET": "test-secret",
        },
        clear=False,
    ):
        client = RemoteSyncClient(broker)
    return client


# ===========================================================================
# 3b-1 — Durable Remote Outbox
# ===========================================================================


class TestOutboxIdempotentInit:
    """BC guard: table creation must be safe on a pre-existing DB."""

    def test_creates_table_on_fresh_db(self, mesh_db):
        """RemoteOutbox creates its table when the DB has only broker tables."""
        from superlocalmemory.mesh.outbox_remote import RemoteOutbox
        ob = RemoteOutbox(str(mesh_db))
        assert ob._active is True

    def test_idempotent_on_existing_table(self, mesh_db):
        """Opening RemoteOutbox twice on the same DB raises no error (IF NOT EXISTS)."""
        from superlocalmemory.mesh.outbox_remote import RemoteOutbox
        ob1 = RemoteOutbox(str(mesh_db))
        assert ob1._active is True
        # Second open — table already exists, IF NOT EXISTS must not fail
        ob2 = RemoteOutbox(str(mesh_db))
        assert ob2._active is True

    def test_pre_created_table_leaves_existing_data(self, mesh_db):
        """If the table already has rows, re-opening does not wipe them."""
        from superlocalmemory.mesh.outbox_remote import RemoteOutbox
        now = time.time()
        ob = RemoteOutbox(str(mesh_db))
        ob.enqueue("http://peer:9000", "p1", {"msg": "hello"}, None, now=now)
        assert _row_count(mesh_db) == 1

        # Re-open — data must survive
        ob2 = RemoteOutbox(str(mesh_db))
        assert ob2._active is True
        assert _row_count(mesh_db) == 1


class TestEnqueueOnFailure:
    """3b-1: send_to_remote must enqueue on every failure branch."""

    def test_enqueues_on_request_error(self, sync_client, mesh_db):
        """When httpx.RequestError occurs, a row appears in the outbox."""
        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.side_effect = (
                httpx.RequestError("connection refused")
            )
            result = sync_client.send_to_remote(
                "peer-1", {"from_peer": "me", "content": "hello", "type": "text"}
            )

        assert result["ok"] is False
        assert _row_count(mesh_db) == 1

    def test_enqueues_on_non_2xx(self, sync_client, mesh_db):
        """When the server returns non-2xx, a row appears in the outbox."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503", request=MagicMock(), response=MagicMock()
        )
        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            result = sync_client.send_to_remote(
                "peer-1", {"from_peer": "me", "content": "hi", "type": "text"}
            )

        assert result["ok"] is False
        assert _row_count(mesh_db) == 1

    def test_enqueues_on_generic_exception(self, sync_client, mesh_db):
        """Any unexpected exception also triggers enqueue."""
        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.side_effect = (
                RuntimeError("boom")
            )
            result = sync_client.send_to_remote(
                "peer-1", {"from_peer": "me", "content": "test", "type": "text"}
            )

        assert result["ok"] is False
        assert _row_count(mesh_db) == 1

    def test_success_does_not_enqueue(self, sync_client, mesh_db):
        """BC: a successful send must NOT add a row to the outbox."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True, "id": 42}
        mock_resp.raise_for_status.return_value = None
        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            result = sync_client.send_to_remote(
                "peer-1", {"from_peer": "me", "content": "hi", "type": "text"}
            )

        assert result["ok"] is True
        assert _row_count(mesh_db) == 0, "success must not enqueue"


class TestOutboxDrain:
    """3b-1: drain logic — success deletes, failure schedules retry."""

    def _prefill_row(self, mesh_db, peer_url="http://192.168.1.5:8765"):
        """Insert a due outbox row directly via the RemoteOutbox API."""
        from superlocalmemory.mesh.outbox_remote import RemoteOutbox
        now = time.time()
        ob = RemoteOutbox(str(mesh_db))
        ob.enqueue(
            peer_url=peer_url,
            to_peer="peer-1",
            payload={"from_peer": "me", "to_peer": "peer-1", "content": "retry-me", "type": "text"},
            headers=None,
            now=now - 60,  # next_retry_at already past
        )
        return ob

    def test_drain_deletes_row_on_success(self, sync_client, mesh_db):
        """After a successful drain POST, the outbox row is deleted."""
        self._prefill_row(mesh_db)
        assert _row_count(mesh_db) == 1

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status.return_value = None
        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
            sync_client._drain_outbox()

        assert _row_count(mesh_db) == 0, "successful drain must delete row"

    def test_drain_marks_retry_on_failure(self, sync_client, mesh_db):
        """After a failed drain POST, retry_count increments and next_retry moves forward."""
        self._prefill_row(mesh_db)
        row_before = _first_row(mesh_db)
        assert row_before is not None

        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.side_effect = (
                httpx.RequestError("still down")
            )
            sync_client._drain_outbox()

        row_after = _first_row(mesh_db)
        assert row_after is not None, "row should still exist after failed drain"
        assert row_after["retry_count"] == 1
        assert row_after["next_retry_at"] > row_before["next_retry_at"]

    def test_drain_does_not_requeue_undrained_rows(self, sync_client, mesh_db):
        """A row whose next_retry_at is in the future must NOT be drained."""
        from superlocalmemory.mesh.outbox_remote import RemoteOutbox
        ob = RemoteOutbox(str(mesh_db))
        far_future = time.time() + 3600
        ob.enqueue(
            peer_url="http://192.168.1.5:8765",
            to_peer="peer-1",
            payload={"from_peer": "me", "to_peer": "peer-1", "content": "wait", "type": "text"},
            headers=None,
            now=far_future,  # next_retry_at = far_future → not due yet
        )
        assert _row_count(mesh_db) == 1

        # Drain should do nothing (no POST, row stays)
        post_mock = MagicMock()
        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post = post_mock
            sync_client._drain_outbox()

        assert _row_count(mesh_db) == 1, "not-yet-due row must remain untouched"
        post_mock.assert_not_called()


class TestOutboxTTLAndCap:
    """3b-1: TTL prune and per-peer cap."""

    def test_prune_expired_removes_stale_rows(self, mesh_db):
        """prune_expired deletes rows whose expires_at is in the past."""
        from superlocalmemory.mesh.outbox_remote import RemoteOutbox
        ob = RemoteOutbox(str(mesh_db))
        expired_now = time.time() - 3600  # enqueued an hour ago with TTL already elapsed
        # Manually insert an already-expired row
        conn = sqlite3.connect(str(mesh_db))
        conn.execute(
            """INSERT INTO mesh_outbox_remote
               (peer_url, to_peer, payload, headers, retry_count, next_retry_at, created_at, expires_at)
               VALUES (?, ?, ?, NULL, 0, ?, ?, ?)""",
            ("http://p:1", "x", '{"k":"v"}', expired_now, expired_now, expired_now - 1),
        )
        conn.commit()
        conn.close()
        assert _row_count(mesh_db) == 1

        ob.prune_expired(time.time())
        assert _row_count(mesh_db) == 0

    def test_per_peer_cap_enforced(self, mesh_db):
        """When a peer_url has _CAP_PER_PEER rows, adding one more drops the oldest."""
        from superlocalmemory.mesh.outbox_remote import RemoteOutbox, _CAP_PER_PEER
        ob = RemoteOutbox(str(mesh_db))
        peer = "http://peer-a:9000"
        now = time.time()

        # Fill to cap
        for i in range(_CAP_PER_PEER):
            ob.enqueue(peer, f"p{i}", {"msg": i}, None, now=now + i)

        assert _row_count(mesh_db, peer_url=peer) == _CAP_PER_PEER

        # One more — oldest must be evicted
        ob.enqueue(peer, "p-new", {"msg": "newest"}, None, now=now + _CAP_PER_PEER + 1)
        assert _row_count(mesh_db, peer_url=peer) == _CAP_PER_PEER

        # Newest message must be present (oldest evicted)
        conn = sqlite3.connect(str(mesh_db))
        conn.row_factory = sqlite3.Row
        latest = conn.execute(
            "SELECT payload FROM mesh_outbox_remote WHERE peer_url=? ORDER BY created_at DESC LIMIT 1",
            (peer,),
        ).fetchone()
        conn.close()
        assert latest is not None
        assert json.loads(latest["payload"])["msg"] == "newest"


class TestDeadLetter:
    """3b-1: rows exceeding MAX_RETRIES are deleted."""

    def test_dead_letter_after_max_retries(self, mesh_db):
        """mark_retry deletes a row once retry_count exceeds _MAX_RETRIES."""
        from superlocalmemory.mesh.outbox_remote import RemoteOutbox, _MAX_RETRIES
        ob = RemoteOutbox(str(mesh_db))
        now = time.time()
        ob.enqueue("http://peer:1", "p1", {"msg": "doom"}, None, now=now)
        assert _row_count(mesh_db) == 1

        row = _first_row(mesh_db)
        row_id = row["id"]

        # Drive mark_retry to the dead-letter threshold
        for _ in range(_MAX_RETRIES + 1):
            ob.mark_retry(row_id, now)
            now += 1

        assert _row_count(mesh_db) == 0, "row must be deleted after MAX_RETRIES"

    def test_dead_letter_on_expired_ttl(self, mesh_db):
        """mark_retry deletes a row when its TTL has elapsed."""
        from superlocalmemory.mesh.outbox_remote import RemoteOutbox
        ob = RemoteOutbox(str(mesh_db))
        now = time.time()
        ob.enqueue("http://peer:1", "p1", {"msg": "expire"}, None, now=now)
        row = _first_row(mesh_db)
        row_id = row["id"]

        # Call mark_retry with now well past expires_at
        ob.mark_retry(row_id, now + 48 * 3600 + 1)
        assert _row_count(mesh_db) == 0


# ===========================================================================
# 3b-3 — TLS-pinned transport
# ===========================================================================


class TestHttpClientTLS:
    """3b-3: _http_client respects TLS env vars."""

    def test_default_no_env_produces_plain_client(self, broker):
        """Without any TLS env, _http_client returns a default client (verify=True)."""
        from superlocalmemory.mesh.remote_sync import RemoteSyncClient
        with patch.dict("os.environ", {}, clear=False):
            # Remove TLS CA if somehow set
            env_clean = {k: v for k, v in __import__("os").environ.items()
                         if k != "SLM_MESH_TLS_CA"}
        with patch.dict("os.environ", env_clean, clear=True):
            client = RemoteSyncClient(broker)
            http = client._http_client(timeout=5)
            # Default httpx.Client has verify=True (system CAs)
            assert http is not None
            http.close()

    def test_custom_ca_is_passed_to_client(self, broker, tmp_path):
        """When SLM_MESH_TLS_CA is set, _http_client passes verify=<path>."""
        ca_file = tmp_path / "fake_ca.pem"
        ca_file.write_text("FAKE CA")  # existence check only

        from superlocalmemory.mesh.remote_sync import RemoteSyncClient
        with patch.dict("os.environ", {"SLM_MESH_TLS_CA": str(ca_file)}):
            client = RemoteSyncClient(broker)
            # Mock httpx.Client to capture the verify kwarg
            with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_cls:
                mock_cls.return_value = MagicMock()
                client._http_client(timeout=5)
                call_kwargs = mock_cls.call_args.kwargs
                assert call_kwargs.get("verify") == str(ca_file)


class TestCertPinning:
    """3b-3: cert pinning rejects mismatched certs; allows matching ones."""

    _GOOD_PIN = "a" * 64   # 64 hex chars = 32 bytes = SHA-256
    _BAD_PIN = "b" * 64

    def test_pin_mismatch_blocks_send_no_post_made(self, sync_client, mesh_db):
        """When the cert hash doesn't match the configured pin, send fails
        and the actual POST must never be delivered."""
        post_mock = MagicMock()
        with patch("superlocalmemory.mesh.remote_sync._get_cert_sha256",
                   return_value=self._GOOD_PIN):
            with patch.dict(
                "os.environ",
                {"SLM_MESH_TLS_PIN": self._BAD_PIN},
            ):
                # Force peer_url to https:// so pinning activates
                sync_client._peer_url = "https://192.168.1.5:8765"
                with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_cls:
                    mock_cls.return_value.__enter__.return_value.post = post_mock
                    result = sync_client.send_to_remote(
                        "peer-1",
                        {"from_peer": "me", "content": "pin-test", "type": "text"},
                    )

        assert result["ok"] is False
        assert "pin" in result["error"].lower()
        post_mock.assert_not_called()

    def test_pin_match_allows_send(self, sync_client, mesh_db):
        """When the cert hash matches the configured pin, send proceeds normally."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True, "id": 77}
        mock_resp.raise_for_status.return_value = None
        with patch("superlocalmemory.mesh.remote_sync._get_cert_sha256",
                   return_value=self._GOOD_PIN):
            with patch.dict(
                "os.environ",
                {"SLM_MESH_TLS_PIN": self._GOOD_PIN},
            ):
                sync_client._peer_url = "https://192.168.1.5:8765"
                with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_cls:
                    mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
                    result = sync_client.send_to_remote(
                        "peer-1",
                        {"from_peer": "me", "content": "secure", "type": "text"},
                    )

        assert result["ok"] is True

    def test_pin_comparison_is_case_insensitive(self, sync_client, mesh_db):
        """Pin comparison normalises to lowercase so UPPERCASE pins also match."""
        # Cert returns lowercase hex; configured pin uses uppercase
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status.return_value = None
        with patch("superlocalmemory.mesh.remote_sync._get_cert_sha256",
                   return_value=self._GOOD_PIN.lower()):
            with patch.dict(
                "os.environ",
                {"SLM_MESH_TLS_PIN": self._GOOD_PIN.upper()},
            ):
                sync_client._peer_url = "https://192.168.1.5:8765"
                with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_cls:
                    mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
                    result = sync_client.send_to_remote(
                        "peer-1",
                        {"from_peer": "me", "content": "case-test", "type": "text"},
                    )

        assert result["ok"] is True

    def test_pin_ignored_for_plaintext_url(self, sync_client, mesh_db):
        """When the URL is http://, cert pinning is skipped (no TLS connection)."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status.return_value = None
        with patch.dict("os.environ", {"SLM_MESH_TLS_PIN": self._BAD_PIN}):
            # http:// URL — pin check bypassed
            sync_client._peer_url = "http://192.168.1.5:8765"
            with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_cls:
                mock_cls.return_value.__enter__.return_value.post.return_value = mock_resp
                result = sync_client.send_to_remote(
                    "peer-1",
                    {"from_peer": "me", "content": "plain", "type": "text"},
                )

        assert result["ok"] is True


# ===========================================================================
# Backward compatibility proofs
# ===========================================================================


class TestBackwardCompatibility:
    """Ensure no new env / no TLS config = byte-for-byte old behaviour."""

    def test_successful_send_no_tls_no_outbox_env(self, broker, mesh_db):
        """With no TLS or outbox env, a successful send:
        - makes exactly one POST to /mesh/send
        - returns {"ok": True, ...} from the server
        - leaves the outbox empty
        """
        from superlocalmemory.mesh.remote_sync import RemoteSyncClient
        with patch.dict(
            "os.environ",
            {
                "SLM_MESH_PEER_URL": "http://192.168.1.5:8765",
                "SLM_MESH_SHARED_SECRET": "bc-secret",
            },
            clear=False,
        ):
            # Scrub any TLS env that might be set in the test environment
            for k in ["SLM_MESH_TLS_CA", "SLM_MESH_TLS_PIN", "SLM_MESH_TLS"]:
                __import__("os").environ.pop(k, None)
            client = RemoteSyncClient(broker)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True, "id": 1}
        mock_resp.raise_for_status.return_value = None
        post_mock = MagicMock(return_value=mock_resp)

        with patch("superlocalmemory.mesh.remote_sync.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post = post_mock
            result = client.send_to_remote(
                "peer-1",
                {"from_peer": "me", "content": "bc-test", "type": "text"},
            )

        assert result == {"ok": True, "id": 1}
        post_mock.assert_called_once()
        # No rows in outbox
        assert _row_count(mesh_db) == 0

    def test_http_scheme_for_discovered_peers_by_default(self, broker):
        """Without SLM_MESH_TLS=on, mDNS-discovered peers use http://."""
        from superlocalmemory.mesh.remote_sync import RemoteSyncClient
        with patch.dict("os.environ", {}, clear=False):
            __import__("os").environ.pop("SLM_MESH_TLS", None)
            client = RemoteSyncClient(broker)

        client._peer_url_from_config = False
        client._update_peer_url("10.0.0.1", 8765)
        assert client._peer_url is not None
        assert client._peer_url.startswith("http://"), (
            "BC: discovered peers must default to http:// when SLM_MESH_TLS is unset"
        )

    def test_https_scheme_for_discovered_peers_when_tls_on(self, broker):
        """When SLM_MESH_TLS=on, mDNS-discovered peers use https://."""
        from superlocalmemory.mesh.remote_sync import RemoteSyncClient
        # Both construction and _update_peer_url must see SLM_MESH_TLS=on
        # because the env var is read at call time in _update_peer_url.
        with patch.dict("os.environ", {"SLM_MESH_TLS": "on"}):
            client = RemoteSyncClient(broker)
            client._peer_url_from_config = False
            client._update_peer_url("10.0.0.2", 8765)

        assert client._peer_url is not None
        assert client._peer_url.startswith("https://"), (
            "SLM_MESH_TLS=on must produce https:// for discovered peers"
        )
