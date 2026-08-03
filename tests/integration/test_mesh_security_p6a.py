# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Phase 6a Sub-Wave 3a — Mesh Security Spine tests.

TDD test file — written RED first, implemented GREEN.

Covers:
  3a-1  Authenticated peer/tenant identity (HMAC sign/verify, replay defense,
          strict_identity mode, loopback trust)
  3a-2  Inbound admission + content-scrub parity (redact_secrets before storage,
          admission gate for remote inbound)
  3a-3  Restart-safe monotonic fencing (counter seeded from DB on restart)

Backward-compat invariant: with strict_identity=False (default), unsigned legacy
remote messages and loopback calls behave exactly as before.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Shared DB setup helpers (mirrors existing test_mesh.py pattern)
# ─────────────────────────────────────────────────────────────────────────────


def _init_mesh_schema(db_path: str) -> None:
    from superlocalmemory.storage.schema_v343 import (
        _MESH_DDL,
        _MESH_V346_ALTERS,
        _MESH_V346_DDL,
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


@pytest.fixture
def mesh_db(tmp_path):
    db_path = tmp_path / "mesh_test.db"
    _init_mesh_schema(str(db_path))
    return db_path


@pytest.fixture
def broker(mesh_db):
    from superlocalmemory.mesh.broker import MeshBroker

    return MeshBroker(str(mesh_db))


# ─────────────────────────────────────────────────────────────────────────────
# 3a-1: sign / verify helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestSignVerify:
    """Unit tests for sign_mesh_message / verify_mesh_message."""

    def test_sign_returns_hex_string(self):
        from superlocalmemory.mesh.broker_security import sign_mesh_message

        sig = sign_mesh_message("secret", "peerA", "peerB", "hello", "nonce1", "1000")
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA-256 hex digest

    def test_verify_valid_signature_succeeds(self):
        from superlocalmemory.mesh.broker_security import (
            sign_mesh_message,
            verify_mesh_message,
        )

        nonce = "abc123"
        ts = "1000000"
        sig = sign_mesh_message("mysecret", "alice", "bob", "payload", nonce, ts)
        assert verify_mesh_message("mysecret", "alice", "bob", "payload", nonce, ts, sig) is True

    def test_verify_tampered_content_fails(self):
        from superlocalmemory.mesh.broker_security import (
            sign_mesh_message,
            verify_mesh_message,
        )

        sig = sign_mesh_message("mysecret", "alice", "bob", "original", "n1", "1000")
        assert verify_mesh_message("mysecret", "alice", "bob", "tampered", "n1", "1000", sig) is False

    def test_verify_tampered_from_peer_fails(self):
        from superlocalmemory.mesh.broker_security import (
            sign_mesh_message,
            verify_mesh_message,
        )

        sig = sign_mesh_message("mysecret", "alice", "bob", "hello", "n2", "1000")
        assert verify_mesh_message("mysecret", "mallory", "bob", "hello", "n2", "1000", sig) is False

    def test_verify_wrong_secret_fails(self):
        from superlocalmemory.mesh.broker_security import (
            sign_mesh_message,
            verify_mesh_message,
        )

        sig = sign_mesh_message("correct-secret", "a", "b", "hi", "n3", "1000")
        assert verify_mesh_message("wrong-secret", "a", "b", "hi", "n3", "1000", sig) is False

    def test_canonical_payload_uses_sha256_of_content(self):
        """Signature must cover sha256(content) not raw content, to prevent length extension."""
        from superlocalmemory.mesh.broker_security import sign_mesh_message

        # Two messages with the same content hash should produce the same sig
        content = "important data"
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        sig1 = sign_mesh_message("s", "a", "b", content, "n", "1")
        # Manually compute expected
        payload = "|".join(["a", "b", content_hash, "n", "1"])
        expected = hmac.new("s".encode(), payload.encode(), hashlib.sha256).hexdigest()
        assert sig1 == expected


# ─────────────────────────────────────────────────────────────────────────────
# 3a-1: check_mesh_message_signature — replay, skew, strict mode
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckMeshMessageSignature:
    """Tests for check_mesh_message_signature gate function."""

    def setup_method(self):
        # Clear nonce store between tests to prevent cross-test pollution
        from superlocalmemory.mesh.broker_security import _clear_nonce_store

        _clear_nonce_store()

    def _sign(
        self,
        secret: str = "shared",
        from_peer: str = "peerA",
        to: str = "peerB",
        content: str = "hello",
        nonce: str | None = None,
        ts: str | None = None,
    ) -> tuple[str, str, str]:
        from superlocalmemory.mesh.broker_security import sign_mesh_message

        nonce = nonce or secrets.token_hex(8)
        ts = ts or str(int(time.time()))
        sig = sign_mesh_message(secret, from_peer, to, content, nonce, ts)
        return sig, nonce, ts

    def test_valid_signature_accepted(self):
        from superlocalmemory.mesh.broker_security import check_mesh_message_signature

        sig, nonce, ts = self._sign()
        result = check_mesh_message_signature(
            "shared", "peerA", "peerB", "hello", sig, nonce, ts,
            is_loopback=False, strict=True,
        )
        assert result is None, f"Expected None (ok), got: {result}"

    def test_tampered_content_rejected(self):
        from superlocalmemory.mesh.broker_security import check_mesh_message_signature

        sig, nonce, ts = self._sign(content="original")
        result = check_mesh_message_signature(
            "shared", "peerA", "peerB", "tampered", sig, nonce, ts,
            is_loopback=False, strict=True,
        )
        assert result is not None
        assert result.get("ok") is False

    def test_tampered_from_peer_rejected(self):
        from superlocalmemory.mesh.broker_security import check_mesh_message_signature

        sig, nonce, ts = self._sign(from_peer="legitimate-sender")
        result = check_mesh_message_signature(
            "shared", "impersonator", "peerB", "hello", sig, nonce, ts,
            is_loopback=False, strict=True,
        )
        assert result is not None
        assert result.get("ok") is False

    def test_replayed_nonce_rejected(self):
        from superlocalmemory.mesh.broker_security import check_mesh_message_signature

        sig, nonce, ts = self._sign()
        # First use: should pass
        first = check_mesh_message_signature(
            "shared", "peerA", "peerB", "hello", sig, nonce, ts,
            is_loopback=False, strict=True,
        )
        assert first is None, f"First use should pass, got: {first}"

        # Second use with same nonce: must be rejected (replay)
        sig2, _, ts2 = self._sign(nonce=nonce)  # same nonce, fresh sig+ts
        second = check_mesh_message_signature(
            "shared", "peerA", "peerB", "hello", sig2, nonce, ts2,
            is_loopback=False, strict=True,
        )
        assert second is not None
        assert "replay" in second.get("error", "").lower() or "nonce" in second.get("error", "").lower()

    def test_stale_timestamp_rejected(self):
        from superlocalmemory.mesh.broker_security import (
            MESH_SIG_SKEW_SECONDS,
            check_mesh_message_signature,
        )

        stale_ts = str(int(time.time()) - MESH_SIG_SKEW_SECONDS - 60)
        sig, nonce, _ = self._sign()
        # Re-sign with stale ts
        from superlocalmemory.mesh.broker_security import sign_mesh_message

        stale_sig = sign_mesh_message("shared", "peerA", "peerB", "hello", nonce, stale_ts)
        result = check_mesh_message_signature(
            "shared", "peerA", "peerB", "hello", stale_sig, nonce, stale_ts,
            is_loopback=False, strict=True,
        )
        assert result is not None
        assert result.get("ok") is False

    def test_loopback_always_trusted_no_signature_needed(self):
        """Loopback callers must never be asked for a signature (existing trust model)."""
        from superlocalmemory.mesh.broker_security import check_mesh_message_signature

        result = check_mesh_message_signature(
            "shared", "peerA", "peerB", "hello", None, None, None,
            is_loopback=True, strict=True,  # even strict=True, loopback wins
        )
        assert result is None, "Loopback must always be trusted"

    def test_strict_false_unsigned_legacy_accepted(self):
        """With strict=False, unsigned messages from remote are still accepted (backward compat)."""
        from superlocalmemory.mesh.broker_security import check_mesh_message_signature

        result = check_mesh_message_signature(
            "shared", "peerA", "peerB", "hello", None, None, None,
            is_loopback=False, strict=False,
        )
        assert result is None, "strict=False must accept unsigned legacy remote messages"

    def test_strict_false_bad_signature_still_rejected(self):
        """With strict=False, if a sig IS present but wrong, it must still be rejected."""
        from superlocalmemory.mesh.broker_security import check_mesh_message_signature

        result = check_mesh_message_signature(
            "shared", "peerA", "peerB", "hello", "badsig", "nonce", str(int(time.time())),
            is_loopback=False, strict=False,
        )
        assert result is not None
        assert result.get("ok") is False

    def test_strict_true_unsigned_rejected(self):
        """With strict=True, unsigned non-loopback messages are rejected."""
        from superlocalmemory.mesh.broker_security import check_mesh_message_signature

        result = check_mesh_message_signature(
            "shared", "peerA", "peerB", "hello", None, None, None,
            is_loopback=False, strict=True,
        )
        assert result is not None
        assert result.get("ok") is False


# ─────────────────────────────────────────────────────────────────────────────
# 3a-2: Content redaction before storage
# ─────────────────────────────────────────────────────────────────────────────


class TestContentScrub:
    """Verify secrets are redacted before message storage."""

    def test_scrub_message_content_redacts_api_keys(self):
        from superlocalmemory.mesh.broker_security import scrub_message_content

        # An Anthropic key pattern (from security_primitives._SECRET_PATTERNS)
        dangerous = "Here is my key: sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxx"
        scrubbed = scrub_message_content(dangerous)
        assert "sk-ant" not in scrubbed
        assert "[" in scrubbed  # redacted placeholder

    def test_scrub_message_content_leaves_safe_content_unchanged(self):
        from superlocalmemory.mesh.broker_security import scrub_message_content

        safe = "Agent A finished task 42 — results in /tmp/results.json"
        assert scrub_message_content(safe) == safe

    def test_send_message_redacts_secret_before_storage(self, broker):
        """broker.send_message must NOT store raw secrets in mesh_messages."""
        sender = broker.register_peer("sender-session")
        receiver = broker.register_peer("receiver-session")

        dangerous_content = "sk-ant-api03-" + "x" * 60
        result = broker.send_message(
            sender["peer_id"], receiver["peer_id"], dangerous_content,
        )
        assert result["ok"] is True

        inbox = broker.get_inbox(receiver["peer_id"])
        assert len(inbox) == 1
        stored_content = inbox[0]["content"]
        assert "sk-ant" not in stored_content, (
            f"Secret leaked into storage: {stored_content!r}"
        )

    def test_send_message_safe_content_not_corrupted(self, broker):
        """Non-secret messages must be stored verbatim."""
        sender = broker.register_peer("s1")
        receiver = broker.register_peer("s2")
        safe_msg = "task completed, file saved at /workspace/out.json"
        broker.send_message(sender["peer_id"], receiver["peer_id"], safe_msg)
        inbox = broker.get_inbox(receiver["peer_id"])
        assert inbox[0]["content"] == safe_msg


# ─────────────────────────────────────────────────────────────────────────────
# 3a-3: Restart-safe monotonic fencing
# ─────────────────────────────────────────────────────────────────────────────


class TestRestartFence:
    """Fencing counter must survive broker restarts."""

    def test_seed_fencing_counter_from_db(self, mesh_db):
        from superlocalmemory.mesh.broker_security import (
            apply_security_schema,
            seed_fencing_counter,
        )

        conn = sqlite3.connect(str(mesh_db))
        conn.row_factory = sqlite3.Row
        apply_security_schema(conn)  # add fencing_token column
        conn.execute(
            "INSERT INTO mesh_locks (profile_id, file_path, locked_by, locked_at, "
            "expires_at, fencing_token) VALUES ('default', '/a', 'p1', '2026-01-01', "
            "'9999-12-31', 42)"
        )
        conn.commit()
        conn.close()

        seeded = seed_fencing_counter(str(mesh_db))
        assert seeded == 42

    def test_seed_empty_db_returns_zero(self, mesh_db):
        from superlocalmemory.mesh.broker_security import seed_fencing_counter

        seeded = seed_fencing_counter(str(mesh_db))
        assert seeded == 0

    def test_restart_counter_exceeds_surviving_token(self, mesh_db):
        """Core invariant: post-restart tokens are always > any surviving DB token."""
        from superlocalmemory.mesh.broker import MeshBroker

        # First broker: acquire a lock (gets a fencing token)
        broker1 = MeshBroker(str(mesh_db))
        peer = broker1.register_peer("peer-session")
        lock_result = broker1.lock_action("/shared/resource", peer["peer_id"], "acquire")
        assert lock_result.get("ok") is True
        token_before_restart = lock_result["fencing_token"]
        assert token_before_restart >= 1

        # Simulate restart: new broker on same DB
        broker2 = MeshBroker(str(mesh_db))

        # Next token from restarted broker must exceed the surviving DB token
        peer2 = broker2.register_peer("peer-session-2")
        lock_result2 = broker2.lock_action(
            "/different/resource", peer2["peer_id"], "acquire"
        )
        assert lock_result2.get("ok") is True
        token_after_restart = lock_result2["fencing_token"]

        assert token_after_restart > token_before_restart, (
            f"After restart, next token {token_after_restart} must exceed "
            f"surviving DB token {token_before_restart}"
        )

    def test_stale_high_token_holder_rejected_after_restart(self, mesh_db):
        """A stale high-token holder cannot bypass the fence after restart."""
        from superlocalmemory.mesh.broker import MeshBroker
        from superlocalmemory.mesh.broker_security import apply_security_schema

        # Pre-seed the DB with a high fencing token (simulates old lock)
        conn = sqlite3.connect(str(mesh_db))
        conn.row_factory = sqlite3.Row
        apply_security_schema(conn)  # ensure fencing_token column exists
        conn.execute(
            "INSERT INTO mesh_locks (profile_id, file_path, locked_by, locked_at, "
            "expires_at, fencing_token) VALUES ('default', '/a/file', 'old-peer', "
            "'2026-01-01', '9999-12-31', 99)"
        )
        conn.commit()
        conn.close()

        # Restart broker (should seed counter >= 99)
        broker = MeshBroker(str(mesh_db))

        # validate_lock_fence with a LOWER token must be rejected
        reject = broker.validate_lock_fence("/a/file", 50)
        assert reject.get("ok") is False
        assert "stale" in reject.get("error", "").lower()


# ─────────────────────────────────────────────────────────────────────────────
# 3a-2: HTTP route — admission gate + signature verification
# ─────────────────────────────────────────────────────────────────────────────

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi import FastAPI  # noqa: E402  (after importorskip)
from fastapi.testclient import TestClient  # noqa: E402


DAEMON_HEADERS = {
    "X-SLM-Daemon-Capability": "mesh-capability",
    "X-SLM-Target-Instance": "mesh-instance",
}


def _make_app(secret: str | None = None) -> tuple[FastAPI, object]:
    """Build minimal FastAPI app with mesh router."""
    from superlocalmemory.mesh.broker import MeshBroker
    from superlocalmemory.server.routes import mesh as mesh_routes

    td = tempfile.mkdtemp()
    db_path = str(Path(td) / "mesh_p6a.db")
    _init_mesh_schema(db_path)
    broker = MeshBroker(db_path)
    broker._shared_secret = secret
    app = FastAPI()
    app.state.mesh_broker = broker
    app.state.config = None
    app.state.daemon_descriptor = SimpleNamespace(
        capability="mesh-capability",
        instance_id="mesh-instance",
        capability_fingerprint="fp",
    )
    app.include_router(mesh_routes.router)
    return app, broker


class TestHTTPSendAdmission:
    """Admission gate is applied to inbound remote /mesh/send (3a-2 parity)."""

    def _register_peer(self, client: TestClient, secret: str) -> str:
        """Register a peer via the HTTP route and return peer_id."""
        r = client.post(
            "/mesh/register",
            json={"session_id": "recv-sess"},
            headers={"Authorization": f"Bearer {secret}"},
        )
        assert r.status_code == 200, f"register failed: {r.text}"
        return r.json()["peer_id"]

    def test_remote_send_with_valid_auth_is_admitted(self):
        """Remote send with valid bearer token (authenticated peer) is allowed."""
        app, broker = _make_app(secret="topsecret")
        c = TestClient(app)
        peer_id = self._register_peer(c, "topsecret")

        r = c.post(
            "/mesh/send",
            json={"from_peer": "remote-agent", "to_peer": peer_id, "content": "hi"},
            headers={"Authorization": "Bearer topsecret"},
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        assert r.json().get("ok") is True

    def test_remote_send_without_auth_rejected(self):
        """Remote send without credentials is rejected (401)."""
        app, _ = _make_app(secret="topsecret")
        c = TestClient(app)
        r = c.post(
            "/mesh/send",
            json={"from_peer": "rogue", "to_peer": "anyone", "content": "hack"},
        )
        assert r.status_code == 401

    def test_loopback_send_bypasses_mesh_secret_check(self):
        """Loopback sends (no secret) still use daemon-capability auth."""
        app, broker = _make_app(secret=None)
        c = TestClient(app)
        sender = broker.register_peer("sender-local")
        receiver = broker.register_peer("receiver-local")
        r = c.post(
            "/mesh/send",
            json={
                "from_peer": sender["peer_id"],
                "to_peer": receiver["peer_id"],
                "content": "coordination update",
            },
            headers=DAEMON_HEADERS,
        )
        assert r.status_code == 200


class TestHTTPSignatureVerification:
    """HMAC signature is verified on the /mesh/send route for remote callers."""

    def setup_method(self):
        from superlocalmemory.mesh.broker_security import _clear_nonce_store
        _clear_nonce_store()

    def _register_peer(self, client: TestClient, secret: str) -> str:
        r = client.post(
            "/mesh/register",
            json={"session_id": "recv2"},
            headers={"Authorization": f"Bearer {secret}"},
        )
        assert r.status_code == 200, r.text
        return r.json()["peer_id"]

    def _make_sig_headers(
        self,
        secret: str,
        from_peer: str,
        to: str,
        content: str,
        nonce: str | None = None,
        ts: str | None = None,
    ) -> dict[str, str]:
        from superlocalmemory.mesh.broker_security import sign_mesh_message

        nonce = nonce or secrets.token_hex(8)
        ts = ts or str(int(time.time()))
        sig = sign_mesh_message(secret, from_peer, to, content, nonce, ts)
        return {
            "Authorization": f"Bearer {secret}",
            "X-Mesh-Sig": sig,
            "X-Mesh-Nonce": nonce,
            "X-Mesh-Ts": ts,
        }

    def test_strict_mode_accepts_valid_signature(self, monkeypatch):
        """In strict mode, a correctly signed message passes the route."""
        monkeypatch.setenv("SLM_MESH_STRICT_IDENTITY", "1")
        app, broker = _make_app(secret="s3cr3t")
        c = TestClient(app)
        peer_id = self._register_peer(c, "s3cr3t")

        from_peer = "remote-sender"
        content = "task complete"
        headers = self._make_sig_headers("s3cr3t", from_peer, peer_id, content)
        r = c.post(
            "/mesh/send",
            json={"from_peer": from_peer, "to_peer": peer_id, "content": content},
            headers=headers,
        )
        assert r.status_code == 200, f"strict mode, valid sig: {r.status_code} {r.text}"
        monkeypatch.delenv("SLM_MESH_STRICT_IDENTITY", raising=False)

    def test_strict_mode_rejects_tampered_content(self, monkeypatch):
        """In strict mode, content tampered after signing is rejected (401)."""
        monkeypatch.setenv("SLM_MESH_STRICT_IDENTITY", "1")
        app, broker = _make_app(secret="s3cr3t")
        c = TestClient(app)
        peer_id = self._register_peer(c, "s3cr3t")

        from_peer = "remote-sender"
        # Sign original content, send tampered content
        headers = self._make_sig_headers("s3cr3t", from_peer, peer_id, "original content")
        headers["Authorization"] = "Bearer s3cr3t"
        r = c.post(
            "/mesh/send",
            json={"from_peer": from_peer, "to_peer": peer_id, "content": "tampered content"},
            headers=headers,
        )
        assert r.status_code == 401, f"tampered content must be rejected: {r.status_code} {r.text}"
        monkeypatch.delenv("SLM_MESH_STRICT_IDENTITY", raising=False)

    def test_strict_mode_rejects_unsigned_message(self, monkeypatch):
        """In strict mode, unsigned remote messages are rejected."""
        monkeypatch.setenv("SLM_MESH_STRICT_IDENTITY", "1")
        app, broker = _make_app(secret="s3cr3t")
        c = TestClient(app)
        peer_id = self._register_peer(c, "s3cr3t")

        r = c.post(
            "/mesh/send",
            json={"from_peer": "sender", "to_peer": peer_id, "content": "unsigned"},
            headers={"Authorization": "Bearer s3cr3t"},
        )
        assert r.status_code == 401, f"unsigned in strict mode must 401: {r.status_code}"
        monkeypatch.delenv("SLM_MESH_STRICT_IDENTITY", raising=False)

    def test_compat_mode_accepts_unsigned_legacy(self):
        """Default compat mode (strict=False) accepts unsigned remote messages."""
        # No SLM_MESH_STRICT_IDENTITY set → default False
        app, broker = _make_app(secret="s3cr3t")
        c = TestClient(app)
        peer_id = self._register_peer(c, "s3cr3t")

        r = c.post(
            "/mesh/send",
            json={"from_peer": "legacy-sender", "to_peer": peer_id, "content": "hello"},
            headers={"Authorization": "Bearer s3cr3t"},
        )
        assert r.status_code == 200, f"unsigned in compat mode must succeed: {r.status_code} {r.text}"

    def test_compat_mode_rejects_bad_signature_if_present(self, monkeypatch):
        """Even in compat mode, a present-but-wrong signature is rejected."""
        app, broker = _make_app(secret="s3cr3t")
        c = TestClient(app)
        peer_id = self._register_peer(c, "s3cr3t")

        r = c.post(
            "/mesh/send",
            json={"from_peer": "sender", "to_peer": peer_id, "content": "hi"},
            headers={
                "Authorization": "Bearer s3cr3t",
                "X-Mesh-Sig": "deadbeef" * 8,  # 64 chars, wrong
                "X-Mesh-Nonce": "somenonce",
                "X-Mesh-Ts": str(int(time.time())),
            },
        )
        assert r.status_code == 401, f"bad sig must be rejected even in compat mode: {r.status_code}"


# ─────────────────────────────────────────────────────────────────────────────
# 3a-1: Outbound signing in RemoteSyncClient
# ─────────────────────────────────────────────────────────────────────────────


class TestOutboundSigning:
    """RemoteSyncClient.send_to_remote must sign messages when secret configured."""

    def test_send_to_remote_adds_signature_headers(self, monkeypatch):
        """When a shared secret exists, send_to_remote adds X-Mesh-Sig headers."""
        import json
        from unittest.mock import MagicMock, patch

        from superlocalmemory.mesh.remote_sync import RemoteSyncClient

        captured_headers: dict = {}

        class _FakeResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"ok": True, "id": 1}

        class _FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def post(self, url, *, json, headers, timeout):
                captured_headers.update(headers)
                return _FakeResponse()

        monkeypatch.setenv("SLM_MESH_SHARED_SECRET", "mysecret")
        monkeypatch.setenv("SLM_MESH_PEER_URL", "http://127.0.0.1:9999")

        broker_mock = MagicMock()
        broker_mock._shared_secret = "mysecret"

        client = RemoteSyncClient(broker_mock)
        client._peer_url = "http://127.0.0.1:9999"
        client._shared_secret = "mysecret"
        client._peer_url_trusted = True

        with patch("httpx.Client", return_value=_FakeClient()):
            result = client.send_to_remote(
                "peerB",
                {"from_peer": "peerA", "content": "hello", "type": "text"},
            )

        assert result.get("ok") is True
        assert "X-Mesh-Sig" in captured_headers, f"Missing X-Mesh-Sig in: {captured_headers}"
        assert "X-Mesh-Nonce" in captured_headers
        assert "X-Mesh-Ts" in captured_headers

        # Verify the signature is valid
        from superlocalmemory.mesh.broker_security import verify_mesh_message

        valid = verify_mesh_message(
            "mysecret",
            "peerA",
            "peerB",
            "hello",
            captured_headers["X-Mesh-Nonce"],
            captured_headers["X-Mesh-Ts"],
            captured_headers["X-Mesh-Sig"],
        )
        assert valid is True, "Outbound signature must be verifiable"


# ─────────────────────────────────────────────────────────────────────────────
# Backward compatibility: existing behaviors unchanged with strict=False
# ─────────────────────────────────────────────────────────────────────────────


class TestBackwardCompat:
    """Prove that existing behaviors are fully preserved with strict_identity=False."""

    def test_loopback_broker_send_unaffected(self, broker):
        """Direct broker.send_message (in-process, loopback) works exactly as before."""
        s = broker.register_peer("compat-sender")
        r = broker.register_peer("compat-receiver")
        result = broker.send_message(s["peer_id"], r["peer_id"], "hello compat")
        assert result["ok"] is True
        inbox = broker.get_inbox(r["peer_id"])
        assert len(inbox) == 1
        assert inbox[0]["content"] == "hello compat"

    def test_loopback_http_send_no_signature_required(self):
        """Loopback HTTP /send doesn't need a signature even in strict mode."""
        from superlocalmemory.mesh.broker_security import _clear_nonce_store
        _clear_nonce_store()

        app, broker = _make_app(secret=None)
        c = TestClient(app)
        sender = broker.register_peer("compat-s")
        receiver = broker.register_peer("compat-r")

        r = c.post(
            "/mesh/send",
            json={
                "from_peer": sender["peer_id"],
                "to_peer": receiver["peer_id"],
                "content": "no sig needed on loopback",
            },
            headers=DAEMON_HEADERS,
        )
        assert r.status_code == 200

    def test_existing_bearer_auth_still_works(self):
        """Existing X-Mesh-Secret / Authorization: Bearer remote auth is unaffected."""
        app, broker = _make_app(secret="existing-secret")
        c = TestClient(app)
        r = c.post(
            "/mesh/register",
            json={"session_id": "compat-remote"},
            headers={"Authorization": "Bearer existing-secret"},
        )
        assert r.status_code == 200

    def test_fencing_counter_still_monotonic_without_restart(self, broker):
        """Normal (non-restart) token generation is still strictly monotonic."""
        peer = broker.register_peer("mono-peer")
        tokens = []
        for path in ("/f1", "/f2", "/f3"):
            r = broker.lock_action(path, peer["peer_id"], "acquire")
            assert r.get("ok") is True
            tokens.append(r["fencing_token"])
        # All tokens strictly increasing
        assert tokens == sorted(tokens)
        assert len(set(tokens)) == len(tokens), "All tokens must be unique"
