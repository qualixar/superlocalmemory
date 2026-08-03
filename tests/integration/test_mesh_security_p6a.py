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

        # Two messages with the same content hash should produce the same sig.
        # SEC-2: delimiter is now NUL (\x00), not pipe.
        content = "important data"
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        sig1 = sign_mesh_message("s", "a", "b", content, "n", "1")
        # Manually compute expected with NUL delimiter
        payload = "\x00".join(["a", "b", content_hash, "n", "1"])
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

    def test_compat_mode_accepts_fleet_signed_unregistered(self, monkeypatch):
        """Backward-compat (default, non-strict): a valid fleet-secret-signed message
        from an UNregistered from_peer is accepted (legacy remote path preserved).
        (Strict-mode acceptance of a REGISTERED peer signing with its own peer_key is
        covered by TestPerPeerIdentityBinding.test_legitimate_peer_key_holder_accepted.)"""
        monkeypatch.delenv("SLM_MESH_STRICT_IDENTITY", raising=False)
        monkeypatch.delenv("SLM_MESH_PRODUCTION", raising=False)
        app, broker = _make_app(secret="s3cr3t")
        c = TestClient(app)
        peer_id = self._register_peer(c, "s3cr3t")

        from_peer = "legacy-sender"
        content = "task complete"
        headers = self._make_sig_headers("s3cr3t", from_peer, peer_id, content)
        r = c.post(
            "/mesh/send",
            json={"from_peer": from_peer, "to_peer": peer_id, "content": content},
            headers=headers,
        )
        assert r.status_code == 200, f"compat fleet-signed: {r.status_code} {r.text}"

    def test_strict_rejects_unregistered_from_peer(self, monkeypatch):
        """P1 (hardened): strict mode rejects an UNregistered from_peer even with a
        valid fleet-secret signature — the fleet-fallback escape hatch is closed."""
        monkeypatch.setenv("SLM_MESH_STRICT_IDENTITY", "1")
        app, broker = _make_app(secret="s3cr3t")
        c = TestClient(app)
        recv_id = self._register_peer(c, "s3cr3t")
        from_peer = "not-registered"
        content = "x"
        headers = self._make_sig_headers("s3cr3t", from_peer, recv_id, content)
        r = c.post(
            "/mesh/send",
            json={"from_peer": from_peer, "to_peer": recv_id, "content": content},
            headers=headers,
        )
        assert r.status_code == 401, (
            f"strict must reject unregistered from_peer (no fleet fallback): "
            f"{r.status_code} {r.text}"
        )
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


# ─────────────────────────────────────────────────────────────────────────────
# SEC-2: NUL delimiter — canonicalization collision prevention
# ─────────────────────────────────────────────────────────────────────────────


class TestCanonicalPayloadSecurity:
    """SEC-2: NUL delimiter prevents field injection / canonicalization collisions."""

    def test_pipe_in_from_peer_does_not_collide_with_pipe_in_to(self):
        """(from='a|b', to='c') must differ from (from='a', to='b|c') — pipe collision."""
        from superlocalmemory.mesh.broker_security import sign_mesh_message

        sig1 = sign_mesh_message("s", "a|b", "c", "content", "n", "1")
        sig2 = sign_mesh_message("s", "a", "b|c", "content", "n", "1")
        assert sig1 != sig2, (
            "Canonicalization collision detected: ('a|b','c') == ('a','b|c'). "
            "Delimiter must be a NUL byte, not pipe."
        )

    def test_nul_byte_in_from_peer_rejected(self):
        """from_peer containing NUL byte must be rejected before HMAC (SEC-2)."""
        from superlocalmemory.mesh.broker_security import check_mesh_message_signature, sign_mesh_message

        ts = str(int(time.time()))
        nonce = "test-nonce-nul"
        sig = sign_mesh_message("s", "peer\x00hack", "to", "msg", nonce, ts)
        result = check_mesh_message_signature(
            "s", "peer\x00hack", "to", "msg", sig, nonce, ts,
            is_loopback=False, strict=True,
        )
        assert result is not None, "NUL in from_peer must be rejected"
        assert result.get("ok") is False

    def test_nul_byte_in_nonce_rejected(self):
        """nonce containing NUL byte must be rejected (SEC-2)."""
        from superlocalmemory.mesh.broker_security import check_mesh_message_signature, sign_mesh_message

        ts = str(int(time.time()))
        nonce = "no\x00nce"
        sig = sign_mesh_message("s", "peerA", "peerB", "msg", nonce, ts)
        result = check_mesh_message_signature(
            "s", "peerA", "peerB", "msg", sig, nonce, ts,
            is_loopback=False, strict=True,
        )
        assert result is not None, "NUL in nonce must be rejected"
        assert result.get("ok") is False

    def test_control_char_in_from_peer_rejected(self):
        """Control characters (< 0x20) in from_peer must be rejected (SEC-2)."""
        from superlocalmemory.mesh.broker_security import check_mesh_message_signature, sign_mesh_message

        ts = str(int(time.time()))
        nonce = "ctrl-nonce"
        from_peer = "peer\x01id"  # SOH control character
        sig = sign_mesh_message("s", from_peer, "to", "msg", nonce, ts)
        result = check_mesh_message_signature(
            "s", from_peer, "to", "msg", sig, nonce, ts,
            is_loopback=False, strict=True,
        )
        assert result is not None, "Control char in from_peer must be rejected"
        assert result.get("ok") is False


# ─────────────────────────────────────────────────────────────────────────────
# SEC-1: Nonce boundary — strict expiry so nonce outlives skew window
# ─────────────────────────────────────────────────────────────────────────────


class TestNonceBoundaryReplay:
    """SEC-1: Nonce at exactly T+skew must still be in store (strict < not <=)."""

    def setup_method(self):
        from superlocalmemory.mesh.broker_security import _clear_nonce_store
        _clear_nonce_store()

    def test_nonce_at_exact_skew_boundary_not_pruned(self):
        """A nonce whose expiry == now must NOT be pruned (strict < means it stays)."""
        from superlocalmemory.mesh.broker_security import (
            MESH_SIG_SKEW_SECONDS,
            _nonce_lock,
            _nonce_store,
            _prune_nonces,
        )
        nonce = "boundary-nonce"
        now = 1_000_000.0
        # Set nonce to expire exactly at now (exp = now)
        with _nonce_lock:
            _nonce_store[nonce] = now  # expires_at == now

        # Prune at exactly 'now' — strict < means exp==now is NOT expired yet
        _prune_nonces(now)

        with _nonce_lock:
            remaining = nonce in _nonce_store
        assert remaining, (
            "Nonce at exp==now must survive pruning (strict < prevents boundary replay)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# SEC-3: SQLite nonce durability — survives in-memory clear
# ─────────────────────────────────────────────────────────────────────────────


class TestSQLiteNonceDurability:
    """SEC-3: Nonces in SQLite survive in-memory cache clear (restart-safe)."""

    def setup_method(self):
        from superlocalmemory.mesh.broker_security import _clear_nonce_store
        _clear_nonce_store()

    def test_nonce_reuse_rejected_after_memory_clear_with_broker(self, mesh_db):
        """After clearing only in-memory nonce cache, SQLite still rejects replay."""
        from superlocalmemory.mesh.broker import MeshBroker
        from superlocalmemory.mesh.broker_security import (
            _nonce_lock,
            _nonce_store,
            check_mesh_message_signature,
            sign_mesh_message,
        )

        # Broker init wires SQLite nonce storage
        _broker = MeshBroker(str(mesh_db))  # noqa: F841 — side effect: wires DB path

        nonce = "sqlite-durability-nonce"
        ts = str(int(time.time()))
        sig = sign_mesh_message("s", "a", "b", "content", nonce, ts)

        # First use — must succeed
        r1 = check_mesh_message_signature(
            "s", "a", "b", "content", sig, nonce, ts,
            is_loopback=False, strict=True,
        )
        assert r1 is None, f"First use must pass, got: {r1}"

        # Clear ONLY the in-memory dict (simulates process restart losing RAM state)
        with _nonce_lock:
            _nonce_store.clear()

        # Re-sign for the replay attempt (same nonce, fresh signature)
        sig2 = sign_mesh_message("s", "a", "b", "content", nonce, ts)
        r2 = check_mesh_message_signature(
            "s", "a", "b", "content", sig2, nonce, ts,
            is_loopback=False, strict=True,
        )
        assert r2 is not None, (
            "SQLite must remember the nonce and reject replay even after in-memory clear. "
            "Current in-memory implementation would accept (bug)."
        )
        assert "replay" in r2.get("error", "").lower() or "nonce" in r2.get("error", "").lower()

    def test_clear_nonce_store_clears_sqlite_table(self, mesh_db):
        """_clear_nonce_store must also clear the SQLite mesh_nonces table."""
        import sqlite3 as _sqlite3
        from superlocalmemory.mesh.broker import MeshBroker
        from superlocalmemory.mesh.broker_security import (
            _clear_nonce_store,
            check_mesh_message_signature,
            sign_mesh_message,
        )

        _broker = MeshBroker(str(mesh_db))  # noqa: F841 — wires DB path

        nonce = "clear-test-nonce"
        ts = str(int(time.time()))
        sig = sign_mesh_message("s", "a", "b", "c", nonce, ts)
        r1 = check_mesh_message_signature(
            "s", "a", "b", "c", sig, nonce, ts,
            is_loopback=False, strict=True,
        )
        assert r1 is None, "First use must pass"

        # Clear store — must remove from SQLite too
        _clear_nonce_store()

        # Now the nonce should be accepted again (table cleared)
        sig2 = sign_mesh_message("s", "a", "b", "c", nonce, ts)
        r2 = check_mesh_message_signature(
            "s", "a", "b", "c", sig2, nonce, ts,
            is_loopback=False, strict=True,
        )
        assert r2 is None, "_clear_nonce_store must flush SQLite so test isolation works"


# ─────────────────────────────────────────────────────────────────────────────
# SEC-4: Per-peer identity binding
# ─────────────────────────────────────────────────────────────────────────────


class TestPerPeerIdentityBinding:
    """SEC-4: Per-peer keys make from_peer unspoofable without the registrant's key."""

    def setup_method(self):
        from superlocalmemory.mesh.broker_security import _clear_nonce_store
        _clear_nonce_store()

    def test_register_returns_peer_key(self, broker):
        """register_peer must return a 'peer_key' in the result dict."""
        result = broker.register_peer("key-test-session")
        assert "peer_key" in result, (
            f"register_peer must return peer_key. Got keys: {list(result.keys())}"
        )
        pk = result["peer_key"]
        assert isinstance(pk, str) and len(pk) == 64, (
            f"peer_key must be 64-char hex string (32 bytes token_hex). Got: {pk!r}"
        )

    def test_register_idempotent_peer_id_but_key_not_reexported(self, broker):
        """P0: re-registering the same session keeps the same peer_id but MUST NOT
        re-export the peer_key. The legitimate owner keeps the key issued at first
        registration; a fleet-secret holder who learns the session_id (via
        /mesh/peers) therefore cannot re-register to steal the victim's key."""
        r1 = broker.register_peer("idem-session")
        r2 = broker.register_peer("idem-session")
        assert r1["peer_id"] == r2["peer_id"], "Same session → same peer_id"
        assert "peer_key" in r1, "First registration must issue the peer_key"
        assert "peer_key" not in r2, "Re-registration must NOT re-export the peer_key (P0)"

    def test_p0_reregister_does_not_reexport_peer_key_http(self, monkeypatch):
        """P0 end-to-end: attacker re-registers a victim's session_id over HTTP with
        the fleet secret → response carries NO peer_key (identity-theft closure)."""
        monkeypatch.delenv("SLM_MESH_STRICT_IDENTITY", raising=False)
        app, _broker = _make_app(secret="fleet-p0")
        c = TestClient(app)
        r1 = c.post("/mesh/register", json={"session_id": "victim-sess"},
                    headers={"Authorization": "Bearer fleet-p0"})
        assert r1.status_code == 200 and "peer_key" in r1.json(), r1.text
        r2 = c.post("/mesh/register", json={"session_id": "victim-sess"},
                    headers={"Authorization": "Bearer fleet-p0"})
        assert r2.status_code == 200, r2.text
        assert "peer_key" not in r2.json(), (
            "Re-registration must NOT re-export peer_key (P0 identity-theft closure)"
        )
        assert r1.json()["peer_id"] == r2.json()["peer_id"], "same peer_id (idempotent)"

    def test_spoof_with_fleet_secret_rejected_in_strict_mode(self, monkeypatch):
        """Strict mode: fleet secret cannot forge a registered peer's from_peer identity."""
        monkeypatch.setenv("SLM_MESH_STRICT_IDENTITY", "1")
        app, _broker = _make_app(secret="fleet-secret")
        c = TestClient(app)

        # Register peer X and receiver
        rx = c.post("/mesh/register", json={"session_id": "px-sess"},
                    headers={"Authorization": "Bearer fleet-secret"})
        assert rx.status_code == 200, rx.text
        peer_x_id = rx.json()["peer_id"]

        rv = c.post("/mesh/register", json={"session_id": "recv-sess"},
                    headers={"Authorization": "Bearer fleet-secret"})
        recv_id = rv.json()["peer_id"]

        from superlocalmemory.mesh.broker_security import sign_mesh_message

        content = "spoofed message"
        nonce = secrets.token_hex(8)
        ts = str(int(time.time()))
        # Attacker uses FLEET SECRET to sign as peer_x_id — must fail in strict mode
        spoof_sig = sign_mesh_message("fleet-secret", peer_x_id, recv_id, content, nonce, ts)

        r = c.post(
            "/mesh/send",
            json={"from_peer": peer_x_id, "to_peer": recv_id, "content": content},
            headers={
                "Authorization": "Bearer fleet-secret",
                "X-Mesh-Sig": spoof_sig,
                "X-Mesh-Nonce": nonce,
                "X-Mesh-Ts": ts,
            },
        )
        assert r.status_code == 401, (
            f"Fleet secret spoofing from_peer of registered peer must fail in strict mode. "
            f"Got: {r.status_code} {r.text}"
        )
        monkeypatch.delenv("SLM_MESH_STRICT_IDENTITY", raising=False)

    def test_legitimate_peer_key_holder_accepted(self, monkeypatch):
        """Strict mode: the legitimate peer_key holder can send as their peer_id."""
        monkeypatch.setenv("SLM_MESH_STRICT_IDENTITY", "1")
        app, _broker = _make_app(secret="fleet-s2")
        c = TestClient(app)

        rx = c.post("/mesh/register", json={"session_id": "legit-sess"},
                    headers={"Authorization": "Bearer fleet-s2"})
        assert rx.status_code == 200, rx.text
        peer_id = rx.json()["peer_id"]
        peer_key = rx.json().get("peer_key")
        assert peer_key is not None, "register must return peer_key (SEC-4)"

        rv = c.post("/mesh/register", json={"session_id": "legit-recv"},
                    headers={"Authorization": "Bearer fleet-s2"})
        recv_id = rv.json()["peer_id"]

        from superlocalmemory.mesh.broker_security import sign_mesh_message

        content = "legit message"
        nonce = secrets.token_hex(8)
        ts = str(int(time.time()))
        # Sign with OWN peer_key — must succeed
        legit_sig = sign_mesh_message(peer_key, peer_id, recv_id, content, nonce, ts)

        r = c.post(
            "/mesh/send",
            json={"from_peer": peer_id, "to_peer": recv_id, "content": content},
            headers={
                "Authorization": "Bearer fleet-s2",
                "X-Mesh-Sig": legit_sig,
                "X-Mesh-Nonce": nonce,
                "X-Mesh-Ts": ts,
            },
        )
        assert r.status_code == 200, (
            f"Legitimate peer_key holder must succeed. Got: {r.status_code} {r.text}"
        )
        monkeypatch.delenv("SLM_MESH_STRICT_IDENTITY", raising=False)

    def test_compat_mode_fleet_secret_still_works_for_unregistered_sender(self, monkeypatch):
        """In compat mode (strict=False), unregistered from_peer + fleet secret sig is accepted."""
        monkeypatch.delenv("SLM_MESH_STRICT_IDENTITY", raising=False)
        monkeypatch.delenv("SLM_MESH_PRODUCTION", raising=False)
        app, _broker = _make_app(secret="fleet-compat")
        c = TestClient(app)

        rv = c.post("/mesh/register", json={"session_id": "compat-recv"},
                    headers={"Authorization": "Bearer fleet-compat"})
        recv_id = rv.json()["peer_id"]

        from superlocalmemory.mesh.broker_security import sign_mesh_message

        content = "compat content"
        nonce = secrets.token_hex(8)
        ts = str(int(time.time()))
        # Unregistered from_peer using fleet secret — compat mode must accept
        sig = sign_mesh_message("fleet-compat", "unregistered-peer", recv_id, content, nonce, ts)

        r = c.post(
            "/mesh/send",
            json={"from_peer": "unregistered-peer", "to_peer": recv_id, "content": content},
            headers={
                "Authorization": "Bearer fleet-compat",
                "X-Mesh-Sig": sig,
                "X-Mesh-Nonce": nonce,
                "X-Mesh-Ts": ts,
            },
        )
        assert r.status_code == 200, (
            f"Compat mode + unregistered from_peer + valid fleet sig must succeed. "
            f"Got: {r.status_code} {r.text}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SEC-5: Admission actor + deny path exercised
# ─────────────────────────────────────────────────────────────────────────────


class TestAdmissionActorGate:
    """SEC-5: Admission gate deny path is real (not a dead code path)."""

    def setup_method(self):
        from superlocalmemory.mesh.broker_security import _clear_nonce_store
        _clear_nonce_store()

    def test_admission_deny_returns_403(self):
        """Mocked admission denial must return HTTP 403 from /mesh/send."""
        from unittest.mock import patch

        from superlocalmemory.core.admission import AdmissionDenied
        from superlocalmemory.core.operation_policy_registry import PolicyDecision

        app, _broker = _make_app(secret="sec5-secret")
        c = TestClient(app)

        rv = c.post("/mesh/register", json={"session_id": "deny-recv"},
                    headers={"Authorization": "Bearer sec5-secret"})
        assert rv.status_code == 200, rv.text
        recv_id = rv.json()["peer_id"]

        denial = PolicyDecision(allowed=False, reason="test_denial_unauthorized_actor")

        with patch("superlocalmemory.core.admission.admit",
                   side_effect=AdmissionDenied(denial)):
            r = c.post(
                "/mesh/send",
                json={"from_peer": "attacker", "to_peer": recv_id, "content": "denied-msg"},
                headers={"Authorization": "Bearer sec5-secret"},
            )
            assert r.status_code == 403, (
                f"Admission denial must return 403. Got: {r.status_code} {r.text}"
            )
            assert "test_denial" in r.text or "unauthorized" in r.text.lower() or "deny" in r.text.lower()

    def test_admission_authorized_allowed(self):
        """No admission mock → personal mode → OWNER → admitted → 200."""
        app, _broker = _make_app(secret="sec5-authz")
        c = TestClient(app)

        rv = c.post("/mesh/register", json={"session_id": "authz-recv"},
                    headers={"Authorization": "Bearer sec5-authz"})
        recv_id = rv.json()["peer_id"]

        r = c.post(
            "/mesh/send",
            json={"from_peer": "authorized-peer", "to_peer": recv_id, "content": "admitted"},
            headers={"Authorization": "Bearer sec5-authz"},
        )
        assert r.status_code == 200, f"Authorized personal mode must admit. Got: {r.status_code} {r.text}"


# ─────────────────────────────────────────────────────────────────────────────
# SEC-6: Fencing fail-closed when table exists but query errors
# ─────────────────────────────────────────────────────────────────────────────


class TestFencingFailClosed:
    """SEC-6: seed_fencing_counter raises if mesh_locks exists but MAX query errors."""

    def test_fail_closed_when_table_exists_but_max_query_errors(self, tmp_path):
        """Fail-closed: raise RuntimeError instead of silently returning 0."""
        import sqlite3 as _sqlite3
        from unittest.mock import MagicMock, patch

        from superlocalmemory.mesh.broker_security import seed_fencing_counter

        db_path = str(tmp_path / "fence_fc.db")
        _init_mesh_schema(db_path)  # creates mesh_locks table

        # Build a mock connection that reports mesh_locks exists but fails on MAX
        mock_conn = MagicMock()

        def execute_side_effect(sql, params=()):
            mock_result = MagicMock()
            if "sqlite_master" in sql.lower():
                # Report table exists
                mock_result.fetchone.return_value = MagicMock()  # truthy
            elif "MAX" in sql.upper() and "mesh_locks" in sql.lower():
                raise _sqlite3.OperationalError("simulated MAX query failure")
            else:
                mock_result.fetchone.return_value = None
            return mock_result

        mock_conn.execute.side_effect = execute_side_effect
        mock_conn.close = MagicMock()
        mock_conn.row_factory = None

        with patch("sqlite3.connect", return_value=mock_conn):
            with pytest.raises((RuntimeError, _sqlite3.OperationalError)):
                seed_fencing_counter(db_path)

    def test_missing_table_returns_zero_not_raises(self, tmp_path):
        """First-run: mesh_locks absent → return 0, no exception (normal startup)."""
        import sqlite3 as _sqlite3
        from superlocalmemory.mesh.broker_security import seed_fencing_counter

        # Empty DB with no tables
        db_path = str(tmp_path / "empty_fence.db")
        conn = _sqlite3.connect(db_path)
        conn.close()

        result = seed_fencing_counter(db_path)
        assert result == 0, f"Missing table must return 0 (first-run). Got: {result}"


# ─────────────────────────────────────────────────────────────────────────────
# SEC-7: Config wiring for mesh.strict_identity
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigWiredStrict:
    """SEC-7: is_strict_identity reads mesh.strict_identity config key."""

    def test_reads_config_attribute_when_no_env(self, monkeypatch):
        """Config object with mesh_strict_identity=True → strict mode on."""
        monkeypatch.delenv("SLM_MESH_PRODUCTION", raising=False)
        monkeypatch.delenv("SLM_MESH_STRICT_IDENTITY", raising=False)

        from superlocalmemory.mesh.broker_security import is_strict_identity

        class FakeConfig:
            mesh_strict_identity = True

        assert is_strict_identity(FakeConfig()) is True

    def test_config_false_does_not_override_env(self, monkeypatch):
        """Env var SLM_MESH_STRICT_IDENTITY=1 wins even if config says False."""
        monkeypatch.setenv("SLM_MESH_STRICT_IDENTITY", "1")

        from superlocalmemory.mesh.broker_security import is_strict_identity

        class FakeConfig:
            mesh_strict_identity = False

        assert is_strict_identity(FakeConfig()) is True
        monkeypatch.delenv("SLM_MESH_STRICT_IDENTITY", raising=False)

    def test_no_config_no_env_returns_false(self, monkeypatch):
        """Default (no env, no config): strict is False — backward compat."""
        monkeypatch.delenv("SLM_MESH_PRODUCTION", raising=False)
        monkeypatch.delenv("SLM_MESH_STRICT_IDENTITY", raising=False)

        from superlocalmemory.mesh.broker_security import is_strict_identity

        assert is_strict_identity() is False
        assert is_strict_identity(None) is False


# ─────────────────────────────────────────────────────────────────────────────
# TEST-1: Real loopback — 127.0.0.1 client actually exercises _is_lb=True branch
# ─────────────────────────────────────────────────────────────────────────────


class TestRealLoopback:
    """TEST-1: Real loopback (client=127.0.0.1) correctly bypasses signature check."""

    def setup_method(self):
        from superlocalmemory.mesh.broker_security import _clear_nonce_store
        _clear_nonce_store()

    def test_real_loopback_bypasses_strict_sig_check(self, monkeypatch):
        """127.0.0.1 client with strict mode: no signature required (loopback trusted)."""
        monkeypatch.setenv("SLM_MESH_STRICT_IDENTITY", "1")
        app, lb_broker = _make_app(secret="lb-secret")
        # client=("127.0.0.1", 50000) makes request.client.host = "127.0.0.1"
        c = TestClient(app, client=("127.0.0.1", 50000))

        receiver = lb_broker.register_peer("lb-recv-sess")

        r = c.post(
            "/mesh/send",
            json={
                "from_peer": "lb-sender",
                "to_peer": receiver["peer_id"],
                "content": "loopback strict test",
            },
            headers={"Authorization": "Bearer lb-secret"},  # for _get_broker auth
        )
        assert r.status_code == 200, (
            f"Loopback (127.0.0.1) must bypass strict sig requirement. "
            f"Got: {r.status_code} {r.text}"
        )
        monkeypatch.delenv("SLM_MESH_STRICT_IDENTITY", raising=False)

    def test_non_loopback_testclient_requires_auth(self):
        """TestClient without loopback override ('testclient' host) goes through auth path."""
        app, nb_broker = _make_app(secret="nb-secret")
        c = TestClient(app)  # host = "testclient" — NOT loopback

        receiver = nb_broker.register_peer("nb-recv-sess")
        # No Authorization header → _get_broker must reject
        r = c.post(
            "/mesh/send",
            json={"from_peer": "x", "to_peer": receiver["peer_id"], "content": "test"},
        )
        assert r.status_code == 401, (
            f"Non-loopback without auth must be rejected. Got: {r.status_code}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST-2: Future timestamp rejection
# ─────────────────────────────────────────────────────────────────────────────


class TestFutureTimestamp:
    """TEST-2: Timestamps far in the future are rejected (skew window applies both ways)."""

    def setup_method(self):
        from superlocalmemory.mesh.broker_security import _clear_nonce_store
        _clear_nonce_store()

    def test_future_timestamp_rejected(self):
        """ts > now + MESH_SIG_SKEW_SECONDS must be rejected (prevents future replay)."""
        from superlocalmemory.mesh.broker_security import (
            MESH_SIG_SKEW_SECONDS,
            check_mesh_message_signature,
            sign_mesh_message,
        )

        future_ts = str(int(time.time()) + MESH_SIG_SKEW_SECONDS + 60)
        nonce = secrets.token_hex(8)
        sig = sign_mesh_message("shared", "peerA", "peerB", "hello", nonce, future_ts)

        result = check_mesh_message_signature(
            "shared", "peerA", "peerB", "hello", sig, nonce, future_ts,
            is_loopback=False, strict=True,
        )
        assert result is not None, "Future timestamp beyond skew must be rejected"
        assert result.get("ok") is False
        assert "skew" in result.get("error", "").lower() or "timestamp" in result.get("error", "").lower()
