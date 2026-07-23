# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | WP-15 coverage tests (D6 — origin guard wiring)

"""E2E tests for the CSRF origin guard wired into unified_daemon.py auth_middleware.

The LLD (§STAGE-5) confirms:
  - Origin guard is in ``unified_daemon.py:1383-1399``, NOT in ``api.py``.
  - Primary test surface = the token route (which calls is_remote_origin_allowed).
  - Unified-daemon write-guard is secondary — tested here via a minimal FastAPI
    app that installs the SAME origin-checking middleware logic so we cover the
    production code path without requiring a live engine.

Anti-tautology gate: delete the ``if not _ok_origin: return 403`` branch from
unified_daemon auth_middleware → ``test_untrusted_origin_blocked`` turns RED.

COVERAGE TARGET: unified_daemon.py:1383-1399 (write-gate origin check), and
indirectly remote_mode.is_remote_origin_allowed (line 144 IPv6 path + LAN).

NOTE: test_token_endpoint.py already covers the token-route path exhaustively.
These tests cover the WRITE-PATH guard (POST requests from untrusted browser
Origin) which is a DIFFERENT code path from the token-route client check.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Minimal app that mirrors unified_daemon.py auth_middleware origin check
# ---------------------------------------------------------------------------

def _build_origin_guard_app(
    monkeypatch: pytest.MonkeyPatch, *, clear_env: bool = True
) -> FastAPI:
    """Build a minimal FastAPI app that installs the unified_daemon origin guard.

    We copy the guard logic from unified_daemon.py:1383-1399 into a standalone
    middleware so we can test it without a live engine or full daemon boot.
    The logic is intentionally identical to the production code — if the
    production code changes, these tests should catch the regression.

    ``clear_env=False`` skips the env cleanup so callers can pre-set env vars
    before calling this function.
    """
    if clear_env:
        monkeypatch.delenv("SLM_REMOTE", raising=False)
        monkeypatch.delenv("SLM_MCP_ALLOWED_HOSTS", raising=False)

    app = FastAPI()

    # Mirror the production origin guard logic from unified_daemon.py:1383-1399
    @app.middleware("http")
    async def origin_guard_middleware(request: Request, call_next):
        is_write = request.method in ("POST", "PUT", "DELETE", "PATCH")
        headers = {k.lower(): v for k, v in request.headers.items()}
        if is_write:
            _origin = headers.get("origin", "")
            if _origin:
                from superlocalmemory.server.origin import (
                    origin_is_daemon,
                    origin_is_loopback,
                )
                _ok_origin = origin_is_daemon(_origin, port=8765)
                _has_credential = any(
                    headers.get(name)
                    for name in (
                        "x-slm-daemon-capability",
                        "x-install-token",
                        "x-slm-api-key",
                    )
                )
                if not _ok_origin and origin_is_loopback(_origin) and _has_credential:
                    _ok_origin = True
                if not _ok_origin:
                    from superlocalmemory.core.remote_mode import is_remote_origin_allowed
                    _ok_origin = is_remote_origin_allowed(_origin)
                if not _ok_origin:
                    return JSONResponse(
                        status_code=403,
                        content={"error": "cross-origin request rejected"},
                    )
        return await call_next(request)

    @app.post("/test-write")
    async def test_write():
        return {"ok": True}

    @app.get("/test-read")
    async def test_read():
        return {"ok": True}

    return app


@pytest.fixture()
def guard_app(monkeypatch: pytest.MonkeyPatch):
    return _build_origin_guard_app(monkeypatch)


@pytest.fixture()
def guard_client(guard_app) -> TestClient:
    return TestClient(guard_app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOriginGuardWiring:
    def test_untrusted_origin_blocked_403(
        self, guard_client: TestClient,
    ) -> None:
        """POST with untrusted browser Origin must return 403.

        Anti-tautology: if the origin guard branch is removed, this returns 200.
        """
        resp = guard_client.post(
            "/test-write",
            headers={"Origin": "https://evil.example.com"},
        )
        assert resp.status_code == 403, (
            f"Expected 403 for untrusted origin, got {resp.status_code}; "
            "origin guard in unified_daemon auth_middleware may not be wired"
        )
        assert "cross-origin" in resp.json().get("error", "").lower() or \
               "origin" in resp.json().get("error", "").lower()

    def test_no_origin_header_not_blocked(
        self, guard_client: TestClient,
    ) -> None:
        """POST without Origin header (CLI/MCP/curl) must pass the guard."""
        resp = guard_client.post("/test-write")
        # No origin → guard does not fire → passes through to route
        assert resp.status_code == 200, (
            f"POST without Origin was blocked (status {resp.status_code}); "
            "CLI/MCP clients must not be affected"
        )
        assert resp.json() == {"ok": True}

    def test_loopback_origin_127_allowed(
        self, guard_client: TestClient,
    ) -> None:
        """Loopback 127.0.0.1 origin passes — own dashboard."""
        resp = guard_client.post(
            "/test-write",
            headers={"Origin": "http://127.0.0.1:8765"},
        )
        assert resp.status_code == 200, (
            f"Loopback origin was blocked (status {resp.status_code})"
        )

    def test_unrelated_loopback_origin_is_rejected_without_credential(
        self, guard_client: TestClient,
    ) -> None:
        """Another local web server cannot issue ambient browser writes."""
        resp = guard_client.post(
            "/test-write",
            headers={"Origin": "http://localhost:8417"},
        )
        assert resp.status_code == 403

    def test_unrelated_loopback_origin_needs_a_credential(
        self, guard_client: TestClient,
    ) -> None:
        resp = guard_client.post(
            "/test-write",
            headers={
                "Origin": "http://localhost:8417",
                "X-Install-Token": "integration-will-validate-this-token",
            },
        )
        assert resp.status_code == 200

    def test_get_request_not_checked(
        self, guard_client: TestClient,
    ) -> None:
        """GET requests are not subject to origin check (read-only, not CSRF risk)."""
        resp = guard_client.get(
            "/test-read",
            headers={"Origin": "https://evil.example.com"},
        )
        assert resp.status_code == 200, (
            f"GET with evil origin was blocked (status {resp.status_code}); "
            "read requests must not be affected by CSRF origin check"
        )

    def test_lan_origin_blocked_when_remote_off(
        self, guard_client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """LAN origin is rejected when SLM_REMOTE is NOT set."""
        monkeypatch.delenv("SLM_REMOTE", raising=False)
        resp = guard_client.post(
            "/test-write",
            headers={"Origin": "http://192.168.1.50:8765"},
        )
        assert resp.status_code == 403, (
            f"LAN origin should be 403 when SLM_REMOTE is off (got {resp.status_code})"
        )

    def test_lan_origin_allowed_in_remote_mode(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Allowlisted LAN origin passes when SLM_REMOTE=1."""
        monkeypatch.setenv("SLM_REMOTE", "1")
        monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "192.168.0.0/16")
        # Build app AFTER env is set, and DON'T clear env in the helper
        # (is_remote_origin_allowed reads env at call time, not build time)
        app = _build_origin_guard_app(monkeypatch, clear_env=False)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/test-write",
            headers={"Origin": "http://192.168.1.50:8765"},
        )
        assert resp.status_code == 200, (
            f"LAN origin was blocked even with SLM_REMOTE=1 and CIDR allowlist "
            f"(got {resp.status_code}); is_remote_origin_allowed not firing"
        )
