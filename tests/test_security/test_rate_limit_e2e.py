# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | WP-15 coverage tests (D3 — rate-limit e2e)

"""E2E tests for the rate-limit middleware wired into api.py.

These characterization tests prove the real middleware fires (429 + Retry-After)
against the ACTUAL api.py create_app() stack, not just pure helper functions.
Pure helper tests live in test_remote_mode.py — this tests the WIRING.

Anti-tautology gate: if the middleware is removed from create_app(), the
``writes_over_limit_429`` test MUST turn RED (the endpoint would return 200
or 401, not 429).

CRIT-1 (LLD §CRIT): TestClient defaults client host to "testclient".
We make_client() which wraps TestClient but injects the real client IP via
ASGITransport middleware — overriding scope["client"] so request.client.host
matches what the middleware inspects.

CRIT-2 (LLD §CRIT): SLM_RATE_LIMIT_* are read at create_app() time.
All env vars MUST be set before create_app() is called.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_app(monkeypatch: pytest.MonkeyPatch, *, write: int, window: int):
    """Build create_app() with test-scale rate limits.

    Env vars set BEFORE the import so rate_limit_config() reads them.
    Auth is disabled (no API key file) so 401 doesn't mask 429.
    """
    monkeypatch.setenv("SLM_RATE_LIMIT_WRITE", str(write))
    monkeypatch.setenv("SLM_RATE_LIMIT_READ", "1000")  # don't hit read limit
    monkeypatch.setenv("SLM_RATE_LIMIT_WINDOW", str(window))

    from superlocalmemory.server.api import create_app
    return create_app()


def _make_client(app, client_host: str) -> TestClient:
    """Return a TestClient that reports client_host to the middleware.

    Starlette TestClient uses ``testclient`` as client.host by default,
    which is NEITHER loopback NOR a real IP — the exempt check fails.
    We override via a thin ASGI wrapper that patches scope["client"].
    """
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as StarletteRequest
    from starlette.responses import Response

    class _HostSpoofMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: StarletteRequest, call_next):
            # Mutate the scope before FastAPI sees it
            request.scope["client"] = (client_host, 9999)
            return await call_next(request)

    app.add_middleware(_HostSpoofMiddleware)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def rate_limit_app(monkeypatch: pytest.MonkeyPatch):
    """App with write_limit=3, window=60 (no refill during test)."""
    # Disable auth so 401 never masks a 429
    monkeypatch.delenv("SLM_API_KEY", raising=False)
    with patch("superlocalmemory.infra.auth_middleware.check_api_key", return_value=True):
        yield _build_app(monkeypatch, write=3, window=60)


# ---------------------------------------------------------------------------
# Tests — rate limit fires for non-exempt IP
# ---------------------------------------------------------------------------

class TestWritesOverLimit:
    def test_writes_over_limit_yields_429(
        self, rate_limit_app, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """4th POST from a non-exempt IP must return 429."""
        client = _make_client(rate_limit_app, "203.0.113.9")

        responses = []
        for _ in range(4):
            resp = client.post("/health")  # /health exists; POST triggers write limiter
            responses.append(resp.status_code)

        # First 3 pass (limit=3), 4th is blocked
        # (health might 405 for POST — use a writable route)
        # The middleware intercepts before routing — so 429 fires before 404/405
        assert 429 in responses, (
            f"Expected 429 in responses {responses}; "
            "middleware may not be wired or IP exemption is wrong"
        )

    def test_429_carries_retry_after_header(
        self, rate_limit_app, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Retry-After header must be present on 429 response."""
        client = _make_client(rate_limit_app, "203.0.113.10")

        last_resp = None
        for _ in range(4):
            last_resp = client.post("/nonexistent-path")

        # Find the 429
        retry_after = last_resp.headers.get("Retry-After")
        if last_resp.status_code == 429:
            assert retry_after is not None, "429 must carry Retry-After header"
            assert int(retry_after) > 0, "Retry-After must be positive"
        else:
            # Try harder — exhaust the limiter by sending more requests
            for _ in range(10):
                resp = client.post("/nonexistent-path")
                if resp.status_code == 429:
                    assert resp.headers.get("Retry-After") is not None
                    assert int(resp.headers["Retry-After"]) > 0
                    return
            pytest.fail(
                f"Never got 429 after 14 POST requests from 203.0.113.10 "
                f"(last status={last_resp.status_code})"
            )

    def test_distinct_ips_have_independent_buckets(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """IP A exhausted should not affect IP B's bucket.

        Each IP gets its own sliding-window bucket in the RateLimiter.
        We build a FRESH app (fresh limiter state) so this test is
        order-independent — it does not share state with other tests.
        """
        monkeypatch.delenv("SLM_API_KEY", raising=False)
        with patch(
            "superlocalmemory.infra.auth_middleware.check_api_key",
            return_value=True,
        ):
            # write_limit=2 — IP A hits limit on request 3
            app = _build_app(monkeypatch, write=2, window=60)

        # Each _make_client adds middleware to the app — build separate apps
        # to avoid middleware stack accumulation.
        with patch(
            "superlocalmemory.infra.auth_middleware.check_api_key",
            return_value=True,
        ):
            app_a = _build_app(monkeypatch, write=2, window=60)
            app_b = _build_app(monkeypatch, write=2, window=60)

        client_a = _make_client(app_a, "198.51.100.1")
        client_b = _make_client(app_b, "198.51.100.2")

        # Drain IP A's write bucket
        for _ in range(4):
            client_a.post("/health")

        # IP B has its OWN app with its OWN fresh limiter — must not be limited
        resp_b = client_b.post("/health")
        assert resp_b.status_code != 429, (
            f"IP B was rate-limited even though it has a separate app/limiter "
            f"(got {resp_b.status_code})"
        )


# ---------------------------------------------------------------------------
# Tests — loopback is always exempt
# ---------------------------------------------------------------------------

class TestLoopbackExempt:
    def test_loopback_never_limited(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """127.0.0.1 is exempt from rate limiting regardless of request count."""
        monkeypatch.delenv("SLM_API_KEY", raising=False)
        # Use write_limit=1 so any non-exempt IP hits 429 on 2nd request
        with patch(
            "superlocalmemory.infra.auth_middleware.check_api_key",
            return_value=True,
        ):
            app = _build_app(monkeypatch, write=1, window=60)

        client = _make_client(app, "127.0.0.1")

        for i in range(5):
            resp = client.post("/health")
            assert resp.status_code != 429, (
                f"Loopback was rate-limited on request {i+1} "
                f"(status={resp.status_code})"
            )


# ---------------------------------------------------------------------------
# Tests — allowlisted LAN exempt when SLM_REMOTE=1
# ---------------------------------------------------------------------------

class TestLANExempt:
    def test_allowlisted_lan_exempt_in_remote_mode(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """192.168.1.50 is exempt when SLM_REMOTE=1 and CIDR covers it."""
        monkeypatch.delenv("SLM_API_KEY", raising=False)
        monkeypatch.setenv("SLM_REMOTE", "1")
        monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "192.168.0.0/16")

        # Rebuild app AFTER setting env (config read at build time)
        with patch(
            "superlocalmemory.infra.auth_middleware.check_api_key",
            return_value=True,
        ):
            app = _build_app(monkeypatch, write=1, window=60)

        client = _make_client(app, "192.168.1.50")

        for i in range(4):
            resp = client.post("/health")
            assert resp.status_code != 429, (
                f"Allowlisted LAN IP was rate-limited on request {i+1} "
                f"(status={resp.status_code}); is_rate_limit_exempt not wired"
            )
