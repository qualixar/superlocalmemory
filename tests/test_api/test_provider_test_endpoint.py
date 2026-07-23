"""Tests for ``POST /api/v3/provider/test`` — LLM provider connectivity check.

Issue #39 Issue 2 regression: a custom OpenAI-compatible endpoint
(llama.cpp / LM Studio) with an EMPTY api key must be probed WITHOUT an
``Authorization`` header — never fall through to the official OpenAI path
and 401. The dashboard now sends ``base_url``; the backend must honor it.
"""

from __future__ import annotations

import httpx
import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.server.routes import v3_api


class _FakeResp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=self)


class _FakeClient:
    """Captures the headers/url of the last request for assertions."""

    last_post_headers: dict | None = None
    last_post_url: str | None = None
    post_status: int = 200

    def __init__(self, *a, **k) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a) -> None:
        return None

    def post(self, url, headers=None, json=None):
        _FakeClient.last_post_headers = dict(headers or {})
        _FakeClient.last_post_url = url
        return _FakeResp(_FakeClient.post_status)

    def get(self, url, headers=None):
        # Official path would call api.openai.com — fail loudly so a regression
        # (custom endpoint ignored) surfaces instead of silently passing.
        raise AssertionError(f"unexpected official-endpoint GET to {url}")


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    _FakeClient.last_post_headers = None
    _FakeClient.last_post_url = None
    _FakeClient.post_status = 200
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    # The SSRF guard (v3.6.12) blocks private targets for non-loopback callers,
    # and TestClient's client host is "testclient" (non-loopback). The real
    # dashboard calls from loopback, so bypass the guard here to test the
    # connectivity logic; the guard itself is unit-tested separately below.
    monkeypatch.setattr(v3_api, "_validate_provider_url", lambda url, host: None)
    app = FastAPI()
    app.include_router(v3_api.router)
    return TestClient(app)


# -- SSRF guard unit tests (v3.6.12 ssrf-1) ----------------------------------

def test_ssrf_loopback_caller_may_target_private() -> None:
    assert v3_api._validate_provider_url("http://192.168.50.140:8043/v1", "127.0.0.1") is None


def test_ssrf_remote_caller_blocked_from_private() -> None:
    assert v3_api._validate_provider_url("http://192.168.50.140:8043/v1", "10.0.0.9") is not None
    assert v3_api._validate_provider_url("http://127.0.0.1:11434", "10.0.0.9") is not None


def test_ssrf_metadata_always_blocked() -> None:
    assert v3_api._validate_provider_url("http://169.254.169.254/latest/meta-data", "127.0.0.1") is not None


def test_ssrf_scheme_enforced() -> None:
    assert v3_api._validate_provider_url("file:///etc/passwd", "127.0.0.1") is not None
    assert v3_api._validate_provider_url("gopher://x/", "10.0.0.9") is not None


def test_ssrf_remote_caller_public_ok() -> None:
    assert v3_api._validate_provider_url("https://api.openai.com/v1", "10.0.0.9") is None


def test_ssrf_allowlisted_lan_client_may_target_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#40 residue: in SLM_REMOTE mode an allowlisted LAN dashboard may probe
    its own LAN LLM endpoint, exactly like the loopback dashboard does."""
    monkeypatch.setenv("SLM_REMOTE", "1")
    monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "192.168.50.144")
    assert v3_api._validate_provider_url(
        "http://192.168.50.140:8043/v1", "192.168.50.144",
    ) is None


def test_ssrf_non_allowlisted_remote_still_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote mode must NOT open the guard to clients outside the allowlist."""
    monkeypatch.setenv("SLM_REMOTE", "1")
    monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "192.168.50.144")
    assert v3_api._validate_provider_url(
        "http://192.168.50.140:8043/v1", "10.9.9.9",
    ) is not None


def test_ssrf_allowlist_ignored_when_remote_mode_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With SLM_REMOTE unset, the allowlist is inert and the guard holds."""
    monkeypatch.delenv("SLM_REMOTE", raising=False)
    monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "192.168.50.144")
    assert v3_api._validate_provider_url(
        "http://192.168.50.140:8043/v1", "192.168.50.144",
    ) is not None


def test_embedding_probe_reuses_private_target_ssrf_guard() -> None:
    """Embedding connectivity probes must not become an alternate SSRF path."""
    app = FastAPI()
    app.include_router(v3_api.router)
    response = TestClient(app).post(
        "/api/v3/embedding/test",
        json={"api_endpoint": "http://127.0.0.1:11434"},
    )
    assert response.status_code == 400
    assert "Internal/private" in response.json()["error"]


def test_custom_endpoint_empty_key_no_auth_header(client: TestClient) -> None:
    resp = client.post("/api/v3/provider/test", json={
        "provider": "openai",
        "model": "local-model",
        "base_url": "http://192.168.50.140:8043/v1",
        # no api_key
    })
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    # The critical assertion: NO Authorization header on an empty key.
    assert "Authorization" not in (_FakeClient.last_post_headers or {})
    assert "chat/completions" in (_FakeClient.last_post_url or "")


def test_custom_endpoint_with_key_sends_auth(client: TestClient) -> None:
    resp = client.post("/api/v3/provider/test", json={
        "provider": "openai",
        "model": "local-model",
        "base_url": "http://192.168.50.140:8043/v1",
        "api_key": "sk-test",
    })
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert _FakeClient.last_post_headers.get("Authorization") == "Bearer sk-test"


def test_custom_endpoint_via_endpoint_field(client: TestClient) -> None:
    """Backend accepts the legacy ``endpoint`` field too (frontend sends both)."""
    resp = client.post("/api/v3/provider/test", json={
        "provider": "openai",
        "model": "local-model",
        "endpoint": "http://192.168.50.140:8043/v1",
    })
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert "Authorization" not in (_FakeClient.last_post_headers or {})
