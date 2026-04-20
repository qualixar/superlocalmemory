"""Tests for ``/internal/token`` — the auto-inject install-token route.

Contract (v3.4.22 post-ship UX fix):
- Loopback caller (absent or loopback Origin) gets 200 + ``{token: "..."}``
- Non-loopback caller → 403 "loopback only"
- Loopback caller with non-loopback Origin → 403 "origin not allowed"
- Missing / empty token file → 500 "token_unavailable" (never echoes path)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterator

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.core import security_primitives as sp
from superlocalmemory.hooks import prewarm_auth as auth_mod
from superlocalmemory.server.routes import token as token_mod


@pytest.fixture()
def tmp_token_dir(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as td:
        token_path = Path(td) / ".install_token"
        monkeypatch.setattr(sp, "_install_token_path", lambda: token_path)
        yield token_path


@pytest.fixture()
def install_token(tmp_token_dir: Path) -> str:
    return sp.ensure_install_token()


@pytest.fixture()
def app() -> FastAPI:
    application = FastAPI()
    application.include_router(token_mod.router)
    return application


@pytest.fixture()
def loopback_client_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """FastAPI TestClient uses ``testclient`` as the client host.

    Production ``is_loopback`` only accepts real loopback literals. For
    the endpoint happy-path tests we monkeypatch the authority so the
    TestClient reaches the handler body. The explicit "non-loopback"
    test overrides this back to ``lambda _: False``.
    """
    monkeypatch.setattr(auth_mod, "is_loopback", lambda host: True)


@pytest.fixture()
def client(
    app: FastAPI, loopback_client_host: None,
) -> TestClient:
    return TestClient(app)


def test_loopback_no_origin_returns_token(
    client: TestClient, install_token: str,
) -> None:
    resp = client.get("/internal/token")
    assert resp.status_code == 200
    body = resp.json()
    assert body["token"] == install_token


def test_loopback_with_loopback_origin_returns_token(
    client: TestClient, install_token: str,
) -> None:
    resp = client.get(
        "/internal/token",
        headers={"Origin": "http://127.0.0.1:8765"},
    )
    assert resp.status_code == 200
    assert resp.json()["token"] == install_token


def test_loopback_with_localhost_origin_returns_token(
    client: TestClient, install_token: str,
) -> None:
    resp = client.get(
        "/internal/token",
        headers={"Origin": "http://localhost:8765"},
    )
    assert resp.status_code == 200
    assert resp.json()["token"] == install_token


def test_foreign_origin_rejected(
    client: TestClient, install_token: str,
) -> None:
    resp = client.get(
        "/internal/token",
        headers={"Origin": "https://evil.example.com"},
    )
    assert resp.status_code == 403
    assert resp.json()["error"] == "origin not allowed"


def test_non_loopback_rejected(
    client: TestClient,
    install_token: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth_mod, "is_loopback", lambda _host: False)
    resp = client.get("/internal/token")
    assert resp.status_code == 403
    assert resp.json()["error"] == "loopback only"


def test_missing_token_file_returns_500(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sp, "_install_token_path", lambda: tmp_path / "does-not-exist",
    )
    resp = client.get("/internal/token")
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "token_unavailable"
    # Never leak the real path or system info.
    assert "does-not-exist" not in str(body)
    assert str(tmp_path) not in str(body)


def test_empty_token_file_returns_500(
    client: TestClient,
    tmp_token_dir: Path,
) -> None:
    tmp_token_dir.write_text("", encoding="utf-8")
    resp = client.get("/internal/token")
    assert resp.status_code == 500
    assert resp.json()["error"] == "token_unavailable"
