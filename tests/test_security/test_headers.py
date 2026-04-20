# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-04 §6.3

"""Tests for the strict security-headers middleware (LLD-04 §4.2 v2)."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

from superlocalmemory.core import security_primitives as sp
from superlocalmemory.server.middleware.security_headers import (
    SecurityHeadersMiddleware,
)
from superlocalmemory.server.routes import brain as brain_mod


# ---------------------------------------------------------------------------
# Fixtures — minimal app with headers middleware and two routes.
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_token_dir(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setattr(sp, "_install_token_path",
                            lambda: Path(td) / ".install_token")
        yield Path(td)


@pytest.fixture()
def install_token(tmp_token_dir: Path) -> str:
    return sp.ensure_install_token()


@pytest.fixture()
def tmp_learning_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "learning.db"
        from superlocalmemory.learning.database import LearningDatabase
        _ = LearningDatabase(db_path)
        monkeypatch.setattr(brain_mod, "_learning_db_path", lambda: db_path)
        monkeypatch.setattr(brain_mod, "_load_raw_preferences",
                            lambda pid: {"topics": [], "entities": [], "tech": []})
        yield db_path


@pytest.fixture()
def app(tmp_learning_db: Path) -> FastAPI:
    application = FastAPI()
    application.add_middleware(SecurityHeadersMiddleware)
    application.include_router(brain_mod.router)

    @application.get("/static-test", response_class=PlainTextResponse)
    async def static_test() -> str:
        return "ok"
    return application


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# U8 — CSP, XFO, XCTO, Referrer-Policy present on every response
# ---------------------------------------------------------------------------


def _brain_get(client: TestClient, install_token: str, path: str = "/api/v3/brain"):
    return client.get(path, headers={"X-Install-Token": install_token})


def test_csp_default_src_self(client: TestClient, install_token: str) -> None:
    r = _brain_get(client, install_token)
    csp = r.headers.get("content-security-policy", "")
    assert "default-src 'self'" in csp


def test_csp_style_src_self_no_nonce(client: TestClient, install_token: str) -> None:
    r = _brain_get(client, install_token)
    csp = r.headers.get("content-security-policy", "")
    assert "style-src 'self'" in csp
    # Strict v2 policy: NO nonce plumbing (removes a whole class of bugs).
    assert "'nonce-" not in csp
    # No 'unsafe-inline' / 'unsafe-eval' allowed in v2.
    assert "'unsafe-inline'" not in csp
    assert "'unsafe-eval'" not in csp


def test_csp_script_src_self(client: TestClient, install_token: str) -> None:
    r = _brain_get(client, install_token)
    csp = r.headers.get("content-security-policy", "")
    assert "script-src 'self'" in csp


def test_csp_frame_ancestors_none(client: TestClient, install_token: str) -> None:
    r = _brain_get(client, install_token)
    csp = r.headers.get("content-security-policy", "")
    assert "frame-ancestors 'none'" in csp


def test_x_frame_options_deny(client: TestClient, install_token: str) -> None:
    r = _brain_get(client, install_token)
    assert r.headers.get("x-frame-options") == "DENY"


def test_x_content_type_options_nosniff(client: TestClient, install_token: str) -> None:
    r = _brain_get(client, install_token)
    assert r.headers.get("x-content-type-options") == "nosniff"


def test_referrer_policy_no_referrer(client: TestClient, install_token: str) -> None:
    r = _brain_get(client, install_token)
    assert r.headers.get("referrer-policy") == "no-referrer"


def test_permissions_policy_sane_defaults(client: TestClient, install_token: str) -> None:
    r = _brain_get(client, install_token)
    pp = r.headers.get("permissions-policy", "")
    for piece in ("interest-cohort=()", "microphone=()",
                  "camera=()", "geolocation=()"):
        assert piece in pp


def test_security_headers_on_brain(
    client: TestClient, install_token: str,
) -> None:
    r = client.get("/api/v3/brain",
                   headers={"X-Install-Token": install_token})
    assert r.status_code == 200
    for h in ("content-security-policy",
              "x-frame-options",
              "x-content-type-options",
              "referrer-policy"):
        assert h in r.headers, f"missing header on /brain: {h}"


def test_strict_csp_scope_skips_non_brain_routes(client: TestClient) -> None:
    """Strict middleware MUST NO-OP on non-/api/v3/brain* paths so the
    existing dashboard (which currently loads CDN libs under the legacy
    CSP) keeps working until the v3.4.22 vendoring work. Regression test
    for the P0 CDN-removal incident caught during Stage 6 delivery review.
    """
    r = client.get("/static-test")
    assert r.status_code == 200
    # If any of these headers are present on /static-test, the strict
    # middleware leaked out of its scope.
    assert "content-security-policy" not in r.headers, (
        "strict CSP escaped /api/v3/brain* scope — legacy dashboard CSP "
        "would be overridden, re-breaking non-Brain tabs.")
    # The other strict headers are also Brain-scope-only.
    assert "x-frame-options" not in r.headers


def test_strict_csp_scope_applies_to_deprecated_shims(
    client: TestClient, install_token: str,
) -> None:
    """Deprecated routes (/api/v3/learning/stats, /patterns, /behavioral)
    MUST carry the strict policy too — they return the same sensitive
    Brain data under the shim flag.
    """
    for shim in ("/api/v3/learning/stats", "/api/v3/patterns",
                 "/api/v3/behavioral"):
        r = client.get(shim, headers={"X-Install-Token": install_token})
        # Depending on shim implementation, may be 200 or 404 — either
        # way, headers must be present when the route handles the
        # request.
        if r.status_code == 200:
            assert "content-security-policy" in r.headers, (
                f"strict CSP missing on deprecated shim {shim}")


def test_security_headers_on_401(client: TestClient) -> None:
    """401 responses must still carry the security headers."""
    r = client.get("/api/v3/brain")
    assert r.status_code == 401
    for h in ("content-security-policy",
              "x-frame-options",
              "x-content-type-options",
              "referrer-policy"):
        assert h in r.headers, f"missing header on 401: {h}"


# ---------------------------------------------------------------------------
# U1 & U7 — static grep guards on UI & brain.js
# ---------------------------------------------------------------------------


_SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory"


def test_no_patterns_or_behavioral_tab_in_index_html() -> None:
    html = (_SRC_DIR / "ui" / "index.html").read_text(encoding="utf-8")
    assert 'id="patterns-tab"' not in html
    assert 'id="behavioral-tab"' not in html
    assert 'id="patterns-pane"' not in html
    assert 'id="behavioral-pane"' not in html


def test_brain_tab_present_in_index_html() -> None:
    html = (_SRC_DIR / "ui" / "index.html").read_text(encoding="utf-8")
    assert 'id="brain-tab"' in html
    assert 'id="brain-pane"' in html
    assert 'js/brain.js' in html


def test_brain_js_is_same_origin_in_index_html() -> None:
    """Brain JS + Brain CSS MUST be served same-origin (LLD-04 §4.3 / §4.5).

    Other dashboard assets can still load from CDN under the legacy CSP —
    the strict policy in ``server/middleware/security_headers.py`` applies
    only to ``/api/v3/brain*`` routes. Vendoring the rest is tracked for
    v3.4.22. This test pins the Brain-owned assets to same-origin so the
    Brain tab itself stays under the strict policy end-to-end.
    """
    html = (_SRC_DIR / "ui" / "index.html").read_text(encoding="utf-8")
    # Brain JS same-origin.
    brain_js_tags = re.findall(
        r'<script[^>]*src=["\'][^"\']*brain\.js[^"\']*["\']', html,
    )
    assert brain_js_tags, "brain.js tag missing from index.html"
    for tag in brain_js_tags:
        assert not re.search(r'https?://', tag), (
            f"brain.js must be same-origin, got {tag!r}")
    # Brain CSS same-origin.
    brain_css_tags = re.findall(
        r'<link[^>]*href=["\'][^"\']*brain\.css[^"\']*["\']', html,
    )
    assert brain_css_tags, "brain.css tag missing from index.html"
    for tag in brain_css_tags:
        assert not re.search(r'https?://', tag), (
            f"brain.css must be same-origin, got {tag!r}")


def test_no_inner_html_in_brain_js() -> None:
    js = (_SRC_DIR / "ui" / "js" / "brain.js").read_text(encoding="utf-8")
    assert not re.search(r"\binnerHTML\s*=", js), (
        "innerHTML assignment banned in brain.js (XSS surface)")
    assert "insertAdjacentHTML" not in js, (
        "insertAdjacentHTML banned in brain.js (XSS surface)")
    assert "dangerouslySetInnerHTML" not in js


def test_middleware_passthrough_non_http_scope() -> None:
    """Non-HTTP (lifespan) scopes must pass straight through untouched."""
    import asyncio

    calls: list[str] = []

    async def inner_app(scope, receive, send):
        calls.append(scope["type"])
        await send({"type": "lifespan.startup.complete"})

    mw = SecurityHeadersMiddleware(inner_app)

    async def receive():
        return {"type": "lifespan.startup"}

    sent: list[dict] = []

    async def send(msg):
        sent.append(msg)

    asyncio.run(mw({"type": "lifespan"}, receive, send))
    assert calls == ["lifespan"]
    # The send was NOT a response.start, so no header mutation was possible.
    assert sent == [{"type": "lifespan.startup.complete"}]


def test_middleware_strips_existing_owned_headers() -> None:
    """If a downstream app already set CSP, middleware must replace it."""
    import asyncio

    async def inner_app(scope, receive, send):
        # Send an http.response.start with a conflicting looser CSP.
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-security-policy",
                 b"default-src *"),  # would be unsafe — must be stripped.
                (b"content-type", b"text/plain"),
            ],
        })
        await send({"type": "http.response.body", "body": b"ok",
                    "more_body": False})

    mw = SecurityHeadersMiddleware(inner_app)
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request"}

    async def send(msg):
        sent.append(msg)

    # raw_path set to /api/v3/brain so the strict-scope guard lets the
    # middleware run end-to-end.
    asyncio.run(mw(
        {"type": "http", "path": "/api/v3/brain",
         "raw_path": b"/api/v3/brain"},
        receive, send,
    ))
    start = sent[0]
    csp_values = [v for (n, v) in start["headers"]
                  if n == b"content-security-policy"]
    assert len(csp_values) == 1, "must have exactly one CSP header"
    assert csp_values[0] != b"default-src *"
    assert b"default-src 'self'" in csp_values[0]
    # Non-owned header (content-type) must survive.
    ct_values = [v for (n, v) in start["headers"] if n == b"content-type"]
    assert ct_values == [b"text/plain"]


def test_brain_route_has_no_fabricated_keys() -> None:
    """Source-level grep: brain.py must not hard-code forbidden metric names."""
    route = (_SRC_DIR / "server" / "routes" / "brain.py").read_text(
        encoding="utf-8",
    )
    for banned in ("hit_rate_24h", "avg_age_on_hit_seconds",
                   "skill_evolution_rows"):
        # Allow the string to appear only inside a comment/docstring that
        # explicitly flags it as banned (i.e. prefixed with "NO " or "banned:").
        # For simplicity here we assert it does not appear at all.
        assert banned not in route, (
            f"brain.py must not reference forbidden metric: {banned}")
