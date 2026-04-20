# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-01 §4.5 auth primitives

"""Tests for the /internal/prewarm authentication primitives.

LLD-01 §4.5 requires four gates BEFORE any work is done:
  1. Loopback-only (client addr in 127.0.0.1 / ::1).
  2. Origin-header CSRF guard (browser origin → reject).
  3. Install-token match (X-SLM-Hook-Token header).
  4. Request-size + schema validation.

These tests exercise the gates in isolation via
``superlocalmemory.hooks.prewarm_auth`` so they can run without a live
FastAPI daemon. The HTTP route composes these primitives.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.core import security_primitives as sp
from superlocalmemory.hooks import prewarm_auth as pa


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    slm_home = tmp_path / ".superlocalmemory"
    slm_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sp, "_install_token_path",
                        lambda: slm_home / ".install_token")
    return slm_home


# ---------------------------------------------------------------------------
# Loopback check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_is_loopback_accepts_loopback(host: str) -> None:
    assert pa.is_loopback(host) is True


@pytest.mark.parametrize("host", [
    "0.0.0.0", "192.168.1.5", "10.0.0.1", "8.8.8.8", "example.com", "",
])
def test_is_loopback_rejects_non_loopback(host: str) -> None:
    assert pa.is_loopback(host) is False


# ---------------------------------------------------------------------------
# Origin CSRF guard
# ---------------------------------------------------------------------------


def test_origin_guard_blocks_browser_request() -> None:
    headers = {"Origin": "http://evil.com"}
    assert pa.is_browser_originated(headers) is True


def test_origin_guard_case_insensitive_header_name() -> None:
    headers = {"origin": "http://evil.com"}
    assert pa.is_browser_originated(headers) is True


def test_origin_guard_allows_hook_request() -> None:
    # Hooks using urllib do NOT set an Origin header.
    headers = {"Content-Type": "application/json",
                "X-SLM-Hook-Token": "deadbeef"}
    assert pa.is_browser_originated(headers) is False


def test_origin_guard_allows_empty_origin() -> None:
    # An explicit empty-string Origin is treated as "no browser" per spec:
    # real browsers always send a non-empty Origin for CORS-protected calls.
    headers = {"Origin": ""}
    assert pa.is_browser_originated(headers) is False


# ---------------------------------------------------------------------------
# Install-token auth gate
# ---------------------------------------------------------------------------


def test_authorize_accepts_valid_token(home: Path) -> None:
    token = sp.ensure_install_token()
    decision = pa.authorize(
        client_host="127.0.0.1",
        headers={"X-SLM-Hook-Token": token, "Content-Type": "application/json"},
    )
    assert decision.allowed is True
    assert decision.status == 200


def test_authorize_rejects_missing_token(home: Path) -> None:
    sp.ensure_install_token()
    decision = pa.authorize(
        client_host="127.0.0.1",
        headers={"Content-Type": "application/json"},
    )
    assert decision.allowed is False
    assert decision.status == 401
    assert "unauthorized" in decision.reason.lower()


def test_authorize_rejects_wrong_token(home: Path) -> None:
    sp.ensure_install_token()
    decision = pa.authorize(
        client_host="127.0.0.1",
        headers={"X-SLM-Hook-Token": "a" * 64},
    )
    assert decision.allowed is False
    assert decision.status == 401


def test_authorize_rejects_browser_origin_before_token(home: Path) -> None:
    token = sp.ensure_install_token()
    decision = pa.authorize(
        client_host="127.0.0.1",
        headers={"X-SLM-Hook-Token": token, "Origin": "http://evil.com"},
    )
    assert decision.allowed is False
    assert decision.status == 403
    assert "origin" in decision.reason.lower()


def test_authorize_rejects_non_loopback_client(home: Path) -> None:
    token = sp.ensure_install_token()
    decision = pa.authorize(
        client_host="10.0.0.5",
        headers={"X-SLM-Hook-Token": token},
    )
    assert decision.allowed is False
    assert decision.status == 403
    assert "loopback" in decision.reason.lower()


def test_authorize_rejects_when_token_file_missing(home: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure no token exists.
    token_path = home / ".install_token"
    if token_path.exists():
        token_path.unlink()
    decision = pa.authorize(
        client_host="127.0.0.1",
        headers={"X-SLM-Hook-Token": "anything"},
    )
    assert decision.allowed is False
    assert decision.status == 401


# ---------------------------------------------------------------------------
# Body-size gate
# ---------------------------------------------------------------------------


def test_check_body_size_rejects_oversize() -> None:
    body = b"x" * (pa.MAX_BODY_BYTES + 1)
    ok, reason = pa.check_body_size(body)
    assert ok is False
    assert "size" in reason.lower()


def test_check_body_size_accepts_in_range() -> None:
    body = b"x" * 128
    ok, reason = pa.check_body_size(body)
    assert ok is True
    assert reason == ""


# ---------------------------------------------------------------------------
# Header-name case sensitivity — the realistic middleware dict is usually
# case-insensitive, but some frameworks normalize differently. Our primitive
# accepts either casing.
# ---------------------------------------------------------------------------


def test_get_token_accepts_variant_casing() -> None:
    for header_name in ("X-SLM-Hook-Token", "x-slm-hook-token",
                         "X-Slm-Hook-Token"):
        assert pa._extract_token({header_name: "abc"}) == "abc"


def test_get_token_returns_empty_for_missing() -> None:
    assert pa._extract_token({"Content-Type": "json"}) == ""


def test_get_token_returns_empty_for_empty_headers() -> None:
    assert pa._extract_token({}) == ""


def test_origin_guard_handles_empty_headers() -> None:
    assert pa.is_browser_originated({}) is False


def test_check_body_size_rejects_non_bytes() -> None:
    ok, reason = pa.check_body_size("a string")  # type: ignore[arg-type]
    assert ok is False
    assert "bytes" in reason.lower()
