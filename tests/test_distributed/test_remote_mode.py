"""Tests for SLM_REMOTE distributed/LAN mode (issue #39).

Contract:
- Default (no env): remote OFF, stateless OFF, no LAN client allowed.
- SLM_REMOTE=1 + allowlist: matching LAN clients/origins allowed; non-matching denied.
- SLM_MCP_STATELESS=1 alone: stateless ON, remote still OFF (no token opening).
- Allowlist matching supports exact IP, CIDR, prefix wildcard, and '*'.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core import remote_mode as rm


# -- is_remote_mode / mcp_stateless ------------------------------------------

def test_remote_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLM_REMOTE", raising=False)
    assert rm.is_remote_mode() is False


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
def test_remote_truthy_values(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    monkeypatch.setenv("SLM_REMOTE", val)
    assert rm.is_remote_mode() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "", "off", "maybe"])
def test_remote_falsy_values(monkeypatch: pytest.MonkeyPatch, val: str) -> None:
    monkeypatch.setenv("SLM_REMOTE", val)
    assert rm.is_remote_mode() is False


def test_stateless_implied_by_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_REMOTE", "1")
    monkeypatch.delenv("SLM_MCP_STATELESS", raising=False)
    assert rm.mcp_stateless() is True


def test_stateless_granular_without_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLM_REMOTE", raising=False)
    monkeypatch.setenv("SLM_MCP_STATELESS", "1")
    assert rm.mcp_stateless() is True
    assert rm.is_remote_mode() is False  # stateless must NOT open the token endpoint


def test_stateless_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLM_REMOTE", raising=False)
    monkeypatch.delenv("SLM_MCP_STATELESS", raising=False)
    assert rm.mcp_stateless() is False


# -- is_lan_client_allowed ----------------------------------------------------

def test_lan_denied_when_remote_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLM_REMOTE", raising=False)
    monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "*")
    assert rm.is_lan_client_allowed("192.168.50.144") is False


def test_lan_denied_when_allowlist_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_REMOTE", "1")
    monkeypatch.delenv("SLM_MCP_ALLOWED_HOSTS", raising=False)
    assert rm.is_lan_client_allowed("192.168.50.144") is False


def test_lan_wildcard_star(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_REMOTE", "1")
    monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "*")
    assert rm.is_lan_client_allowed("192.168.50.144") is True


def test_lan_exact_ip_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_REMOTE", "1")
    monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "192.168.50.144:*")
    assert rm.is_lan_client_allowed("192.168.50.144") is True
    assert rm.is_lan_client_allowed("192.168.50.99") is False


def test_lan_cidr_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_REMOTE", "1")
    monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "192.168.50.0/24")
    assert rm.is_lan_client_allowed("192.168.50.144") is True
    assert rm.is_lan_client_allowed("192.168.51.1") is False


def test_lan_prefix_wildcard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_REMOTE", "1")
    monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "192.168.*")
    assert rm.is_lan_client_allowed("192.168.50.144") is True
    assert rm.is_lan_client_allowed("10.0.0.1") is False


def test_lan_empty_client_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_REMOTE", "1")
    monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "*")
    assert rm.is_lan_client_allowed("") is False


# -- is_remote_origin_allowed -------------------------------------------------

def test_origin_allowed_lan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_REMOTE", "1")
    monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "192.168.50.144:*")
    assert rm.is_remote_origin_allowed("http://192.168.50.144:8765") is True


def test_origin_denied_foreign(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_REMOTE", "1")
    monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "192.168.50.144:*")
    assert rm.is_remote_origin_allowed("https://evil.example.com") is False


def test_origin_denied_when_remote_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLM_REMOTE", raising=False)
    monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "*")
    assert rm.is_remote_origin_allowed("http://192.168.50.144:8765") is False


# -- rate_limit_config (issue #40 Issue 3) -----------------------------------

def test_rate_limit_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("SLM_RATE_LIMIT_WRITE", "SLM_RATE_LIMIT_READ", "SLM_RATE_LIMIT_WINDOW"):
        monkeypatch.delenv(k, raising=False)
    assert rm.rate_limit_config() == (30, 120, 60)


def test_rate_limit_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_RATE_LIMIT_WRITE", "100")
    monkeypatch.setenv("SLM_RATE_LIMIT_READ", "500")
    monkeypatch.setenv("SLM_RATE_LIMIT_WINDOW", "30")
    assert rm.rate_limit_config() == (100, 500, 30)


def test_rate_limit_invalid_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_RATE_LIMIT_WRITE", "abc")
    monkeypatch.setenv("SLM_RATE_LIMIT_READ", "-5")
    monkeypatch.setenv("SLM_RATE_LIMIT_WINDOW", "0")
    assert rm.rate_limit_config() == (30, 120, 60)


# -- is_rate_limit_exempt -----------------------------------------------------

@pytest.mark.parametrize("ip", ["127.0.0.1", "::1", "localhost"])
def test_loopback_always_exempt(monkeypatch: pytest.MonkeyPatch, ip: str) -> None:
    monkeypatch.delenv("SLM_REMOTE", raising=False)
    assert rm.is_rate_limit_exempt(ip) is True


def test_lan_not_exempt_without_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLM_REMOTE", raising=False)
    monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "*")
    assert rm.is_rate_limit_exempt("192.168.50.144") is False


def test_lan_exempt_in_remote_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_REMOTE", "1")
    monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "192.168.50.0/24")
    assert rm.is_rate_limit_exempt("192.168.50.144") is True
    assert rm.is_rate_limit_exempt("10.9.9.9") is False
