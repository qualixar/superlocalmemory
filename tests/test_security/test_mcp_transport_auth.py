# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Security contract for the Streamable-HTTP MCP transport.

An MCP protocol session is routing state, not caller authentication.  A
non-loopback HTTP client must therefore prove possession of the configured SLM
API key before it can reach tools that borrow the daemon's local capability.
"""

from __future__ import annotations

import inspect


def test_non_loopback_mcp_requires_a_configured_api_key(tmp_path) -> None:
    from superlocalmemory.infra.auth_middleware import authorize_http_mcp_request

    missing_key_file = tmp_path / "api_key"
    assert authorize_http_mcp_request(
        {}, client_host="192.168.50.20", key_file=missing_key_file
    ) is False


def test_allowlisted_remote_host_is_not_mcp_authentication(
    tmp_path, monkeypatch,
) -> None:
    from superlocalmemory.infra.auth_middleware import authorize_http_mcp_request

    monkeypatch.setenv("SLM_REMOTE", "1")
    monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "192.168.50.0/24")
    assert authorize_http_mcp_request(
        {}, client_host="192.168.50.20", key_file=tmp_path / "api_key"
    ) is False


def test_non_loopback_mcp_accepts_only_the_configured_api_key(tmp_path) -> None:
    from superlocalmemory.infra.auth_middleware import authorize_http_mcp_request

    key_file = tmp_path / "api_key"
    key_file.write_text("correct-secret\n")

    assert authorize_http_mcp_request(
        {"x-slm-api-key": "wrong"},
        client_host="203.0.113.9",
        key_file=key_file,
    ) is False
    assert authorize_http_mcp_request(
        {"x-slm-api-key": "correct-secret"},
        client_host="203.0.113.9",
        key_file=key_file,
    ) is True


def test_loopback_mcp_keeps_local_transport_compatibility(tmp_path) -> None:
    from superlocalmemory.infra.auth_middleware import authorize_http_mcp_request

    assert authorize_http_mcp_request(
        {}, client_host="127.0.0.1", key_file=tmp_path / "api_key"
    ) is True
    assert authorize_http_mcp_request(
        {}, client_host="::1", key_file=tmp_path / "api_key"
    ) is True


def test_unified_daemon_does_not_exempt_mcp_from_authentication() -> None:
    from superlocalmemory.server import unified_daemon

    source = inspect.getsource(unified_daemon._register_dashboard_routes)
    assert '_AUTH_EXEMPT_PREFIXES = ("/v1/", "/v1beta/")' in source
    assert 'startswith(("/v1/", "/v1beta/", "/mcp"))' not in source
    assert "authorize_http_mcp_request" in source
