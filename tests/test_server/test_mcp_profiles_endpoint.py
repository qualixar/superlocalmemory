# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""TDD: GET /api/v3/mcp/profiles — MCP profile summary endpoint.

Written RED-first: all tests fail before the endpoint exists.
Run: SLM_TEST_ISOLATION=1 pytest tests/test_server/test_mcp_profiles_endpoint.py -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from superlocalmemory.server.profile_runtime import bind_profile_runtime
from superlocalmemory.server.unified_daemon import create_app

_ENDPOINT = "/api/v3/mcp/profiles"

_EXPECTED_COUNTS = {
    "core": 14,
    "code": 21,
    "full": 39,
    "power": 51,
    "mesh": 8,
}


@pytest.fixture
def trusted_client(engine_with_mock_deps):
    """Daemon app bound to the seeded engine; TestClient == trusted loopback."""
    engine = engine_with_mock_deps
    engine.profile_id = "default"
    engine._config.active_profile = "default"
    engine._db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
        ("default", "default"),
    )
    if hasattr(engine._db, "commit"):
        engine._db.commit()

    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    bind_profile_runtime(app.state, engine, engine._config)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def test_mcp_profiles_returns_200(trusted_client, monkeypatch):
    """Trusted loopback caller must get 200."""
    monkeypatch.delenv("SLM_MCP_PROFILE", raising=False)
    resp = trusted_client.get(_ENDPOINT)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"


def test_mcp_profiles_not_forbidden_for_local_owner(trusted_client, monkeypatch):
    """Must not 403 the machine owner — regression guard (matches auth pattern)."""
    monkeypatch.delenv("SLM_MCP_PROFILE", raising=False)
    resp = trusted_client.get(_ENDPOINT)
    assert resp.status_code != 403, (
        "Local owner was denied — a header-required gate was applied to a "
        "header-less dashboard GET."
    )


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------

def test_mcp_profiles_response_shape(trusted_client, monkeypatch):
    """Response must have current, profiles, aliases, total_tools."""
    monkeypatch.delenv("SLM_MCP_PROFILE", raising=False)
    data = trusted_client.get(_ENDPOINT).json()
    assert "current" in data, "missing 'current'"
    assert "profiles" in data, "missing 'profiles'"
    assert "aliases" in data, "missing 'aliases'"
    assert "total_tools" in data, "missing 'total_tools'"


def test_mcp_profiles_all_named_profiles_present(trusted_client, monkeypatch):
    """All five named profiles must appear in the response."""
    monkeypatch.delenv("SLM_MCP_PROFILE", raising=False)
    data = trusted_client.get(_ENDPOINT).json()
    assert set(data["profiles"].keys()) == {"core", "code", "full", "power", "mesh"}, (
        f"profile keys mismatch: {set(data['profiles'].keys())}"
    )


# ---------------------------------------------------------------------------
# Counts (the hard numbers from the spec)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("profile,expected_count", list(_EXPECTED_COUNTS.items()))
def test_mcp_profile_count(trusted_client, monkeypatch, profile, expected_count):
    """Each profile must report the documented tool count."""
    monkeypatch.delenv("SLM_MCP_PROFILE", raising=False)
    data = trusted_client.get(_ENDPOINT).json()
    actual = data["profiles"][profile]["count"]
    assert actual == expected_count, (
        f"profile '{profile}': expected {expected_count} tools, got {actual}"
    )


# ---------------------------------------------------------------------------
# Tool lists
# ---------------------------------------------------------------------------

def test_mcp_profiles_tool_lists_non_empty(trusted_client, monkeypatch):
    """Every profile must expose a non-empty tool list."""
    monkeypatch.delenv("SLM_MCP_PROFILE", raising=False)
    data = trusted_client.get(_ENDPOINT).json()
    for name, info in data["profiles"].items():
        assert info["tools"], f"profile '{name}' has empty tool list"
        assert isinstance(info["tools"], list)


def test_mcp_profiles_tool_lists_sorted(trusted_client, monkeypatch):
    """Tool lists must be sorted alphabetically (deterministic UI ordering)."""
    monkeypatch.delenv("SLM_MCP_PROFILE", raising=False)
    data = trusted_client.get(_ENDPOINT).json()
    for name, info in data["profiles"].items():
        tools = info["tools"]
        assert tools == sorted(tools), f"profile '{name}' tool list is not sorted"


def test_mcp_profiles_descriptions_present(trusted_client, monkeypatch):
    """Each profile must have a non-empty plain-English description."""
    monkeypatch.delenv("SLM_MCP_PROFILE", raising=False)
    data = trusted_client.get(_ENDPOINT).json()
    for name, info in data["profiles"].items():
        assert "description" in info, f"profile '{name}' missing description"
        assert info["description"], f"profile '{name}' has empty description"


# ---------------------------------------------------------------------------
# Current profile resolution
# ---------------------------------------------------------------------------

def test_mcp_profiles_current_defaults_to_core(trusted_client, monkeypatch):
    """When SLM_MCP_PROFILE is unset, current must be 'core'."""
    monkeypatch.delenv("SLM_MCP_PROFILE", raising=False)
    data = trusted_client.get(_ENDPOINT).json()
    assert data["current"] == "core", (
        f"expected 'core' for unset SLM_MCP_PROFILE, got {data['current']!r}"
    )


def test_mcp_profiles_current_respects_env(trusted_client, monkeypatch):
    """When SLM_MCP_PROFILE is set to a known profile, current must match."""
    monkeypatch.setenv("SLM_MCP_PROFILE", "full")
    data = trusted_client.get(_ENDPOINT).json()
    assert data["current"] == "full", (
        f"expected 'full', got {data['current']!r}"
    )


def test_mcp_profiles_current_resolves_alias(trusted_client, monkeypatch):
    """Legacy alias names (e.g. code21) must resolve to canonical name."""
    monkeypatch.setenv("SLM_MCP_PROFILE", "code21")
    data = trusted_client.get(_ENDPOINT).json()
    assert data["current"] == "code", (
        f"expected 'code' for alias 'code21', got {data['current']!r}"
    )


def test_mcp_profiles_current_is_a_valid_profile_key(trusted_client, monkeypatch):
    """'current' value must always be a key in the profiles dict."""
    monkeypatch.delenv("SLM_MCP_PROFILE", raising=False)
    data = trusted_client.get(_ENDPOINT).json()
    assert data["current"] in data["profiles"], (
        f"'current' ({data['current']!r}) not found in profiles keys"
    )


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------

def test_mcp_profiles_aliases_non_empty(trusted_client, monkeypatch):
    """aliases dict must exist and have at least one entry."""
    monkeypatch.delenv("SLM_MCP_PROFILE", raising=False)
    data = trusted_client.get(_ENDPOINT).json()
    assert isinstance(data["aliases"], dict)
    assert len(data["aliases"]) > 0, "aliases dict is empty"


def test_mcp_profiles_total_tools_is_positive(trusted_client, monkeypatch):
    """total_tools must be a positive integer."""
    monkeypatch.delenv("SLM_MCP_PROFILE", raising=False)
    data = trusted_client.get(_ENDPOINT).json()
    assert isinstance(data["total_tools"], int)
    assert data["total_tools"] > 0
