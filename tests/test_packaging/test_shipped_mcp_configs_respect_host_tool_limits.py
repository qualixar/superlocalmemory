# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A host that refuses an MCP server above its own tool ceiling is a shipped
config away from simply not starting.

antigravity-plugin/mcp_config.json set `SLM_MCP_PROFILE=power` (66 tools)
*and* `SLM_MCP_ALL_TOOLS=1`, which bypasses the profile filter entirely and
registers all 101 tools — above Antigravity's reported 100-tool ceiling.
Nothing compared a shipped config's resolved tool count against the host it
ships for, so the one host declared with a hard ceiling drifted straight
past it.

ide/configs/cursor-mcp.json carries no `env` block at all, so it falls back
to the no-profile default (the essential surface, which mirrors `full` at
54 tools) — above Cursor's reported ~40-tool ceiling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from superlocalmemory.mcp.profiles import _PROFILE_ALIASES, _PROFILE_DEFINITIONS

REPO_ROOT = Path(__file__).resolve().parents[2]

# `_ESSENTIAL_TOOLS` (the no-profile MCP default) is asserted equal to `full`
# (56 tools) by tests/test_mcp/test_mcp_exposure_contract.py
# (`test_registration_exposure_is_exact_and_duplicate_free`, exposure
# "essential"). Reuse that invariant here instead of importing
# `superlocalmemory.mcp.server`, which triggers heavier module-level setup.
_NO_PROFILE_DEFAULT_COUNT = len(_PROFILE_DEFINITIONS["full"])

# `SLM_MCP_ALL_TOOLS=1` (or `SLM_MCP_PROFILE=whole`) bypasses every profile
# filter and registers every tool FastMCP knows about. Pinned by the same
# exposure contract test above (exposure "whole", expected_count 105).
_ALL_TOOLS_COUNT = 105

# Declared, vendor-reported per-host MCP tool ceilings. Only hosts with an
# actual reported limit are listed; a host absent here is not asserted.
HOST_TOOL_LIMITS: dict[str, int] = {
    "antigravity-plugin/mcp_config.json": 100,
    "ide/configs/cursor-mcp.json": 40,
}


def _resolve_tool_count(config_path: Path) -> int:
    """Resolve the MCP tool count a shipped config would register.

    Mirrors the precedence documented at the top of
    ``src/superlocalmemory/mcp/server.py``: ``ALL > TOOLS > PROFILE > default``.
    """
    data = json.loads(config_path.read_text(encoding="utf-8"))
    servers = data.get("mcpServers") or data.get("servers") or {}
    server = servers.get("superlocalmemory", {})
    env = server.get("env", {}) or {}

    if env.get("SLM_MCP_ALL_TOOLS") == "1":
        return _ALL_TOOLS_COUNT

    user_tools = env.get("SLM_MCP_TOOLS", "").strip()
    if user_tools:
        return len({t.strip() for t in user_tools.split(",") if t.strip()})

    profile = env.get("SLM_MCP_PROFILE", "").strip().lower()
    if profile == "whole":
        return _ALL_TOOLS_COUNT
    if profile:
        canonical = _PROFILE_ALIASES.get(profile, profile)
        assert canonical in _PROFILE_DEFINITIONS, (
            f"{config_path}: unknown SLM_MCP_PROFILE {profile!r}"
        )
        return len(_PROFILE_DEFINITIONS[canonical])

    return _NO_PROFILE_DEFAULT_COUNT


@pytest.mark.parametrize(
    ("rel_path", "limit"), sorted(HOST_TOOL_LIMITS.items()), ids=lambda v: str(v)
)
def test_shipped_config_stays_within_its_hosts_declared_tool_limit(
    rel_path: str, limit: int
) -> None:
    config_path = REPO_ROOT / rel_path
    assert config_path.exists(), f"{rel_path} is gone; update this test or restore the file"

    resolved = _resolve_tool_count(config_path)
    assert resolved <= limit, (
        f"{rel_path} resolves to {resolved} tools, above its host's "
        f"reported {limit}-tool ceiling"
    )
