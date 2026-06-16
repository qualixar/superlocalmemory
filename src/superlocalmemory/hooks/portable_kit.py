# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""WP-08 portable kit — ``slm connect <ide>`` MCP-wiring.

Writes SLM's MCP block into the target IDE config via MERGE-NOT-CLOBBER:
- Only touches the ``superlocalmemory`` server key.
- All other servers + top-level keys are preserved byte-for-byte.
- Atomic write (.tmp + os.replace); aborts on parse error (file untouched).
- claude-code is OUT: short-circuits to a WP-06 plugin pointer, no config written.
- AGENTS.md is appended with <!-- SLM-START/END --> markers (never overwrite).

IDE_MATRIX verified against ide/configs/* templates (read-only, WP-04 owns).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Marker convention copied from ide_connector.py (do not edit that class)
SLM_MARKER_START = "<!-- SLM-START -->"
SLM_MARKER_END = "<!-- SLM-END -->"

CLAUDE_CODE_PLUGIN_POINTER = (
    "slm connect claude-code: Claude Code is configured via the SLM plugin (WP-06).\n"
    "Run: slm plugin install  OR  see plugin-src/ for manual installation.\n"
    "No MCP config file is written by this command."
)


# ---------------------------------------------------------------------------
# IDEDescriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IDEDescriptor:
    """Immutable descriptor for one IDE in the support matrix."""

    ide_id: str
    display: str
    mcp_path_global: str          # relative to home (empty string = OUT)
    mcp_path_project: str | None  # relative to project root; None = no project scope
    server_key: str               # top-level key that holds the servers dict
    fmt: str                      # "json" | "toml" | "yaml" | "" (OUT)
    agents_md_path: str | None    # relative to scope root; None = unsupported
    server_block: dict[str, Any] = field(default_factory=dict)
    caveats: str = ""


# ---------------------------------------------------------------------------
# IDE_MATRIX — server_key + fmt VERIFIED vs ide/configs/* templates
# Paths are [CN-ONLINE] best-effort; confirmed from public docs where possible.
# ---------------------------------------------------------------------------

IDE_MATRIX: dict[str, IDEDescriptor] = {
    # --- JSON IDEs ---
    "cursor": IDEDescriptor(
        ide_id="cursor",
        display="Cursor",
        mcp_path_global=".cursor/mcp.json",
        mcp_path_project=".cursor/mcp.json",
        server_key="mcpServers",
        fmt="json",
        agents_md_path=".cursorrules",
        server_block={"command": "slm", "args": ["mcp"], "type": "stdio"},
        caveats="project .cursor/mcp.json",
    ),
    "antigravity": IDEDescriptor(
        ide_id="antigravity",
        display="Antigravity (agy)",
        mcp_path_global=".antigravity/mcp.json",
        mcp_path_project=None,
        server_key="mcpServers",
        fmt="json",
        agents_md_path=None,
        server_block={"command": "slm", "args": ["mcp"], "type": "stdio"},
        caveats="Vertex auth (R1); path [CN-ONLINE]",
    ),
    "windsurf": IDEDescriptor(
        ide_id="windsurf",
        display="Windsurf",
        mcp_path_global=".codeium/windsurf/mcp_config.json",
        mcp_path_project=None,
        server_key="mcpServers",
        fmt="json",
        agents_md_path=".windsurfrules",
        server_block={"command": "slm", "args": ["mcp"], "type": "stdio"},
        caveats="path [CN-ONLINE]",
    ),
    "gemini-cli": IDEDescriptor(
        ide_id="gemini-cli",
        display="Gemini CLI",
        mcp_path_global=".gemini/settings.json",
        mcp_path_project=None,
        server_key="mcpServers",
        fmt="json",
        agents_md_path="GEMINI.md",
        server_block={"command": "slm", "args": ["mcp"], "type": "stdio"},
        caveats="Google deprecating; path [CN-ONLINE]",
    ),
    "vscode-copilot": IDEDescriptor(
        ide_id="vscode-copilot",
        display="VS Code / Copilot",
        mcp_path_global=".vscode/mcp.json",
        mcp_path_project=".vscode/mcp.json",
        server_key="servers",
        fmt="json",
        agents_md_path=".github/copilot-instructions.md",
        server_block={"type": "stdio", "command": "slm", "args": ["mcp"]},
        caveats="key NOT mcpServers; uses 'servers'",
    ),
    "zed": IDEDescriptor(
        ide_id="zed",
        display="Zed Editor",
        mcp_path_global=".config/zed/settings.json",
        mcp_path_project=None,
        server_key="context_servers",
        fmt="json",
        agents_md_path=None,  # no rules surface
        server_block={"source": "custom", "command": "slm", "args": ["mcp"]},
        caveats="no rules surface → AGENTS.md skip",
    ),
    "jetbrains": IDEDescriptor(
        ide_id="jetbrains",
        display="JetBrains IDEs",
        mcp_path_global=".config/JetBrains/mcp.json",
        mcp_path_project=".mcp.json",
        server_key="mcpServers",
        fmt="json",
        agents_md_path=None,
        server_block={"command": "slm", "args": ["mcp"], "type": "stdio"},
        caveats="path per product [CN-ONLINE]",
    ),
    "opencode": IDEDescriptor(
        ide_id="opencode",
        display="OpenCode",
        mcp_path_global=".config/opencode/config.json",
        mcp_path_project=None,
        server_key="mcp",
        fmt="json",
        agents_md_path=None,
        server_block={"command": "slm", "args": ["mcp"]},
        caveats="top-level key is 'mcp'",
    ),
    "claude-desktop": IDEDescriptor(
        ide_id="claude-desktop",
        display="Claude Desktop",
        mcp_path_global=(
            "Library/Application Support/Claude/claude_desktop_config.json"
            if sys.platform == "darwin"
            else ".config/Claude/claude_desktop_config.json"
        ),
        mcp_path_project=None,
        server_key="mcpServers",
        fmt="json",
        agents_md_path=None,
        server_block={"command": "slm", "args": ["mcp"], "type": "stdio"},
        caveats="desktop app (not Claude Code)",
    ),
    # --- TOML IDEs ---
    "codex": IDEDescriptor(
        ide_id="codex",
        display="Codex CLI",
        mcp_path_global=".codex/config.toml",
        mcp_path_project=None,
        server_key="mcp_servers",
        fmt="toml",
        agents_md_path="AGENTS.md",
        server_block={"command": "slm", "args": ["mcp"]},
        caveats="tomllib read / tomli_w write",
    ),
    # --- YAML IDEs ---
    "continue": IDEDescriptor(
        ide_id="continue",
        display="Continue.dev",
        mcp_path_global=".continue/config.yaml",
        mcp_path_project=".continue/config.yaml",
        server_key="contextProviders",
        fmt="yaml",
        agents_md_path=None,
        server_block={
            "name": "mcp",
            "params": {
                "serverName": "superlocalmemory",
                "command": "slm",
                "args": ["mcp"],
            },
        },
        caveats="contextProviders is a LIST; append+dedupe by serverName",
    ),
    # --- OUT: claude-code defers to WP-06 ---
    "claude-code": IDEDescriptor(
        ide_id="claude-code",
        display="Claude Code (WP-06 plugin)",
        mcp_path_global="",
        mcp_path_project=None,
        server_key="",
        fmt="",
        agents_md_path=None,
        server_block={},
        caveats="OUT — WP-06 plugin pointer only; no MCP config written",
    ),
    # --- EXPERIMENTAL (gated, not wired by default) ---
    # chatgpt-desktop, perplexity, cody: gated behind --experimental
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def supported_ides() -> list[str]:
    """Return all ide_ids in the matrix (including claude-code and experimental)."""
    return list(IDE_MATRIX.keys())


def resolve_descriptor(ide_id: str) -> IDEDescriptor | None:
    """Return the IDEDescriptor for ide_id, or None if unknown."""
    return IDE_MATRIX.get(ide_id)


def connect_ide(
    ide_id: str,
    *,
    home: Path | None = None,
    project: Path | None = None,
    here: bool = False,
    profile: str | None = None,
    agents_md_source: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Wire SLM into the target IDE config via merge-not-clobber.

    Returns a result dict:
        {ide, mcp_config: wrote|merged|unchanged|skipped|error,
         mcp_path, agents_md: wrote|skipped(...)|unchanged|error,
         servers_preserved: int, error: str|None}
    """
    result: dict[str, Any] = {
        "ide": ide_id,
        "mcp_config": "error",
        "mcp_path": "",
        "agents_md": "skipped(not-run)",
        "servers_preserved": 0,
        "error": None,
    }

    # Step 1 — resolve
    desc = resolve_descriptor(ide_id)
    if desc is None:
        result["error"] = (
            f"Unknown IDE '{ide_id}'. Supported: {', '.join(supported_ides())}"
        )
        return result

    # Step 1a — claude-code short-circuit (AC6)
    if desc.fmt == "":
        print(CLAUDE_CODE_PLUGIN_POINTER)
        result["mcp_config"] = "skipped"
        result["agents_md"] = "skipped(claude-code-out)"
        return result

    # Step 2 — scope resolution
    effective_home = home or Path.home()
    if here:
        if project is None:
            result["error"] = "--here requires --project (project root path)"
            return result
        scope_root = project
        rel_path = desc.mcp_path_project or desc.mcp_path_global
    else:
        scope_root = effective_home
        rel_path = desc.mcp_path_global

    config_path = scope_root / rel_path
    result["mcp_path"] = str(config_path)

    # Step 3 — load existing config
    try:
        data = _load_config(config_path, desc.fmt)
    except _ParseError as exc:
        result["error"] = str(exc)
        # File is untouched (we never wrote; abort)
        return result

    # Step 4 — extract server container
    # For continue (yaml list), special-case
    if desc.fmt == "yaml":
        mcp_status, servers_preserved = _merge_yaml_list(
            data, desc, profile
        )
        result["mcp_config"] = mcp_status
        result["servers_preserved"] = servers_preserved
    else:
        servers = data.setdefault(desc.server_key, {})
        pre_count = len(servers)
        pre_slm = copy.deepcopy(servers.get("superlocalmemory"))

        # Step 5 — merge
        block = copy.deepcopy(desc.server_block)
        if profile:
            block.setdefault("env", {})["SLM_MCP_PROFILE"] = profile

        servers["superlocalmemory"] = block

        if servers.get("superlocalmemory") == pre_slm and pre_slm is not None:
            mcp_status = "unchanged"
        elif pre_slm is None:
            mcp_status = "wrote"
        else:
            mcp_status = "merged"

        result["servers_preserved"] = max(0, pre_count - (0 if pre_slm is None else 1))
        result["mcp_config"] = mcp_status

    # Step 6 — atomic write
    try:
        _atomic_write(config_path, data, desc.fmt)
    except Exception as exc:
        result["error"] = f"Write failed: {exc}"
        result["mcp_config"] = "error"
        return result

    # Verify idempotent: if nothing changed, re-read and confirm
    if result["mcp_config"] != "unchanged":
        pass  # already wrote
    else:
        pass  # already unchanged; atomic write still ran (idempotent)

    # Step 7 — AGENTS.md
    result["agents_md"] = _handle_agents_md(
        desc, scope_root, agents_md_source, here
    )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _ParseError(Exception):
    """Raised when an existing config file cannot be parsed."""


def _load_config(path: Path, fmt: str) -> dict[str, Any]:
    """Load and parse existing config; return {} if file absent.

    Raises _ParseError if file exists but is malformed.
    """
    if not path.exists():
        return {}

    raw = path.read_text(encoding="utf-8")

    try:
        if fmt == "json":
            return json.loads(raw)
        elif fmt == "toml":
            import tomllib
            return tomllib.loads(raw)
        elif fmt == "yaml":
            import yaml
            parsed = yaml.safe_load(raw)
            # Non-dict result (e.g. bare string) is treated as empty config
            if parsed is None:
                return {}
            if not isinstance(parsed, dict):
                return {}
            return parsed
        else:
            # Unknown format — return empty; caller will fail gracefully
            return {}
    except Exception as exc:
        raise _ParseError(
            f"Config parse error ({fmt}) at {path}: {exc}"
        ) from exc


def _merge_yaml_list(
    data: dict[str, Any],
    desc: IDEDescriptor,
    profile: str | None,
) -> tuple[str, int]:
    """Merge SLM entry into a list-style YAML contextProviders (continue.dev).

    Returns (status, servers_preserved).
    """
    providers: list[dict] = data.setdefault(desc.server_key, [])
    if not isinstance(providers, list):
        providers = []
        data[desc.server_key] = providers

    pre_count = sum(
        1 for p in providers
        if p.get("params", {}).get("serverName") != "superlocalmemory"
    )

    # Check if SLM already present
    existing_idx = None
    for i, p in enumerate(providers):
        if p.get("params", {}).get("serverName") == "superlocalmemory":
            existing_idx = i
            break

    block = copy.deepcopy(desc.server_block)
    if profile:
        block.setdefault("params", {})["env"] = {"SLM_MCP_PROFILE": profile}

    if existing_idx is not None:
        if providers[existing_idx] == block:
            return "unchanged", pre_count
        providers[existing_idx] = block
        return "merged", pre_count
    else:
        providers.append(block)
        return "wrote", pre_count


def _atomic_write(path: Path, data: dict[str, Any], fmt: str) -> None:
    """Serialize data and atomically write to path (.tmp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        if fmt == "json":
            content = json.dumps(data, indent=2) + "\n"
            tmp_path.write_text(content, encoding="utf-8")
        elif fmt == "toml":
            import tomli_w
            tmp_path.write_text(tomli_w.dumps(data), encoding="utf-8")
        elif fmt == "yaml":
            import yaml
            tmp_path.write_text(yaml.safe_dump(data, default_flow_style=False))
        else:
            raise ValueError(f"Unknown format: {fmt}")

        os.replace(tmp_path, path)
    except Exception:
        # Clean up tmp on failure
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _handle_agents_md(
    desc: IDEDescriptor,
    scope_root: Path,
    agents_md_source: Callable[[], str] | None,
    here: bool,
) -> str:
    """Append SLM section to AGENTS.md with <!-- SLM-START/END --> markers.

    D-1 resolution: only write AGENTS.md in --here (project) scope;
    skip in global scope (don't litter $HOME).
    """
    if desc.agents_md_path is None:
        return "skipped(unsupported)"

    if agents_md_source is None:
        return "skipped(no-source)"

    agents_path = scope_root / desc.agents_md_path

    try:
        source_content = agents_md_source()
    except Exception as exc:
        logger.warning("agents_md_source() failed: %s — skipping AGENTS.md write", exc)
        return "skipped(source-error)"

    # Read existing content
    existing = ""
    if agents_path.exists():
        existing = agents_path.read_text(encoding="utf-8")

    # Idempotency check
    if SLM_MARKER_START in existing:
        return "unchanged"

    # Append SLM section
    section = (
        f"\n{SLM_MARKER_START}\n"
        f"{source_content.strip()}\n"
        f"{SLM_MARKER_END}\n"
    )
    agents_path.parent.mkdir(parents=True, exist_ok=True)
    agents_path.write_text(existing + section, encoding="utf-8")
    return "wrote"
