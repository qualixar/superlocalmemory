# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""Portable kit — ``slm connect <ide>`` MCP-wiring.

Writes SLM's MCP block into the target IDE config via MERGE-NOT-CLOBBER:
- Only touches the ``superlocalmemory`` server key.
- All other servers + top-level keys are preserved byte-for-byte.
- Atomic write (.tmp + os.replace); aborts on parse error (file untouched).
- claude-code is OUT: short-circuits to a plugin pointer, no config written.
- AGENTS.md is appended with <!-- SLM-START/END --> markers (never overwrite).

IDE_MATRIX follows the linked official host schemas; ide/configs contains matching examples.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Marker convention copied from ide_connector.py (do not edit that class)
SLM_MARKER_START = "<!-- SLM-START -->"
SLM_MARKER_END = "<!-- SLM-END -->"

# Valid transport values for connect_ide() and slm connect --transport.
# "stdio"           — default, zero regression; uses desc.server_block as-is.
# "http"            — native MCP Streamable-HTTP; requires daemon at :daemon_port.
# "http-mcp-remote" — mcp-remote stdio bridge for stdio-only clients.
VALID_TRANSPORTS: frozenset[str] = frozenset({"stdio", "http", "http-mcp-remote"})

#: The commands below are Claude Code's own, not ours -- ``slm`` has no
#: ``plugin`` subcommand and never has, so the instruction this used to print
#: ended at ``invalid choice: 'plugin'`` with nothing else offered. The working
#: path already existed in ``cli/setup_wizard.py``; it just was not the one
#: anybody was told about.
CLAUDE_CODE_PLUGIN_POINTER = (
    "slm connect claude-code: Claude Code is configured via the SLM plugin, "
    "which Claude Code installs itself.\n"
    "Run:  claude plugin marketplace add qualixar/superlocalmemory\n"
    "then: claude plugin install superlocalmemory@qualixar\n"
    "Or run `slm setup` to do both, or see plugin-src/ to install by hand.\n"
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


_CLAUDE_DESKTOP_FILE = "claude_desktop_config.json"


def global_config_path(desc: "IDEDescriptor", home: Path, *, platform: str | None = None,
                       environ: "dict[str, str] | os._Environ[str] | None" = None) -> Path:
    """Where ``desc`` keeps its user-level config.

    Every host is ``home / mcp_path_global`` except Claude Desktop, whose place depends on the
    platform: ``~/Library/Application Support/Claude`` on macOS, ``%APPDATA%\\Claude`` on Windows,
    ``~/.config/Claude`` on Linux. ``environ`` is where APPDATA is read from (default: the real
    environment); pass ``{}`` to keep a given home self-contained.
    """
    env = os.environ if environ is None else environ
    if desc.ide_id == "muse" and env.get("XDG_CONFIG_HOME"):
        return Path(env["XDG_CONFIG_HOME"]) / "muse" / "settings.json"
    if desc.ide_id == "grok-cli" and env.get("GROK_HOME"):
        return Path(env["GROK_HOME"]) / "config.toml"
    if desc.ide_id == "openclaw" and env.get("OPENCLAW_CONFIG_PATH"):
        return Path(env["OPENCLAW_CONFIG_PATH"]).expanduser()
    if desc.ide_id == "copilot-cli" and env.get("COPILOT_HOME"):
        return Path(env["COPILOT_HOME"]) / "mcp-config.json"
    if desc.ide_id != "claude-desktop":
        return home / desc.mcp_path_global
    platform = sys.platform if platform is None else platform
    if platform == "darwin":
        return home / "Library" / "Application Support" / "Claude" / _CLAUDE_DESKTOP_FILE
    if platform.startswith("win"):
        appdata = env.get("APPDATA")
        base = Path(appdata) if appdata else home / "AppData" / "Roaming"
        return base / "Claude" / _CLAUDE_DESKTOP_FILE
    return home / ".config" / "Claude" / _CLAUDE_DESKTOP_FILE


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
    # https://www.antigravity.google/docs/mcp/ (opened 2026-10-11)
    "antigravity": IDEDescriptor(
        ide_id="antigravity",
        display="Antigravity (agy)",
        mcp_path_global=".gemini/config/mcp_config.json",
        mcp_path_project=".agents/mcp_config.json",
        server_key="mcpServers",
        fmt="json",
        agents_md_path=None,
        server_block={"command": "slm", "args": ["mcp"]},
        caveats="Antigravity 2.0 global/workspace MCP contract",
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
    # https://geminicli.com/docs/tools/mcp-server/
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
    # https://code.visualstudio.com/docs/agent-customization/mcp-servers
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
        agents_md_path=".junie/AGENTS.md",  # Junie guidelines for memory and optimize
        server_block={"command": "slm", "args": ["mcp"], "type": "stdio"},
        caveats="path per product [CN-ONLINE]",
    ),
    # https://opencode.ai/docs/mcp-servers/ and https://opencode.ai/docs/config/
    "opencode": IDEDescriptor(
        ide_id="opencode",
        display="OpenCode",
        mcp_path_global=".config/opencode/opencode.json",
        mcp_path_project="opencode.json",
        server_key="mcp",
        fmt="json",
        agents_md_path=None,
        server_block={"type": "local", "command": ["slm", "mcp"]},
        caveats="top-level key is 'mcp'",
    ),
    # https://qwenlm.github.io/qwen-code-docs (settings.md, opened 2026-10-16)
    "qwen-code": IDEDescriptor(
        ide_id="qwen-code",
        display="Qwen Code",
        mcp_path_global=".qwen/settings.json",
        mcp_path_project=".qwen/settings.json",
        server_key="mcpServers",
        fmt="json",
        agents_md_path=None,
        server_block={"command": "slm", "args": ["mcp"]},
        caveats="settings.json also holds other Qwen settings; only mcpServers is touched",
    ),
    # MoonshotAI/kimi-code docs/en/customization/mcp.md (opened 2026-10-16)
    "kimi-code": IDEDescriptor(
        ide_id="kimi-code",
        display="Kimi Code",
        mcp_path_global=".kimi-code/mcp.json",
        mcp_path_project=".kimi-code/mcp.json",
        server_key="mcpServers",
        fmt="json",
        agents_md_path=None,
        server_block={"command": "slm", "args": ["mcp"]},
        caveats="$KIMI_CODE_HOME overrides ~/.kimi-code",
    ),
    # Legacy Kimi CLI (MoonshotAI/kimi-cli, archived): ~/.kimi/mcp.json
    "kimi-cli": IDEDescriptor(
        ide_id="kimi-cli",
        display="Kimi CLI (legacy)",
        mcp_path_global=".kimi/mcp.json",
        mcp_path_project=None,
        server_key="mcpServers",
        fmt="json",
        agents_md_path=None,
        server_block={"command": "slm", "args": ["mcp"]},
        caveats="archived upstream; Kimi Code is the successor",
    ),
    # https://kiro.dev/docs/mcp/configuration/ (opened 2026-10-16)
    "kiro": IDEDescriptor(
        ide_id="kiro",
        display="Kiro",
        mcp_path_global=".kiro/settings/mcp.json",
        mcp_path_project=".kiro/settings/mcp.json",
        server_key="mcpServers",
        fmt="json",
        agents_md_path=None,
        server_block={"command": "slm", "args": ["mcp"]},
        caveats="workspace settings win over user settings",
    ),
    "claude-desktop": IDEDescriptor(
        ide_id="claude-desktop",
        display="Claude Desktop",
        # Relative to home on macOS and Linux; Windows is resolved by
        # global_config_path() (APPDATA). This default is what it falls back to.
        mcp_path_global=(
            "Library/Application Support/Claude/claude_desktop_config.json"
            if sys.platform == "darwin"
            else "AppData/Roaming/Claude/claude_desktop_config.json"
            if sys.platform.startswith("win")
            else ".config/Claude/claude_desktop_config.json"
        ),
        mcp_path_project=None,
        server_key="mcpServers",
        fmt="json",
        agents_md_path=None,
        server_block={"command": "slm", "args": ["mcp"], "type": "stdio"},
        caveats="desktop app (not Claude Code)",
    ),
    # Muse Code 1.4.7, built-in MCP setup guide: mcpServers in settings.json,
    # new file starts at schema_version 1; XDG_CONFIG_HOME is honoured.
    "muse": IDEDescriptor(
        ide_id="muse",
        display="Muse CLI",
        mcp_path_global=".config/muse/settings.json",
        mcp_path_project=None,
        server_key="mcpServers",
        fmt="json",
        agents_md_path=None,
        # mode optional: Muse's default `required` aborts the whole run when
        # the server cannot start (dev.meta.ai/docs/muse-code/configuration).
        server_block={"type": "stdio", "command": "slm", "args": ["mcp"], "mode": "optional"},
        caveats="legacy mcp_servers key is renamed; both keys drop every server",
    ),
    # GitHub Copilot CLI, and VS Code's Agents window (its Agent Host reads this
    # file natively; extension-provided servers do not reach it). GitHub docs,
    # "Adding MCP servers for GitHub Copilot CLI": type local, tools ["*"].
    "copilot-cli": IDEDescriptor(
        ide_id="copilot-cli",
        display="GitHub Copilot CLI / VS Code Agents window",
        mcp_path_global=".copilot/mcp-config.json",
        mcp_path_project=None,
        server_key="mcpServers",
        fmt="json",
        agents_md_path=None,
        server_block={"type": "local", "command": "slm", "args": ["mcp"], "tools": ["*"]},
        caveats="COPILOT_HOME moves the folder",
    ),
    # OpenClaw: optional JSON5 config ~/.openclaw/openclaw.json
    # (OPENCLAW_CONFIG_PATH), servers under mcp.servers (docs.openclaw.ai:
    # gateway/configuration, tools/mcp). A file that is not plain JSON is never
    # rewritten (comments would be lost); the error names `openclaw mcp add`.
    "openclaw": IDEDescriptor(
        ide_id="openclaw",
        display="OpenClaw",
        mcp_path_global=".openclaw/openclaw.json",
        mcp_path_project=None,
        server_key="mcp.servers",
        fmt="json",
        agents_md_path=None,
        server_block={"command": "slm", "args": ["mcp"]},
        caveats="JSON5 with comments is left alone; skills go to ~/.agents/skills",
    ),
    # --- TOML IDEs ---
    # Grok CLI 1.0.50 (docs.x.ai/build/features/mcp-servers): [mcp_servers.<name>]
    # with command, args, env; $GROK_HOME moves the folder (docs.x.ai/build/settings).
    "grok-cli": IDEDescriptor(
        ide_id="grok-cli",
        display="Grok CLI",
        mcp_path_global=".grok/config.toml",
        mcp_path_project=None,
        server_key="mcp_servers",
        fmt="toml",
        agents_md_path=None,
        server_block={"command": "slm", "args": ["mcp"]},
        caveats="the plugin (grok plugin install) adds skills and hooks",
    ),
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
    # https://docs.continue.dev/reference (mcpServers, opened 2026-10-11)
    "continue": IDEDescriptor(
        ide_id="continue", display="Continue.dev",
        mcp_path_global=".continue/config.yaml", mcp_path_project=".continue/config.yaml",
        server_key="mcpServers", fmt="yaml", agents_md_path=None,
        server_block={"name": "superlocalmemory", "command": "slm", "args": ["mcp"]},
        caveats="mcpServers is a list; merge by name",
    ),
    # https://hermes-agent.nousresearch.com/docs/user-guide/features/plugins/
    "hermes": IDEDescriptor(
        ide_id="hermes", display="Hermes (native plugin)",
        mcp_path_global="", mcp_path_project=None, server_key="", fmt="", agents_md_path=None,
        caveats="hermes plugins install 'https://github.com/qualixar/superlocalmemory.git#hermes-plugin'",
    ),
    # --- OUT: claude-code defers to the plugin installer ---
    "claude-code": IDEDescriptor(
        ide_id="claude-code",
        display="Claude Code (plugin)",
        mcp_path_global="",
        mcp_path_project=None,
        server_key="",
        fmt="",
        agents_md_path=None,
        server_block={},
        caveats="OUT — plugin pointer only; no MCP config written",
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
    dry_run: bool = False,
    transport: str = "stdio",
    daemon_port: int = 8765,
) -> dict[str, Any]:
    """Merge a schema-specific MCP config; host registration remains pending."""
    from superlocalmemory.hooks.portable_install import install

    return install(
        sys.modules[__name__], ide_id, home=home, project=project, here=here,
        profile=profile, source=agents_md_source, dry_run=dry_run,
        transport=transport, port=daemon_port,
    )


def connect_many(
    ide_ids: list[str],
    *,
    home: Path | None = None,
    project: Path | None = None,
    here: bool = False,
    profile: str | None = None,
    agents_md_source: Callable[[], str] | None = None,
    transport: str = "stdio",
    daemon_port: int = 8765,
) -> list[dict[str, Any]]:
    """Connect each requested host independently through the same registry."""
    return [
        connect_ide(
            ide_id,
            home=home,
            project=project,
            here=here,
            profile=profile,
            agents_md_source=agents_md_source,
            transport=transport,
            daemon_port=daemon_port,
        )
        for ide_id in ide_ids
    ]


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
    *, block: dict[str, Any] | None = None,
) -> tuple[str, int]:
    """Merge Continue's current mcpServers list without clobbering user settings."""
    before = copy.deepcopy(data)
    # https://docs.continue.dev/reference
    for key, value in {"name": "SLM Config", "version": "1.0.0", "schema": "v1"}.items():
        data.setdefault(key, value)
    providers = data.setdefault(desc.server_key, [])
    if not isinstance(providers, list) or any(not isinstance(p, dict) for p in providers):
        raise ValueError("mcpServers must be a list of server objects")
    block = copy.deepcopy(
        block if block is not None else _build_server_block(desc, "stdio", 8765, profile)
    )
    if profile:
        block.setdefault("env", {})["SLM_MCP_PROFILE"] = profile
    merged: list[dict[str, Any]] = []
    replaced = False
    for entry in providers:
        if entry.get("name") != "superlocalmemory":
            merged.append(entry)
        elif not replaced:
            merged.append(block)
            replaced = True
    preserved = len(merged) - int(replaced)
    if not replaced:
        merged.append(block)
    data[desc.server_key] = merged
    migrated = _remove_legacy_slm_provider(data)
    status = "unchanged" if data == before else "merged" if replaced or migrated else "wrote"
    return status, preserved


def _remove_legacy_slm_provider(data: dict[str, Any]) -> bool:
    """Remove only SLM's obsolete Continue entries, preserving other providers."""
    legacy = data.get("contextProviders")
    if not isinstance(legacy, list):
        return False
    remaining = [entry for entry in legacy if not (
        isinstance(entry, dict) and isinstance(entry.get("params"), dict)
        and entry["params"].get("serverName") == "superlocalmemory"
    )]
    if remaining == legacy:
        return False
    if remaining:
        data["contextProviders"] = remaining
    else:
        del data["contextProviders"]
    return True


def _build_server_block(
    desc: IDEDescriptor,
    transport: str,
    daemon_port: int,
    profile: str | None,
) -> dict[str, Any]:
    """Use each host's declared transport and environment keys."""
    from urllib.parse import urlencode

    url = f"http://127.0.0.1:{daemon_port}/mcp/"
    if profile:
        url += "?" + urlencode({"profile": profile})
    if transport == "http":
        # https://raw.githubusercontent.com/continuedev/continue/main/packages/config-yaml/src/schemas/mcp/index.ts
        if desc.ide_id == "continue":
            return {"name": "superlocalmemory", "type": "streamable-http", "url": url}
        if desc.ide_id == "opencode":
            return {"type": "remote", "url": url}
        if desc.ide_id == "antigravity":
            return {"serverUrl": url}
        if desc.ide_id == "gemini-cli":
            return {"httpUrl": url}
        return {"type": "http", "url": url}
    if transport == "http-mcp-remote":
        block = ({"type": "local", "command": ["mcp-remote", url]}
                 if desc.ide_id == "opencode" else {"command": "mcp-remote", "args": [url]})
        if desc.ide_id == "continue":
            block["name"] = "superlocalmemory"
        if desc.ide_id not in {"opencode", "antigravity", "continue"}:
            block["type"] = "stdio"
    else:
        block = copy.deepcopy(desc.server_block)
        _pin_stdio_runtime(block, desc.ide_id)
    if profile:
        env_key = "environment" if desc.ide_id == "opencode" else "env"
        block.setdefault(env_key, {})["SLM_MCP_PROFILE"] = profile
    return block


def _pin_stdio_runtime(block: dict[str, Any], host: str) -> None:
    """Pin the process while preserving each host's command/environment schema."""
    from superlocalmemory.cli.deployment import security_environment
    from superlocalmemory.infra.data_root import canonical_data_root

    command = block.get("command")
    if command == "slm":
        block["command"] = sys.executable
        block["args"] = ["-m", "superlocalmemory.cli", *block.get("args", [])]
    elif isinstance(command, list) and command and command[0] == "slm":
        block["command"] = [sys.executable, "-m", "superlocalmemory.cli", *command[1:]]
    env_key = "environment" if host == "opencode" else "env"
    block.setdefault(env_key, {}).update(
        SLM_MCP_HOST=host, SLM_DATA_DIR=str(canonical_data_root()),
        **security_environment(),
    )


def _check_daemon_health(daemon_port: int, timeout: float = 2.0) -> bool:
    """Non-blocking probe of the SLM daemon health endpoint.

    Returns True if the daemon responds with HTTP 200, False otherwise.
    Never raises — all exceptions are caught and treated as "unreachable".
    """
    try:
        import urllib.error
        import urllib.request

        url = f"http://127.0.0.1:{daemon_port}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            payload = json.loads(resp.read())
            if payload.get("status") != "ok":
                return False
        # Reachable is not enough on a computer shared by several accounts:
        # the port must be this account's own SuperLocalMemory.
        from superlocalmemory.cli.daemon import owned_daemon_answers

        return owned_daemon_answers(daemon_port)
    except Exception:
        return False


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
            tmp_path.write_text(yaml.safe_dump(data, default_flow_style=False), encoding="utf-8")
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
    """Append a marked SLM section without overwriting user rules."""
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
    if not isinstance(source_content, str) or not source_content.strip():
        logger.warning("agents_md_source() returned no content — skipping AGENTS.md write")
        return "skipped(no-source-content)"

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
    # Atomic write: agents_path is the user's hand-written rules file
    # (.cursorrules, AGENTS.md, copilot-instructions.md, ...). A crash mid-write
    # must NOT truncate it. tmp in the same dir → os.replace is atomic.
    _tmp = agents_path.with_suffix(agents_path.suffix + ".tmp")
    _tmp.write_text(existing + section, encoding="utf-8")
    os.replace(_tmp, agents_path)
    return "wrote"


# ---------------------------------------------------------------------------
# __main__ — `python -m superlocalmemory.hooks.portable_kit [ids...]`
#
# Used by the npm installer to execute non-destructive IDE connects when
# Python is available at postinstall time. If Python / the package is not
# importable the installer records a pending_ide_connections.json instead.
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import argparse as _argparse

    _parser = _argparse.ArgumentParser(
        description="SLM IDE connector — non-destructive merge-not-clobber."
    )
    _parser.add_argument(
        "ide_ids",
        nargs="*",
        metavar="IDE",
        help=(
            "IDE ids to connect.  Pass 'all' to connect every supported IDE "
            "(excluding claude-code which uses the plugin installer)."
        ),
    )
    _parser.add_argument("--home", help="Override home directory (test hook).")
    _parser.add_argument("--profile", help="SLM_MCP_PROFILE to inject.")
    _parser.add_argument(
        "--list", action="store_true",
        help="Print supported IDE ids with display names and exit.",
    )
    _args = _parser.parse_args()

    if _args.list:
        for _id, _desc in IDE_MATRIX.items():
            _flag = "[OUT]" if not _desc.fmt else ""
            print(f"  {_id:22s} {_desc.display} {_flag}".rstrip())
        sys.exit(0)

    _home = Path(_args.home) if _args.home else None

    # Resolve targets: "all" → every MCP-capable IDE (fmt != ""), else explicit list.
    if "all" in _args.ide_ids:
        _targets = [k for k, d in IDE_MATRIX.items() if d.fmt]
    else:
        _targets = [t for t in _args.ide_ids if t]

    if not _targets:
        print("No IDE ids given. Pass ide ids or 'all'. Use --list to see options.")
        sys.exit(0)

    _results = connect_many(_targets, home=_home, profile=_args.profile)
    _ok = 0
    _fail = 0
    for _r in _results:
        _err = _r.get("error")
        _status = _r.get("mcp_config", "error")
        if _err:
            print(f"  ERROR     {_r['ide']}: {_err}", file=sys.stderr)
            _fail += 1
        else:
            print(f"  {_status.upper():9s} {_r['ide']} → {_r.get('mcp_path', '')}")
            _ok += 1
    print(f"\n{_ok} connected, {_fail} errors.")
    sys.exit(0 if _fail == 0 else 1)
