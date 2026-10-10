"""an earlier stage portable-host certification boundaries.

This module makes two deliberately narrow claims, and no more:

* ``slm connect`` can add the SLM-owned MCP entry to supported portable host
  configs without losing unrelated structured configuration.
* A standards-compliant stdio MCP client can negotiate with the real ``slm
  mcp`` process and discover the lifecycle/memory tools.

It does *not* claim that an external proprietary host has been launched.  A
valid generated file and a valid MCP process are necessary compatibility
evidence, not proof that Cursor, Antigravity, Gemini, Continue, Grok, Muse,
or Hermes has accepted a particular local configuration.
"""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from superlocalmemory.hooks.portable_kit import IDE_MATRIX, connect_ide

ROOT = Path(__file__).resolve().parents[2]
PORTABLE_HOSTS = ("cursor", "antigravity", "gemini-cli", "continue")
GENERIC_STDIO_CLIENTS = ("grok", "muse", "hermes")
# A real, non-default profile: since 2d7a6a17 ``slm connect`` refuses an
# unknown profile name before writing any host config, so the certified write
# must use one SLM defines. "code" also proves the value is carried verbatim.
CERT_PROFILE = "code"
REQUIRED_LIFECYCLE_TOOLS = {
    "session_init",
    "recall",
    "remember",
    "close_session",
}


def _mcp_server_python() -> str:
    """Return an interpreter that can run SLM's pinned MCP 2 server.

    CI normally uses the project interpreter.  Local certification may supply
    ``SLM_TEST_MCP_PYTHON`` to exercise the exact isolated package runtime.
    Refusing an incompatible interpreter is deliberate: a successful client
    handshake with the wrong dependency stack is not release evidence.
    """
    configured = os.environ.get("SLM_TEST_MCP_PYTHON")
    if configured:
        candidate = Path(configured)
        if not candidate.is_file():
            pytest.fail("SLM_TEST_MCP_PYTHON must name an executable interpreter")
        return str(candidate)
    if importlib.util.find_spec("mcp.server.mcpserver") is None:
        pytest.skip(
            "portable MCP certification requires the declared mcp==2.0.0 "
            "server runtime; set SLM_TEST_MCP_PYTHON for an isolated artifact"
        )
    return sys.executable


def _read(path: Path, fmt: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if fmt == "json":
        value = json.loads(text)
    elif fmt == "yaml":
        value = yaml.safe_load(text)
    else:  # The an earlier stage portable subset is JSON + Continue YAML.
        raise AssertionError(f"unexpected portable host format: {fmt}")
    assert isinstance(value, dict)
    return value


def _write_preserved_config(path: Path, fmt: str, server_key: str) -> tuple[dict[str, Any], Any]:
    """Write unrelated config which SLM has no authority to alter."""
    other_server = {
        "command": "unrelated-memory",
        "args": ["--keep", "all"],
        "env": {"UNRELATED_SETTING": "preserve-me"},
        "nested": {"enabled": True, "priority": 7},
    }
    top_level = {"hostOwned": {"theme": "dark", "preserve": [1, 2, 3]}}
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        payload = {**top_level, server_key: {"unrelated": copy.deepcopy(other_server)}}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return top_level, other_server

    assert fmt == "yaml"
    provider = {
        "name": "mcp",
        "params": {
            "serverName": "unrelated",
            **copy.deepcopy(other_server),
        },
    }
    payload = {**top_level, server_key: [provider]}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return top_level, provider


@pytest.mark.parametrize("host", PORTABLE_HOSTS)
def test_portable_host_connect_is_additive_and_idempotent(host: str, tmp_path: Path) -> None:
    """Supported portable hosts keep all configuration SLM does not own."""
    descriptor = IDE_MATRIX[host]
    config_path = tmp_path / descriptor.mcp_path_global
    top_level, preserved = _write_preserved_config(
        config_path, descriptor.fmt, descriptor.server_key
    )

    first = connect_ide(host, home=tmp_path, profile=CERT_PROFILE)
    second = connect_ide(host, home=tmp_path, profile=CERT_PROFILE)

    assert first["error"] is None
    assert second["error"] is None
    assert first["mcp_config"] == "wrote"
    assert second["mcp_config"] == "unchanged"
    assert first["servers_preserved"] == 1

    rendered = _read(config_path, descriptor.fmt)
    assert rendered["hostOwned"] == top_level["hostOwned"]
    if descriptor.fmt == "json":
        assert rendered[descriptor.server_key]["unrelated"] == preserved
        slm = rendered[descriptor.server_key]["superlocalmemory"]
        assert slm["command"] == "slm"
        assert slm["args"] == ["mcp"]
        assert slm["env"]["SLM_MCP_PROFILE"] == CERT_PROFILE
    else:
        providers = rendered[descriptor.server_key]
        assert providers[0] == preserved
        matches = [
            provider
            for provider in providers
            if provider.get("params", {}).get("serverName") == "superlocalmemory"
        ]
        assert len(matches) == 1
        slm = matches[0]["params"]
        assert slm["command"] == "slm"
        assert slm["args"] == ["mcp"]
        assert slm["env"]["SLM_MCP_PROFILE"] == CERT_PROFILE


@pytest.mark.asyncio
@pytest.mark.parametrize("client_id", GENERIC_STDIO_CLIENTS)
async def test_generic_cli_clients_can_use_the_standard_stdio_mcp_contract(
    client_id: str, tmp_path: Path
) -> None:
    """A generic client identity completes MCP init plus tool discovery.

    The subprocess uses a test-only SLM data directory and no credentials.
    This certifies the documented generic *stdio* route, not an unverified
    native configuration file for any provider-specific CLI.
    """
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    data_dir = tmp_path / f"slm-data-{client_id}"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "SLM_DATA_DIR": str(data_dir),
        "SLM_AGENT_ID": client_id,
        "SLM_DISABLE_WARMUP_SIDE_EFFECTS": "1",
        "SLM_MCP_EMBEDDED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    params = StdioServerParameters(
        command=_mcp_server_python(),
        args=["-m", "superlocalmemory.cli.main", "mcp"],
        env=env,
        cwd=ROOT,
    )

    # A real file for the server's stderr. stdio_client's default (errlog=sys.stderr)
    # is bound when mcp.client.stdio is first imported; if an earlier test imported it
    # under capsys, that default is a dead in-memory buffer with no fileno.
    with open(tmp_path / f"mcp-server-{client_id}.err", "w", encoding="utf-8") as errlog:
        async with stdio_client(params, errlog=errlog) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                initialized = await asyncio.wait_for(session.initialize(), timeout=15)
                listed = await asyncio.wait_for(session.list_tools(), timeout=15)

    protocol_version = getattr(
        initialized, "protocol_version", getattr(initialized, "protocolVersion", None)
    )
    assert protocol_version
    names = {tool.name for tool in listed.tools}
    assert REQUIRED_LIFECYCLE_TOOLS <= names
    assert not data_dir.exists(), "MCP discovery must not initialize a data store"
