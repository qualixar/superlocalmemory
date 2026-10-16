"""Native Qwen Code and Kimi Code manifests ship at the plugin root.

Qwen's converter of the Claude manifest nested ``mcpServers`` one level too
deep and kept the Windows-only ``${ComSpec}`` launch line, so the server could
not start. Each host now gets its own manifest that launches slm-launch.
"""

from __future__ import annotations

import json

from tests.test_plugin._plugin_sandbox import BUILT_PLUGIN, REPO


def _load(name: str) -> dict:
    return json.loads((BUILT_PLUGIN / name).read_text(encoding="utf-8"))


def _version() -> str:
    return json.loads((REPO / "plugin-src" / "manifest.json").read_text())["version"]


def test_qwen_manifest_launches_the_plugin_launcher() -> None:
    data = _load("qwen-extension.json")
    assert data["name"] == "superlocalmemory"
    assert data["version"] == _version()
    server = data["mcpServers"]["superlocalmemory"]
    assert server["command"] == "${extensionPath}/scripts/slm-launch"
    assert "mcpServers" not in data["mcpServers"], "servers must not be nested twice"
    assert server["env"]["SLM_AGENT_ID"] == "qwen_code"


def test_kimi_manifest_uses_paths_inside_the_plugin_root() -> None:
    data = _load("kimi.plugin.json")
    assert data["name"] == "superlocalmemory"
    assert data["version"] == _version()
    server = data["mcpServers"]["superlocalmemory"]
    assert server["command"] == "./scripts/slm-launch"
    assert data["skills"] == "./skills/" and data["agents"] == "./agents/"
    assert server["env"]["SLM_AGENT_ID"] == "kimi_code"


def test_no_manifest_carries_the_windows_launch_line() -> None:
    for name in ("qwen-extension.json", "kimi.plugin.json"):
        assert "ComSpec" not in (BUILT_PLUGIN / name).read_text(encoding="utf-8")


def test_agent_plugins_manifest_for_kiro_and_cross_tool_hosts() -> None:
    plugin = _load("plugin.json")
    assert plugin["$schema"] == "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
    assert plugin["name"] == "superlocalmemory" and plugin["version"] == _version()
    assert plugin["license"] == "AGPL-3.0-or-later" and plugin["keywords"]
    mcp = _load("mcp.json")
    assert mcp["$schema"] == "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"
    server = mcp["mcpServers"]["superlocalmemory"]
    assert server == {"type": "stdio", "command": "slm", "args": ["mcp"],
                      "env": {"SLM_AGENT_ID": "agent_plugins"}}


def test_muse_build_does_not_carry_other_hosts_manifests() -> None:
    muse = REPO / "muse-plugin"
    for name in ("plugin.json", "mcp.json", "qwen-extension.json", "kimi.plugin.json"):
        assert not (muse / name).exists(), f"muse-plugin must not ship {name}"
