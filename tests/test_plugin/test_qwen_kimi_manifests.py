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
