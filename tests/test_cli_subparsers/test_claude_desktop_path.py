"""Where Claude Desktop keeps its config on each platform (audit F7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from superlocalmemory.hooks import portable_kit
from superlocalmemory.hooks.portable_kit import IDE_MATRIX, connect_ide, global_config_path

FILE = "claude_desktop_config.json"
HOME = Path("/home/someone")


def _path(platform: str, environ: dict[str, str]) -> Path:
    return global_config_path(IDE_MATRIX["claude-desktop"], HOME, platform=platform, environ=environ)


def test_macos_keeps_application_support() -> None:
    assert _path("darwin", {}) == HOME / "Library" / "Application Support" / "Claude" / FILE


def test_linux_keeps_dot_config() -> None:
    assert _path("linux", {}) == HOME / ".config" / "Claude" / FILE


def test_windows_uses_appdata() -> None:
    appdata = "C:\\Users\\Someone\\AppData\\Roaming"
    assert _path("win32", {"APPDATA": appdata}) == Path(appdata) / "Claude" / FILE


def test_windows_without_appdata_falls_back_to_the_default_roaming_folder() -> None:
    assert _path("win32", {}) == HOME / "AppData" / "Roaming" / "Claude" / FILE


def test_other_hosts_are_unchanged_on_windows() -> None:
    cursor = IDE_MATRIX["cursor"]
    assert global_config_path(cursor, HOME, platform="win32", environ={"APPDATA": "X:\\a"}) == HOME / cursor.mcp_path_global


def test_connect_on_windows_writes_under_appdata(monkeypatch, tmp_path: Path) -> None:
    appdata = tmp_path / "Roaming"
    monkeypatch.setattr(portable_kit.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(appdata))
    result = connect_ide("claude-desktop")  # no home given: the real environment decides
    assert result["error"] is None, result
    target = appdata / "Claude" / FILE
    assert Path(result["mcp_path"]) == target
    assert "superlocalmemory" in json.loads(target.read_text())["mcpServers"]


def test_a_given_home_stays_sandboxed_even_on_windows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(portable_kit.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "elsewhere"))
    result = connect_ide("claude-desktop", home=tmp_path / "home")
    assert result["error"] is None, result
    assert Path(result["mcp_path"]) == tmp_path / "home" / "AppData" / "Roaming" / "Claude" / FILE


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_connect_on_other_platforms_uses_the_home(monkeypatch, tmp_path: Path, platform: str) -> None:
    monkeypatch.setattr(portable_kit.sys, "platform", platform)
    result = connect_ide("claude-desktop", home=tmp_path)
    expected = (tmp_path / "Library" / "Application Support" / "Claude" if platform == "darwin"
                else tmp_path / ".config" / "Claude") / FILE
    assert Path(result["mcp_path"]) == expected


def test_upgrade_detection_looks_where_connect_wrote(monkeypatch, tmp_path: Path) -> None:
    from superlocalmemory.cli import host_upgrades

    monkeypatch.setattr(portable_kit.sys, "platform", "win32")
    monkeypatch.delenv("APPDATA", raising=False)
    connect_ide("claude-desktop", home=tmp_path)
    assert "claude-desktop" in host_upgrades._detected_hosts(tmp_path)
