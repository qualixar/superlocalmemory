# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for Claude Code hook installer — v3.3.6 hybrid hooks.

Covers: install_hooks, remove_hooks, check_status, auto_install_if_needed,
upgrade_hooks, _merge_hooks, _is_slm_hook_entry, _hook_definitions.

Uses tmp_path with monkeypatch to redirect all file I/O to temp dirs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from superlocalmemory.hooks import claude_code_hooks as hooks_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_paths(tmp_path, monkeypatch):
    """Redirect all module-level paths to tmp_path so tests never touch real files."""
    settings_path = tmp_path / ".claude" / "settings.json"
    version_dir = tmp_path / ".superlocalmemory" / "hooks"
    version_file = version_dir / ".version"
    disabled_file = version_dir / ".hooks-disabled"

    monkeypatch.setattr(hooks_mod, "CLAUDE_SETTINGS", settings_path)
    monkeypatch.setattr(hooks_mod, "VERSION_DIR", version_dir)
    monkeypatch.setattr(hooks_mod, "VERSION_FILE", version_file)
    monkeypatch.setattr(hooks_mod, "DISABLED_FILE", disabled_file)


@pytest.fixture
def settings_path():
    """Return the current (patched) CLAUDE_SETTINGS path."""
    return hooks_mod.CLAUDE_SETTINGS


@pytest.fixture
def version_dir():
    """Return the current (patched) VERSION_DIR path."""
    return hooks_mod.VERSION_DIR


@pytest.fixture
def version_file():
    """Return the current (patched) VERSION_FILE path."""
    return hooks_mod.VERSION_FILE


@pytest.fixture
def disabled_file():
    """Return the current (patched) DISABLED_FILE path."""
    return hooks_mod.DISABLED_FILE


def _write_settings(settings_path: Path, data: dict) -> None:
    """Helper: write settings.json with given data."""
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(data, indent=2) + "\n")


def _read_settings(settings_path: Path) -> dict:
    """Helper: read settings.json."""
    return json.loads(settings_path.read_text())


# ---------------------------------------------------------------------------
# _is_slm_hook_entry
# ---------------------------------------------------------------------------


class TestIsSLMHookEntry:
    """Detect SLM vs non-SLM hook entries."""

    def test_detects_slm_hook_command(self):
        entry = {"hooks": [{"type": "command", "command": "slm hook start 2>/dev/null || true"}]}
        assert hooks_mod._is_slm_hook_entry(entry) is True

    def test_detects_session_marker(self):
        entry = {"hooks": [{"type": "command", "command": "test -f /tmp/slm-session-initialized"}]}
        assert hooks_mod._is_slm_hook_entry(entry) is True

    def test_detects_legacy_script_path(self):
        entry = {"hooks": [{"type": "command", "command": "bash ~/.superlocalmemory/hooks/start.sh"}]}
        assert hooks_mod._is_slm_hook_entry(entry) is True

    def test_detects_slm_session_start_time(self):
        entry = {"hooks": [{"type": "command", "command": "test ! -f /tmp/slm-session-start-time"}]}
        assert hooks_mod._is_slm_hook_entry(entry) is True

    def test_rejects_non_slm_entry(self):
        entry = {"hooks": [{"type": "command", "command": "prettier --write $FILE"}]}
        assert hooks_mod._is_slm_hook_entry(entry) is False

    def test_rejects_empty_hooks_list(self):
        entry = {"hooks": []}
        assert hooks_mod._is_slm_hook_entry(entry) is False

    def test_rejects_entry_without_hooks_key(self):
        entry = {"matcher": "Write"}
        assert hooks_mod._is_slm_hook_entry(entry) is False

    def test_rejects_entry_without_command(self):
        entry = {"hooks": [{"type": "command"}]}
        assert hooks_mod._is_slm_hook_entry(entry) is False


# ---------------------------------------------------------------------------
# _hook_definitions
# ---------------------------------------------------------------------------


class TestHookDefinitions:
    """Verify hook structure with and without gate."""

    def test_default_no_gate(self):
        defs = hooks_mod._hook_definitions(include_gate=False)
        assert "SessionStart" in defs
        assert "PostToolUse" in defs
        assert "Stop" in defs
        # v3.4.43: PreToolUse now present by default for pre-web recall on
        # WebSearch/WebFetch. The gate entry is added on top only when
        # include_gate=True (see test_with_gate).
        assert "PreToolUse" in defs
        assert len(defs["PreToolUse"]) == 1
        assert defs["PreToolUse"][0]["matcher"] == "WebSearch|WebFetch"

    def test_with_gate(self):
        defs = hooks_mod._hook_definitions(include_gate=True)
        assert "PreToolUse" in defs
        # v3.4.43: gate is prepended; pre-web recall stays after it.
        # PreToolUse entries:
        #   [0] gate (matcher _GATED_TOOLS)
        #   [1] before_web (matcher WebSearch|WebFetch)
        assert len(defs["PreToolUse"]) == 2
        # PostToolUse entries:
        #   [0] init-done (matcher mcp__superlocalmemory__session_init)
        #   [1] checkpoint (matcher Write|Edit)
        #   [2] post_tool_outcome (LLD-09 Track A.2 — all gated tools)
        assert len(defs["PostToolUse"]) == 3

    def test_default_userpromptsubmit_includes_topic_shift(self):
        """v3.4.43: UserPromptSubmit should fire user_prompt_rehash AND topic_shift."""
        defs = hooks_mod._hook_definitions(include_gate=False)
        ups = defs["UserPromptSubmit"]
        assert len(ups) == 2
        cmds = [m["hooks"][0]["command"] for m in ups]
        assert any("user_prompt_rehash" in c for c in cmds)
        assert any("topic_shift" in c for c in cmds)

    def test_default_pretooluse_includes_before_web(self):
        """v3.4.43: PreToolUse should include before_web for WebSearch/WebFetch."""
        defs = hooks_mod._hook_definitions(include_gate=False)
        pre = defs["PreToolUse"]
        assert pre[0]["matcher"] == "WebSearch|WebFetch"
        assert "before_web" in pre[0]["hooks"][0]["command"]

    def test_session_start_has_timeout(self):
        defs = hooks_mod._hook_definitions()
        # v3.6.18: [0] = mandate (5000ms), [1] = start (15000ms)
        mandate_hook = defs["SessionStart"][0]["hooks"][0]
        assert mandate_hook["timeout"] == 5000
        assert mandate_hook["type"] == "command"
        start_hook = defs["SessionStart"][1]["hooks"][0]
        assert start_hook["timeout"] == 15000
        assert start_hook["type"] == "command"

    def test_stop_has_timeout(self):
        defs = hooks_mod._hook_definitions()
        stop_hook = defs["Stop"][0]["hooks"][0]
        assert stop_hook["timeout"] == 10000

    def test_checkpoint_matcher(self):
        defs = hooks_mod._hook_definitions()
        checkpoint = defs["PostToolUse"][0]
        assert checkpoint["matcher"] == "Write|Edit"

    def test_gate_matcher_covers_tools(self):
        defs = hooks_mod._hook_definitions(include_gate=True)
        pre_tool = defs["PreToolUse"][0]
        assert "Bash" in pre_tool["matcher"]
        assert "Read" in pre_tool["matcher"]
        assert "Write" in pre_tool["matcher"]

    def test_gate_timeout_is_fast(self):
        defs = hooks_mod._hook_definitions(include_gate=True)
        gate_hook = defs["PreToolUse"][0]["hooks"][0]
        assert gate_hook["timeout"] == 500

    def test_init_done_timeout_is_fast(self):
        defs = hooks_mod._hook_definitions(include_gate=True)
        init_done = defs["PostToolUse"][0]
        assert init_done["matcher"] == "mcp__superlocalmemory__session_init"
        assert init_done["hooks"][0]["timeout"] == 500

    def test_commands_use_error_wrapping(self):
        """Value-add hooks should have error absorption wrappers."""
        defs = hooks_mod._hook_definitions()
        start_cmd = defs["SessionStart"][0]["hooks"][0]["command"]
        assert "|| true" in start_cmd or "|| exit /b 0" in start_cmd

    def test_default_gate_is_false(self):
        """Calling without arguments defaults to gate=False (only before_web in PreToolUse)."""
        defs = hooks_mod._hook_definitions()
        # v3.4.43: PreToolUse now present by default for pre-web recall.
        # The gate-specific entry (matcher _GATED_TOOLS) is NOT added unless
        # include_gate=True. Verify only the before_web entry is present.
        assert "PreToolUse" in defs
        assert len(defs["PreToolUse"]) == 1
        assert defs["PreToolUse"][0]["matcher"] == "WebSearch|WebFetch"


# ---------------------------------------------------------------------------
# _merge_hooks
# ---------------------------------------------------------------------------


class TestMergeHooks:
    """Merge SLM hooks into settings, preserving non-SLM hooks."""

    def test_merge_into_empty_settings(self):
        settings = {}
        hook_defs = hooks_mod._hook_definitions()
        result = hooks_mod._merge_hooks(settings, hook_defs)
        assert "hooks" in result
        assert "SessionStart" in result["hooks"]
        assert "PostToolUse" in result["hooks"]
        assert "Stop" in result["hooks"]

    def test_preserves_non_slm_hooks(self):
        non_slm_entry = {
            "matcher": "Write",
            "hooks": [{"type": "command", "command": "prettier --write $FILE"}],
        }
        settings = {"hooks": {"PostToolUse": [non_slm_entry]}}
        hook_defs = hooks_mod._hook_definitions()
        result = hooks_mod._merge_hooks(settings, hook_defs)

        post_tool = result["hooks"]["PostToolUse"]
        # First entry should be the non-SLM one (preserved), followed by SLM
        commands = [e["hooks"][0]["command"] for e in post_tool]
        assert any("prettier" in c for c in commands)
        assert any("slm hook" in c for c in commands)

    def test_replaces_existing_slm_hooks(self):
        old_slm = {
            "hooks": [{"type": "command", "command": "slm hook start 2>/dev/null || true"}],
        }
        settings = {"hooks": {"SessionStart": [old_slm]}}
        hook_defs = hooks_mod._hook_definitions()
        result = hooks_mod._merge_hooks(settings, hook_defs)

        # v3.6.18: 2 SessionStart entries after merge — mandate + start (not triplicated)
        assert len(result["hooks"]["SessionStart"]) == 2

    def test_preserves_non_hook_settings(self):
        settings = {"allowedTools": ["Bash"], "hooks": {}}
        hook_defs = hooks_mod._hook_definitions()
        result = hooks_mod._merge_hooks(settings, hook_defs)
        assert result["allowedTools"] == ["Bash"]

    def test_merge_is_idempotent(self):
        settings = {}
        hook_defs = hooks_mod._hook_definitions()
        first = hooks_mod._merge_hooks(settings, hook_defs)
        second = hooks_mod._merge_hooks(first, hook_defs)
        assert first == second


# ---------------------------------------------------------------------------
# install_hooks
# ---------------------------------------------------------------------------


class TestInstallHooks:
    """Install SLM hooks into Claude Code settings.json."""

    def test_install_creates_settings_file(self, settings_path):
        result = hooks_mod.install_hooks()
        assert result["success"] is True
        assert settings_path.exists()

    def test_install_writes_version_file(self, version_file):
        hooks_mod.install_hooks()
        assert version_file.exists()
        assert version_file.read_text().strip() == hooks_mod.HOOKS_VERSION

    def test_install_returns_hooks_added(self):
        result = hooks_mod.install_hooks()
        assert "SessionStart" in result["hooks_added"]
        assert "PostToolUse" in result["hooks_added"]
        assert "Stop" in result["hooks_added"]

    def test_install_default_no_gate(self):
        result = hooks_mod.install_hooks()
        assert result["gate_enabled"] is False
        # v3.4.43: PreToolUse IS now in hooks_added (pre-web recall is the
        # default entry). Gate=False just means the _GATED_TOOLS firewall
        # entry isn't added on top.
        assert "PreToolUse" in result["hooks_added"]

    def test_install_with_gate(self, settings_path):
        result = hooks_mod.install_hooks(include_gate=True)
        assert result["gate_enabled"] is True
        assert "PreToolUse" in result["hooks_added"]
        data = _read_settings(settings_path)
        assert "PreToolUse" in data["hooks"]

    def test_install_preserves_non_slm_hooks(self, settings_path):
        non_slm = {
            "hooks": {
                "PostToolUse": [
                    {"hooks": [{"type": "command", "command": "eslint --fix $FILE"}]}
                ]
            }
        }
        _write_settings(settings_path, non_slm)
        hooks_mod.install_hooks()
        data = _read_settings(settings_path)
        commands = [e["hooks"][0]["command"] for e in data["hooks"]["PostToolUse"]]
        assert any("eslint" in c for c in commands)

    def test_install_clears_disabled_marker(self, disabled_file, version_dir):
        version_dir.mkdir(parents=True, exist_ok=True)
        disabled_file.write_text("removed by user\n")
        assert disabled_file.exists()

        hooks_mod.install_hooks()
        assert not disabled_file.exists()

    def test_install_idempotent(self, settings_path):
        hooks_mod.install_hooks()
        first = _read_settings(settings_path)
        hooks_mod.install_hooks()
        second = _read_settings(settings_path)
        assert first == second

    def test_install_twice_same_result(self, settings_path):
        """Install twice = exactly one set of SLM hooks, not duplicated."""
        hooks_mod.install_hooks()
        hooks_mod.install_hooks()
        data = _read_settings(settings_path)
        # v3.6.18: 2 SessionStart entries — mandate (first) + start
        assert len(data["hooks"]["SessionStart"]) == 2
        assert len(data["hooks"]["Stop"]) == 1

    def test_install_no_errors_on_success(self):
        result = hooks_mod.install_hooks()
        assert result["errors"] == []

    def test_install_preserves_other_settings_keys(self, settings_path):
        _write_settings(settings_path, {"allowedTools": ["Bash", "Read"], "theme": "dark"})
        hooks_mod.install_hooks()
        data = _read_settings(settings_path)
        assert data["allowedTools"] == ["Bash", "Read"]
        assert data["theme"] == "dark"


# ---------------------------------------------------------------------------
# remove_hooks
# ---------------------------------------------------------------------------


class TestRemoveHooks:
    """Remove SLM hooks from settings.json."""

    def test_remove_after_install(self, settings_path):
        hooks_mod.install_hooks()
        result = hooks_mod.remove_hooks()
        assert result["success"] is True
        data = _read_settings(settings_path)
        # All SLM hooks gone — "hooks" key should be empty or absent
        assert "hooks" not in data or data["hooks"] == {}

    def test_remove_writes_disabled_marker(self, disabled_file):
        hooks_mod.install_hooks()
        hooks_mod.remove_hooks()
        assert disabled_file.exists()
        assert "removed by user" in disabled_file.read_text()

    def test_remove_deletes_version_file(self, version_file):
        hooks_mod.install_hooks()
        assert version_file.exists()
        hooks_mod.remove_hooks()
        assert not version_file.exists()

    def test_remove_preserves_non_slm_hooks(self, settings_path):
        non_slm = {"hooks": [{"type": "command", "command": "prettier --write $FILE"}]}
        slm_entry = {"hooks": [{"type": "command", "command": "slm hook checkpoint 2>/dev/null || true"}]}
        _write_settings(settings_path, {"hooks": {"PostToolUse": [non_slm, slm_entry]}})

        hooks_mod.remove_hooks()
        data = _read_settings(settings_path)
        post = data["hooks"]["PostToolUse"]
        assert len(post) == 1
        assert "prettier" in post[0]["hooks"][0]["command"]

    def test_remove_on_empty_settings(self, settings_path):
        _write_settings(settings_path, {})
        result = hooks_mod.remove_hooks()
        assert result["success"] is True

    def test_remove_on_no_settings_file(self):
        result = hooks_mod.remove_hooks()
        assert result["success"] is True

    def test_remove_idempotent(self, settings_path):
        hooks_mod.install_hooks()
        hooks_mod.remove_hooks()
        first = _read_settings(settings_path)
        hooks_mod.remove_hooks()
        second = _read_settings(settings_path)
        assert first == second

    def test_remove_cleans_empty_hook_types(self, settings_path):
        """If a hook type has only SLM entries, remove the whole key."""
        hooks_mod.install_hooks()
        hooks_mod.remove_hooks()
        data = _read_settings(settings_path)
        assert "SessionStart" not in data.get("hooks", {})
        assert "Stop" not in data.get("hooks", {})


# ---------------------------------------------------------------------------
# check_status
# ---------------------------------------------------------------------------


class TestCheckStatus:
    """Report installation status."""

    def test_status_when_not_installed(self):
        status = hooks_mod.check_status()
        assert status["installed"] is False
        assert status["version"] == ""
        assert status["gate_enabled"] is False

    def test_status_after_install(self):
        hooks_mod.install_hooks()
        status = hooks_mod.check_status()
        assert status["installed"] is True
        assert status["version"] == hooks_mod.HOOKS_VERSION
        assert status["latest_version"] == hooks_mod.HOOKS_VERSION
        assert status["needs_upgrade"] is False
        assert status["gate_enabled"] is False

    def test_status_with_gate(self):
        hooks_mod.install_hooks(include_gate=True)
        status = hooks_mod.check_status()
        assert status["gate_enabled"] is True

    def test_status_detects_hook_types(self):
        hooks_mod.install_hooks()
        status = hooks_mod.check_status()
        assert "SessionStart" in status["hook_types"]
        assert "PostToolUse" in status["hook_types"]
        assert "Stop" in status["hook_types"]

    def test_status_needs_upgrade(self, version_file, version_dir):
        hooks_mod.install_hooks()
        # Simulate older version
        version_dir.mkdir(parents=True, exist_ok=True)
        version_file.write_text("3.3.5")
        status = hooks_mod.check_status()
        assert status["needs_upgrade"] is True
        assert status["version"] == "3.3.5"

    def test_status_after_remove(self):
        hooks_mod.install_hooks()
        hooks_mod.remove_hooks()
        status = hooks_mod.check_status()
        assert status["installed"] is False
        assert status["version"] == ""

    def test_status_installed_requires_3_hook_types(self, settings_path):
        """installed=True only when at least 3 hook types contain SLM entries."""
        _write_settings(settings_path, {
            "hooks": {
                "SessionStart": [{"hooks": [{"type": "command", "command": "slm hook start 2>/dev/null || true"}]}],
            }
        })
        status = hooks_mod.check_status()
        assert status["installed"] is False  # Only 1 hook type


# ---------------------------------------------------------------------------
# auto_install_if_needed
# ---------------------------------------------------------------------------


class TestAutoInstallIfNeeded:
    """Auto-install: respects disabled marker, fast path on version match."""

    def test_auto_install_fresh(self, settings_path):
        """Installs when no version file and no disabled marker."""
        result = hooks_mod.auto_install_if_needed()
        assert result is not None
        assert result["success"] is True
        assert settings_path.exists()

    def test_auto_install_respects_disabled(self, disabled_file, version_dir):
        """Returns None when .hooks-disabled marker exists."""
        version_dir.mkdir(parents=True, exist_ok=True)
        disabled_file.write_text("removed by user\n")
        result = hooks_mod.auto_install_if_needed()
        assert result is None

    def test_fast_path_version_match(self, version_file, version_dir):
        """Returns None immediately when installed version matches current."""
        version_dir.mkdir(parents=True, exist_ok=True)
        version_file.write_text(hooks_mod.HOOKS_VERSION)
        result = hooks_mod.auto_install_if_needed()
        assert result is None

    def test_upgrades_on_version_mismatch(self, version_file, version_dir, settings_path):
        """Installs when version file exists but differs from current."""
        # Pre-install to have valid settings
        hooks_mod.install_hooks()
        # Simulate older version
        version_file.write_text("3.3.5")
        result = hooks_mod.auto_install_if_needed()
        assert result is not None
        assert result["success"] is True
        assert version_file.read_text().strip() == hooks_mod.HOOKS_VERSION

    def test_auto_install_uses_no_gate(self, settings_path):
        """Auto-install always uses include_gate=False.

        v3.4.43: PreToolUse is now populated by default (pre-web recall).
        The gate-specific PreToolUse entry (matcher _GATED_TOOLS) is NOT
        added. Verify only the WebSearch|WebFetch entry exists.
        """
        result = hooks_mod.auto_install_if_needed()
        assert result is not None
        assert result["gate_enabled"] is False
        data = _read_settings(settings_path)
        pre = data.get("hooks", {}).get("PreToolUse", [])
        # Exactly 1 entry (pre-web recall), no gate.
        assert len(pre) == 1
        assert pre[0]["matcher"] == "WebSearch|WebFetch"

    def test_auto_install_clears_disabled_on_install(self, disabled_file):
        """install_hooks clears .hooks-disabled, but auto_install never
        runs when disabled, so disabled should remain absent after auto_install."""
        result = hooks_mod.auto_install_if_needed()
        assert result is not None
        assert not disabled_file.exists()

    def test_auto_install_exception_returns_none(self, monkeypatch):
        """If something raises, return None (don't crash MCP startup)."""
        def _boom_read_settings():
            raise PermissionError("no access")

        # Ensure not disabled and no version file so it attempts install
        # Then blow up during install_hooks -> _read_settings
        monkeypatch.setattr(hooks_mod, "_read_settings", _boom_read_settings)
        result = hooks_mod.auto_install_if_needed()
        # The exception in install_hooks is caught, but install returns errors.
        # However the outer try/except in auto_install_if_needed catches too.
        # Either way, it should NOT raise.
        assert result is None or (isinstance(result, dict) and len(result.get("errors", [])) > 0)


# ---------------------------------------------------------------------------
# upgrade_hooks
# ---------------------------------------------------------------------------


class TestUpgradeHooks:
    """Upgrade existing hooks preserving gate setting."""

    def test_upgrade_when_not_installed(self):
        result = hooks_mod.upgrade_hooks()
        assert result["upgraded"] is False
        assert result["reason"] == "No hooks installed"

    def test_upgrade_preserves_gate_off(self, version_file):
        hooks_mod.install_hooks(include_gate=False)
        # Simulate older version
        version_file.write_text("3.3.5")
        result = hooks_mod.upgrade_hooks()
        assert result["upgraded"] is True
        assert result["from_version"] == "3.3.5"
        assert result["to_version"] == hooks_mod.HOOKS_VERSION
        assert result["gate_enabled"] is False

    def test_upgrade_preserves_gate_on(self, version_file, settings_path):
        hooks_mod.install_hooks(include_gate=True)
        version_file.write_text("3.3.5")
        result = hooks_mod.upgrade_hooks()
        assert result["upgraded"] is True
        assert result["gate_enabled"] is True
        data = _read_settings(settings_path)
        assert "PreToolUse" in data["hooks"]

    def test_upgrade_writes_current_version(self, version_file):
        hooks_mod.install_hooks()
        version_file.write_text("3.3.4")
        hooks_mod.upgrade_hooks()
        assert version_file.read_text().strip() == hooks_mod.HOOKS_VERSION

    def test_upgrade_preserves_non_slm_hooks(self, settings_path, version_file):
        non_slm = {"hooks": [{"type": "command", "command": "mypy --strict $FILE"}]}
        hooks_mod.install_hooks()
        data = _read_settings(settings_path)
        data["hooks"]["PostToolUse"].insert(0, non_slm)
        _write_settings(settings_path, data)
        version_file.write_text("3.3.5")

        hooks_mod.upgrade_hooks()
        data = _read_settings(settings_path)
        commands = [e["hooks"][0]["command"] for e in data["hooks"]["PostToolUse"]]
        assert any("mypy" in c for c in commands)


# ---------------------------------------------------------------------------
# .hooks-disabled lifecycle
# ---------------------------------------------------------------------------


class TestDisabledMarkerLifecycle:
    """Full lifecycle: install clears, remove writes, auto_install respects."""

    def test_full_cycle(self, disabled_file, version_dir):
        # 1. Install — no marker
        hooks_mod.install_hooks()
        assert not disabled_file.exists()

        # 2. Remove — marker written
        hooks_mod.remove_hooks()
        assert disabled_file.exists()

        # 3. Auto-install — respects marker, does nothing
        result = hooks_mod.auto_install_if_needed()
        assert result is None
        assert disabled_file.exists()

        # 4. Explicit install — clears marker
        hooks_mod.install_hooks()
        assert not disabled_file.exists()

    def test_remove_then_install_then_auto(self, disabled_file):
        hooks_mod.install_hooks()
        hooks_mod.remove_hooks()
        assert disabled_file.exists()
        hooks_mod.install_hooks()
        assert not disabled_file.exists()
        # Auto-install should see version match → None
        result = hooks_mod.auto_install_if_needed()
        assert result is None


# ---------------------------------------------------------------------------
# Cross-platform commands
# ---------------------------------------------------------------------------


class TestCrossPlatformCommands:
    """Verify platform-specific command generation."""

    def test_unix_gate_cmd(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        cmd = hooks_mod._gate_cmd()
        assert "test -f" in cmd
        assert "exit 2" in cmd

    def test_unix_init_done_cmd(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        cmd = hooks_mod._init_done_cmd()
        assert cmd.startswith("touch ")

    def test_unix_wrap_python_cmd(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        cmd = hooks_mod._wrap_python_cmd("start")
        assert "slm hook start" in cmd
        assert "2>/dev/null || true" in cmd

    def test_windows_gate_cmd(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        cmd = hooks_mod._gate_cmd()
        assert "cmd /c" in cmd
        assert "exit /b 0" in cmd
        assert "exit /b 2" in cmd

    def test_windows_init_done_cmd(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        cmd = hooks_mod._init_done_cmd()
        assert "cmd /c" in cmd
        assert "echo." in cmd

    def test_windows_wrap_python_cmd(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        cmd = hooks_mod._wrap_python_cmd("checkpoint")
        assert "cmd /c" in cmd
        assert "slm hook checkpoint" in cmd
        assert "2>NUL" in cmd
        assert "exit /b 0" in cmd

    def test_windows_hooks_use_backslash(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        defs = hooks_mod._hook_definitions(include_gate=True)
        gate_cmd = defs["PreToolUse"][0]["hooks"][0]["command"]
        assert "\\" in gate_cmd


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Error handling, malformed input, boundary conditions."""

    def test_install_with_malformed_settings(self, settings_path):
        """If settings.json has non-JSON content, install should handle it."""
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text("not json at all")
        result = hooks_mod.install_hooks()
        # _read_settings will raise, caught in install_hooks
        assert result["success"] is False or len(result["errors"]) > 0

    def test_install_creates_parent_dirs(self, settings_path):
        """Installs even when .claude/ directory doesn't exist yet."""
        assert not settings_path.parent.exists()
        result = hooks_mod.install_hooks()
        assert result["success"] is True
        assert settings_path.parent.exists()

    def test_merge_mixed_slm_and_non_slm(self):
        """Mixed entries: SLM replaced, non-SLM preserved, order correct."""
        non_slm_1 = {"hooks": [{"type": "command", "command": "eslint $FILE"}]}
        slm_old = {"hooks": [{"type": "command", "command": "slm hook checkpoint 2>/dev/null || true"}]}
        non_slm_2 = {"hooks": [{"type": "command", "command": "black $FILE"}]}

        settings = {"hooks": {"PostToolUse": [non_slm_1, slm_old, non_slm_2]}}
        hook_defs = hooks_mod._hook_definitions()
        result = hooks_mod._merge_hooks(settings, hook_defs)

        post = result["hooks"]["PostToolUse"]
        commands = [e["hooks"][0]["command"] for e in post]
        # non-SLM preserved in order, SLM old replaced by new at end
        assert commands[0] == "eslint $FILE"
        assert commands[1] == "black $FILE"
        assert "slm hook" in commands[2]

    def test_settings_json_pretty_formatted(self, settings_path):
        """settings.json should be pretty-printed (indent=2)."""
        hooks_mod.install_hooks()
        content = settings_path.read_text()
        assert "  " in content  # indented
        assert content.endswith("\n")

    def test_remove_slm_hooks_leaves_empty_hooks_deleted(self):
        """_remove_slm_hooks deletes 'hooks' key when all types empty."""
        slm_entry = {"hooks": [{"type": "command", "command": "slm hook start 2>/dev/null || true"}]}
        settings = {"hooks": {"SessionStart": [slm_entry]}}
        result = hooks_mod._remove_slm_hooks(settings)
        assert "hooks" not in result

    def test_check_status_with_corrupt_version_file(self, version_file, version_dir):
        """Status handles unreadable version file gracefully."""
        version_dir.mkdir(parents=True, exist_ok=True)
        version_file.write_text("")
        status = hooks_mod.check_status()
        assert status["version"] == ""

    def test_version_file_content(self, version_file):
        """Version file should contain the exact HOOKS_VERSION string."""
        hooks_mod.install_hooks()
        assert version_file.read_text() == hooks_mod.HOOKS_VERSION


# ---------------------------------------------------------------------------
# Integration: full install → status → upgrade → remove → status
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    """End-to-end lifecycle integration tests."""

    def test_install_status_remove_status(self):
        # Install
        install_result = hooks_mod.install_hooks()
        assert install_result["success"] is True

        # Status: installed
        status = hooks_mod.check_status()
        assert status["installed"] is True
        assert status["version"] == hooks_mod.HOOKS_VERSION

        # Remove
        remove_result = hooks_mod.remove_hooks()
        assert remove_result["success"] is True

        # Status: not installed
        status = hooks_mod.check_status()
        assert status["installed"] is False

    def test_install_upgrade_remove(self, version_file, settings_path):
        # Install with gate
        hooks_mod.install_hooks(include_gate=True)

        # Simulate older version
        version_file.write_text("3.3.4")

        # Upgrade
        result = hooks_mod.upgrade_hooks()
        assert result["upgraded"] is True
        assert result["from_version"] == "3.3.4"
        assert result["to_version"] == hooks_mod.HOOKS_VERSION
        # Gate preserved
        data = _read_settings(settings_path)
        assert "PreToolUse" in data["hooks"]

        # Remove
        hooks_mod.remove_hooks()
        status = hooks_mod.check_status()
        assert status["installed"] is False

    def test_auto_install_then_explicit_remove_blocks_auto(self, disabled_file):
        """Auto-install → remove → auto-install should be blocked."""
        result = hooks_mod.auto_install_if_needed()
        assert result is not None
        assert result["success"] is True

        hooks_mod.remove_hooks()
        assert disabled_file.exists()

        result = hooks_mod.auto_install_if_needed()
        assert result is None
