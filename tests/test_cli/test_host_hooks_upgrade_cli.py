# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""4.2.30: ``slm upgrade-hosts`` and ``slm doctor`` on SLM's own hook entries.

``upgrade-hosts --apply`` runs the same merge-only upgrade the daemon runs by itself, instead of
answering "plugin-managed". Doctor says per host whether SLM's hooks are current. Plugin-only hosts
get the exact command that moves them. Everything runs against tmp files.
"""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from superlocalmemory.cli import host_upgrades
from superlocalmemory.hooks import claude_code_hooks as claude
from superlocalmemory.hooks import codex_assets
from superlocalmemory.hooks import codex_hooks as codex
from tests.test_hooks.test_host_hook_upgrade import (
    _old_claude_settings,
    _old_codex_hooks,
    _write,
)


@pytest.fixture
def files(tmp_path, monkeypatch):
    settings = tmp_path / "home" / ".claude" / "settings.json"
    hooks = tmp_path / "home" / ".codex" / "hooks.json"
    monkeypatch.setattr(claude, "CLAUDE_SETTINGS", settings)
    monkeypatch.setattr(codex, "DEFAULT_HOOKS_PATH", hooks)
    monkeypatch.setattr(claude, "VERSION_DIR", tmp_path / "hooks-dir")
    monkeypatch.setattr(claude, "VERSION_FILE", tmp_path / "hooks-dir" / ".version")
    monkeypatch.setattr(claude, "DISABLED_FILE", tmp_path / "hooks-dir" / ".hooks-disabled")
    return settings, hooks


def test_upgrade_hosts_apply_updates_claude_settings_merge_only(files) -> None:
    settings, _ = files
    _write(settings, _old_claude_settings())

    result = host_upgrades._upgrade_host("claude-code", apply=True, already_integrated=True)

    assert result["status"] == "updated" and "plugin-managed" not in result["detail"]
    text = settings.read_text()
    assert "auto_recall" in text and "my-lint" in text and "my-notify" in text
    assert "backup" in result["detail"]


def test_upgrade_hosts_preview_changes_nothing(files) -> None:
    settings, _ = files
    _write(settings, _old_claude_settings())
    before = settings.read_bytes()

    result = host_upgrades._upgrade_host("claude-code", apply=False, already_integrated=True)

    assert result["status"] == "preview" and settings.read_bytes() == before


def test_upgrade_hosts_does_not_create_hooks_for_a_host_without_them(files) -> None:
    settings, _ = files
    _write(settings, {"theme": "dark"})
    before = settings.read_bytes()

    result = host_upgrades._upgrade_host("claude-code", apply=True, already_integrated=False)

    assert result["status"] == "verified" and "none added" in result["detail"]
    assert settings.read_bytes() == before


def test_upgrade_hosts_codex_lands_the_timeout_and_keeps_user_hooks(files, monkeypatch) -> None:
    _, hooks = files
    _write(hooks, _old_codex_hooks())
    monkeypatch.setattr(codex_assets, "install_assets",
                        lambda **_k: {"success": True, "skills_written": [], "agents_written": []})

    result = host_upgrades._upgrade_host("codex", apply=True, already_integrated=True)

    assert result["status"] == "updated"
    data = json.loads(hooks.read_text())
    prompt = data["hooks"]["UserPromptSubmit"]
    assert any(e["hooks"][0].get("timeout") == 7 for e in prompt if codex.is_slm_hook_entry(e))
    assert {"hooks": [{"type": "command", "command": "my-prompt"}]} in prompt


def test_detection_counts_a_settings_file_with_any_slm_hook(files) -> None:
    settings, _ = files
    _write(settings, {"hooks": {"Stop": [{"hooks": [{"type": "command",
                                                      "command": "slm hook stop 2>/dev/null || true"}]}]}})
    assert "claude-code" in host_upgrades._detected_hosts(home=settings.parents[1])


def test_plugin_only_hosts_print_the_exact_command(files, monkeypatch, capsys) -> None:
    monkeypatch.setattr(host_upgrades, "_detected_hosts",
                        lambda home=None: ["antigravity", "hermes", "muse", "claude-code", "cursor"])
    monkeypatch.setattr(host_upgrades, "_report_reconnect_apps", lambda _t: None)
    monkeypatch.setattr(host_upgrades, "_upgrade_host",
                        lambda host, **_k: {"status": "verified", "detail": "x"})
    monkeypatch.setattr("superlocalmemory.cli.measurement_cmd.announce_if_pending", lambda: None)

    host_upgrades.cmd_upgrade_hosts(Namespace(hosts=[], apply=False, all_detected=True))

    out = capsys.readouterr().out
    assert 'agy plugin install "$(npm root -g)/superlocalmemory/antigravity-plugin"' in out
    assert 'muse plugins install "$(npm root -g)/superlocalmemory/muse-plugin"' in out
    assert "claude plugin update superlocalmemory@qualixar" in out
    assert "hermes plugins install" in out and "--force --ref" in out
    assert host_upgrades.plugin_update_commands("cursor") == []


def test_hermes_command_names_the_running_release() -> None:
    (command,) = host_upgrades.plugin_update_commands("hermes", version="4.2.30")
    assert "refs/tags/v4.2.30^{}" in command


def test_doctor_reports_stale_current_and_absent(files) -> None:
    settings, hooks = files
    assert host_upgrades.host_hooks_finding()[0] == "PASS"
    assert "no SLM hooks" in host_upgrades.host_hooks_finding()[1]

    _write(settings, _old_claude_settings())
    status, detail, fix = host_upgrades.host_hooks_finding()
    assert status == "WARN" and "Claude Code settings.json" in detail
    assert fix == "slm upgrade-hosts --host claude-code --apply"
    assert "auto_recall" not in settings.read_text(), "doctor never writes"

    host_upgrades._upgrade_host("claude-code", apply=True, already_integrated=True)
    status, detail, _ = host_upgrades.host_hooks_finding()
    assert status == "PASS" and "current" in detail

    hooks.parent.mkdir(parents=True)
    hooks.write_text("{ broken", encoding="utf-8")
    assert host_upgrades.host_hooks_finding()[0] == "WARN"


def test_plugin_hosts_finding_lists_only_connected_plugin_hosts(monkeypatch) -> None:
    monkeypatch.setattr(host_upgrades, "_detected_hosts", lambda home=None: ["cursor", "muse"])
    status, detail, _ = host_upgrades.plugin_hosts_finding()
    assert status == "PASS" and "muse plugins install" in detail and "cursor" not in detail
    monkeypatch.setattr(host_upgrades, "_detected_hosts", lambda home=None: ["cursor"])
    assert host_upgrades.plugin_hosts_finding() is None


def test_slm_doctor_prints_the_host_hooks_and_plugin_host_rows(monkeypatch) -> None:
    import io
    from unittest import mock

    from superlocalmemory.cli.commands import cmd_doctor

    monkeypatch.setattr(host_upgrades, "host_hooks_finding",
                        lambda: ("WARN", "SLM hooks are older in: Claude Code settings.json",
                                 "slm upgrade-hosts --host claude-code --apply"))
    monkeypatch.setattr(host_upgrades, "plugin_hosts_finding",
                        lambda: ("PASS", "muse: muse plugins install x", ""))
    captured = io.StringIO()
    with mock.patch("sys.stdout", captured):
        try:
            cmd_doctor(Namespace(json=False, quick=True, fix=False))
        except SystemExit:
            pass
    out = captured.getvalue()
    assert "Host hooks: SLM hooks are older in: Claude Code settings.json" in out
    assert "Fix: slm upgrade-hosts --host claude-code --apply" in out
    assert "Plugin hosts: muse: muse plugins install x" in out


def test_codex_plugin_update_refreshes_the_marketplace_snapshot() -> None:
    """`codex plugin marketplace add` leaves an existing snapshot as it was ("already
    added"), so an upgrading user re-installed the old plugin (found on the author's Mac
    after 4.2.30: the cache stayed at 4.2.19). `upgrade` refreshes the snapshot; `add`
    first keeps the same commands working for someone who never added it."""
    cmds = host_upgrades.plugin_update_commands("codex", "9.9.9")
    assert cmds == [
        "codex plugin marketplace add qualixar/superlocalmemory --ref main",
        "codex plugin marketplace upgrade qualixar",
        "codex plugin add superlocalmemory-codex@qualixar",
    ]


@pytest.mark.parametrize("apply", [False, True])
def test_upgrade_hosts_reports_plugin_managed_claude_code(files, monkeypatch, apply) -> None:
    """Found on the owner's Mac, 20 Nov 2026 (4.2.31): with SLM's Claude Code plugin
    installed, ``slm upgrade-hosts --all-detected`` crashed with KeyError 'plugin-managed'
    before reaching any other host, so plugin users could not refresh Codex, Cursor or the
    rest from one command. The plugin owns SLM's hooks; the report must say so and go on."""
    from superlocalmemory.hooks import host_hook_upgrade

    settings, _ = files
    monkeypatch.setattr(host_hook_upgrade, "claude_plugin_present", lambda _path: True)
    _write(settings, _old_claude_settings())
    before = settings.read_bytes()

    result = host_upgrades._upgrade_host("claude-code", apply=apply, already_integrated=True)

    assert result["status"] == "verified", result
    assert "plugin" in result["detail"] and "claude plugin update" in result["detail"]
    assert settings.read_bytes() == before
