# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm upgrade-hosts --host claude-code`` on a Claude Code wired without the plugin.

The command printed "[verified] claude-code: Claude Code settings.json: already current" while every
skill and advisor was from an older release and four newer skills were missing. It now looks at the
skills, subagents and ``/slm-loop`` too, and says "already current" only when all of it is. Every
test uses a tmp ``~/.claude``.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from superlocalmemory.cli import host_upgrades
from superlocalmemory.hooks import claude_code_assets as assets
from superlocalmemory.hooks import claude_code_hooks as claude
from superlocalmemory.hooks import codex_hooks as codex

PLAN = assets.planned_files(assets.source_root())


@pytest.fixture
def claude_dir(tmp_path, monkeypatch) -> Path:
    folder = tmp_path / "home" / ".claude"
    folder.mkdir(parents=True)
    settings = claude._merge_hooks({"theme": "dark"}, claude._hook_definitions())
    (folder / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(claude, "CLAUDE_SETTINGS", folder / "settings.json")
    monkeypatch.setattr(codex, "DEFAULT_HOOKS_PATH", tmp_path / "home" / ".codex" / "hooks.json")
    monkeypatch.setattr(claude, "VERSION_DIR", tmp_path / "hooks-dir")
    monkeypatch.setattr(claude, "VERSION_FILE", tmp_path / "hooks-dir" / ".version")
    monkeypatch.setattr(claude, "DISABLED_FILE", tmp_path / "hooks-dir" / ".hooks-disabled")
    return folder


def _old_install(folder: Path, skip: str = "slm-media") -> None:
    for rel in PLAN:
        if skip in rel:
            continue
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_text("the 4.1.24 text\n", encoding="utf-8")


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_stale_assets_are_never_reported_current(claude_dir) -> None:
    _old_install(claude_dir)

    result = host_upgrades._upgrade_host("claude-code", apply=False, already_integrated=True)

    assert result["status"] == "preview"
    assert "already current" not in result["detail"].split("skills/agents/commands:")[1]
    assert "would add: skills/slm-media/SKILL.md" in result["detail"]
    assert "would refresh: skills/slm-recall/SKILL.md" in result["detail"]
    assert "would refresh: agents/slm-memory-advisor.md" in result["detail"]
    assert "would refresh: commands/slm-loop.md" in result["detail"]


def test_preview_changes_nothing(claude_dir) -> None:
    _old_install(claude_dir)
    before = _tree(claude_dir)

    host_upgrades._upgrade_host("claude-code", apply=False, already_integrated=True)

    assert _tree(claude_dir) == before


def test_apply_refreshes_adds_and_keeps_a_backup(claude_dir) -> None:
    _old_install(claude_dir)

    result = host_upgrades._upgrade_host("claude-code", apply=True, already_integrated=True)

    assert result["status"] == "updated"
    for rel, data in PLAN.items():
        assert (claude_dir / rel).read_bytes() == data, rel
    (backup,) = (claude_dir / "skills/slm-recall").glob("SKILL.md.bak-slm-*")
    assert backup.read_text() == "the 4.1.24 text\n"
    assert "added: skills/slm-media/SKILL.md" in result["detail"]


def test_current_only_when_hooks_and_assets_all_match(claude_dir) -> None:
    host_upgrades._upgrade_host("claude-code", apply=True, already_integrated=True)

    result = host_upgrades._upgrade_host("claude-code", apply=True, already_integrated=True)

    assert result["status"] == "verified"
    assert result["detail"].count("already current") == 2


def test_an_edited_advisor_and_an_outside_link_are_reported_not_overwritten(claude_dir, tmp_path) -> None:
    host_upgrades._upgrade_host("claude-code", apply=True, already_integrated=True)
    (claude_dir / "agents/slm-memory-advisor.md").write_text("my advisor\n")
    (claude_dir / "skills/slm-cache/SKILL.md").unlink()
    elsewhere = tmp_path / "checkout"
    elsewhere.mkdir()
    (elsewhere / "SKILL.md").write_text("checkout copy\n")
    (claude_dir / "skills/slm-cache").rmdir()
    (claude_dir / "skills/slm-cache").symlink_to(elsewhere, target_is_directory=True)

    result = host_upgrades._upgrade_host("claude-code", apply=True, already_integrated=True)

    assert (claude_dir / "agents/slm-memory-advisor.md").read_text() == "my advisor\n"
    assert (elsewhere / "SKILL.md").read_text() == "checkout copy\n"
    assert "resolve outside ~/.claude and were NOT refreshed" in result["detail"]
    assert "kept 1 file(s) you changed" in result["detail"]
    assert result["status"] == "verified" and "already current" not in result["detail"].split(
        "skills/agents/commands:")[1]


def test_with_the_plugin_installed_nothing_is_written_and_nothing_crashes(claude_dir, capsys) -> None:
    settings = json.loads((claude_dir / "settings.json").read_text())
    settings["enabledPlugins"] = {"superlocalmemory@qualixar": True}
    (claude_dir / "settings.json").write_text(json.dumps(settings))
    _old_install(claude_dir)
    before = _tree(claude_dir)

    for apply in (False, True):
        result = host_upgrades._upgrade_host("claude-code", apply=apply, already_integrated=True)
        assert result["status"] == "verified" and "plugin" in result["detail"]
    assert _tree(claude_dir) == before

    host_upgrades.cmd_upgrade_hosts(Namespace(hosts=["claude-code"], apply=False, all_detected=False))
    assert "claude plugin update superlocalmemory@qualixar" in capsys.readouterr().out


def test_a_claude_without_slm_gets_nothing_added(claude_dir) -> None:
    (claude_dir / "settings.json").write_text('{"theme": "dark"}\n')
    before = _tree(claude_dir)

    result = host_upgrades._upgrade_host("claude-code", apply=True, already_integrated=False)

    assert result["status"] == "verified" and "none added" in result["detail"]
    assert _tree(claude_dir) == before


def test_the_printed_line_names_each_file(claude_dir, capsys, monkeypatch) -> None:
    _old_install(claude_dir)
    monkeypatch.setattr(host_upgrades, "_report_reconnect_apps", lambda _t: None)
    monkeypatch.setattr("superlocalmemory.cli.measurement_cmd.announce_if_pending", lambda: None)

    host_upgrades.cmd_upgrade_hosts(Namespace(hosts=["claude-code"], apply=False, all_detected=False))

    out = capsys.readouterr().out
    assert "[preview] claude-code:" in out and "would add: skills/slm-media/SKILL.md" in out
    assert "would refresh: skills/slm-recall/SKILL.md" in out


def test_doctor_names_stale_assets_without_writing(claude_dir) -> None:
    _old_install(claude_dir)
    before = _tree(claude_dir)

    status, detail, fix = host_upgrades.claude_assets_finding()

    assert status == "WARN" and "21 SLM skill" in detail and "slm upgrade-hosts --host claude-code --apply" in fix
    assert _tree(claude_dir) == before

    host_upgrades._upgrade_host("claude-code", apply=True, already_integrated=True)
    assert host_upgrades.claude_assets_finding()[0] == "PASS"


def test_doctor_stays_quiet_for_a_plugin_install_or_no_slm(claude_dir) -> None:
    (claude_dir / "settings.json").write_text('{"theme": "dark"}\n')
    assert host_upgrades.claude_assets_finding() is None


def test_a_skills_only_claude_code_is_detected_for_all_detected(claude_dir, tmp_path, monkeypatch) -> None:
    (claude_dir / "settings.json").write_text('{"theme": "dark"}\n', encoding="utf-8")
    for rel in PLAN:
        if "slm-recall" in rel:
            (claude_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (claude_dir / rel).write_text("the 4.1.24 text\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert "claude-code" in host_upgrades._detected_hosts(tmp_path / "home")


def test_a_claude_code_with_nothing_of_slm_is_still_not_detected(claude_dir, tmp_path, monkeypatch) -> None:
    (claude_dir / "settings.json").write_text('{"theme": "dark"}\n', encoding="utf-8")
    (claude_dir / "skills" / "my-own-skill").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert "claude-code" not in host_upgrades._detected_hosts(tmp_path / "home")
