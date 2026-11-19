# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""4.2.30: SLM's hooks reach every upgrading user by replacement and addition only.

The owner's rule: hooks already live in people's harnesses, so an upgrade may replace SLM-owned
entries and add missing SLM entries, and must leave every other entry, key and the key order alone.
It happens by itself (the daemon, once per version) and on demand (``slm upgrade-hosts --apply``).

Every test works in a tmp folder; none touches a real host configuration.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import pytest

from superlocalmemory.hooks import claude_code_hooks as claude
from superlocalmemory.hooks import codex_hooks as codex
from superlocalmemory.hooks import host_hook_upgrade as upgrade

ROOT = Path(__file__).resolve().parents[2]
VERSION = "4.2.30"
ACTION = re.compile(r"hook (\w+)")


def _user_entry(command: str, matcher: str | None = None) -> dict:
    entry: dict = {"hooks": [{"type": "command", "command": command}]}
    if matcher:
        entry["matcher"] = matcher
    return entry


def _old_claude_settings() -> dict:
    """A 4.2.19 settings.json: the SLM entries as 4.2.19 wrote them, plus the user's own."""
    old = claude._hook_definitions()
    old["PreToolUse"] = [old["PreToolUse"][0]]  # before_web only, no cue
    old["PostToolUse"] = old["PostToolUse"][:2]  # checkpoint + post_tool_outcome, no cue
    old["UserPromptSubmit"] = old["UserPromptSubmit"][:2]  # no auto_recall
    return {
        "theme": "dark",
        "env": {"FOO": "bar"},
        "hooks": {
            "PreToolUse": [_user_entry("my-lint", "Bash"), *old["PreToolUse"]],
            "PostToolUse": [*old["PostToolUse"], _user_entry("my-after")],
            "UserPromptSubmit": [_user_entry("my-prompt"), *old["UserPromptSubmit"]],
            "SessionStart": old["SessionStart"],
            "Stop": old["Stop"],
            "Notification": [_user_entry("my-notify")],
        },
        "permissions": {"allow": ["Bash(ls)"]},
    }


def _write(path: Path, data: dict, indent: int | str = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent) + "\n", encoding="utf-8")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _slm_actions(entries: list) -> list[str]:
    found = []
    for entry in entries:
        for hook in entry.get("hooks", []):
            found.extend(ACTION.findall(hook.get("command", "")))
    return found


# --------------------------------------------------------------------------------------------
# 1. The settings.json definitions equal the plugin's hooks.json, for every SLM hook
# --------------------------------------------------------------------------------------------

def _flatten(hooks: dict) -> set[tuple]:
    rows = set()
    for event, entries in hooks.items():
        for entry in entries:
            for hook in entry.get("hooks", []):
                match = ACTION.search(hook.get("command", ""))
                if match and match.group(1) not in {"session"}:
                    rows.add((event, entry.get("matcher"), match.group(1), hook.get("timeout")))
    return rows


def test_settings_definitions_equal_the_plugin_hooks_json() -> None:
    plugin = json.loads((ROOT / "plugin-src" / "hooks" / "hooks.json").read_text())["hooks"]
    plugin_rows = _flatten(plugin)
    settings_rows = _flatten(claude._hook_definitions())
    assert plugin_rows, "the plugin hooks.json must declare SLM hooks"
    assert plugin_rows == settings_rows, (
        f"only in plugin: {sorted(plugin_rows - settings_rows, key=str)}; "
        f"only in settings: {sorted(settings_rows - plugin_rows, key=str)}")


def test_the_4230_hooks_are_in_the_settings_definitions() -> None:
    rows = _flatten(claude._hook_definitions())
    assert ("PreToolUse", "Write|Edit|MultiEdit|Bash", "cue", 5) in rows
    assert ("PostToolUse", "Write|Edit|MultiEdit|NotebookEdit|Bash", "cue", 5) in rows
    assert ("UserPromptSubmit", None, "auto_recall", 7) in rows


# --------------------------------------------------------------------------------------------
# 2. Claude Code settings.json: replace and add SLM entries, nothing else
# --------------------------------------------------------------------------------------------

def test_claude_old_install_gains_the_new_hooks_and_keeps_everything_else(tmp_path) -> None:
    path = tmp_path / ".claude" / "settings.json"
    original = _old_claude_settings()
    _write(path, original)

    result = upgrade.upgrade_claude_settings(path, version=VERSION)

    assert result["status"] == "updated"
    data = _load(path)
    assert data["theme"] == "dark" and data["env"] == {"FOO": "bar"}
    assert data["permissions"] == {"allow": ["Bash(ls)"]}
    assert list(data) == list(original), "top-level key order must not change"
    assert list(data["hooks"]) == list(original["hooks"]), "event order must not change"
    for event, entries in original["hooks"].items():
        kept = [e for e in entries if not claude._is_slm_hook_entry(e)]
        got = [e for e in data["hooks"][event] if not claude._is_slm_hook_entry(e)]
        assert got == kept, f"user entries of {event} changed"
    assert _flatten(data["hooks"]) == _flatten(claude._hook_definitions())


def test_claude_stale_slm_entry_is_replaced_in_place_not_moved(tmp_path) -> None:
    path = tmp_path / "settings.json"
    stale = {"matcher": "WebSearch|WebFetch",
             "hooks": [{"type": "command", "command": "slm hook before_web 2>/dev/null || true",
                        "timeout": 5000}]}  # a release before 4.1.22 wrote milliseconds
    _write(path, {"hooks": {"PreToolUse": [_user_entry("first"), stale, _user_entry("last")]}})

    upgrade.upgrade_claude_settings(path, version=VERSION)

    entries = _load(path)["hooks"]["PreToolUse"]
    commands = [e["hooks"][0]["command"] for e in entries]
    assert commands[0] == "first"
    assert commands[1].startswith("slm hook before_web") and entries[1]["hooks"][0]["timeout"] == 5
    assert "slm hook cue" in commands[2]
    assert commands[3] == "last", "the user's later entry keeps its place after SLM's"
    assert _slm_actions(entries).count("before_web") == 1


def test_claude_mixed_group_keeps_the_users_own_hook(tmp_path) -> None:
    path = tmp_path / "settings.json"
    mixed = {"matcher": "Bash", "hooks": [
        {"type": "command", "command": "slm hook cue 2>/dev/null || true", "timeout": 1},
        {"type": "command", "command": "my-own-bash-hook"}]}
    _write(path, {"hooks": {"PreToolUse": [mixed]}})

    upgrade.upgrade_claude_settings(path, version=VERSION)

    commands = [h["command"] for e in _load(path)["hooks"]["PreToolUse"] for h in e["hooks"]]
    assert "my-own-bash-hook" in commands
    assert commands.count("slm hook cue 2>/dev/null || true") == 1


def test_claude_second_run_changes_nothing_not_even_the_mtime(tmp_path) -> None:
    path = tmp_path / "settings.json"
    _write(path, _old_claude_settings())
    upgrade.upgrade_claude_settings(path, version=VERSION)
    before_bytes, before_mtime = path.read_bytes(), path.stat().st_mtime_ns

    result = upgrade.upgrade_claude_settings(path, version=VERSION)

    assert result["status"] == "current"
    assert path.read_bytes() == before_bytes and path.stat().st_mtime_ns == before_mtime


def test_claude_file_without_slm_entries_is_never_touched(tmp_path) -> None:
    path = tmp_path / "settings.json"
    _write(path, {"hooks": {"PreToolUse": [_user_entry("mine")]}, "theme": "light"})
    before, mtime = path.read_bytes(), path.stat().st_mtime_ns

    result = upgrade.upgrade_claude_settings(path, version=VERSION)

    assert result["status"] == "not-integrated"
    assert path.read_bytes() == before and path.stat().st_mtime_ns == mtime
    assert not list(tmp_path.glob("*.bak-slm-*"))


def test_claude_missing_file_is_not_created(tmp_path) -> None:
    path = tmp_path / ".claude" / "settings.json"
    assert upgrade.upgrade_claude_settings(path, version=VERSION)["status"] == "not-integrated"
    assert not path.exists() and not path.parent.exists()


def test_claude_backup_is_written_once_before_the_first_change(tmp_path) -> None:
    path = tmp_path / "settings.json"
    _write(path, _old_claude_settings())
    original = path.read_bytes()

    upgrade.upgrade_claude_settings(path, version=VERSION)
    backup = tmp_path / f"settings.json.bak-slm-{VERSION}"
    assert backup.read_bytes() == original

    # Make it stale again: the one-time backup keeps the pre-upgrade file.
    data = _load(path)
    data["hooks"]["PreToolUse"] = [e for e in data["hooks"]["PreToolUse"]
                                   if "cue" not in json.dumps(e)]
    _write(path, data)
    upgrade.upgrade_claude_settings(path, version=VERSION)
    assert backup.read_bytes() == original


def test_claude_keeps_the_users_indent_and_non_ascii_text(tmp_path) -> None:
    path = tmp_path / "settings.json"
    data = _old_claude_settings()
    data["note"] = "cafe — éè"
    _write(path, data, indent=4)

    upgrade.upgrade_claude_settings(path, version=VERSION)

    text = path.read_text(encoding="utf-8")
    assert '\n    "theme"' in text and '\n  "theme"' not in text, "top-level keys stay at 4 spaces"
    assert "—" in text and text.endswith("}\n")


def test_claude_unreadable_file_is_reported_and_left_alone(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{ "hooks": { "slm hook": ', encoding="utf-8")
    before = path.read_bytes()

    result = upgrade.upgrade_claude_settings(path, version=VERSION)

    assert result["status"] == "error" and path.read_bytes() == before


def test_claude_preview_does_not_write(tmp_path) -> None:
    path = tmp_path / "settings.json"
    _write(path, _old_claude_settings())
    before = path.read_bytes()

    result = upgrade.upgrade_claude_settings(path, version=VERSION, apply=False)

    assert result["status"] == "would-update" and path.read_bytes() == before
    assert not list(tmp_path.glob("*.bak-slm-*"))


def test_claude_write_goes_through_a_temp_file_and_rename(tmp_path, monkeypatch) -> None:
    path = tmp_path / "settings.json"
    _write(path, _old_claude_settings())
    seen: list[tuple[str, str]] = []
    real = os.replace

    def spy(src, dst):
        seen.append((str(src), str(dst)))
        return real(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    upgrade.upgrade_claude_settings(path, version=VERSION)
    assert seen and seen[-1][1] == str(path.resolve()) and seen[-1][0] != seen[-1][1]
    assert not list(tmp_path.glob("*.slm_tmp"))


def test_claude_gate_that_the_user_enabled_stays_enabled(tmp_path) -> None:
    path = tmp_path / "settings.json"
    _write(path, claude._merge_hooks({}, claude._hook_definitions(include_gate=True)))
    upgrade.upgrade_claude_settings(path, version=VERSION)
    assert "Call mcp__superlocalmemory__session_init first" in path.read_text()


def test_claude_symlinked_settings_keep_the_link(tmp_path) -> None:
    real = tmp_path / "dotfiles" / "settings.json"
    _write(real, _old_claude_settings())
    link = tmp_path / ".claude" / "settings.json"
    link.parent.mkdir()
    link.symlink_to(real)

    upgrade.upgrade_claude_settings(link, version=VERSION)

    assert link.is_symlink()
    assert "auto_recall" in real.read_text()


# --------------------------------------------------------------------------------------------
# 3. Codex hooks.json: replace by marker, add missing, keep the rest
# --------------------------------------------------------------------------------------------

def _old_codex_hooks() -> dict:
    defs = codex.hook_definitions()
    old_prompt = json.loads(json.dumps(defs["UserPromptSubmit"]))
    old_prompt[0]["hooks"][0]["timeout"] = 5  # 4.2.19
    return {
        "custom": {"keep": True},
        "hooks": {
            "SessionStart": [_user_entry("my-start"), *defs["SessionStart"]],
            "UserPromptSubmit": [*old_prompt, _user_entry("my-prompt")],
            "PostToolUse": defs["PostToolUse"],
            "Stop": [*defs["Stop"], _user_entry("my-stop")],
        },
    }


def test_codex_timeout_change_lands_in_place_and_user_hooks_stay(tmp_path) -> None:
    path = tmp_path / ".codex" / "hooks.json"
    original = _old_codex_hooks()
    _write(path, original)

    result = upgrade.upgrade_codex_hooks(path, version=VERSION)

    assert result["status"] == "updated"
    data = _load(path)
    prompt = data["hooks"]["UserPromptSubmit"]
    assert prompt[0]["hooks"][0]["timeout"] == 7 and "codex-prompt" in prompt[0]["hooks"][0]["command"]
    assert prompt[1] == _user_entry("my-prompt"), "replaced in place, user entry stays after it"
    assert data["custom"] == {"keep": True} and list(data) == list(original)
    assert data["hooks"]["SessionStart"][0] == _user_entry("my-start")
    assert [e for e in data["hooks"]["Stop"]][-1] == _user_entry("my-stop")


def test_codex_install_hooks_itself_replaces_the_stale_entry(tmp_path) -> None:
    """``slm codex install`` and ``upgrade-hosts --host codex`` share this path."""
    path = tmp_path / "hooks.json"
    _write(path, _old_codex_hooks())

    codex.install_hooks(hooks_path=path)

    prompt = _load(path)["hooks"]["UserPromptSubmit"]
    assert [e["hooks"][0]["timeout"] for e in prompt if codex.is_slm_hook_entry(e)] == [7]
    assert sum(codex.is_slm_hook_entry(e) for e in prompt) == 1


def test_codex_adds_a_missing_slm_entry(tmp_path) -> None:
    path = tmp_path / "hooks.json"
    data = _old_codex_hooks()
    del data["hooks"]["Stop"][0]  # SLM's Stop entry gone, the user's stays
    _write(path, data)

    upgrade.upgrade_codex_hooks(path, version=VERSION)

    stop = _load(path)["hooks"]["Stop"]
    assert any(codex.is_slm_hook_entry(e) for e in stop) and _user_entry("my-stop") in stop


def test_codex_second_run_is_a_no_op_and_install_hooks_does_not_rewrite(tmp_path) -> None:
    path = tmp_path / "hooks.json"
    _write(path, _old_codex_hooks())
    upgrade.upgrade_codex_hooks(path, version=VERSION)
    before, mtime = path.read_bytes(), path.stat().st_mtime_ns

    assert upgrade.upgrade_codex_hooks(path, version=VERSION)["status"] == "current"
    codex.install_hooks(hooks_path=path)
    assert path.read_bytes() == before and path.stat().st_mtime_ns == mtime


def test_codex_file_without_slm_entries_is_untouched(tmp_path) -> None:
    path = tmp_path / "hooks.json"
    _write(path, {"hooks": {"Stop": [_user_entry("my-stop")]}})
    before, mtime = path.read_bytes(), path.stat().st_mtime_ns

    result = upgrade.upgrade_codex_hooks(path, version=VERSION)

    assert result["status"] == "not-integrated"
    assert path.read_bytes() == before and path.stat().st_mtime_ns == mtime
    assert not list(tmp_path.glob("*.bak-slm-*"))


def test_codex_backup_written_once_with_the_version_name(tmp_path) -> None:
    path = tmp_path / "hooks.json"
    _write(path, _old_codex_hooks())
    original = path.read_bytes()

    upgrade.upgrade_codex_hooks(path, version=VERSION)

    assert (tmp_path / f"hooks.json.bak-slm-{VERSION}").read_bytes() == original
    assert not (tmp_path / "hooks.json.slm.bak").exists(), "one backup, not two"


def test_codex_malformed_file_is_reported_and_left_alone(tmp_path) -> None:
    path = tmp_path / "hooks.json"
    path.write_text("{ not json slm_codex_hook", encoding="utf-8")
    assert upgrade.upgrade_codex_hooks(path, version=VERSION)["status"] == "error"
    assert path.read_text(encoding="utf-8") == "{ not json slm_codex_hook"


# --------------------------------------------------------------------------------------------
# 4. Once per version, by the daemon; opt-out; guards
# --------------------------------------------------------------------------------------------

@pytest.fixture
def host_files(tmp_path):
    claude_path = tmp_path / "home" / ".claude" / "settings.json"
    codex_path = tmp_path / "home" / ".codex" / "hooks.json"
    _write(claude_path, _old_claude_settings())
    _write(codex_path, _old_codex_hooks())
    root = tmp_path / "slm-data"
    root.mkdir()
    return {"claude_settings": claude_path, "codex_hooks_path": codex_path, "data_root": root}


def test_isolation_plugin_turns_the_automatic_upgrade_off_for_every_test() -> None:
    assert os.environ.get("SLM_NO_HOOK_UPGRADE") == "1"


def test_run_once_upgrades_every_integrated_host_and_writes_the_marker(host_files) -> None:
    outcome = upgrade.run_once(version=VERSION, env={}, **host_files)

    assert outcome["hosts"]["claude-code"]["status"] == "updated"
    assert outcome["hosts"]["codex"]["status"] == "updated"
    marker = json.loads((host_files["data_root"] / "hooks-upgraded.json").read_text())
    assert marker["version"] == VERSION
    assert "auto_recall" in host_files["claude_settings"].read_text()


def test_run_once_runs_once_per_version(host_files) -> None:
    upgrade.run_once(version=VERSION, env={}, **host_files)
    claude_bytes = host_files["claude_settings"].read_bytes()
    # A user edit after the upgrade is never overwritten again by the same version.
    data = _load(host_files["claude_settings"])
    data["hooks"]["PreToolUse"] = [e for e in data["hooks"]["PreToolUse"]
                                   if "cue" not in json.dumps(e)]
    _write(host_files["claude_settings"], data)
    edited = host_files["claude_settings"].read_bytes()
    assert edited != claude_bytes

    assert upgrade.run_once(version=VERSION, env={}, **host_files) is None
    assert host_files["claude_settings"].read_bytes() == edited

    again = upgrade.run_once(version="4.2.31", env={}, **host_files)
    assert again is not None and again["hosts"]["claude-code"]["status"] == "updated"


def test_run_once_respects_the_opt_out(host_files) -> None:
    before = host_files["claude_settings"].read_bytes()

    assert upgrade.run_once(version=VERSION, env={"SLM_NO_HOOK_UPGRADE": "1"}, **host_files) is None

    assert host_files["claude_settings"].read_bytes() == before
    assert not (host_files["data_root"] / "hooks-upgraded.json").exists()


def test_run_once_does_not_mark_the_version_done_after_an_error(host_files) -> None:
    host_files["codex_hooks_path"].write_text("{ broken", encoding="utf-8")

    outcome = upgrade.run_once(version=VERSION, env={}, **host_files)

    assert outcome["hosts"]["codex"]["status"] == "error"
    assert outcome["hosts"]["claude-code"]["status"] == "updated"
    assert not (host_files["data_root"] / "hooks-upgraded.json").exists(), "retry next start"


def test_run_once_never_creates_a_host_file(tmp_path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    outcome = upgrade.run_once(version=VERSION, env={}, data_root=root,
                               claude_settings=tmp_path / "c" / "settings.json",
                               codex_hooks_path=tmp_path / "x" / "hooks.json")
    assert {h["status"] for h in outcome["hosts"].values()} == {"not-integrated"}
    assert not (tmp_path / "c").exists() and not (tmp_path / "x").exists()
    assert (root / "hooks-upgraded.json").exists()


def test_run_once_respects_the_users_hooks_removed_marker(host_files, tmp_path, monkeypatch) -> None:
    disabled = tmp_path / "hooks-dir" / ".hooks-disabled"
    disabled.parent.mkdir()
    disabled.write_text("removed by user\n")
    monkeypatch.setattr(claude, "DISABLED_FILE", disabled)
    before = host_files["claude_settings"].read_bytes()

    outcome = upgrade.run_once(version=VERSION, env={}, **host_files)

    assert outcome["hosts"]["claude-code"]["status"] == "skipped"
    assert host_files["claude_settings"].read_bytes() == before


def test_the_daemon_loop_runs_it_once_after_start_up_settles(host_files) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    calls: list[str] = []
    real = upgrade.run_once

    def counting(**kwargs):
        calls.append("run")
        return real(**kwargs)

    outcome = asyncio.run(upgrade.run(first_delay_s=42.0, sleep=fake_sleep, run_once_fn=counting,
                                      version=VERSION, env={}, **host_files))

    assert sleeps == [42.0] and calls == ["run"]
    assert outcome["hosts"]["claude-code"]["status"] == "updated"


def test_the_daemon_loop_does_nothing_when_disabled(host_files) -> None:
    async def fake_sleep(seconds: float) -> None:
        raise AssertionError("must not even wait")

    assert asyncio.run(upgrade.run(sleep=fake_sleep, version=VERSION,
                                   env={"SLM_NO_HOOK_UPGRADE": "1"}, **host_files)) is None


def test_the_daemon_starts_and_stops_the_task() -> None:
    source = (ROOT / "src" / "superlocalmemory" / "server" / "unified_daemon.py").read_text()
    assert "host_hook_upgrade" in source and "_host_hook_upgrade_task" in source
    assert "host_hook_upgrade.run()" in source, "the daemon must start the task"
    assert "_hook_up.cancel()" in source, "and cancel it at shutdown"
    # It is only ever started from the daemon, never from a hook, a recall or the CLI.
    for module in ("hook_handlers", "auto_recall_hook", "cue_hook"):
        text = (ROOT / "src" / "superlocalmemory" / "hooks" / f"{module}.py").read_text()
        assert "host_hook_upgrade" not in text, module


# -- the Claude Code plugin owns SLM's hooks (found on the owner's Mac, 19 Nov 2026) ----------

def _owner_like_settings(tmp_path, *, enabled: bool):
    import json as _json

    settings = {
        "enabledPlugins": {"superlocalmemory@qualixar": True} if enabled else {},
        "hooks": {
            "Stop": [{"hooks": [{"type": "command", "command": "slm session close 2>/dev/null || true"}]}],
            "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "my-own-guard.sh"}]}],
            "PostToolUse": [],
        },
    }
    path = tmp_path / "settings.json"
    path.write_text(_json.dumps(settings, indent=2), encoding="utf-8")
    return path


def test_with_the_slm_plugin_enabled_settings_json_is_left_exactly_as_it_was(tmp_path):
    from superlocalmemory.hooks import host_hook_upgrade as up

    path = _owner_like_settings(tmp_path, enabled=True)
    before = path.read_bytes()
    res = up.upgrade_claude_settings(path, version="4.2.30")
    assert res["status"] == "plugin-managed"
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("settings.json.bak-slm-*"))


def test_an_installed_plugin_listed_beside_the_settings_also_counts(tmp_path):
    import json as _json

    from superlocalmemory.hooks import host_hook_upgrade as up

    path = _owner_like_settings(tmp_path, enabled=False)
    (tmp_path / "plugins").mkdir()
    (tmp_path / "plugins" / "installed_plugins.json").write_text(
        _json.dumps({"version": 2, "plugins": {"superlocalmemory@qualixar": [{}]}}), encoding="utf-8")
    before = path.read_bytes()
    assert up.upgrade_claude_settings(path, version="4.2.30")["status"] == "plugin-managed"
    assert path.read_bytes() == before


def test_without_the_plugin_a_settings_install_is_still_upgraded(tmp_path):
    from superlocalmemory.hooks import host_hook_upgrade as up

    path = _owner_like_settings(tmp_path, enabled=False)
    res = up.upgrade_claude_settings(path, version="4.2.30")
    assert res["status"] == "updated"
    text = path.read_text(encoding="utf-8")
    assert "slm hook cue" in text and "my-own-guard.sh" in text
