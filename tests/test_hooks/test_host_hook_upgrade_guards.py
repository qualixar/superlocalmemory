# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# ruff: noqa: F811 - pytest fixtures imported by name
"""F-13 (19 Nov 2026): the hook upgrade never overwrites a file that changed, and obeys the kill switch.

(a) It read the file, merged, and replaced it with no check that the file was still what it read:
    an edit Claude Code or the person made in that window was lost. Now the file is compared with what
    was read just before the replace; if it differs, nothing is written and the next daemon start
    tries again (the version is not marked done).
(b) A person who threw the kill switch ("turn off everything new in 4.2.30") still got two new hooks
    written into ``settings.json``. The automatic upgrade now does nothing while it is on, and does
    not mark the version done, so it runs when the switch is lifted.
"""

from __future__ import annotations

import json

from superlocalmemory.core.feature_flags import KILL_ENV, set_kill_switch
from superlocalmemory.hooks import host_hook_upgrade as upgrade
from tests.test_hooks.test_host_hook_upgrade import (  # noqa: F401
    VERSION,
    _load,
    _slm_actions,
    host_files,
)

# -- (b) the kill switch ------------------------------------------------------------------------------

def test_the_kill_switch_stops_the_automatic_upgrade_and_leaves_both_files_alone(host_files) -> None:
    before = {k: host_files[k].read_bytes() for k in ("claude_settings", "codex_hooks_path")}
    set_kill_switch(host_files["data_root"], True)
    assert upgrade.run_once(version=VERSION, env={}, **host_files) is None
    assert {k: host_files[k].read_bytes() for k in before} == before
    assert upgrade.marker_version(host_files["data_root"]) is None  # not done: it runs once lifted


def test_the_environment_kill_switch_counts_too(host_files) -> None:
    before = host_files["claude_settings"].read_bytes()
    assert upgrade.run_once(version=VERSION, env={KILL_ENV: "on"}, **host_files) is None
    assert host_files["claude_settings"].read_bytes() == before


def test_lifting_the_kill_switch_lets_the_upgrade_run(host_files) -> None:
    set_kill_switch(host_files["data_root"], True)
    assert upgrade.run_once(version=VERSION, env={}, **host_files) is None
    set_kill_switch(host_files["data_root"], False)
    outcome = upgrade.run_once(version=VERSION, env={}, **host_files)
    assert outcome and outcome["hosts"]["claude-code"]["status"] == "updated"


def test_an_explicit_upgrade_by_a_person_is_not_stopped_by_the_switch(host_files) -> None:
    set_kill_switch(host_files["data_root"], True)
    result = upgrade.upgrade_claude_settings(host_files["claude_settings"], version=VERSION)
    assert result["status"] == "updated"  # `slm upgrade-hosts --apply` is the person's own call


# -- (a) a file that changed while the upgrade worked on it -------------------------------------------

def _edit_during_the_backup(monkeypatch, path, extra_hook="my-new-hook"):
    """Claude Code (or the person) writes the file at the moment the upgrade is about to."""
    real = upgrade._backup_once

    def backup_then_someone_edits(target, version):
        out = real(target, version)
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("hooks", {}).setdefault("Notification", []).append(
            {"hooks": [{"type": "command", "command": extra_hook}]})
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return out

    monkeypatch.setattr(upgrade, "_backup_once", backup_then_someone_edits)


def test_a_file_edited_after_it_was_read_is_not_overwritten(host_files, monkeypatch) -> None:
    path = host_files["claude_settings"]
    _edit_during_the_backup(monkeypatch, path)
    result = upgrade.upgrade_claude_settings(path, version=VERSION)
    assert result["status"] == "changed-meanwhile"
    after = _load(path)
    assert "my-new-hook" in json.dumps(after)  # the edit that happened in the window is still there
    assert "cue" not in _slm_actions(after["hooks"].get("PreToolUse", []))  # and nothing of ours was written
    assert not list(path.parent.glob("*.slm_tmp"))  # no temp file left behind


def test_a_changed_file_is_tried_again_at_the_next_start(host_files, monkeypatch) -> None:
    path = host_files["claude_settings"]
    _edit_during_the_backup(monkeypatch, path)
    first = upgrade.run_once(version=VERSION, env={}, **host_files)
    assert first["hosts"]["claude-code"]["status"] == "changed-meanwhile"
    assert upgrade.marker_version(host_files["data_root"]) is None  # not marked done
    monkeypatch.undo()  # the next start: nobody is editing now
    second = upgrade.run_once(version=VERSION, env={}, **host_files)
    assert second["hosts"]["claude-code"]["status"] == "updated"
    final = _load(path)
    assert "my-new-hook" in json.dumps(final) and "cue" in _slm_actions(final["hooks"]["PreToolUse"])


def test_an_unchanged_file_is_still_written_as_before(host_files) -> None:
    result = upgrade.upgrade_claude_settings(host_files["claude_settings"], version=VERSION)
    assert result["status"] == "updated"
    assert not list(host_files["claude_settings"].parent.glob("*.slm_tmp"))


def test_a_kill_switch_that_cannot_be_read_skips_the_start_and_marks_nothing(host_files, monkeypatch) -> None:
    from superlocalmemory.core import feature_flags

    def unreadable(*_a, **_k):
        raise RuntimeError("flags unreadable")

    monkeypatch.setattr(feature_flags, "FeatureFlags", unreadable)
    before = host_files["claude_settings"].read_bytes()
    assert upgrade.run_once(version=VERSION, env={}, **host_files) is None
    assert host_files["claude_settings"].read_bytes() == before
    assert upgrade.marker_version(host_files["data_root"]) is None
