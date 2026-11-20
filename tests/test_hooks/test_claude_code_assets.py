# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A Claude Code wired by hand (no plugin) gets new and updated SLM skills, agents and command.

Before this, ``slm upgrade-hosts`` refreshed only the hook entries in ``settings.json`` and said
"already current" while every skill and advisor was still from the release that installed them and
the skills added since were missing. Every test runs in a tmp ``~/.claude``; none touches a real one.
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

import pytest

from superlocalmemory.hooks import claude_code_assets as assets
from superlocalmemory.hooks import claude_code_hooks as claude
from superlocalmemory.hooks import host_hook_upgrade as hook_upgrade

VERSION = "4.2.32"
PLAN = assets.planned_files(assets.source_root())
SKILL_RECALL = "skills/slm-recall/SKILL.md"
ADVISOR = "agents/slm-memory-advisor.md"
COMMAND = "commands/slm-loop.md"


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def home(tmp_path) -> Path:
    return tmp_path / "home"


@pytest.fixture
def claude_dir(home) -> Path:
    """A Claude Code with SLM's hooks in settings.json and nothing else of SLM's yet."""
    folder = home / ".claude"
    folder.mkdir(parents=True)
    settings = claude._merge_hooks({"theme": "dark"}, claude._hook_definitions())
    (folder / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return folder


def _upgrade(claude_dir: Path, **kwargs):
    kwargs.setdefault("version", VERSION)
    return assets.upgrade_claude_assets(claude_dir / "settings.json", **kwargs)


def _seed_old_install(claude_dir: Path, keep_out: tuple[str, ...] = ()) -> dict[str, bytes]:
    """An install from an older release: every shipped file present but stale, some missing."""
    old = {}
    for rel in PLAN:
        if any(name in rel for name in keep_out):
            continue
        old[rel] = b"---\nname: old\n---\nthe 4.1.24 text of " + rel.encode() + b"\n"
        target = claude_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(old[rel])
    return old


def test_the_release_ships_the_skills_the_plugin_ships() -> None:
    skills = {rel.split("/")[1] for rel in PLAN if rel.startswith("skills/")}
    assert {"slm-media", "slm-web-access", "slm-bot-memory", "slm-getting-started-bot",
            "slm-recall"} <= skills
    assert ADVISOR in PLAN and COMMAND in PLAN


def test_the_wheel_is_told_to_ship_every_file_this_upgrade_installs() -> None:
    """Every ``plugin-src`` skill, agent and command has a data-file entry the installed
    package reads from; a file left out would be silently missing from a pip install."""
    root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"][
        "data-files"]
    shipped = {}
    for dest, sources in data.items():
        if dest.startswith("share/superlocalmemory/claude-code/"):
            kind = dest.split("/")[3]
            for source in sources:
                name = source.rsplit("/", 1)[-1] if kind != "skills" else dest.rsplit("/", 1)[-1]
                shipped[f"{kind}/{name}" if kind != "skills" else f"skills/{name}/SKILL.md"] = source
    assert set(shipped) == set(PLAN)
    for rel, source in shipped.items():
        if rel.startswith("agents/"):
            assert source == f"claude-code-assets/{rel}", (rel, source)  # never plugin-src's bare names
    for rel, source in shipped.items():
        assert (root / source).read_bytes() == PLAN[rel]


def test_the_installed_layout_is_read_when_there_is_no_checkout(tmp_path, monkeypatch) -> None:
    import sysconfig

    share = tmp_path / "prefix" / "share" / "superlocalmemory" / "claude-code"
    for rel, data in PLAN.items():
        (share / rel).parent.mkdir(parents=True, exist_ok=True)
        (share / rel).write_bytes(data)
    monkeypatch.setattr(assets, "__file__", str(tmp_path / "site" / "a" / "b" / "claude_code_assets.py"))
    monkeypatch.setattr(sysconfig, "get_path", lambda name, *a, **k: str(tmp_path / "prefix"))

    assert assets.source_root() == share
    assert assets.planned_files(share) == PLAN


def test_stale_skills_are_refreshed_and_new_skills_added(claude_dir) -> None:
    old = _seed_old_install(claude_dir, keep_out=("slm-media", "slm-web-access"))

    result = _upgrade(claude_dir)

    assert result["status"] == "updated"
    assert (claude_dir / SKILL_RECALL).read_bytes() == PLAN[SKILL_RECALL]
    assert (claude_dir / ADVISOR).read_bytes() == PLAN[ADVISOR]
    assert (claude_dir / COMMAND).read_bytes() == PLAN[COMMAND]
    assert (claude_dir / "skills/slm-media/SKILL.md").read_bytes() == PLAN["skills/slm-media/SKILL.md"]
    assert "skills/slm-web-access/SKILL.md" in result["added"]
    assert SKILL_RECALL in result["adopted"]
    backup = claude_dir / "skills/slm-recall/SKILL.md.bak-slm-4.2.32"
    assert backup.read_bytes() == old[SKILL_RECALL], "the older copy is kept, never lost"


def test_preview_lists_exactly_what_would_change_and_writes_nothing(claude_dir) -> None:
    _seed_old_install(claude_dir, keep_out=("slm-media",))
    before = _tree(claude_dir)

    result = _upgrade(claude_dir, apply=False)

    assert result["status"] == "would-update"
    assert result["added"] == ["skills/slm-media/SKILL.md"]
    assert set(result["adopted"]) == set(PLAN) - {"skills/slm-media/SKILL.md"}
    assert _tree(claude_dir) == before
    assert not (claude_dir / assets.MANIFEST_NAME).exists()


def test_only_current_when_every_file_matches_this_release(claude_dir) -> None:
    _seed_old_install(claude_dir)
    assert _upgrade(claude_dir)["status"] == "updated"

    again = _upgrade(claude_dir)
    assert again["status"] == "current" and not again["added"] and not again["adopted"]

    (claude_dir / SKILL_RECALL).write_bytes(PLAN[SKILL_RECALL] + b"\nedited\n")
    stale_source = _source_with(claude_dir.parent / "src", {ADVISOR: b"new advisor\n"})
    third = _upgrade(claude_dir, source=stale_source)
    assert third["status"] == "updated" and third["refreshed"] == [ADVISOR]


def _source_with(root: Path, overrides: dict[str, bytes]) -> Path:
    """This release's source, copied, with some files replaced: stands in for the next release."""
    for rel, data in {**PLAN, **overrides}.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return root


def test_a_file_the_person_edited_after_slm_wrote_it_is_kept(claude_dir) -> None:
    assert _upgrade(claude_dir)["status"] == "updated"  # adds everything, records digests
    mine = (claude_dir / ADVISOR)
    mine.write_text("my own advisor text\n", encoding="utf-8")
    next_release = _source_with(claude_dir.parent / "next", {ADVISOR: b"new advisor text\n",
                                                             SKILL_RECALL: b"new recall\n"})

    result = _upgrade(claude_dir, source=next_release)

    assert mine.read_text(encoding="utf-8") == "my own advisor text\n"
    assert result["preserved"] == [ADVISOR]
    assert (claude_dir / SKILL_RECALL).read_bytes() == b"new recall\n", "SLM's own copy still moves"
    assert not list((claude_dir / "agents").glob("*.bak-slm-*")), "nothing was replaced, so no backup"


def test_files_slm_does_not_ship_are_never_touched(claude_dir) -> None:
    (claude_dir / "skills/my-skill").mkdir(parents=True)
    (claude_dir / "skills/my-skill/SKILL.md").write_text("mine\n")
    (claude_dir / "agents").mkdir()
    (claude_dir / "agents/slm-custom.md").write_text("an agent named like SLM's but not shipped\n")
    (claude_dir / "agents/reviewer.md").write_text("mine too\n")
    (claude_dir / "skills/slm-recall").mkdir()
    (claude_dir / "skills/slm-recall/notes.txt").write_text("my notes beside the skill\n")

    assert _upgrade(claude_dir)["status"] == "updated"

    assert (claude_dir / "skills/my-skill/SKILL.md").read_text() == "mine\n"
    assert (claude_dir / "agents/slm-custom.md").read_text().startswith("an agent named")
    assert (claude_dir / "agents/reviewer.md").read_text() == "mine too\n"
    assert (claude_dir / "skills/slm-recall/notes.txt").read_text() == "my notes beside the skill\n"


def test_a_symlink_pointing_outside_is_reported_and_left_alone(claude_dir, tmp_path) -> None:
    checkout = tmp_path / "my-checkout" / "slm-recall"
    checkout.mkdir(parents=True)
    (checkout / "SKILL.md").write_text("my checkout copy\n")
    (claude_dir / "skills").mkdir()
    link = claude_dir / "skills" / "slm-recall"
    link.symlink_to(checkout, target_is_directory=True)
    file_target = tmp_path / "elsewhere.md"
    file_target.write_text("a linked advisor\n")
    (claude_dir / "agents").mkdir()
    (claude_dir / ADVISOR).symlink_to(file_target)

    result = _upgrade(claude_dir)

    assert set(result["outside"]) == {SKILL_RECALL, ADVISOR}
    assert link.is_symlink() and (checkout / "SKILL.md").read_text() == "my checkout copy\n"
    assert (claude_dir / ADVISOR).is_symlink() and file_target.read_text() == "a linked advisor\n"
    assert not list(checkout.glob("*.bak*")) and not list(tmp_path.glob("*.slm_tmp"))
    assert (claude_dir / "skills/slm-media/SKILL.md").is_file(), "the rest is still added"


def test_a_skills_folder_kept_elsewhere_is_left_alone_entirely(claude_dir, tmp_path) -> None:
    dotfiles = tmp_path / "dotfiles-skills"
    dotfiles.mkdir()
    (claude_dir / "skills").symlink_to(dotfiles, target_is_directory=True)

    result = _upgrade(claude_dir)

    assert not any(dotfiles.iterdir())
    assert {rel for rel in PLAN if rel.startswith("skills/")} <= set(result["outside"])


def test_with_the_plugin_installed_claude_is_left_alone(claude_dir) -> None:
    settings = json.loads((claude_dir / "settings.json").read_text())
    settings["enabledPlugins"] = {"superlocalmemory@qualixar": True}
    (claude_dir / "settings.json").write_text(json.dumps(settings))
    _seed_old_install(claude_dir)
    before = _tree(claude_dir)

    for apply in (False, True):
        result = _upgrade(claude_dir, apply=apply)
        assert result["status"] == "plugin-managed"
    assert _tree(claude_dir) == before


def test_a_skill_the_person_removed_is_not_put_back(claude_dir) -> None:
    assert _upgrade(claude_dir)["status"] == "updated"
    (claude_dir / "skills/slm-mesh/SKILL.md").unlink()

    result = _upgrade(claude_dir)

    assert result["status"] == "current" and result["removed"] == ["skills/slm-mesh/SKILL.md"]
    assert not (claude_dir / "skills/slm-mesh/SKILL.md").exists()


def test_the_automatic_step_adds_but_never_replaces_an_unrecorded_file(claude_dir) -> None:
    old = _seed_old_install(claude_dir, keep_out=("slm-media",))

    result = _upgrade(claude_dir, adopt_unrecorded=False)

    assert result["added"] == ["skills/slm-media/SKILL.md"]
    assert result["status"] == "updated" and set(result["held"]) == set(old)
    assert (claude_dir / SKILL_RECALL).read_bytes() == old[SKILL_RECALL]
    assert not list(claude_dir.rglob("*.bak-slm-*"))
    only_held = _upgrade(claude_dir, adopt_unrecorded=False)
    assert only_held["status"] == "needs-review", "held-back files are never reported as current"


def test_nothing_is_created_for_a_claude_without_slm(home) -> None:
    folder = home / ".claude"
    folder.mkdir(parents=True)
    (folder / "settings.json").write_text('{"theme": "dark"}\n')
    before = _tree(folder)

    assert _upgrade(folder)["status"] == "not-integrated"
    assert _tree(folder) == before
    assert assets.upgrade_claude_assets(home / "none" / "settings.json", version=VERSION)[
        "status"] == "not-integrated"
    assert not (home / "none").exists()


def test_the_manifest_records_posix_paths_and_the_digest_of_what_was_written(claude_dir) -> None:
    _upgrade(claude_dir)

    recorded = json.loads((claude_dir / assets.MANIFEST_NAME).read_text())["files"]
    assert set(recorded) == set(PLAN) and "\\" not in "".join(recorded)
    assert recorded[SKILL_RECALL] == assets._digest(PLAN[SKILL_RECALL])


def test_crlf_files_are_compared_as_bytes(claude_dir) -> None:
    crlf = PLAN[SKILL_RECALL].replace(b"\n", b"\r\n")
    (claude_dir / SKILL_RECALL).parent.mkdir(parents=True)
    (claude_dir / SKILL_RECALL).write_bytes(crlf)

    result = _upgrade(claude_dir, apply=False)

    assert SKILL_RECALL in result["adopted"], "a CRLF copy is a different file, kept as a backup"


def test_an_unwritable_target_is_an_error_not_a_crash(claude_dir) -> None:
    (claude_dir / "skills").write_text("a file where the skills folder should be\n")

    result = _upgrade(claude_dir)

    assert result["status"] == "error"


# -- once per version, by the daemon ---------------------------------------------------------

@pytest.fixture
def daemon_files(claude_dir, tmp_path):
    root = tmp_path / "slm-data"
    root.mkdir()
    return {"claude_settings": claude_dir / "settings.json",
            "codex_hooks_path": tmp_path / "home" / ".codex" / "hooks.json", "data_root": root}


def test_the_daemon_step_adds_new_skills_once_per_version(daemon_files, claude_dir, monkeypatch) -> None:
    monkeypatch.setattr(claude, "DISABLED_FILE", daemon_files["data_root"] / "no-such-marker")
    old = _seed_old_install(claude_dir, keep_out=("slm-media",))

    first = hook_upgrade.run_once(version=VERSION, env={}, **daemon_files)

    assert first["hosts"]["claude-code-assets"]["added"] == ["skills/slm-media/SKILL.md"]
    assert (claude_dir / SKILL_RECALL).read_bytes() == old[SKILL_RECALL], "unrecorded: not replaced"
    (claude_dir / "skills/slm-media/SKILL.md").unlink()
    assert hook_upgrade.run_once(version=VERSION, env={}, **daemon_files) is None
    assert not (claude_dir / "skills/slm-media/SKILL.md").exists(), "not run again for this version"

    again = hook_upgrade.run_once(version="4.2.33", env={}, **daemon_files)
    assert again is not None and again["hosts"]["claude-code-assets"]["status"] in ("current", "needs-review")


def test_the_daemon_step_never_creates_a_claude_folder(tmp_path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    outcome = hook_upgrade.run_once(version=VERSION, env={}, data_root=root,
                                    claude_settings=tmp_path / "c" / ".claude" / "settings.json",
                                    codex_hooks_path=tmp_path / "x" / "hooks.json")
    assert {h["status"] for h in outcome["hosts"].values()} == {"not-integrated"}
    assert not (tmp_path / "c").exists()


def test_a_failed_asset_write_is_retried_at_the_next_start(daemon_files, claude_dir) -> None:
    (claude_dir / "skills").write_text("blocks the folder\n")

    outcome = hook_upgrade.run_once(version=VERSION, env={}, **daemon_files)

    assert outcome["hosts"]["claude-code-assets"]["status"] == "error"
    assert not (daemon_files["data_root"] / "hooks-upgraded.json").exists()


# -- review findings on the first version of this change ---------------------------------------

def _recorded_install(claude_dir: Path) -> None:
    assert _upgrade(claude_dir)["status"] == "updated"  # adds everything and records the digests


def test_an_edit_saved_between_the_check_and_the_write_is_kept(claude_dir, tmp_path, monkeypatch) -> None:
    _recorded_install(claude_dir)
    nxt = _source_with(tmp_path / "next", {ADVISOR: b"new advisor text\n"})
    real = assets._decide

    def decide_then_person_types(folder, rel, *args, **kwargs):
        entry = real(folder, rel, *args, **kwargs)
        if rel == ADVISOR:
            (folder / rel).write_text("typed while the upgrade ran\n", encoding="utf-8")
        return entry

    monkeypatch.setattr(assets, "_decide", decide_then_person_types)

    result = _upgrade(claude_dir, source=nxt)

    assert (claude_dir / ADVISOR).read_text(encoding="utf-8") == "typed while the upgrade ran\n"
    assert ADVISOR in result["preserved"] and ADVISOR not in result["refreshed"]
    assert not list(claude_dir.rglob("*.slm_tmp")), "no half-written temp file is left behind"
    monkeypatch.setattr(assets, "_decide", real)
    again = _upgrade(claude_dir, source=nxt)  # the manifest must not have learned the new digest
    assert (claude_dir / ADVISOR).read_text(encoding="utf-8") == "typed while the upgrade ran\n"
    assert ADVISOR in again["preserved"]


def test_a_file_created_between_the_check_and_the_write_is_kept(claude_dir, monkeypatch) -> None:
    real = assets._decide

    def decide_then_person_creates(folder, rel, *args, **kwargs):
        entry = real(folder, rel, *args, **kwargs)
        if rel == COMMAND and entry.action == "add":
            (folder / rel).parent.mkdir(parents=True, exist_ok=True)
            (folder / rel).write_text("my own /slm-loop\n", encoding="utf-8")
        return entry

    monkeypatch.setattr(assets, "_decide", decide_then_person_creates)

    result = _upgrade(claude_dir)

    assert (claude_dir / COMMAND).read_text(encoding="utf-8") == "my own /slm-loop\n"
    assert COMMAND in result["preserved"]


def test_an_edit_made_while_the_backup_was_copied_is_not_lost(claude_dir, monkeypatch) -> None:
    old = _seed_old_install(claude_dir)
    real = assets._backup

    def backup_then_person_types(target, version):
        made = real(target, version)
        if target.name == "SKILL.md" and target.parent.name == "slm-recall":
            target.write_text("typed after the backup\n", encoding="utf-8")
        return made

    monkeypatch.setattr(assets, "_backup", backup_then_person_types)

    result = _upgrade(claude_dir)

    assert (claude_dir / SKILL_RECALL).read_text(encoding="utf-8") == "typed after the backup\n"
    assert SKILL_RECALL in result["preserved"]
    assert old[SKILL_RECALL] != b"typed after the backup\n"


def _failing_replace(monkeypatch, marker: str) -> None:
    real = os.replace

    def replace(src, dst, *args, **kwargs):
        if marker in str(dst) and not str(dst).endswith(assets.MANIFEST_NAME):
            raise OSError(28, "No space left on device")
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(assets.os, "replace", replace)


def test_a_failed_write_leaves_no_orphan_backup_and_repeats_do_not_stack(claude_dir, monkeypatch) -> None:
    _seed_old_install(claude_dir)
    _failing_replace(monkeypatch, "slm-recall")

    for _ in range(3):
        result = _upgrade(claude_dir)
        assert result["status"] == "error"

    assert not list((claude_dir / "skills/slm-recall").glob("*.bak-slm-*"))
    assert result["backups"] == [] or all("slm-recall" not in b for b in result["backups"])


def test_an_identical_backup_is_reused_not_copied_again(tmp_path) -> None:
    target = tmp_path / "SKILL.md"
    target.write_text("mine\n")

    first = assets._backup(target, "4.2.32")
    second = assets._backup(target, "4.2.32")

    assert first == second and len(list(tmp_path.glob("*.bak-slm-*"))) == 1
    target.write_text("mine, edited again\n")
    third = assets._backup(target, "4.2.32")
    assert third != first and third.read_text() == "mine, edited again\n"
    assert first.read_text() == "mine\n", "an earlier backup is never overwritten"


def test_a_backup_that_was_already_there_survives_a_failed_write(claude_dir, monkeypatch) -> None:
    old = _seed_old_install(claude_dir)
    first = _upgrade(claude_dir)
    assert first["status"] == "updated"
    # put the old text back and fail the next write: the earlier (identical) backup must stay
    (claude_dir / SKILL_RECALL).write_bytes(old[SKILL_RECALL])
    (claude_dir / assets.MANIFEST_NAME).unlink()
    _failing_replace(monkeypatch, "slm-recall")

    result = _upgrade(claude_dir)

    assert result["status"] == "error"
    kept = list((claude_dir / "skills/slm-recall").glob("*.bak-slm-*"))
    assert len(kept) == 1 and kept[0].read_bytes() == old[SKILL_RECALL]


def test_a_file_that_cannot_be_read_is_an_honest_retryable_error(claude_dir, monkeypatch) -> None:
    _recorded_install(claude_dir)
    real = Path.read_bytes

    def read_bytes(self):
        if self == claude_dir / ADVISOR:
            raise PermissionError(13, "Permission denied")
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    result = _upgrade(claude_dir, source=_source_with(claude_dir.parent / "next", {ADVISOR: b"new\n"}))

    assert result["status"] == "error"
    assert ADVISOR in result["detail"] and "read" in result["detail"]
    assert ADVISOR not in result["preserved"], "the person did not change it; do not say they did"
    assert result["status"] in hook_upgrade._RETRY_STATUSES


def test_an_unreadable_file_keeps_the_version_from_being_marked_done(daemon_files, claude_dir, monkeypatch) -> None:
    monkeypatch.setattr(claude, "DISABLED_FILE", daemon_files["data_root"] / "no-such-marker")
    _recorded_install(claude_dir)
    real = Path.read_bytes

    def read_bytes(self):
        if self == claude_dir / ADVISOR:
            raise PermissionError(13, "Permission denied")
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    outcome = hook_upgrade.run_once(version="4.2.40", env={}, **daemon_files)

    assert outcome["hosts"]["claude-code-assets"]["status"] == "error"
    assert not (daemon_files["data_root"] / "hooks-upgraded.json").exists()


def test_backups_are_reported_as_paths_under_claude_that_exist(claude_dir) -> None:
    _seed_old_install(claude_dir)

    result = _upgrade(claude_dir)

    backups = result["backups"]
    assert len(backups) == len(set(backups)) and len(backups) > 1
    for rel in backups:
        assert rel.startswith(("skills/", "agents/", "commands/")), rel
        assert (claude_dir / rel).is_file()


def test_a_folder_where_a_file_belongs_is_blocked_the_same_way_in_preview_and_apply(claude_dir) -> None:
    (claude_dir / SKILL_RECALL).mkdir(parents=True)

    preview = _upgrade(claude_dir, apply=False)
    applied = _upgrade(claude_dir)

    for result in (preview, applied):
        assert result["blocked"] == [SKILL_RECALL]
        assert SKILL_RECALL not in result["added"]
    assert applied["status"] != "error"
    assert (claude_dir / SKILL_RECALL).is_dir()
    assert (claude_dir / ADVISOR).is_file(), "the other files are still added"
    assert _upgrade(claude_dir)["status"] == "needs-review"


# -- CRIT: the re-check must not stop the refresh it exists to protect ------------------------

def test_an_untouched_file_slm_wrote_is_still_refreshed_with_its_mode_and_no_leftovers(claude_dir, tmp_path) -> None:
    _recorded_install(claude_dir)
    (claude_dir / ADVISOR).chmod(0o640)
    nxt = _source_with(tmp_path / "next", {ADVISOR: b"new advisor text\r\nwith CRLF\r\n"})

    result = _upgrade(claude_dir, source=nxt)

    assert result["status"] == "updated" and result["refreshed"] == [ADVISOR]
    assert (claude_dir / ADVISOR).read_bytes() == b"new advisor text\r\nwith CRLF\r\n"
    assert (claude_dir / ADVISOR).stat().st_mode & 0o777 == 0o640
    assert json.loads((claude_dir / assets.MANIFEST_NAME).read_text())["files"][ADVISOR] == \
        assets._digest(b"new advisor text\r\nwith CRLF\r\n")
    assert not list(claude_dir.rglob("*.slm_tmp")) and not list(claude_dir.rglob("*.bak-slm-*"))


def test_a_refresh_through_a_link_that_stays_inside_claude_still_works(claude_dir, tmp_path) -> None:
    _recorded_install(claude_dir)
    real = claude_dir / "kept" / "advisor.md"
    real.parent.mkdir()
    real.write_bytes((claude_dir / ADVISOR).read_bytes())
    (claude_dir / ADVISOR).unlink()
    (claude_dir / ADVISOR).symlink_to(real)
    nxt = _source_with(tmp_path / "next", {ADVISOR: b"new advisor text\n"})

    result = _upgrade(claude_dir, source=nxt)

    assert result["refreshed"] == [ADVISOR] and (claude_dir / ADVISOR).is_symlink()
    assert real.read_bytes() == b"new advisor text\n"


def test_a_file_that_appeared_already_identical_to_this_release_is_not_called_an_edit(claude_dir, monkeypatch) -> None:
    real = assets._decide

    def decide_then_same_text_appears(folder, rel, *args, **kwargs):
        entry = real(folder, rel, *args, **kwargs)
        if rel == COMMAND and entry.action == "add":
            (folder / rel).parent.mkdir(parents=True, exist_ok=True)
            (folder / rel).write_bytes(PLAN[COMMAND])
        return entry

    monkeypatch.setattr(assets, "_decide", decide_then_same_text_appears)

    result = _upgrade(claude_dir)

    assert COMMAND not in result["preserved"] and result["status"] == "updated"


def test_a_different_backup_already_on_disk_is_neither_reused_nor_removed(claude_dir, monkeypatch) -> None:
    old = _seed_old_install(claude_dir)
    earlier = claude_dir / "skills/slm-recall" / f"SKILL.md.bak-slm-{VERSION}"
    earlier.write_bytes(b"an unrelated earlier backup\n")

    result = _upgrade(claude_dir)

    assert earlier.read_bytes() == b"an unrelated earlier backup\n"
    new = claude_dir / "skills/slm-recall" / f"SKILL.md.bak-slm-{VERSION}.1"
    assert new.read_bytes() == old[SKILL_RECALL]
    assert f"skills/slm-recall/SKILL.md.bak-slm-{VERSION}.1" in result["backups"]
    # and a failed write later removes only the copy it made, not that earlier one
    (claude_dir / SKILL_RECALL).write_bytes(b"third text\n")
    (claude_dir / assets.MANIFEST_NAME).unlink()
    _failing_replace(monkeypatch, "slm-recall")
    assert _upgrade(claude_dir)["status"] == "error"
    assert earlier.exists() and new.exists()
    assert len(list((claude_dir / "skills/slm-recall").glob("*.bak-slm-*"))) == 2


# -- sub-agents name SLM tools the way a non-plugin Claude Code resolves them -----------------

REPO = Path(__file__).resolve().parents[2]
DIRECT = "mcp__superlocalmemory__"
ADVISORS = tuple(rel for rel in PLAN if rel.startswith("agents/"))


def _tools(data: bytes) -> list[str]:
    head = data.decode("utf-8").split("\n---\n", 1)[0]
    line = next(ln for ln in head.splitlines() if ln.startswith("tools:"))
    return [t.strip() for t in line[len("tools:"):].split(",") if t.strip()]


def _slm_tools_of(source: bytes) -> list[str]:
    """The SLM entries of a source agent, in order (host built-ins are capitalised)."""
    return [t for t in _tools(source) if not t[0].isupper()]


def _is_registered_slm_tool(name: str) -> bool:
    import re

    pattern = re.compile(rf"\bdef {re.escape(name)}\(")
    return any(pattern.search(p.read_text(encoding="utf-8"))
               for p in (REPO / "src" / "superlocalmemory" / "mcp").glob("*.py"))


def _direct_form(source: bytes) -> bytes:
    """What Claude Code needs for an SLM it talks to as a plain MCP server (an independent oracle)."""
    text = source.decode("utf-8")
    fm_end = text.index("\n---\n", 4) + 5
    head = text[:fm_end]
    out = []
    for line in head.split("\n"):
        if line.startswith("tools:"):
            entries = [t.strip() for t in line[len("tools:"):].split(",") if t.strip()]
            line = "tools: " + ", ".join(t if t[0].isupper() else DIRECT + t for t in entries)
        out.append(line)
    return ("\n".join(out) + text[fm_end:]).encode("utf-8")


def test_installed_advisors_carry_only_direct_slm_tool_names(claude_dir) -> None:
    assert len(ADVISORS) == 4
    _upgrade(claude_dir)

    for rel in ADVISORS:
        tools = _tools((claude_dir / rel).read_bytes())
        slm = [t for t in tools if not t[0].isupper()]
        assert slm, rel
        assert all(t.startswith(DIRECT) for t in slm), (rel, tools)  # not bare, not plugin form
        assert not any(t.startswith("mcp__plugin_") for t in tools), (rel, tools)
        assert all(_is_registered_slm_tool(t[len(DIRECT):]) for t in slm), (rel, slm)
        assert len(slm) == len(set(slm)), rel


def test_the_shipped_advisors_are_the_direct_form_of_plugin_src(claude_dir) -> None:
    """One source of truth: each shipped advisor is plugin-src's, with only the tool line rewritten."""
    for rel in ADVISORS:
        assert PLAN[rel] == _direct_form((REPO / "plugin-src" / rel).read_bytes()), rel


def test_a_working_direct_advisor_is_left_byte_identical_and_not_backed_up(claude_dir) -> None:
    agents = claude_dir / "agents"
    agents.mkdir()
    # agents that already use mcp__superlocalmemory__, as #330 installed them
    working = {rel: _direct_form((REPO / "plugin-src" / rel).read_bytes()) for rel in ADVISORS}
    for rel, data in working.items():
        (claude_dir / rel).write_bytes(data)

    result = _upgrade(claude_dir, adopt_unrecorded=True)

    for rel, data in working.items():
        assert (claude_dir / rel).read_bytes() == data, rel
    assert sorted(result["current"]) >= sorted(ADVISORS)
    assert not result["adopted"] and not result["refreshed"] and not result.get("backups")
    assert not list(agents.glob("*.bak-slm-*"))


def test_an_older_direct_advisor_is_refreshed_to_the_direct_form_never_to_bare_names(claude_dir) -> None:
    rel = ADVISOR
    (claude_dir / "agents").mkdir()
    working = _direct_form((REPO / "plugin-src" / rel).read_bytes())
    (claude_dir / rel).write_bytes(working.replace(b"Advises", b"Old text that advises"))

    _upgrade(claude_dir, adopt_unrecorded=True)

    now = (claude_dir / rel).read_bytes()
    assert now == PLAN[rel]
    assert all(t.startswith(DIRECT) for t in _slm_tools_of(now))


def test_the_daemon_step_installs_direct_form_advisors_too(daemon_files, claude_dir, monkeypatch) -> None:
    monkeypatch.setattr(claude, "DISABLED_FILE", daemon_files["data_root"] / "no-such-marker")

    hook_upgrade.run_once(version=VERSION, env={}, **daemon_files)

    for rel in ADVISORS:
        slm = _slm_tools_of((claude_dir / rel).read_bytes())
        assert slm and all(t.startswith(DIRECT) for t in slm), rel


def test_an_agents_source_with_bare_names_is_refused_not_installed(claude_dir, tmp_path) -> None:
    """A stale or wrongly built agents folder must never reach ~/.claude."""
    root = tmp_path / "share" / "claude-code"
    for rel, data in PLAN.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    (root / ADVISOR).write_bytes((REPO / "plugin-src" / ADVISOR).read_bytes())  # bare names

    result = _upgrade(claude_dir, source=root)

    assert result["status"] == "error" and "bare" in result["detail"]
    assert not (claude_dir / ADVISOR).exists()


def test_generated_advisors_have_lf_line_endings_and_posix_manifest_keys(claude_dir) -> None:
    """A Windows checkout holds the same bytes (.gitattributes eol=lf) and the manifest uses ``/``."""
    for rel in ADVISORS:
        assert b"\r" not in PLAN[rel], rel
    _upgrade(claude_dir)
    keys = json.loads((claude_dir / assets.MANIFEST_NAME).read_text(encoding="utf-8"))["files"]
    assert ADVISOR in keys and not any("\\" in k for k in keys)
