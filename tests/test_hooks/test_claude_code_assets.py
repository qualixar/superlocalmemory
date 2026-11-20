# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A Claude Code wired by hand (no plugin) gets new and updated SLM skills, agents and command.

Before this, ``slm upgrade-hosts`` refreshed only the hook entries in ``settings.json`` and said
"already current" while every skill and advisor was still from the release that installed them and
the skills added since were missing. Every test runs in a tmp ``~/.claude``; none touches a real one.
"""

from __future__ import annotations

import json
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
