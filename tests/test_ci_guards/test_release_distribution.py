"""Contracts for the release fan-out: every distribution channel updates on a tag.

The release workflow publishes PyPI and npm, then a ``distribute`` matrix runs
one isolated job per channel (scripts/release/<channel>.sh). These tests pin the
workflow shape, the channel scripts' safety rules and the pure helpers in
scripts/release/release_tools.py.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.release import release_tools as rt

ROOT = Path(__file__).resolve().parents[2]
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "pypi-publish.yml"
TEST_WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"
RELEASE_DIR = ROOT / "scripts" / "release"
CHANNELS = (
    "github-release",
    "mcp-registry",
    "openvsx",
    "claude-plugin-repo",
    "hermes",
    "dsh-bundle",
    "clawhub",
    "listing-checklist",
)
SECRET_NAMES = (
    "OVSX_PAT",
    "CLAWHUB_TOKEN",
    "SLM_DEPLOY_KEY",
    "GH_TOKEN",
    "DEPLOY_KEY_SUPERLOCALMEMORY_PLUGIN",
    "DEPLOY_KEY_DSH_SUPERLOCALMEMORY",
)


def _release() -> tuple[str, dict]:
    source = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    return source, yaml.safe_load(source)


def _steps(job: dict) -> list[dict]:
    return job["steps"]


def _step(job: dict, name: str) -> dict:
    matches = [step for step in _steps(job) if step.get("name") == name]
    assert matches, f"no step named {name!r}"
    return matches[0]


# --- npm trusted publishing -------------------------------------------------


def test_npm_publishes_through_trusted_publishing_not_a_token() -> None:
    source, workflow = _release()
    release = workflow["jobs"]["release"]
    publish = _step(release, "Publish to npm")

    assert "NPM_TOKEN" not in source
    assert "NODE_AUTH_TOKEN" not in json.dumps(publish)
    assert "--provenance" in publish["run"]
    assert release["permissions"]["id-token"] == "write"
    assert release["environment"]["name"] == "pypi"


def test_npm_publish_runs_on_a_node_and_npm_that_support_oidc() -> None:
    source, workflow = _release()
    release = workflow["jobs"]["release"]
    names = [step.get("name") for step in _steps(release)]
    setup = _step(release, "Set up npm for trusted publishing")

    assert names.index("Set up npm for trusted publishing") < names.index("Publish to npm")
    node_steps = [s for s in _steps(release) if "setup-node" in str(s.get("uses", ""))]
    publish_node = node_steps[-1]["with"]["node-version"]
    assert int(str(publish_node).split(".")[0]) >= 24
    pinned = re.search(r"npm@(\d+)\.(\d+)\.(\d+)", setup["run"])
    assert pinned, "npm upgrade must pin an exact version"
    assert tuple(int(p) for p in pinned.groups()) >= (11, 5, 1)


# --- distribute fan-out ----------------------------------------------------


def test_release_job_exposes_version_and_tag_sha_to_later_jobs() -> None:
    _source, workflow = _release()
    outputs = workflow["jobs"]["release"]["outputs"]

    assert "steps.version.outputs.version" in outputs["version"]
    assert "steps.tag.outputs.sha" in outputs["tag_sha"]


def test_distribute_runs_every_channel_in_isolation_after_the_release() -> None:
    _source, workflow = _release()
    distribute = workflow["jobs"]["distribute"]

    assert "release" in distribute["needs"]
    assert "needs.release.result == 'success'" in distribute["if"]
    assert distribute["strategy"]["fail-fast"] is False
    channels = [leg["channel"] for leg in distribute["strategy"]["matrix"]["include"]]
    assert sorted(channels) == sorted(CHANNELS)
    assert distribute["permissions"] == {
        "contents": "write",
        "id-token": "write",
    }
    run = _step(distribute, "Publish channel")["run"]
    assert 'scripts/release/${{ matrix.channel }}.sh' in run


def test_each_channel_receives_only_its_own_secret() -> None:
    _source, workflow = _release()
    env = _step(workflow["jobs"]["distribute"], "Publish channel")["env"]

    assert "matrix.channel == 'openvsx' && secrets.OVSX_PAT" in env["OVSX_PAT"]
    assert "matrix.channel == 'clawhub' && secrets.CLAWHUB_TOKEN" in env["CLAWHUB_TOKEN"]
    key = env["SLM_DEPLOY_KEY"]
    assert "matrix.channel == 'claude-plugin-repo' && secrets.DEPLOY_KEY_SUPERLOCALMEMORY_PLUGIN" in key
    assert "matrix.channel == 'dsh-bundle' && secrets.DEPLOY_KEY_DSH_SUPERLOCALMEMORY" in key


def test_every_leg_records_a_result_even_when_it_fails() -> None:
    _source, workflow = _release()
    distribute = workflow["jobs"]["distribute"]
    record = _step(distribute, "Record channel result")
    upload = _step(distribute, "Upload channel result")

    assert record["if"] == "always()"
    assert upload["if"] == "always()"
    assert "steps.publish.outcome" in record["run"]


def test_summary_table_runs_after_all_legs_and_fails_on_any_failed_channel() -> None:
    _source, workflow = _release()
    summary = workflow["jobs"]["distribution-summary"]

    assert "distribute" in summary["needs"]
    assert "always()" in summary["if"]
    step = _step(summary, "Write channel table")
    assert "GITHUB_STEP_SUMMARY" in step["run"]
    assert "release_tools.py summary" in step["run"]
    assert "--fail-on-failed" in step["run"]
    matrix = workflow["jobs"]["distribute"]["strategy"]["matrix"]["include"]
    assert step["env"]["CHANNELS"].split(",") == [leg["channel"] for leg in matrix]


def test_vsix_is_built_once_for_both_the_github_release_and_open_vsx() -> None:
    _source, workflow = _release()
    vsix = workflow["jobs"]["vsix"]

    assert "release" in vsix["needs"]
    assert "scripts/release/build-vsix.sh" in json.dumps(vsix)
    assert "vsix" in workflow["jobs"]["distribute"]["needs"]


# --- PR drift guard ----------------------------------------------------------


def test_pull_requests_fail_when_generated_plugins_drift_from_their_sources() -> None:
    workflow = yaml.safe_load(TEST_WORKFLOW.read_text(encoding="utf-8"))
    guard = workflow["jobs"]["plugin-drift"]
    run = "\n".join(step.get("run", "") for step in guard["steps"])

    assert "node scripts/build-all-plugins.mjs" in run
    assert "git diff --exit-code" in run
    assert "git status --porcelain" in run


# --- channel script safety ---------------------------------------------------


@pytest.mark.parametrize("channel", (*CHANNELS, "build-vsix"))
def test_channel_script_exists_is_strict_and_supports_dry_run(channel: str) -> None:
    script = RELEASE_DIR / f"{channel}.sh"
    source = script.read_text(encoding="utf-8")

    assert os.access(script, os.X_OK), f"{script} is not executable"
    assert "set -euo pipefail" in source
    assert "_lib.sh" in source
    assert "set -x" not in source


def test_shared_library_parses_dry_run_and_never_traces() -> None:
    source = (RELEASE_DIR / "_lib.sh").read_text(encoding="utf-8")

    assert "--dry-run" in source
    assert "set -x" not in source
    assert "xtrace" not in source


@pytest.mark.parametrize("path", sorted(RELEASE_DIR.glob("*.sh")), ids=lambda p: p.name)
def test_scripts_never_echo_a_secret_value(path: Path) -> None:
    # Writing a secret into a private file (">\"$dir/key\"") is how a deploy key
    # reaches ssh; sending it to stdout or stderr is never allowed.
    for line in path.read_text(encoding="utf-8").splitlines():
        for name in SECRET_NAMES:
            if re.search(rf"\b(echo|printf)\b[^\n]*\${{?{name}\b", line):
                assert re.search(r'>\s*"\$[a-z_]+/', line), f"{path.name} prints ${name}: {line.strip()}"


def test_legacy_plugin_publisher_delegates_to_the_channel_script() -> None:
    source = (ROOT / "scripts" / "publish-claude-plugin-repo.sh").read_text(encoding="utf-8")

    assert "scripts/release/claude-plugin-repo.sh" in source or "release/claude-plugin-repo.sh" in source


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")
def test_hermes_dry_run_reports_the_served_subdirectory(tmp_path: Path) -> None:
    status = tmp_path / "status"
    version = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))["version"]
    result = subprocess.run(
        # A ref that cannot be fetched keeps this offline and independent of
        # what origin/main serves today (which lags a release until it merges).
        ["bash", str(RELEASE_DIR / "hermes.sh"), "--dry-run", "--version", version,
         "--ref", "origin/slm-test-ref-that-does-not-exist"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "SLM_CHANNEL_STATUS_FILE": str(status)},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "hermes-plugin" in result.stdout
    assert status.read_text(encoding="utf-8").startswith("status=")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")
def test_scripts_refuse_an_unknown_option() -> None:
    result = subprocess.run(
        ["bash", str(RELEASE_DIR / "hermes.sh"), "--publish-everything"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert "unknown option" in result.stderr


# --- pure helpers ------------------------------------------------------------

CHANGELOG = """# Changelog

Intro text.

## [4.2.8] — Every channel updates on its own

No migration.

### Release

- One tag publishes everywhere.

## [4.2.7] — Older

- old line
"""


def test_changelog_section_returns_only_that_versions_body() -> None:
    body = rt.changelog_section(CHANGELOG, "4.2.8")

    assert body.startswith("No migration.")
    assert "One tag publishes everywhere." in body
    assert "old line" not in body
    assert "## [4.2.7]" not in body


def test_changelog_section_rejects_a_missing_version() -> None:
    with pytest.raises(ValueError, match="4.9.9"):
        rt.changelog_section(CHANGELOG, "4.9.9")


def test_changelog_section_does_not_match_a_version_prefix() -> None:
    with pytest.raises(ValueError):
        rt.changelog_section(CHANGELOG, "4.2")


def test_set_json_version_keeps_formatting_and_reports_change() -> None:
    text = '{\n  "name": "x",\n  "version": "0.1.0",\n  "files": ["a", "b"],\n  "dep": {"version": "9"}\n}\n'
    new_text, changed = rt.set_json_version(text, "4.2.8")

    assert changed is True
    assert '"version": "4.2.8"' in new_text
    assert '"files": ["a", "b"]' in new_text
    assert '{"version": "9"}' in new_text
    assert json.loads(new_text)["version"] == "4.2.8"


def test_set_json_version_is_a_no_op_when_already_current() -> None:
    text = '{\n  "version": "4.2.8"\n}\n'

    assert rt.set_json_version(text, "4.2.8") == (text, False)


def test_set_json_version_requires_a_top_level_version() -> None:
    with pytest.raises(ValueError):
        rt.set_json_version('{"name": "x"}', "1.0.0")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [("4.2.8", "4.2.8", 0), ("4.2.10", "4.2.9", 1), ("4.2.7", "4.2.8", -1), ("5.0.0", "4.99.99", 1)],
)
def test_compare_versions_is_numeric(left: str, right: str, expected: int) -> None:
    assert rt.compare_versions(left, right) == expected


def test_yaml_scalar_reads_a_top_level_key() -> None:
    text = "name: superlocalmemory\nversion: 4.2.8\nnested:\n  version: 1\n"

    assert rt.yaml_scalar(text, "version") == "4.2.8"
    assert rt.yaml_scalar(text, "missing") is None


def test_skill_changed_compares_local_files_against_published_hashes(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text("hello\n", encoding="utf-8")
    local = rt.tree_digest(tmp_path)
    published = {"SKILL.md": local["SKILL.md"], "skill-card.md": "server-generated"}

    assert rt.skill_changed(local, published) is False
    assert rt.skill_changed(local, {"SKILL.md": "0" * 64}) is True
    assert rt.skill_changed({**local, "extra.md": "1" * 64}, published) is True


def test_record_maps_step_outcomes_to_channel_statuses(tmp_path: Path) -> None:
    status_file = tmp_path / "status"
    status_file.write_text("status=skipped\ndetail=4.2.8 already on Open VSX\n", encoding="utf-8")

    assert rt.record("openvsx", "success", status_file) == {
        "channel": "openvsx",
        "status": "skipped",
        "detail": "4.2.8 already on Open VSX",
    }
    assert rt.record("mcp-registry", "failure", tmp_path / "none")["status"] == "failed"
    assert rt.record("hermes", "success", tmp_path / "none")["status"] == "ok"
    assert rt.record("clawhub", "skipped", tmp_path / "none")["status"] == "skipped"


def test_summary_table_lists_every_channel_and_flags_missing_ones() -> None:
    results = [
        {"channel": "openvsx", "status": "ok", "detail": "published"},
        {"channel": "hermes", "status": "failed", "detail": "main | behind"},
    ]
    table, failed = rt.summary_table(results, ("openvsx", "hermes", "clawhub"))

    assert "| Channel | Result | Detail |" in table
    assert "| openvsx | ok | published |" in table
    assert "main \\| behind" in table
    assert "| clawhub | failed | no result recorded" in table
    assert failed == ["hermes", "clawhub"]


def test_checklist_names_every_human_gated_channel() -> None:
    body = rt.checklist_body("4.2.8")

    for needle in (
        "claude.ai/directory/manage",
        "Cursor Marketplace",
        "VS Code Marketplace",
        "Antigravity",
        "Kimi",
        "Kilo-Org/kilo-marketplace",
        "cline/mcp-marketplace",
        "awesome-opencode",
        "awesome-dsh-plugin",
        "xai-org/plugin-marketplace",
    ):
        assert needle in body, needle
    # Hermes was accepted (PR 139715); it is no longer a manual listing.
    assert "NousResearch/hermes-agent" not in body
    assert body.count("- [ ]") >= 10
    assert "4.2.8" in body
