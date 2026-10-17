"""Fail-closed contracts for CI and coordinated release promotion."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TEST_WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "pypi-publish.yml"


def _workflow(path: Path) -> tuple[str, dict]:
    source = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(source)
    assert isinstance(payload, dict)
    return source, payload


def test_gate_failures_are_never_waived() -> None:
    """4.2.9: the gate is small so it can be strict. Any failure fails the job."""
    source, workflow = _workflow(TEST_WORKFLOW)
    gate_script = (ROOT / "scripts" / "ci-gate.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in gate_script
    assert "|| true" not in gate_script
    for path in (TEST_WORKFLOW, RELEASE_WORKFLOW):
        _, payload = _workflow(path)
        for step in payload["jobs"]["gate"]["steps"]:
            assert not step.get("continue-on-error"), (path.name, step.get("name"))
    assert "EXIT_CODE=$?" not in source
    assert "Windows KeyboardInterrupt during cleanup (non-fatal)" not in source


def test_release_uses_only_immutable_action_revisions() -> None:
    source, _workflow_payload = _workflow(RELEASE_WORKFLOW)
    action_uses = re.findall(r"^\s*-?\s*uses:\s*([^\s#]+)", source, re.MULTILINE)

    assert action_uses
    assert all(re.search(r"@[0-9a-f]{40}$", action) for action in action_uses), action_uses
    assert "uvx --from twine==6.2.0 twine check" in source
    assert "uvx twine check" not in source


def test_release_tests_and_records_exact_candidates_before_publication() -> None:
    source, workflow = _workflow(RELEASE_WORKFLOW)
    release = workflow["jobs"]["release"]

    assert release["permissions"] == {"contents": "write", "id-token": "write"}
    assert "uv lock --check" in source
    assert "SLM_RELEASE_PYTHON_DIST" in source
    assert "-m slow" in source
    assert "scripts/release_evidence.py" in source
    assert "SHA256SUMS" in source
    assert "verify-npm" in source
    assert "verify-source-parity" in source
    assert ".cdx.json" in source
    assert "actions/upload-artifact@" in source
    # The GitHub release is a distribution channel now: it runs after the
    # release job, from scripts/release/github-release.sh.
    github_release = (ROOT / "scripts" / "release" / "github-release.sh").read_text(encoding="utf-8")
    assert "gh release create" in github_release
    assert "release" in workflow["jobs"]["distribute"]["needs"]

    build = source.index("Build both release artifacts")
    artifact_tests = source.index("Test exact candidate artifacts")
    evidence = source.index("Generate immutable release evidence")
    publish_pypi = source.index("Publish to PyPI")
    parity = source.index("Verify registry parity")
    distribute = source.index("\n  distribute:")

    assert build < artifact_tests < evidence < publish_pypi < parity < distribute


def test_release_python_lock_is_part_of_the_source_checkout() -> None:
    lockfile = ROOT / "uv.lock"
    ignored = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert lockfile.is_file(), "release CI requires a checked-in uv.lock"
    assert "uv.lock" not in ignored, "uv.lock cannot be ignored when release CI consumes it"


def test_changelog_starts_with_the_current_package_version() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    expected = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    latest = re.search(r"^## \[([^]]+)]", changelog, re.MULTILINE)

    assert expected is not None
    assert latest is not None
    assert latest.group(1) == expected.group(1)
