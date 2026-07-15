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


def test_windows_test_failures_are_not_blanket_waived() -> None:
    source, workflow = _workflow(TEST_WORKFLOW)
    versions = workflow["jobs"]["test"]["strategy"]["matrix"]["python-version"]

    assert "3.14" in versions
    assert "EXIT_CODE=$?" not in source
    assert "Windows KeyboardInterrupt during cleanup (non-fatal)" not in source


def test_release_uses_only_immutable_action_revisions() -> None:
    source, _workflow_payload = _workflow(RELEASE_WORKFLOW)
    action_uses = re.findall(r"^\s*-?\s*uses:\s*([^\s#]+)", source, re.MULTILINE)

    assert action_uses
    assert all(re.search(r"@[0-9a-f]{40}$", action) for action in action_uses), action_uses


def test_release_tests_and_records_exact_candidates_before_publication() -> None:
    source, workflow = _workflow(RELEASE_WORKFLOW)
    release = workflow["jobs"]["release"]

    assert release["permissions"] == {"contents": "write", "id-token": "write"}
    assert "uv lock --check" in source
    assert "SLM_RELEASE_PYTHON_DIST" in source
    assert "-m slow" in source
    assert "scripts/release_evidence.py" in source
    assert "SHA256SUMS" in source
    assert "actions/upload-artifact@" in source
    assert "gh release create" in source

    build = source.index("Build both release artifacts")
    artifact_tests = source.index("Test exact candidate artifacts")
    evidence = source.index("Generate immutable release evidence")
    publish_pypi = source.index("Publish to PyPI")
    parity = source.index("Verify registry parity")
    github_release = source.index("Create GitHub release")

    assert build < artifact_tests < evidence < publish_pypi < parity < github_release


def test_changelog_starts_with_the_current_package_version() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    expected = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    latest = re.search(r"^## \[([^]]+)]", changelog, re.MULTILINE)

    assert expected is not None
    assert latest is not None
    assert latest.group(1) == expected.group(1)
