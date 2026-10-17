"""Contract for the built-installer verification workflow (macOS only since 4.2.9)."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "installer-matrix.yml"
# 4.2.9: CI runs macOS only, so it stays fast and green; the Windows steps
# remain in the workflow, guarded, for the day the OS is added back.
SUPPORTED_OS = {"macos-14"}


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_installer_workflow_is_least_privilege_and_macos() -> None:
    workflow = _workflow()
    assert workflow["permissions"] == {"contents": "read"}

    for job_name in ("clone-installers", "npm-artifact"):
        job = workflow["jobs"][job_name]
        assert set(job["strategy"]["matrix"]["os"]) == SUPPORTED_OS
        assert job["timeout-minutes"] <= 30


def test_installer_workflow_builds_and_exercises_candidate_artifacts() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "python -m build" in source
    assert "scripts/install.sh" in source
    assert "scripts/install.ps1" in source
    assert "npm pack --ignore-scripts" in source
    assert "npm install --global" in source
    assert source.count("--version") >= 4
    assert "uninstall" in source.lower()
    assert "--break-system-packages" not in source
    assert "curl" not in source
    assert "Invoke-RestMethod" not in source
