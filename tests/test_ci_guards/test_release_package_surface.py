"""Release gates for the npm artifact and coordinated registry publication."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_JSON = ROOT / "package.json"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "publish-release.yml"
LEGACY_WORKFLOWS = (
    ROOT / ".github" / "workflows" / "npm-publish.yml",
    ROOT / ".github" / "workflows" / "pypi-publish.yml",
)


def _npm_dry_run() -> dict:
    completed = subprocess.run(
        ["npm", "pack", "--ignore-scripts", "--dry-run", "--json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(completed.stdout)[0]


def test_npm_manifest_allowlists_only_runtime_install_scripts() -> None:
    package = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    files = set(package["files"])

    assert "scripts/" not in files
    assert {
        "scripts/postinstall.js",
        "scripts/preuninstall.js",
    } <= files
    assert "scripts/postinstall_binary.js" not in files
    assert "scripts/build-dmg.sh" not in files
    assert "scripts/test-dmg.sh" not in files


def test_npm_dry_run_contains_no_build_tools_tests_or_compiled_caches() -> None:
    artifact = _npm_dry_run()
    paths = {entry["path"] for entry in artifact["files"]}

    assert "scripts/postinstall.js" in paths
    assert "scripts/preuninstall.js" in paths
    assert not any(path.endswith((".pyc", ".pyo")) for path in paths)
    assert not any("/__pycache__/" in f"/{path}" for path in paths)
    assert not any(path.startswith("scripts/__tests__/") for path in paths)
    assert "scripts/postinstall_binary.js" not in paths
    assert "scripts/build-dmg.sh" not in paths
    assert "scripts/test-dmg.sh" not in paths


def test_one_workflow_coordinates_both_registries_and_supports_recovery() -> None:
    assert RELEASE_WORKFLOW.is_file()
    assert not any(path.exists() for path in LEGACY_WORKFLOWS)

    source = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.safe_load(source)
    release = workflow["jobs"]["release"]

    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert release["environment"]["name"] == "release"
    assert release["permissions"] == {"contents": "read", "id-token": "write"}
    assert "npm publish" in source
    assert "pypa/gh-action-pypi-publish@release/v1" in source
    assert "check-release-registries" in source
    assert "verify-release-registries" in source
    assert "steps.registry.outputs.pypi_exists != 'true'" in source
    assert "steps.registry.outputs.npm_exists != 'true'" in source
    assert source.index("Build both release artifacts") < source.index("Publish to PyPI")
    assert source.index("Build both release artifacts") < source.index("Publish to npm")
    assert source.index("Publish to npm") < source.index("Verify registry parity")
