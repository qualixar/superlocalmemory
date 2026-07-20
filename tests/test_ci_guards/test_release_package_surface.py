"""Release gates for the npm artifact and coordinated registry publication."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from scripts import release_registry_guard as registry_guard


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_JSON = ROOT / "package.json"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "pypi-publish.yml"
LEGACY_WORKFLOWS = (ROOT / ".github" / "workflows" / "npm-publish.yml",)


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
        "scripts/postinstall-interactive.js",
        "scripts/postinstall/validation.js",
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
    assert {path for path in paths if path.startswith("scripts/")} == {
        "scripts/postinstall-interactive.js",
        "scripts/postinstall.js",
        "scripts/postinstall/validation.js",
        "scripts/preuninstall.js",
    }
    assert artifact["entryCount"] <= 650
    assert artifact["unpackedSize"] <= 10 * 1024 * 1024


def test_dmg_distribution_contract_is_absent_from_supported_surfaces() -> None:
    removed_paths = (
        ROOT / "scripts" / "build-dmg.sh",
        ROOT / "scripts" / "dmg_release.py",
        ROOT / "scripts" / "test-dmg.sh",
        ROOT / "docs" / "install-macos-dmg.md",
        ROOT / "tests" / "installers" / "test_dmg_release_contract.py",
    )
    assert not any(path.exists() for path in removed_paths)

    supported_text = [
        (ROOT / "README.md").read_text(encoding="utf-8"),
        *(path.read_text(encoding="utf-8") for path in (ROOT / ".github" / "workflows").glob("*.yml")),
        *(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "docs").rglob("*.md")
            if "audits" not in path.parts and "v2-archive" not in path.parts
        ),
    ]
    assert not any("dmg" in content.lower() for content in supported_text)


def test_one_workflow_coordinates_both_registries_and_supports_recovery() -> None:
    assert RELEASE_WORKFLOW.is_file()
    assert not any(path.exists() for path in LEGACY_WORKFLOWS)

    source = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.safe_load(source)
    release = workflow["jobs"]["release"]

    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert release["environment"]["name"] == "pypi"
    assert release["permissions"] == {"contents": "write", "id-token": "write"}
    assert "npm publish" in source
    assert "pypa/gh-action-pypi-publish@cef221092ed1bacb1cc03d23a2d87d1d172e277b" in source
    assert "check-release-registries" in source
    assert "verify-release-registries" in source
    assert "python -m pytest tests/ -q" in source
    assert "SLM_RELEASE_PYTHON_DIST" in source
    assert "scripts/release_evidence.py" in source
    assert "steps.registry.outputs.pypi_exists != 'true'" in source
    assert "steps.registry.outputs.npm_exists != 'true'" in source
    assert source.index("Build both release artifacts") < source.index("Publish to PyPI")
    assert source.index("Build both release artifacts") < source.index("Publish to npm")
    assert source.index("Publish to npm") < source.index("Verify registry parity")


def test_registry_guard_checks_both_immutable_package_versions(tmp_path: Path) -> None:
    pypi_dist = tmp_path / "python"
    pypi_dist.mkdir()
    wheel = pypi_dist / "superlocalmemory-3.7.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    tarball = tmp_path / "superlocalmemory-3.7.0.tgz"
    tarball.write_bytes(b"npm")
    checked: list[str] = []

    def fetcher(url: str) -> dict | None:
        checked.append(url)
        if "pypi.org" in url:
            return {
                "info": {"version": "3.7.0"},
                "urls": [
                    {
                        "filename": wheel.name,
                        "digests": {"sha256": registry_guard._sha256(wheel)},
                    }
                ],
            }
        return None

    state = registry_guard.registry_state(
        "3.7.0", pypi_dist=pypi_dist, npm_tarball=tarball, fetcher=fetcher
    )

    assert state == registry_guard.RegistryState(pypi_exists=True, npm_exists=False)
    assert len(checked) == 2
    assert any("pypi.org/pypi/superlocalmemory/3.7.0/json" in url for url in checked)
    assert any("registry.npmjs.org/superlocalmemory/3.7.0" in url for url in checked)


def test_registry_guard_writes_idempotent_recovery_outputs(tmp_path: Path) -> None:
    output = tmp_path / "github-output"
    registry_guard._write_github_output(
        output,
        registry_guard.RegistryState(pypi_exists=True, npm_exists=False),
    )

    assert output.read_text(encoding="utf-8") == (
        "pypi_exists=true\nnpm_exists=false\n"
    )


def test_registry_parity_verification_retries_partial_publication(
    monkeypatch, tmp_path: Path
) -> None:
    states = iter(
        (
            registry_guard.RegistryState(pypi_exists=True, npm_exists=False),
            registry_guard.RegistryState(pypi_exists=True, npm_exists=True),
        )
    )
    sleeps: list[float] = []
    monkeypatch.setattr(registry_guard, "registry_state", lambda _version, **_kwargs: next(states))
    monkeypatch.setattr(registry_guard.time, "sleep", sleeps.append)

    result = registry_guard._verify_with_retries(
        "3.7.0",
        attempts=2,
        interval=0.25,
        pypi_dist=tmp_path,
        npm_tarball=tmp_path / "candidate.tgz",
    )

    assert result == registry_guard.RegistryState(True, True)
    assert sleeps == [0.25]


def test_registry_guard_rejects_same_version_with_different_artifact(tmp_path: Path) -> None:
    tarball = tmp_path / "candidate.tgz"
    tarball.write_bytes(b"candidate")

    try:
        registry_guard._assert_npm_identity(
            {"version": "3.7.0", "dist": {"integrity": "sha512-wrong"}},
            "3.7.0",
            tarball,
        )
    except RuntimeError as exc:
        assert "artifact identity mismatch" in str(exc)
    else:
        raise AssertionError("mismatched immutable npm artifact was accepted")
