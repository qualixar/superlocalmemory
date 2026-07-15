"""Release-contract tests for the macOS DMG artifact.

The DMG is a transport for one already-built wheel.  It must never rebuild from
or package a mutable repository checkout.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts" / "dmg_release.py"
BUILD_SCRIPT = ROOT / "scripts" / "build-dmg.sh"
TEST_SCRIPT = ROOT / "scripts" / "test-dmg.sh"


def _load_helper():
    spec = importlib.util.spec_from_file_location("dmg_release", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wheel(tmp_path: Path, *, version: str = "3.6.23") -> Path:
    wheel = tmp_path / f"superlocalmemory-{version}-py3-none-any.whl"
    metadata = (
        "Metadata-Version: 2.1\n"
        "Name: superlocalmemory\n"
        f"Version: {version}\n\n"
    )
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"superlocalmemory-{version}.dist-info/METADATA", metadata
        )
        archive.writestr("superlocalmemory/__init__.py", "")
    return wheel


def test_prepare_stage_uses_one_frozen_wheel_and_agpl_notices(tmp_path):
    helper = _load_helper()
    wheel = _wheel(tmp_path)
    stage = tmp_path / "stage"

    manifest = helper.prepare_stage(
        wheel=wheel,
        project_root=ROOT,
        stage_dir=stage,
    )

    volume = stage / "SuperLocalMemory"
    copied_wheel = volume / "artifacts" / wheel.name
    assert copied_wheel.read_bytes() == wheel.read_bytes()
    assert manifest["product"] == "SuperLocalMemory"
    assert manifest["version"] == "3.6.23"
    assert manifest["license"] == "AGPL-3.0-or-later"
    assert manifest["wheel"]["sha256"] == hashlib.sha256(
        wheel.read_bytes()
    ).hexdigest()
    assert manifest["wheel"]["name"] == wheel.name
    assert manifest["installer"]["manager"] == "uv-or-pipx"
    assert manifest["installer"]["preserves_runtime_data_on_uninstall"] is True
    assert (volume / "LICENSE").is_file()
    assert (volume / "NOTICE").is_file()
    assert (volume / "ATTRIBUTION.md").is_file()
    assert (volume / "INSTALL.command").stat().st_mode & 0o111
    assert (volume / "UNINSTALL.command").stat().st_mode & 0o111
    assert "2.7.4" not in (volume / "README-INSTALLATION.txt").read_text()
    assert helper.validate_stage(volume) == manifest


def test_prepare_stage_rejects_wheel_not_matching_project_version(tmp_path):
    helper = _load_helper()
    wheel = _wheel(tmp_path, version="9.9.9")

    with pytest.raises(ValueError, match="does not match project version"):
        helper.prepare_stage(
            wheel=wheel,
            project_root=ROOT,
            stage_dir=tmp_path / "stage",
        )


def test_validate_stage_rejects_tampered_frozen_artifact(tmp_path):
    helper = _load_helper()
    wheel = _wheel(tmp_path)
    stage = tmp_path / "stage"
    helper.prepare_stage(wheel=wheel, project_root=ROOT, stage_dir=stage)
    copied_wheel = stage / "SuperLocalMemory" / "artifacts" / wheel.name
    copied_wheel.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checksum"):
        helper.validate_stage(stage / "SuperLocalMemory")


def test_sidecar_records_final_dmg_checksum_and_truthful_signing_state(tmp_path):
    helper = _load_helper()
    dmg = tmp_path / "SuperLocalMemory-v3.6.23-macos-universal.dmg"
    dmg.write_bytes(b"candidate-dmg")

    manifest_path, checksum_path = helper.write_release_sidecars(
        dmg=dmg,
        version="3.6.23",
        signed=False,
        notarized=False,
    )

    manifest = json.loads(manifest_path.read_text())
    expected = hashlib.sha256(dmg.read_bytes()).hexdigest()
    assert manifest["dmg"]["sha256"] == expected
    assert manifest["signing"] == "unsigned"
    assert manifest["notarization"] == "not-submitted"
    assert checksum_path.read_text() == f"{expected}  {dmg.name}\n"
    assert helper.validate_release_sidecars(dmg, require_release_ready=False) == manifest
    with pytest.raises(ValueError, match="not signed"):
        helper.validate_release_sidecars(dmg, require_release_ready=True)


def test_release_pair_rejects_sidecar_version_not_matching_embedded_wheel(tmp_path):
    helper = _load_helper()
    wheel = _wheel(tmp_path)
    stage = tmp_path / "stage"
    helper.prepare_stage(wheel=wheel, project_root=ROOT, stage_dir=stage)
    dmg = tmp_path / "SuperLocalMemory-v9.9.9-macos-universal.dmg"
    dmg.write_bytes(b"candidate-dmg")
    helper.write_release_sidecars(
        dmg=dmg,
        version="9.9.9",
        signed=False,
        notarized=False,
    )

    with pytest.raises(ValueError, match="mounted wheel version"):
        helper.validate_release_pair(
            stage / "SuperLocalMemory",
            dmg,
            require_release_ready=False,
        )


@pytest.mark.parametrize("script", [BUILD_SCRIPT, TEST_SCRIPT])
def test_dmg_scripts_are_strict_and_have_no_legacy_contract(script):
    source = script.read_text(encoding="utf-8")
    assert "set -euo pipefail" in source
    assert "SuperLocalMemory V2" not in source
    assert "2.7.4" not in source
    assert "2.1.0" not in source
    assert "MIT License" not in source


def test_build_cli_requires_explicit_frozen_wheel():
    result = subprocess.run(
        ["bash", str(BUILD_SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--wheel" in result.stdout
    assert "--sign-identity" in result.stdout
    assert "--notary-profile" in result.stdout
    assert "--stage-only" in result.stdout


def test_test_cli_exposes_explicit_release_ready_gate():
    result = subprocess.run(
        ["bash", str(TEST_SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--dmg" in result.stdout
    assert "--require-release-ready" in result.stdout
