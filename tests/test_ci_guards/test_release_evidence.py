"""Evidence manifests bind release promotion to exact candidate bytes."""

from __future__ import annotations

import json
import io
import tarfile
from pathlib import Path

import pytest

from scripts.release_evidence import (
    build_cyclonedx_sbom,
    build_evidence,
    verify_evidence,
    verify_npm_tarball,
    write_evidence,
)


def _candidate_files(root: Path) -> list[Path]:
    candidates = [
        root / "python" / "superlocalmemory-3.7.0-py3-none-any.whl",
        root / "python" / "superlocalmemory-3.7.0.tar.gz",
        root / "npm" / "superlocalmemory-3.7.0.tgz",
    ]
    for index, path in enumerate(candidates):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"candidate-{index}".encode())
    return candidates


def test_evidence_round_trip_binds_all_candidate_bytes(tmp_path: Path) -> None:
    artifacts = _candidate_files(tmp_path / "dist")
    evidence = build_evidence(
        version="3.7.0",
        commit="a" * 40,
        artifacts=artifacts,
        artifact_root=tmp_path / "dist",
        generated_at="2026-07-15T00:00:00Z",
        workflow_run="12345",
    )

    manifest, checksums = write_evidence(evidence, tmp_path / "dist")

    assert json.loads(manifest.read_text(encoding="utf-8")) == evidence
    assert len(checksums.read_text(encoding="utf-8").splitlines()) == 3
    assert verify_evidence(evidence, tmp_path / "dist") == []


def test_evidence_verification_rejects_mutated_candidate(tmp_path: Path) -> None:
    artifacts = _candidate_files(tmp_path / "dist")
    evidence = build_evidence(
        version="3.7.0",
        commit="b" * 40,
        artifacts=artifacts,
        artifact_root=tmp_path / "dist",
        generated_at="2026-07-15T00:00:00Z",
    )
    artifacts[0].write_bytes(b"changed-after-gate")

    errors = verify_evidence(evidence, tmp_path / "dist")

    assert any("sha256 mismatch" in error for error in errors)


def test_evidence_requires_wheel_sdist_and_npm_tarball(tmp_path: Path) -> None:
    artifacts = _candidate_files(tmp_path / "dist")[:-1]

    with pytest.raises(ValueError, match="npm tarball"):
        build_evidence(
            version="3.7.0",
            commit="c" * 40,
            artifacts=artifacts,
            artifact_root=tmp_path / "dist",
            generated_at="2026-07-15T00:00:00Z",
        )


def test_lock_derived_sbom_inventories_python_and_npm_dependencies(tmp_path: Path) -> None:
    uv_lock = tmp_path / "uv.lock"
    uv_lock.write_text(
        'version = 1\nrevision = 1\nrequires-python = ">=3.11"\n'
        '[[package]]\nname = "httpx"\nversion = "0.28.1"\n'
        '[[package]]\nname = "superlocalmemory"\nversion = "3.7.0"\n',
        encoding="utf-8",
    )
    package_lock = tmp_path / "package-lock.json"
    package_lock.write_text(
        json.dumps(
            {
                "packages": {
                    "": {"name": "superlocalmemory", "version": "3.7.0"},
                    "node_modules/docx": {"version": "9.5.1"},
                }
            }
        ),
        encoding="utf-8",
    )

    sbom = build_cyclonedx_sbom(
        version="3.7.0",
        commit="d" * 40,
        uv_lock=uv_lock,
        package_lock=package_lock,
        generated_at="2026-07-15T00:00:00Z",
    )

    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["metadata"]["component"]["version"] == "3.7.0"
    purls = {component["purl"] for component in sbom["components"]}
    assert "pkg:pypi/httpx@0.28.1" in purls
    assert "pkg:npm/docx@9.5.1" in purls


def test_npm_candidate_verifier_reads_the_exact_tarball(tmp_path: Path) -> None:
    tarball = tmp_path / "superlocalmemory-3.7.0.tgz"
    members = {
        "package/package.json": json.dumps(
            {
                "name": "superlocalmemory",
                "version": "3.7.0",
                "license": "AGPL-3.0-or-later",
                "bin": {"slm": "bin/slm-npm"},
            }
        ).encode(),
        "package/bin/slm-npm": b"#!/bin/sh\n",
        "package/src/superlocalmemory/__init__.py": b'__version__ = "3.7.0"\n',
        "package/LICENSE": b"AGPL\n",
        "package/NOTICE": b"notice\n",
    }
    with tarfile.open(tarball, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    assert verify_npm_tarball(tarball, "3.7.0") == []


def test_npm_candidate_verifier_rejects_version_drift(tmp_path: Path) -> None:
    tarball = tmp_path / "candidate.tgz"
    payload = json.dumps(
        {
            "name": "superlocalmemory",
            "version": "3.6.23",
            "license": "AGPL-3.0-or-later",
            "bin": {"slm": "bin/slm-npm"},
        }
    ).encode()
    with tarfile.open(tarball, "w:gz") as archive:
        info = tarfile.TarInfo("package/package.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    errors = verify_npm_tarball(tarball, "3.7.0")

    assert any("version mismatch" in error for error in errors)
