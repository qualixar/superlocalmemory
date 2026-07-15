"""Evidence manifests bind release promotion to exact candidate bytes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.release_evidence import build_evidence, verify_evidence, write_evidence


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
