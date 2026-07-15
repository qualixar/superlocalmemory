"""Session fixtures for black-box package builds."""

from __future__ import annotations

from pathlib import Path

import pytest

from ._harness import BuiltArtifacts, build_snapshot, copy_release_snapshot


@pytest.fixture(scope="session")
def release_snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("slm-release-snapshot")
    return copy_release_snapshot(root / "source")


@pytest.fixture(scope="session")
def built_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
    release_snapshot: Path,
) -> BuiltArtifacts:
    root = tmp_path_factory.mktemp("slm-built-artifacts")
    return build_snapshot(release_snapshot, root / "dist")
