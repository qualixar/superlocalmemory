"""Release guard for existing Cozo graph-store compatibility."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
STORE_COMPATIBLE_PIN = "pycozo[embedded]==0.3.0"


def test_all_published_cozo_dependency_surfaces_keep_store_compatible_pin() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]

    assert STORE_COMPATIBLE_PIN in project["dependencies"]
    assert project["optional-dependencies"]["cozo"] == [STORE_COMPATIBLE_PIN]
    assert STORE_COMPATIBLE_PIN in project["optional-dependencies"]["scale"]
