"""Release contracts for Starlette's synchronous TestClient backend."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HTTPX2_PIN = "httpx2==2.5.0"


def test_httpx2_is_pinned_in_both_developer_dependency_surfaces() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    assert HTTPX2_PIN in project["project"]["optional-dependencies"]["dev"]
    assert HTTPX2_PIN in project["dependency-groups"]["dev"]


def test_starlette_testclient_import_has_no_deprecation_warning() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::DeprecationWarning",
            "-W",
            "error::starlette.testclient.StarletteDeprecationWarning",
            "-c",
            (
                "from starlette.applications import Starlette; "
                "from starlette.testclient import TestClient; "
                "client = TestClient(Starlette()); client.close()"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
