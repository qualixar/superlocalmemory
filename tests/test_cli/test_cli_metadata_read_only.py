"""Contract: CLI metadata inspection must not activate SuperLocalMemory."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "arguments",
    [[], ["--help"], ["--version"], ["remember", "--help"]],
)
def test_cli_metadata_commands_create_no_durable_state(
    tmp_path: Path,
    arguments: list[str],
) -> None:
    data_root = tmp_path / "data"
    environment = os.environ.copy()
    environment["SLM_DATA_DIR"] = str(data_root)
    environment["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        [sys.executable, "-m", "superlocalmemory.cli.main", *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert not data_root.exists(), (
        f"metadata invocation {arguments!r} created {data_root}"
    )
