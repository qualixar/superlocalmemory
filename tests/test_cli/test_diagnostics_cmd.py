"""Explicit CLI export contract for privacy-preserving local diagnostics."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from superlocalmemory.cli.main import _command_requires_daemon
from superlocalmemory.infra.local_diagnostics import LocalDiagnostics


ROOT = Path(__file__).resolve().parents[2]


def test_diagnostics_export_is_explicit_and_does_not_require_daemon() -> None:
    assert _command_requires_daemon(Namespace(command="diagnostics")) is False


def test_successful_ide_connection_records_only_aggregate_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superlocalmemory.cli.commands import cmd_connect
    from superlocalmemory.hooks import portable_kit
    from superlocalmemory.infra import local_diagnostics

    monkeypatch.setattr(
        portable_kit,
        "connect_ide",
        lambda *_args, **_kwargs: {
            "error": None,
            "mcp_config": "wrote",
            "mcp_path": "/private/config.json",
            "agents_md": "unchanged",
        },
    )
    recorded: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        local_diagnostics,
        "record_operation",
        lambda operation, **kwargs: recorded.append((operation, kwargs.get("client"))),
    )

    cmd_connect(Namespace(
        ide="cursor", here=False, profile=None, json=False,
        cross_platform=False, disable=None, dry_run=False,
    ))

    assert recorded == [("activation", "cursor")]


def test_cli_exports_existing_local_aggregates_without_network(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    diagnostics = LocalDiagnostics(data_root / "diagnostics.db")
    diagnostics.record("activation", client="codex")
    destination = tmp_path / "diagnostics-export.json"
    env = os.environ.copy()
    env["SLM_DATA_DIR"] = str(data_root)
    env["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "superlocalmemory.cli.main",
            "diagnostics",
            "export",
            str(destination),
            "--json",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    output = json.loads(result.stdout)
    assert output["success"] is True
    assert output["command"] == "diagnostics"
    assert output["data"] == {
        "exported": True,
        "reporting": "manual_export_only",
    }
    assert destination.is_file()
    exported = json.loads(destination.read_text(encoding="utf-8"))
    assert exported["privacy"]["local_only"] is True
    assert exported["privacy"]["reporting"] == "manual_export_only"


def test_privacy_diagnostics_documentation_states_no_automatic_reporting() -> None:
    doc = (ROOT / "docs/privacy-diagnostics.md").read_text(encoding="utf-8")
    assert "No network reporting endpoint" in doc
    assert "never exported automatically" in doc
    assert "slm diagnostics export" in doc
