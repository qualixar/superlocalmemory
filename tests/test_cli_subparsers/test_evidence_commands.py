# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch


def test_evidence_cli_exposes_versioned_workflow() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "superlocalmemory.cli.main", "evidence", "--help"],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
    )
    assert result.returncode == 0, result.stderr
    for command in ("export", "verify", "import", "rebuild"):
        assert command in result.stdout


def test_import_is_dry_run_without_execute(capsys) -> None:
    from superlocalmemory.cli.evidence_cmd import cmd_evidence

    args = Namespace(
        evidence_command="import",
        bundle="/tmp/bundle",
        profile="default",
        replace=True,
        rollback_dir="/tmp/rollback",
        execute=False,
        json=False,
    )
    with patch(
        "superlocalmemory.core.evidence_bundle.import_evidence_bundle"
    ) as importer:
        cmd_evidence(args)
    importer.assert_not_called()
    assert "--execute" in capsys.readouterr().out


def test_rebuild_is_dry_run_without_execute(capsys) -> None:
    from superlocalmemory.cli.evidence_cmd import cmd_evidence

    args = Namespace(
        evidence_command="rebuild",
        profile="default",
        execute=False,
        json=False,
    )
    with patch(
        "superlocalmemory.core.evidence_bundle.rebuild_derived_state"
    ) as rebuild:
        cmd_evidence(args)
    rebuild.assert_not_called()
    assert "--execute" in capsys.readouterr().out
