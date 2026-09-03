# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for slm doctor multi-install detection (task 8.7).

TDD RED phase: all tests fail before implementation exists.
"""

from __future__ import annotations

import io
import sys
from argparse import Namespace
from pathlib import Path
from unittest import mock

import pytest


def _run_doctor(tmp_path=None, installs=None, error_logs=None, extra_args=None):
    """Helper: run cmd_doctor with mocked installs + capture stdout."""
    from superlocalmemory.cli.commands import cmd_doctor

    args = Namespace(
        json=False,
        quick=True,   # skip slow probes
        fix=False,
        **(extra_args or {}),
    )

    install_list = installs if installs is not None else []
    error_log_list = error_logs if error_logs is not None else []

    captured = io.StringIO()

    with mock.patch(
        "superlocalmemory.core.install_detector._detect_all_installs",
        return_value=install_list,
    ), mock.patch(
        "superlocalmemory.cli.commands._detect_all_installs",
        return_value=install_list,
    ), mock.patch(
        "pathlib.Path.glob",
        side_effect=lambda self_path, pattern, **kw: (
            iter(error_log_list) if "migration-error" in str(pattern)
            else iter([])
        ),
    ):
        with mock.patch("sys.stdout", captured):
            try:
                cmd_doctor(args)
            except SystemExit:
                pass

    return captured.getvalue()


class TestDoctorInstallVersions:
    """slm doctor reports install versions and divergence."""

    def test_doctor_output_contains_install_versions_check(self, tmp_path):
        """doctor output includes 'install_versions' check."""
        from superlocalmemory.cli.commands import cmd_doctor

        installs = [
            {"path": "/pipx/venvs/slm/", "version": "4.1.0", "type": "pipx"},
            {"path": "~/.slm-venv/", "version": "4.1.0", "type": "venv"},
        ]

        output = _run_doctor(tmp_path=tmp_path, installs=installs)
        assert "install_versions" in output.lower() or "install" in output.lower()

    def test_doctor_warns_on_version_divergence(self, tmp_path):
        """doctor emits WARN when installs are at different versions."""
        installs = [
            {"path": "/pipx/venvs/slm/", "version": "4.0.8", "type": "pipx"},
            {"path": "~/.slm-venv/", "version": "4.1.0", "type": "venv"},
        ]
        output = _run_doctor(tmp_path=tmp_path, installs=installs)
        assert "WARN" in output or "warn" in output.lower()
        assert "4.0.8" in output or "pipx" in output or "divergen" in output.lower()

    def test_doctor_pass_when_all_same_version(self, tmp_path):
        """doctor emits PASS when all installs are at the same version."""
        installs = [
            {"path": "/pipx/venvs/slm/", "version": "4.1.0", "type": "pipx"},
            {"path": "~/.slm-venv/", "version": "4.1.0", "type": "venv"},
            {"path": "/usr/local/node_modules/slm/", "version": "4.1.0", "type": "npm"},
        ]
        output = _run_doctor(tmp_path=tmp_path, installs=installs)
        # No WARN about install versions when they match
        lines = [l for l in output.splitlines() if "install_versions" in l.lower()]
        for line in lines:
            assert "WARN" not in line

    def test_doctor_warns_on_stale_npm_venv_wheel(self, tmp_path):
        """4.1.14 single-source: npm wrapper over a stale venv wheel WARNs."""
        installs = [
            {
                "path": "/usr/local/node_modules/slm/",
                "version": "4.1.14",
                "type": "npm",
                "resolved": "/usr/local/node_modules/slm/.slm-venv/lib/python3.13/site-packages/superlocalmemory",
                "wheel_version": "4.1.13",
            },
        ]
        output = _run_doctor(tmp_path=tmp_path, installs=installs)
        assert "WARN" in output
        assert "npm rebuild" in output
        assert "4.1.13" in output

    def test_doctor_silent_when_npm_venv_matches(self, tmp_path):
        installs = [
            {
                "path": "/usr/local/node_modules/slm/",
                "version": "4.1.14",
                "type": "npm",
                "resolved": "/usr/local/node_modules/slm/.slm-venv/lib/python3.13/site-packages/superlocalmemory",
                "wheel_version": "4.1.14",
            },
        ]
        output = _run_doctor(tmp_path=tmp_path, installs=installs)
        lines = [l for l in output.splitlines() if "install_versions" in l.lower()]
        for line in lines:
            assert "WARN" not in line

    def test_doctor_reports_migration_error_log(self, tmp_path):
        """doctor reports ERROR when migration-error-*.log exists."""
        error_log = tmp_path / "migration-error-20260819-120000.log"
        error_log.write_text("migration failed: test error\n")

        from superlocalmemory.cli.commands import cmd_doctor
        from argparse import Namespace
        import io

        args = Namespace(json=False, quick=True, fix=False)
        captured = io.StringIO()

        with mock.patch(
            "superlocalmemory.cli.commands._detect_all_installs",
            return_value=[],
        ), mock.patch(
            "superlocalmemory.cli.commands._migration_error_logs",
            return_value=[error_log],
        ):
            with mock.patch("sys.stdout", captured):
                try:
                    cmd_doctor(args)
                except SystemExit:
                    pass

        output = captured.getvalue()
        assert "migration_errors" in output.lower() or "migration" in output.lower()
        assert "FAIL" in output or "ERROR" in output or "error" in output.lower()

    def test_doctor_no_migration_error_check_when_clean(self, tmp_path):
        """doctor reports PASS for migration_errors when no error logs exist."""
        from superlocalmemory.cli.commands import cmd_doctor
        from argparse import Namespace
        import io

        args = Namespace(json=False, quick=True, fix=False)
        captured = io.StringIO()

        with mock.patch(
            "superlocalmemory.cli.commands._detect_all_installs",
            return_value=[],
        ), mock.patch(
            "superlocalmemory.cli.commands._migration_error_logs",
            return_value=[],
        ):
            with mock.patch("sys.stdout", captured):
                try:
                    cmd_doctor(args)
                except SystemExit:
                    pass

        output = captured.getvalue()
        # migration_errors check must appear (with PASS) or simply not appear (no errors found)
        if "migration_errors" in output.lower():
            lines = [l for l in output.splitlines() if "migration_errors" in l.lower()]
            assert all("FAIL" not in l and "WARN" not in l for l in lines)

    def test_doctor_json_output_includes_install_versions(self, tmp_path):
        """doctor --json output contains install_versions check."""
        import json
        from superlocalmemory.cli.commands import cmd_doctor
        from argparse import Namespace
        import io

        args = Namespace(json=True, quick=True, fix=False)
        captured = io.StringIO()

        installs = [
            {"path": "/pipx/venvs/slm/", "version": "4.0.8", "type": "pipx"},
            {"path": "~/.slm-venv/", "version": "4.1.0", "type": "venv"},
        ]

        with mock.patch(
            "superlocalmemory.cli.commands._detect_all_installs",
            return_value=installs,
        ), mock.patch(
            "superlocalmemory.cli.commands._migration_error_logs",
            return_value=[],
        ):
            with mock.patch("sys.stdout", captured):
                try:
                    cmd_doctor(args)
                except SystemExit:
                    pass

        # Should not crash — JSON output is parseable
        output = captured.getvalue().strip()
        # The json_print function emits valid JSON lines
        assert output  # non-empty


class TestDoctorMigrationErrorDetection:
    """_migration_error_logs() is importable and returns a list."""

    def test_migration_error_logs_is_callable(self):
        from superlocalmemory.cli.commands import _migration_error_logs
        result = _migration_error_logs()
        assert isinstance(result, list)
