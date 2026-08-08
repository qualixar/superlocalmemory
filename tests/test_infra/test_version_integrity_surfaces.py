# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""Issue #107 -- staleness detection has to reach a human.

The detection in ``version_integrity`` is worthless if nobody sees it.  These
tests pin the three places it must surface, chosen because they are where
someone actually looks when SLM is behaving oddly:

* ``slm doctor`` -- what a user runs when something seems wrong,
* ``/health`` -- what a client or dashboard polls,
* ``slm mcp`` startup -- the process that actually goes stale.

The MCP one carries an extra hard constraint: that transport speaks JSON-RPC
over stdio, so a diagnostic printed to stdout would corrupt the protocol and
turn a warning into an outage.  It is tested for that explicitly.
"""

from __future__ import annotations

import io
import json
import logging
from argparse import Namespace
from unittest.mock import patch

from superlocalmemory.infra.version_integrity import VersionIntegrity


def _stale() -> VersionIntegrity:
    return VersionIntegrity(
        running="3.8.11",
        installed="3.8.12",
        state="stale",
        detail="running 3.8.11 but 3.8.12 is installed",
        hint="Restart this process to load the installed code.",
    )


def _current() -> VersionIntegrity:
    return VersionIntegrity(
        running="3.8.12", installed="3.8.12", state="current",
        detail="running 3.8.12, matching the installed distribution",
    )


class TestDoctorSurface:
    def _run_doctor_json(self) -> dict:
        from superlocalmemory.cli import commands

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            commands.cmd_doctor(Namespace(json=True, fix=False))
        return json.loads(buf.getvalue())

    def _find(self, report: dict, name: str) -> dict | None:
        # `slm doctor --json` wraps its payload in the standard CLI envelope,
        # so the checks live under "data", not at the top level.
        checks = report.get("data", {}).get("checks", [])
        for check in checks:
            if check.get("name") == name:
                return check
        return None

    def test_doctor_reports_a_stale_process_as_a_warning(self) -> None:
        with patch(
            "superlocalmemory.infra.version_integrity.check_version_integrity",
            return_value=_stale(),
        ):
            report = self._run_doctor_json()

        check = self._find(report, "Version integrity")
        assert check is not None, "doctor must report version integrity"
        assert check["status"] == "WARN"
        assert "3.8.11" in check["detail"] and "3.8.12" in check["detail"]
        assert check["fix"], "a stale result must tell the user what to do"

    def test_doctor_passes_when_versions_match(self) -> None:
        with patch(
            "superlocalmemory.infra.version_integrity.check_version_integrity",
            return_value=_current(),
        ):
            report = self._run_doctor_json()

        check = self._find(report, "Version integrity")
        assert check is not None
        assert check["status"] == "PASS"

    def test_doctor_survives_a_raising_version_check(self) -> None:
        """Doctor is a diagnostic tool; it must not die diagnosing."""
        with patch(
            "superlocalmemory.infra.version_integrity.check_version_integrity",
            side_effect=RuntimeError("boom"),
        ):
            report = self._run_doctor_json()

        check = self._find(report, "Version integrity")
        assert check is not None
        assert check["status"] == "WARN"


class TestHealthSurface:
    def test_health_payload_includes_version_integrity(self) -> None:
        from superlocalmemory.server.unified_daemon import _version_integrity_payload

        with patch(
            "superlocalmemory.infra.version_integrity.check_version_integrity",
            return_value=_stale(),
        ):
            payload = _version_integrity_payload()

        assert payload["state"] == "stale"
        assert payload["is_stale"] is True
        assert payload["running"] == "3.8.11"
        assert payload["installed"] == "3.8.12"

    def test_health_payload_is_json_serialisable(self) -> None:
        from superlocalmemory.server.unified_daemon import _version_integrity_payload

        json.dumps(_version_integrity_payload())

    def test_health_payload_degrades_instead_of_raising(self) -> None:
        """/health decides whether the daemon is usable. A diagnostic failing
        must never turn that into a 500."""
        from superlocalmemory.server.unified_daemon import _version_integrity_payload

        with patch(
            "superlocalmemory.infra.version_integrity.check_version_integrity",
            side_effect=RuntimeError("boom"),
        ):
            payload = _version_integrity_payload()

        assert payload["state"] == "unknown"
        json.dumps(payload)


class TestMcpStartupSurface:
    def test_stale_mcp_server_logs_a_warning(self, caplog) -> None:
        from superlocalmemory.infra.version_integrity import check_version_integrity

        logger = logging.getLogger("slm.test.mcp")
        with caplog.at_level(logging.WARNING, logger="slm.test.mcp"):
            result = check_version_integrity(
                running="3.8.11", installed_reader=lambda: "3.8.12"
            )
            if result.is_stale:
                logger.warning("MCP server version drift: %s. %s",
                               result.detail, result.hint)

        assert any("version drift" in r.message or "version drift" in r.getMessage()
                   for r in caplog.records)

    def test_mcp_startup_check_writes_nothing_to_stdout(self) -> None:
        """MCP speaks JSON-RPC over stdio.

        Any byte on stdout corrupts the protocol, so a staleness *warning*
        printed to stdout would escalate a cosmetic problem into a dead
        session. The check must be silent on stdout regardless of outcome.
        """
        from superlocalmemory.infra.version_integrity import check_version_integrity

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            for running, installed in (
                ("3.8.11", "3.8.12"),   # stale
                ("3.8.12", "3.8.12"),   # current
                ("3.9.0", "3.8.12"),    # ahead
                ("weird", "3.8.12"),    # mismatch
            ):
                check_version_integrity(
                    running=running, installed_reader=lambda i=installed: i
                )

        assert buf.getvalue() == "", (
            "version_integrity wrote to stdout; this corrupts MCP's JSON-RPC "
            "stdio transport"
        )
