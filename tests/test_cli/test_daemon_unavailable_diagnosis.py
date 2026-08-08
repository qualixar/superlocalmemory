# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Issue #104 — DAEMON_UNAVAILABLE must say which failure actually happened.

"owned daemon is unavailable; retry later" described a stopped daemon, a
recycled PID, an unreachable port and an identity mismatch identically, so the
reporter spent days on a problem the CLI already knew the shape of.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import psutil
import pytest

from superlocalmemory.infra.daemon_identity import (
    build_descriptor,
    write_descriptor,
)

RECORDED_CREATE_TIME = 1785680165.85
LIVE_TOKEN = "lx1:9f3a2c1e-0000-4000-8000-000000000001:44219"
OTHER_TOKEN = "lx1:9f3a2c1e-0000-4000-8000-000000000001:98877"


def _running_process(create_time: float) -> MagicMock:
    process = MagicMock()
    process.is_running.return_value = True
    process.status.return_value = psutil.STATUS_RUNNING
    process.create_time.return_value = create_time
    return process


def _write_owned_descriptor(port: int, *, token: str | None):
    root = Path(os.environ["SLM_DATA_DIR"])
    descriptor = build_descriptor(
        data_root=root,
        port=port,
        version="3.8.11",
        pid=612,
        process_create_time=RECORDED_CREATE_TIME,
        process_start_token=token,
        instance_id="diagnosed-instance",
        capability="diagnosed-capability",
        state="ready",
    )
    write_descriptor(descriptor, data_root=root)
    return descriptor


def test_no_descriptor_reports_no_daemon() -> None:
    from superlocalmemory.cli import daemon as _daemon

    diagnosis = _daemon.describe_daemon_unavailability()
    assert diagnosis["reason"] == "no_daemon"
    assert "slm start" in diagnosis["hint"]


def test_unreadable_descriptor_is_named_as_such() -> None:
    from superlocalmemory.cli import daemon as _daemon

    Path(os.environ["SLM_DATA_DIR"], "daemon.json").write_text("not-json")

    diagnosis = _daemon.describe_daemon_unavailability()
    assert diagnosis["reason"] == "descriptor_unusable"


def test_recycled_pid_is_reported_as_pid_reuse() -> None:
    from superlocalmemory.cli import daemon as _daemon

    _write_owned_descriptor(43181, token=LIVE_TOKEN)
    process = _running_process(RECORDED_CREATE_TIME)

    with patch.object(_daemon, "_is_pid_alive", return_value=True), patch(
        "psutil.Process", return_value=process,
    ), patch.object(_daemon, "process_start_token_for", return_value=OTHER_TOKEN):
        diagnosis = _daemon.describe_daemon_unavailability()

    assert diagnosis["reason"] == "pid_reused_by_another_process"
    assert "612" in diagnosis["message"]
    assert "slm restart" in diagnosis["hint"]


def test_exited_daemon_is_reported_as_exited() -> None:
    from superlocalmemory.cli import daemon as _daemon

    _write_owned_descriptor(43182, token=LIVE_TOKEN)

    with patch.object(_daemon, "_is_pid_alive", return_value=False):
        diagnosis = _daemon.describe_daemon_unavailability()

    assert diagnosis["reason"] == "daemon_process_exited"


def test_identity_mismatch_message_points_at_clock_drift() -> None:
    """The exact case issue #104 hit: name the clock, not just "unavailable"."""
    from superlocalmemory.cli import daemon as _daemon

    _write_owned_descriptor(43183, token=None)
    process = _running_process(RECORDED_CREATE_TIME + 35.0)

    with patch.object(_daemon, "_is_pid_alive", return_value=True), patch(
        "psutil.Process", return_value=process,
    ), patch.object(
        _daemon, "process_start_token_for", return_value=None,
    ), patch.object(_daemon, "_fetch_health", return_value=None):
        diagnosis = _daemon.describe_daemon_unavailability()

    assert diagnosis["reason"] == "daemon_identity_mismatch"
    assert "clock" in diagnosis["message"]
    assert "WSL2" in diagnosis["message"]


def test_live_owned_daemon_that_will_not_answer_is_reported_as_unreachable() -> None:
    from superlocalmemory.cli import daemon as _daemon

    _write_owned_descriptor(43184, token=LIVE_TOKEN)
    process = _running_process(RECORDED_CREATE_TIME)

    with patch.object(_daemon, "_is_pid_alive", return_value=True), patch(
        "psutil.Process", return_value=process,
    ), patch.object(
        _daemon, "process_start_token_for", return_value=LIVE_TOKEN,
    ), patch.object(_daemon, "_fetch_health", return_value=None):
        diagnosis = _daemon.describe_daemon_unavailability()

    assert diagnosis["reason"] == "daemon_unreachable"
    assert "43184" in diagnosis["message"]


def test_foreign_daemon_on_the_port_is_distinguished() -> None:
    from superlocalmemory.cli import daemon as _daemon

    descriptor = _write_owned_descriptor(43185, token=LIVE_TOKEN)
    process = _running_process(RECORDED_CREATE_TIME)
    foreign = {"status": "ok", **descriptor.public_health_fields()}
    foreign["instance_id"] = "someone-elses-daemon"

    with patch.object(_daemon, "_is_pid_alive", return_value=True), patch(
        "psutil.Process", return_value=process,
    ), patch.object(
        _daemon, "process_start_token_for", return_value=LIVE_TOKEN,
    ), patch.object(_daemon, "_fetch_health", return_value=foreign):
        diagnosis = _daemon.describe_daemon_unavailability()

    assert diagnosis["reason"] == "port_owned_by_another_daemon"


def test_diagnosis_never_raises_and_degrades_to_a_generic_answer() -> None:
    from superlocalmemory.cli import daemon as _daemon

    with patch.object(
        _daemon, "_describe_daemon_unavailability", side_effect=RuntimeError("boom"),
    ):
        diagnosis = _daemon.describe_daemon_unavailability()

    assert diagnosis["reason"] == "unknown"
    assert diagnosis["message"]
    assert diagnosis["hint"]


def test_cli_stderr_names_the_reason(capsys) -> None:
    """The line the user actually reads must be actionable."""
    from superlocalmemory.cli import commands

    with pytest.raises(SystemExit):
        commands._daemon_unavailable("recall", use_json=False)

    stderr = capsys.readouterr().err
    assert "DAEMON_UNAVAILABLE (no_daemon):" in stderr
    assert "slm start" in stderr
    # The old wording carried no diagnosis at all.
    assert "owned daemon is unavailable; retry later." not in stderr


def test_mcp_envelope_carries_the_same_diagnosis() -> None:
    from superlocalmemory.mcp._daemon_proxy import daemon_unavailable_error

    message = daemon_unavailable_error()
    assert message.startswith("DAEMON_UNAVAILABLE (no_daemon):")


def test_liveness_evidence_is_reported_for_every_branch() -> None:
    """Guard against the diagnosis table drifting away from the resolver."""
    from superlocalmemory.cli import daemon as _daemon

    descriptor = SimpleNamespace(
        pid=612,
        port=8765,
        process_create_time=RECORDED_CREATE_TIME,
        process_start_token=LIVE_TOKEN,
        state="ready",
    )
    with patch.object(_daemon, "_is_pid_alive", return_value=True), patch(
        "psutil.Process", side_effect=psutil.NoSuchProcess(612),
    ):
        assert _daemon._resolve_descriptor_liveness(descriptor) == (
            False, "process_unreadable",
        )
    assert "process_unreadable" in _daemon._LIVENESS_DIAGNOSIS
    assert set(_daemon._LIVENESS_DIAGNOSIS) == {
        "process_exited",
        "process_zombie",
        "process_unreadable",
        "start_token_mismatch",
        "identity_mismatch",
    }
