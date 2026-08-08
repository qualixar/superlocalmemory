# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Issue #104 — a drifting clock must not un-own a healthy daemon.

On WSL2 the boot time that ``psutil.create_time()`` is built from moves
against the wall clock during a session, so the creation time recorded in
``daemon.json`` stops matching the creation time computed for the *same* live
process (the reporter measured ~35s of divergence after ~4 minutes of uptime).
The old ownership check compared those two numbers with a 1.0s tolerance and
declared the daemon foreign, so every CLI command failed with
``DAEMON_UNAVAILABLE`` while the dashboard — which never runs the check — kept
working against the same process and the same database.

These tests drive the real ownership resolver. Nothing under test is mocked:
only the OS probes it consumes (the process table, the boot-independent start
token, and the loopback health endpoint) are substituted, so a regression in
the resolver itself cannot be hidden.
"""

from __future__ import annotations

import json
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

LIVE_TOKEN = "lx1:9f3a2c1e-0000-4000-8000-000000000001:44219"
OTHER_TOKEN = "lx1:9f3a2c1e-0000-4000-8000-000000000001:98877"
RECORDED_CREATE_TIME = 1785680165.85


class _HealthResponse:
    status = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()

    def geturl(self) -> str | None:
        return None


def _running_process(create_time: float) -> MagicMock:
    process = MagicMock()
    process.is_running.return_value = True
    process.status.return_value = psutil.STATUS_RUNNING
    process.create_time.return_value = create_time
    return process


def _descriptor(*, token: str | None, create_time: float = RECORDED_CREATE_TIME):
    """A descriptor shaped exactly like the one issue #104 reported."""
    return SimpleNamespace(
        pid=612,
        port=8765,
        process_create_time=create_time,
        process_start_token=token,
        state="ready",
    )


# ---------------------------------------------------------------------------
# 1. The reported failure: monotonically growing creation-time drift
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "drift_seconds",
    [1.5, 5.0, 35.0, 120.0, 900.0, 86_400.0],
    ids=["just-over-tolerance", "5s", "reported-35s", "2min", "15min", "1day"],
)
def test_growing_create_time_drift_never_disowns_the_live_daemon(
    drift_seconds: float,
) -> None:
    """Drift is unbounded, so no tolerance constant can be the fix.

    The start token is boot-relative and cannot move when the clock moves, so
    the daemon stays owned no matter how far creation time has drifted.
    """
    from superlocalmemory.cli import daemon as _daemon

    descriptor = _descriptor(token=LIVE_TOKEN)
    process = _running_process(RECORDED_CREATE_TIME + drift_seconds)

    with patch.object(_daemon, "_is_pid_alive", return_value=True), patch(
        "psutil.Process", return_value=process,
    ), patch.object(
        _daemon, "process_start_token_for", return_value=LIVE_TOKEN,
    ), patch.object(_daemon, "_fetch_health") as health:
        alive, evidence = _daemon._resolve_descriptor_liveness(descriptor)

    assert alive is True
    assert evidence == "start_token_match"
    # The token settles it outright: no HTTP probe was needed.
    health.assert_not_called()


def test_drift_accumulating_over_a_session_keeps_the_daemon_owned() -> None:
    """Replay the reporter's timeline: drift grows minute over minute."""
    from superlocalmemory.cli import daemon as _daemon

    descriptor = _descriptor(token=LIVE_TOKEN)
    verdicts = []
    for minute in range(0, 11):
        # WSL2's btime slides steadily; every recomputed create_time is worse.
        process = _running_process(RECORDED_CREATE_TIME + minute * 8.75)
        with patch.object(_daemon, "_is_pid_alive", return_value=True), patch(
            "psutil.Process", return_value=process,
        ), patch.object(
            _daemon, "process_start_token_for", return_value=LIVE_TOKEN,
        ):
            verdicts.append(_daemon._descriptor_process_is_alive(descriptor))

    assert verdicts == [True] * 11


def test_is_daemon_running_survives_drift_end_to_end(monkeypatch) -> None:
    """The user-visible symptom: `slm recall` must find its own daemon."""
    from superlocalmemory.cli import daemon as _daemon

    root = Path(os.environ["SLM_DATA_DIR"])
    descriptor = build_descriptor(
        data_root=root,
        port=43171,
        version="3.8.11",
        pid=os.getpid(),
        process_create_time=RECORDED_CREATE_TIME,
        process_start_token=LIVE_TOKEN,
        instance_id="drifting-instance",
        capability="drifting-capability",
        state="ready",
    )
    write_descriptor(descriptor, data_root=root)
    health = {"status": "ok", **descriptor.public_health_fields()}

    # 35 seconds of accumulated drift, exactly as reported.
    process = _running_process(RECORDED_CREATE_TIME + 35.0)
    with patch.object(_daemon, "_is_pid_alive", return_value=True), patch(
        "psutil.Process", return_value=process,
    ), patch.object(
        _daemon, "process_start_token_for", return_value=LIVE_TOKEN,
    ), patch(
        "urllib.request.urlopen", return_value=_HealthResponse(health),
    ):
        assert _daemon.is_daemon_running() is True


# ---------------------------------------------------------------------------
# 2. PID reuse must still be rejected — the safety property the check exists for
# ---------------------------------------------------------------------------

def test_recycled_pid_is_rejected_by_start_token_without_any_probe() -> None:
    """A different process on the same PID has a different start token."""
    from superlocalmemory.cli import daemon as _daemon

    descriptor = _descriptor(token=LIVE_TOKEN)
    # An unrelated process that happens to hold pid 612 now. Its creation time
    # is even within tolerance, so only the token can catch it.
    process = _running_process(RECORDED_CREATE_TIME + 0.2)

    with patch.object(_daemon, "_is_pid_alive", return_value=True), patch(
        "psutil.Process", return_value=process,
    ), patch.object(
        _daemon, "process_start_token_for", return_value=OTHER_TOKEN,
    ), patch.object(_daemon, "_fetch_health") as health:
        alive, evidence = _daemon._resolve_descriptor_liveness(descriptor)

    assert alive is False
    assert evidence == "start_token_mismatch"
    health.assert_not_called()


def test_recycled_pid_after_reboot_is_rejected_even_at_identical_ticks() -> None:
    """A fresh boot changes boot_id, so identical tick counts cannot collide."""
    from superlocalmemory.cli import daemon as _daemon

    before_reboot = "lx1:11111111-1111-4111-8111-111111111111:44219"
    after_reboot = "lx1:22222222-2222-4222-8222-222222222222:44219"
    descriptor = _descriptor(token=before_reboot)
    process = _running_process(RECORDED_CREATE_TIME)

    with patch.object(_daemon, "_is_pid_alive", return_value=True), patch(
        "psutil.Process", return_value=process,
    ), patch.object(
        _daemon, "process_start_token_for", return_value=after_reboot,
    ):
        assert _daemon._descriptor_process_is_alive(descriptor) is False


def test_dead_pid_is_rejected_before_any_identity_work() -> None:
    from superlocalmemory.cli import daemon as _daemon

    descriptor = _descriptor(token=LIVE_TOKEN)
    with patch.object(_daemon, "_is_pid_alive", return_value=False):
        assert _daemon._resolve_descriptor_liveness(descriptor) == (
            False, "process_exited",
        )


def test_zombie_descriptor_process_is_still_not_alive() -> None:
    """Pre-existing guard must survive the rewrite."""
    from superlocalmemory.cli import daemon as _daemon

    process = MagicMock()
    process.is_running.return_value = True
    process.status.return_value = psutil.STATUS_ZOMBIE
    descriptor = _descriptor(token=LIVE_TOKEN)

    with patch.object(_daemon, "_is_pid_alive", return_value=True), patch(
        "psutil.Process", return_value=process,
    ):
        assert _daemon._resolve_descriptor_liveness(descriptor) == (
            False, "process_zombie",
        )


# ---------------------------------------------------------------------------
# 3. Legacy descriptors (no token) and token-less platforms fall back safely
# ---------------------------------------------------------------------------

def test_legacy_descriptor_with_drift_is_rescued_by_health_identity() -> None:
    """No token (older release, or Windows): the daemon proves itself instead.

    Echoing the random instance id and the capability fingerprint out of a
    mode-0600 descriptor is a strictly stronger ownership proof than a
    creation-time comparison, so it may override a creation-time mismatch.
    """
    from superlocalmemory.cli import daemon as _daemon

    descriptor = build_descriptor(
        data_root=Path(os.environ["SLM_DATA_DIR"]),
        port=43172,
        version="3.8.11",
        pid=612,
        process_create_time=RECORDED_CREATE_TIME,
        process_start_token=None,
        instance_id="legacy-instance",
        capability="legacy-capability",
        state="ready",
    )
    process = _running_process(RECORDED_CREATE_TIME + 35.0)
    health = {"status": "ok", **descriptor.public_health_fields()}

    with patch.object(_daemon, "_is_pid_alive", return_value=True), patch(
        "psutil.Process", return_value=process,
    ), patch.object(
        _daemon, "process_start_token_for", return_value=None,
    ), patch.object(_daemon, "_fetch_health", return_value=health):
        alive, evidence = _daemon._resolve_descriptor_liveness(descriptor)

    assert alive is True
    assert evidence == "health_identity_match"


def test_legacy_descriptor_with_drift_rejects_an_impostor_on_the_port() -> None:
    """The health fallback is an identity check, not a liveness check."""
    from superlocalmemory.cli import daemon as _daemon

    descriptor = build_descriptor(
        data_root=Path(os.environ["SLM_DATA_DIR"]),
        port=43173,
        version="3.8.11",
        pid=612,
        process_create_time=RECORDED_CREATE_TIME,
        process_start_token=None,
        instance_id="legacy-instance",
        capability="legacy-capability",
        state="ready",
    )
    process = _running_process(RECORDED_CREATE_TIME + 35.0)
    # Something else answers on the port: right shape, wrong instance.
    foreign = {"status": "ok", **descriptor.public_health_fields()}
    foreign["instance_id"] = "someone-elses-daemon"

    with patch.object(_daemon, "_is_pid_alive", return_value=True), patch(
        "psutil.Process", return_value=process,
    ), patch.object(
        _daemon, "process_start_token_for", return_value=None,
    ), patch.object(_daemon, "_fetch_health", return_value=foreign):
        alive, evidence = _daemon._resolve_descriptor_liveness(descriptor)

    assert alive is False
    assert evidence == "identity_mismatch"


def test_legacy_descriptor_with_drift_and_dead_port_is_rejected() -> None:
    from superlocalmemory.cli import daemon as _daemon

    descriptor = _descriptor(token=None)
    process = _running_process(RECORDED_CREATE_TIME + 35.0)

    with patch.object(_daemon, "_is_pid_alive", return_value=True), patch(
        "psutil.Process", return_value=process,
    ), patch.object(
        _daemon, "process_start_token_for", return_value=None,
    ), patch.object(_daemon, "_fetch_health", return_value=None):
        assert _daemon._resolve_descriptor_liveness(descriptor) == (
            False, "identity_mismatch",
        )


def test_legacy_descriptor_within_tolerance_needs_no_probe() -> None:
    """The cheap path stays cheap for the overwhelmingly common case."""
    from superlocalmemory.cli import daemon as _daemon

    descriptor = _descriptor(token=None)
    process = _running_process(RECORDED_CREATE_TIME + 0.4)

    with patch.object(_daemon, "_is_pid_alive", return_value=True), patch(
        "psutil.Process", return_value=process,
    ), patch.object(
        _daemon, "process_start_token_for", return_value=None,
    ), patch.object(_daemon, "_fetch_health") as health:
        alive, evidence = _daemon._resolve_descriptor_liveness(descriptor)

    assert (alive, evidence) == (True, "create_time_match")
    health.assert_not_called()


def test_token_present_but_process_gone_falls_back_not_crashes() -> None:
    """An unreadable live token must not be read as proof of anything."""
    from superlocalmemory.cli import daemon as _daemon

    descriptor = _descriptor(token=LIVE_TOKEN)
    process = _running_process(RECORDED_CREATE_TIME)

    with patch.object(_daemon, "_is_pid_alive", return_value=True), patch(
        "psutil.Process", return_value=process,
    ), patch.object(
        _daemon, "process_start_token_for", return_value=None,
    ):
        # Token unavailable now -> comparison is undecidable -> creation time
        # decides, and here it agrees.
        assert _daemon._resolve_descriptor_liveness(descriptor) == (
            True, "create_time_match",
        )


def test_cross_scheme_tokens_are_never_compared() -> None:
    """A macOS token and a Linux token say nothing about each other."""
    from superlocalmemory.cli import daemon as _daemon

    descriptor = _descriptor(token="mn1:1785680165.85")
    process = _running_process(RECORDED_CREATE_TIME + 35.0)

    with patch.object(_daemon, "_is_pid_alive", return_value=True), patch(
        "psutil.Process", return_value=process,
    ), patch.object(
        _daemon, "process_start_token_for", return_value=LIVE_TOKEN,
    ), patch.object(_daemon, "_fetch_health", return_value=None):
        alive, evidence = _daemon._resolve_descriptor_liveness(descriptor)

    # Undecidable token -> creation time -> drifted -> health -> unreachable.
    assert (alive, evidence) == (False, "identity_mismatch")
