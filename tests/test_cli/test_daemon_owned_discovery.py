# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Client discovery must attach only to the descriptor-owned daemon."""

from __future__ import annotations

import io
import json
import os
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.infra.daemon_identity import (
    build_descriptor,
    read_descriptor,
    write_descriptor,
)
from superlocalmemory.infra.process_identity import process_start_token_for
from tests._urlopen_fake import patch_urlopen


class _HealthResponse:
    status = 200

    def __init__(self, payload: dict, final_url: str | None = None) -> None:
        self._payload = payload
        self._final_url = final_url

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()

    def geturl(self) -> str | None:
        return self._final_url


def _owned_descriptor(port: int = 43123):
    root = Path(os.environ["SLM_DATA_DIR"])
    descriptor = build_descriptor(
        data_root=root,
        port=port,
        version="3.7.0a1",
        pid=os.getpid(),
        instance_id="owned-instance",
        capability="owned-capability",
        state="ready",
    )
    write_descriptor(descriptor, data_root=root)
    return descriptor


def test_matching_descriptor_and_health_are_adopted() -> None:
    from superlocalmemory.cli import daemon

    descriptor = _owned_descriptor()
    health = {"status": "ok", **descriptor.public_health_fields()}
    with patch_urlopen(return_value=_HealthResponse(health)) as request:
        assert daemon.is_daemon_running()
    assert request.call_count == 1
    assert ":43123/health" in request.call_args.args[0]


def test_foreign_health_is_rejected_without_rewriting_local_state() -> None:
    from superlocalmemory.cli import daemon

    descriptor = _owned_descriptor()
    original = Path(os.environ["SLM_DATA_DIR"], "daemon.json").read_text(encoding="utf-8")
    health = {"status": "ok", **descriptor.public_health_fields()}
    health["namespace_id"] = "foreign"

    with patch_urlopen(return_value=_HealthResponse(health)):
        assert not daemon.is_daemon_running()

    assert Path(os.environ["SLM_DATA_DIR"], "daemon.json").read_text(encoding="utf-8") == original


def test_custom_port_never_falls_through_to_fixed_legacy_port() -> None:
    from superlocalmemory.cli import daemon

    descriptor = _owned_descriptor(port=43124)
    calls: list[str] = []

    def _foreign_only(url: str, timeout: float):
        calls.append(url)
        if ":8767/" in url:
            return _HealthResponse({
                "status": "ok",
                **descriptor.public_health_fields(),
            })
        raise OSError("configured port unavailable")

    with patch_urlopen(side_effect=_foreign_only):
        assert not daemon.is_daemon_running()

    assert calls == ["http://127.0.0.1:43124/health"]


def test_health_redirect_to_another_origin_is_rejected() -> None:
    from superlocalmemory.cli import daemon

    descriptor = _owned_descriptor(port=43124)
    health = {"status": "ok", **descriptor.public_health_fields()}
    redirected = _HealthResponse(health, final_url="http://127.0.0.1:8767/health")

    with patch_urlopen(return_value=redirected):
        assert not daemon.is_daemon_running()


def test_arbitrary_live_pid_without_descriptor_is_rejected() -> None:
    from superlocalmemory.cli import daemon

    root = Path(os.environ["SLM_DATA_DIR"])
    (root / "daemon.pid").write_text(str(os.getpid()), encoding="utf-8")
    (root / "daemon.port").write_text("43125", encoding="utf-8")

    with patch.object(daemon, "_is_verified_legacy_process", return_value=False):
        assert not daemon.is_daemon_running()


def test_reused_pid_with_wrong_process_identity_is_rejected() -> None:
    """A live PID that is not our daemon must be rejected without a probe.

    Issue #104: the descriptor now records a clock-independent start token as
    well as a creation time, so a stale descriptor has to be stale in both --
    otherwise the token would (correctly) prove this very process is the one
    the descriptor names.
    """
    from superlocalmemory.cli import daemon

    root = Path(os.environ["SLM_DATA_DIR"])
    # Same token scheme this platform produces, different value -- i.e. exactly
    # what a recycled PID looks like, on whichever OS the suite is running.
    live_token = process_start_token_for(os.getpid())
    if live_token is None:
        pytest.skip("platform exposes no clock-independent start token")
    scheme, _, value = live_token.partition(":")
    stale_token = f"{scheme}:{value}-recycled"
    descriptor = build_descriptor(
        data_root=root,
        port=43125,
        version="3.7.0a1",
        pid=os.getpid(),
        process_create_time=0.0,
        process_start_token=stale_token,
        instance_id="stale-process",
        capability="stale-capability",
        state="ready",
    )
    write_descriptor(descriptor, data_root=root)

    # The stale record is rejected; the only probe allowed is the same-account
    # health fallback, and an unanswered port cannot rescue it.
    with patch_urlopen(side_effect=OSError("closed")):
        assert not daemon.is_daemon_running()


def test_legacy_descriptor_with_wrong_creation_time_needs_identity_proof() -> None:
    """Descriptors from before v3.8.12 carry no token, so health decides.

    A creation-time mismatch alone is ambiguous -- a recycled PID and a stepped
    clock look identical -- so the daemon is asked to prove ownership. An
    unanswered port is not proof, and the descriptor is still rejected.
    """
    from superlocalmemory.cli import daemon

    root = Path(os.environ["SLM_DATA_DIR"])
    descriptor = build_descriptor(
        data_root=root,
        port=43127,
        version="3.7.0a1",
        pid=os.getpid(),
        process_create_time=0.0,
        process_start_token=None,
        instance_id="legacy-stale-process",
        capability="legacy-stale-capability",
        state="ready",
    )
    write_descriptor(descriptor, data_root=root)

    with patch_urlopen(side_effect=OSError("no listener")):
        assert not daemon.is_daemon_running()


def test_get_port_uses_only_valid_owned_descriptor() -> None:
    from superlocalmemory.cli import daemon

    _owned_descriptor(port=43126)
    assert daemon._get_port() == 43126

    Path(os.environ["SLM_DATA_DIR"], "daemon.json").write_text("malformed", encoding="utf-8")
    assert daemon._get_port() == daemon._DEFAULT_PORT


def test_launcher_passes_one_process_identity_and_leaves_the_record_to_the_child() -> None:
    from superlocalmemory.cli import daemon

    fake_process = MagicMock(pid=54321)
    with (
        patch.object(daemon, "is_daemon_running", return_value=False),
        patch.object(daemon, "_wait_for_daemon", return_value=True),
        patch("subprocess.Popen", return_value=fake_process) as popen,
    ):
        assert daemon._start_daemon_subprocess()

    # The daemon launch, not the last Popen: on Windows, writing the
    # descriptor then runs icacls to make it owner-only.
    [launch] = [c for c in popen.call_args_list
                if "superlocalmemory.server.unified_daemon" in c.args[0]]
    child_env = launch.kwargs["env"]
    # Only the daemon that owns the data folder writes the record.
    assert read_descriptor() is None
    assert child_env["SLM_DAEMON_INSTANCE_ID"]
    assert child_env["SLM_DAEMON_CAPABILITY"]


def test_wait_requires_matching_health_not_only_a_live_starting_pid() -> None:
    from superlocalmemory.cli import daemon

    descriptor = build_descriptor(
        data_root=Path(os.environ["SLM_DATA_DIR"]),
        port=43128,
        version="3.7.0a1",
        pid=os.getpid(),
        instance_id="starting-instance",
        capability="starting-capability",
        state="starting",
    )
    write_descriptor(descriptor)
    foreign = {"status": "ok", **descriptor.public_health_fields()}
    foreign["instance_id"] = "foreign-instance"

    with (
        patch_urlopen(return_value=_HealthResponse(foreign)),
        patch("time.sleep"),
    ):
        assert not daemon._wait_for_daemon(timeout=1)


def test_daemon_request_refuses_foreign_identity_before_write() -> None:
    from superlocalmemory.cli import daemon

    descriptor = _owned_descriptor(port=43129)
    foreign = {"status": "ok", **descriptor.public_health_fields()}
    foreign["namespace_id"] = "foreign"

    with patch_urlopen(return_value=_HealthResponse(foreign)) as request:
        assert daemon.daemon_request("POST", "/remember", {"content": "blocked"}) is None
    assert request.call_count == 1


def test_daemon_request_sends_private_capability_after_identity_match() -> None:
    from superlocalmemory.cli import daemon

    descriptor = _owned_descriptor(port=43130)
    health = {"status": "ok", **descriptor.public_health_fields()}
    seen_headers: list[dict[str, str]] = []

    def _respond(request, timeout: float):
        if isinstance(request, str):
            return _HealthResponse(health)
        seen_headers.append(dict(request.header_items()))
        return _HealthResponse({"ok": True})

    with patch_urlopen(side_effect=_respond):
        result = daemon.daemon_request("POST", "/remember", {"content": "owned"})

    assert result == {"ok": True}
    normalized = {key.lower(): value for key, value in seen_headers[0].items()}
    assert normalized["x-slm-daemon-capability"] == descriptor.capability
    assert normalized["x-slm-target-instance"] == descriptor.instance_id


def test_daemon_request_preserves_profile_conflict() -> None:
    from superlocalmemory.cli import daemon

    descriptor = _owned_descriptor(port=43134)
    health = {"status": "ok", **descriptor.public_health_fields()}
    conflict = urllib.error.HTTPError(
        "http://127.0.0.1:43134/remember",
        409,
        "Conflict",
        {},
        io.BytesIO(b'{"detail":"profile mismatch: expected work"}'),
    )

    with patch_urlopen(side_effect=[_HealthResponse(health), conflict]):
        with pytest.raises(daemon.DaemonConflict) as caught:
            daemon.daemon_request(
                "POST",
                "/remember",
                {"content": "bound"},
                preserve_conflict=True,
            )

    assert "profile mismatch" in str(caught.value)


def test_daemon_request_forwards_explicit_user_session_after_identity_match(
    monkeypatch,
) -> None:
    """Governed CLI requests retain the authenticated dashboard identity."""
    from superlocalmemory.cli import daemon

    _owned_descriptor(port=43131)
    descriptor = read_descriptor()
    assert descriptor is not None
    health = {"status": "ok", **descriptor.public_health_fields()}
    seen_headers: list[dict[str, str]] = []
    monkeypatch.setenv("SLM_USER_SESSION", "session-from-explicit-login")

    def _respond(request, timeout: float):
        if isinstance(request, str):
            return _HealthResponse(health)
        seen_headers.append(dict(request.header_items()))
        return _HealthResponse({"ok": True})

    with patch_urlopen(side_effect=_respond):
        assert daemon.daemon_request("DELETE", "/memories/fact-1") == {"ok": True}

    normalized = {key.lower(): value for key, value in seen_headers[0].items()}
    assert normalized["x-slm-user-session"] == "session-from-explicit-login"


def test_daemon_request_omits_blank_user_session_after_identity_match(monkeypatch) -> None:
    """Whitespace must not create an empty credential header."""
    from superlocalmemory.cli import daemon

    descriptor = _owned_descriptor(port=43132)
    health = {"status": "ok", **descriptor.public_health_fields()}
    seen_headers: list[dict[str, str]] = []
    monkeypatch.setenv("SLM_USER_SESSION", "   ")

    def _respond(request, timeout: float):
        if isinstance(request, str):
            return _HealthResponse(health)
        seen_headers.append(dict(request.header_items()))
        return _HealthResponse({"ok": True})

    with patch_urlopen(side_effect=_respond):
        assert daemon.daemon_request("POST", "/remember", {"content": "owned"}) == {"ok": True}

    normalized = {key.lower(): value for key, value in seen_headers[0].items()}
    assert "x-slm-user-session" not in normalized


def test_stop_never_scans_or_kills_machine_wide_processes() -> None:
    from superlocalmemory.cli import daemon

    descriptor = _owned_descriptor(port=43133)
    with (
        patch.object(
            daemon, "daemon_request", return_value={"status": "stopping"},
        ) as request,
        patch.object(
            daemon, "wait_for_owned_daemon_shutdown", return_value=True,
        ) as wait_for_shutdown,
        patch("psutil.process_iter") as process_iter,
        patch("subprocess.run") as subprocess_run,
    ):
        assert daemon.stop_daemon()

    # v4.1.4: verify_health=False -- stop_daemon() already proved process
    # ownership via _descriptor_process_is_alive() (real PID here, since
    # _owned_descriptor() records os.getpid()), so it no longer requires a
    # separate GET /health round trip to succeed before sending /stop. See
    # tests/test_cli/test_stop_daemon_busy_health.py for the full bug this
    # fixes: a busy-but-alive daemon's event loop can't answer /health, and
    # the old unconditional preflight made stop_daemon() falsely report the
    # daemon as not running.
    request.assert_called_once_with(
        "POST",
        "/stop",
        expected_descriptor=descriptor,
        verify_health=False,
    )
    wait_for_shutdown.assert_called_once_with(descriptor)
    process_iter.assert_not_called()
    subprocess_run.assert_not_called()
    assert descriptor.instance_id == "owned-instance"


def test_stop_without_owned_descriptor_fails_closed() -> None:
    from superlocalmemory.cli import daemon

    with (
        patch("psutil.process_iter") as process_iter,
        patch("subprocess.run") as subprocess_run,
    ):
        assert not daemon.stop_daemon()

    process_iter.assert_not_called()
    subprocess_run.assert_not_called()


def test_stop_waits_for_verified_legacy_pid_and_custom_port() -> None:
    """The compatibility stop path must not wait on the default namespace."""
    from superlocalmemory.cli import daemon

    legacy = {"status": "ok", "pid": 7771, "_legacy_port": 43134}
    with (
        patch.object(daemon, "read_descriptor", return_value=None),
        patch.object(daemon, "_verified_legacy_health", return_value=legacy),
        patch.object(
            daemon, "daemon_request", return_value={"status": "stopping"},
        ) as request,
        patch.object(
            daemon, "wait_for_owned_daemon_shutdown", return_value=True,
        ) as wait_for_shutdown,
    ):
        assert daemon.stop_daemon()

    wait_for_shutdown.assert_called_once_with(
        None,
        legacy_pid=7771,
        legacy_port=43134,
    )
    request.assert_called_once_with(
        "POST",
        "/stop",
        expected_legacy=legacy,
    )


def test_legacy_stop_request_refuses_descriptor_replacement() -> None:
    """A descriptor-backed replacement cannot be adopted by a legacy stop."""
    from superlocalmemory.cli import daemon

    captured = {"status": "ok", "pid": 7772, "_legacy_port": 43136}
    replacement = _owned_descriptor(port=43136)
    with (
        patch.object(daemon, "read_descriptor", return_value=replacement),
        patch_urlopen() as request,
    ):
        assert daemon.daemon_request(
            "POST",
            "/stop",
            expected_legacy=captured,
        ) is None

    request.assert_not_called()


def test_stop_request_stays_bound_to_captured_daemon_instance() -> None:
    """A replacement descriptor cannot redirect an in-progress stop."""
    from superlocalmemory.cli import daemon

    captured = _owned_descriptor(port=43135)
    replacement = build_descriptor(
        data_root=Path(os.environ["SLM_DATA_DIR"]),
        port=43135,
        version="3.8.11",
        pid=os.getpid(),
        instance_id="replacement-instance",
        capability="replacement-capability",
        state="ready",
    )
    write_descriptor(
        replacement,
        data_root=Path(os.environ["SLM_DATA_DIR"]),
    )
    replacement_health = {
        "status": "ok",
        **replacement.public_health_fields(),
    }

    with patch(
        "urllib.request.urlopen",
        return_value=_HealthResponse(replacement_health),
    ) as request:
        assert daemon.daemon_request(
            "POST",
            "/stop",
            expected_descriptor=captured,
        ) is None

    request.assert_called_once_with(
        "http://127.0.0.1:43135/health",
        # 4.1.22: the "nothing proves a start is in progress" health probe
        # reads with the same short, fixed budget as the starting-daemon
        # polling loop (STARTING_PROBE_S) only on Windows, where a closed
        # port cost the old, longer default the full read timeout on a
        # platform slow to refuse it. POSIX (this test's platform) keeps
        # the original 2 s budget -- see daemon_startup.health_probe_timeout
        # and its correction after this test briefly expected 0.5 s here.
        timeout=2.0,
    )


def test_pytest_isolation_cannot_spawn_a_daemon_without_fixture_opt_in(
    monkeypatch,
) -> None:
    from superlocalmemory.cli import daemon

    monkeypatch.delenv("SLM_TEST_ALLOW_DAEMON_SPAWN", raising=False)
    with patch.object(daemon, "_start_daemon_subprocess", return_value=True) as spawn:
        assert not daemon.ensure_daemon()
    spawn.assert_not_called()
