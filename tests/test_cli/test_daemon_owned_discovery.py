# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Client discovery must attach only to the descriptor-owned daemon."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from superlocalmemory.infra.daemon_identity import (
    build_descriptor,
    read_descriptor,
    write_descriptor,
)


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
    with patch("urllib.request.urlopen", return_value=_HealthResponse(health)) as request:
        assert daemon.is_daemon_running()
    assert request.call_count == 1
    assert ":43123/health" in request.call_args.args[0]


def test_foreign_health_is_rejected_without_rewriting_local_state() -> None:
    from superlocalmemory.cli import daemon

    descriptor = _owned_descriptor()
    original = Path(os.environ["SLM_DATA_DIR"], "daemon.json").read_text()
    health = {"status": "ok", **descriptor.public_health_fields()}
    health["namespace_id"] = "foreign"

    with patch("urllib.request.urlopen", return_value=_HealthResponse(health)):
        assert not daemon.is_daemon_running()

    assert Path(os.environ["SLM_DATA_DIR"], "daemon.json").read_text() == original


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

    with patch("urllib.request.urlopen", side_effect=_foreign_only):
        assert not daemon.is_daemon_running()

    assert calls == ["http://127.0.0.1:43124/health"]


def test_health_redirect_to_another_origin_is_rejected() -> None:
    from superlocalmemory.cli import daemon

    descriptor = _owned_descriptor(port=43124)
    health = {"status": "ok", **descriptor.public_health_fields()}
    redirected = _HealthResponse(health, final_url="http://127.0.0.1:8767/health")

    with patch("urllib.request.urlopen", return_value=redirected):
        assert not daemon.is_daemon_running()


def test_arbitrary_live_pid_without_descriptor_is_rejected() -> None:
    from superlocalmemory.cli import daemon

    root = Path(os.environ["SLM_DATA_DIR"])
    (root / "daemon.pid").write_text(str(os.getpid()))
    (root / "daemon.port").write_text("43125")

    with patch.object(daemon, "_is_verified_legacy_process", return_value=False):
        assert not daemon.is_daemon_running()


def test_reused_pid_with_wrong_process_creation_time_is_rejected() -> None:
    from superlocalmemory.cli import daemon

    root = Path(os.environ["SLM_DATA_DIR"])
    descriptor = build_descriptor(
        data_root=root,
        port=43125,
        version="3.7.0a1",
        pid=os.getpid(),
        process_create_time=0.0,
        instance_id="stale-process",
        capability="stale-capability",
        state="ready",
    )
    write_descriptor(descriptor, data_root=root)

    with patch("urllib.request.urlopen") as request:
        assert not daemon.is_daemon_running()
    request.assert_not_called()


def test_get_port_uses_only_valid_owned_descriptor() -> None:
    from superlocalmemory.cli import daemon

    _owned_descriptor(port=43126)
    assert daemon._get_port() == 43126

    Path(os.environ["SLM_DATA_DIR"], "daemon.json").write_text("malformed")
    assert daemon._get_port() == daemon._DEFAULT_PORT


def test_launcher_passes_and_publishes_one_process_identity() -> None:
    from superlocalmemory.cli import daemon

    fake_process = MagicMock(pid=54321)
    with (
        patch.object(daemon, "is_daemon_running", return_value=False),
        patch.object(daemon, "_wait_for_daemon", return_value=True),
        patch("subprocess.Popen", return_value=fake_process) as popen,
    ):
        assert daemon._start_daemon_subprocess()

    child_env = popen.call_args.kwargs["env"]
    descriptor = read_descriptor()
    assert descriptor is not None
    assert descriptor.pid == 54321
    assert descriptor.state == "starting"
    assert child_env["SLM_DAEMON_INSTANCE_ID"] == descriptor.instance_id
    assert child_env["SLM_DAEMON_CAPABILITY"] == descriptor.capability


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
        patch("urllib.request.urlopen", return_value=_HealthResponse(foreign)),
        patch("time.sleep"),
    ):
        assert not daemon._wait_for_daemon(timeout=1)


def test_daemon_request_refuses_foreign_identity_before_write() -> None:
    from superlocalmemory.cli import daemon

    descriptor = _owned_descriptor(port=43129)
    foreign = {"status": "ok", **descriptor.public_health_fields()}
    foreign["namespace_id"] = "foreign"

    with patch("urllib.request.urlopen", return_value=_HealthResponse(foreign)) as request:
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

    with patch("urllib.request.urlopen", side_effect=_respond):
        result = daemon.daemon_request("POST", "/remember", {"content": "owned"})

    assert result == {"ok": True}
    normalized = {key.lower(): value for key, value in seen_headers[0].items()}
    assert normalized["x-slm-daemon-capability"] == descriptor.capability
    assert normalized["x-slm-target-instance"] == descriptor.instance_id


def test_stop_never_scans_or_kills_machine_wide_processes() -> None:
    from superlocalmemory.cli import daemon

    descriptor = _owned_descriptor(port=43133)
    with (
        patch.object(
            daemon, "daemon_request", return_value={"status": "stopping"},
        ) as request,
        patch("psutil.process_iter") as process_iter,
        patch("subprocess.run") as subprocess_run,
    ):
        assert daemon.stop_daemon()

    request.assert_called_once_with("POST", "/stop")
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


def test_pytest_isolation_cannot_spawn_a_daemon_without_fixture_opt_in(
    monkeypatch,
) -> None:
    from superlocalmemory.cli import daemon

    monkeypatch.delenv("SLM_TEST_ALLOW_DAEMON_SPAWN", raising=False)
    with patch.object(daemon, "_start_daemon_subprocess", return_value=True) as spawn:
        assert not daemon.ensure_daemon()
    spawn.assert_not_called()
