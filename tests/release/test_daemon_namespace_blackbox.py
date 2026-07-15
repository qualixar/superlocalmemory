"""Installed-wheel black box for two independent daemon namespaces."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

import psutil
import pytest

from ._harness import (
    BuiltArtifacts,
    create_venv,
    install_artifact,
    run_checked,
    safe_child_env,
)

pytestmark = pytest.mark.slow

_PUBLIC_PORTS = {8765, 8767}
_LIFECYCLE_NAMES = {
    ".install_token",
    ".setup-complete",
    "config.json",
    "daemon.json",
    "daemon.pid",
    "daemon.port",
    "daemon.lock",
    "daemon.log",
    "daemon-error.log",
}


def _random_private_port(excluded: set[int]) -> int:
    """Ask the kernel for a loopback port, excluding public/default ports."""
    for _ in range(20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = int(listener.getsockname()[1])
        if port not in excluded and port not in _PUBLIC_PORTS:
            return port
    raise AssertionError("could not allocate an isolated daemon port")


def _daemon_env(home: Path, port: int) -> dict[str, str]:
    env = safe_child_env(home)
    cache_root = home / "slm-data" / "cache"
    env.update(
        {
            "CI": "1",
            "SLM_NON_INTERACTIVE": "1",
            "SLM_TEST_ISOLATION": "1",
            "SLM_TEST_ALLOW_DAEMON_SPAWN": "1",
            "SLM_DAEMON_PORT": str(port),
            "OMP_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            # Installed-artifact verification must be deterministic and must
            # never fetch models.  Keep any library cache metadata inside the
            # fixture-owned namespace so the root-locality audit remains real.
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": str(cache_root / "huggingface"),
            "SENTENCE_TRANSFORMERS_HOME": str(
                cache_root / "sentence-transformers"
            ),
            "XDG_CACHE_HOME": str(cache_root),
        }
    )
    return env


def _cli(
    python: Path,
    work_dir: Path,
    env: dict[str, str],
    *arguments: str,
    timeout: int = 150,
) -> subprocess.CompletedProcess[str]:
    return run_checked(
        [str(python), "-I", "-m", "superlocalmemory.cli.main", *arguments],
        cwd=work_dir,
        env=env,
        timeout=timeout,
    )


def _read_descriptor(data_root: Path) -> dict[str, Any]:
    path = data_root / "daemon.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _health(port: int) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/health"
    with urllib.request.urlopen(url, timeout=5) as response:
        assert response.status == 200
        assert response.geturl() == url
        payload = json.loads(response.read().decode("utf-8"))
    assert isinstance(payload, dict)
    return payload


def _same_owned_process(descriptor: dict[str, Any]) -> psutil.Process | None:
    """Resolve only the exact PID plus creation-time identity in a descriptor."""
    try:
        process = psutil.Process(int(descriptor["pid"]))
        if abs(process.create_time() - float(descriptor["process_create_time"])) > 1.0:
            return None
        return process
    except (KeyError, TypeError, ValueError, psutil.Error):
        return None


def _wait_owned_exit(descriptor: dict[str, Any], timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _same_owned_process(descriptor) is None:
            return
        time.sleep(0.1)
    raise AssertionError(f"owned daemon PID {descriptor.get('pid')} did not exit")


def _terminate_owned_tree(descriptor: dict[str, Any]) -> None:
    """Last-resort cleanup scoped to one descriptor-owned process tree."""
    process = _same_owned_process(descriptor)
    if process is None:
        return
    try:
        children = process.children(recursive=True)
    except psutil.Error:
        children = []
    for owned in [*children, process]:
        try:
            owned.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs([*children, process], timeout=10)
    for owned in alive:
        try:
            owned.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=5)


def _root_local_state(home: Path, data_root: Path) -> list[Path]:
    state = []
    for path in home.rglob("*"):
        if not path.is_file():
            continue
        if path.name in _LIFECYCLE_NAMES or path.suffix in {".db", ".sqlite", ".sqlite3"}:
            state.append(path)
            assert path.is_relative_to(data_root), f"state escaped data root: {path}"
    return state


def _assert_running_namespace(
    *,
    home: Path,
    data_root: Path,
    port: int,
) -> dict[str, Any]:
    descriptor = _read_descriptor(data_root)
    assert Path(descriptor["data_root"]).resolve() == data_root.resolve()
    assert descriptor["port"] == port
    assert descriptor["port"] not in _PUBLIC_PORTS
    assert descriptor["state"] == "ready"
    assert _same_owned_process(descriptor) is not None
    assert (data_root / "daemon.pid").read_text(encoding="utf-8").strip() == str(
        descriptor["pid"]
    )
    assert (data_root / "daemon.port").read_text(encoding="utf-8").strip() == str(port)
    assert (data_root / "logs" / "daemon.log").is_file()
    assert (data_root / "memory.db").is_file()
    assert (data_root / ".setup-complete").is_file()
    assert not (home / ".superlocalmemory").exists()
    if os.name != "nt":
        assert (data_root / "daemon.json").stat().st_mode & 0o777 == 0o600

    health = _health(port)
    for field in (
        "daemon_protocol",
        "namespace_id",
        "instance_id",
        "capability_fingerprint",
        "owner_id",
        "pid",
        "port",
        "version",
    ):
        assert str(health[field]) == str(descriptor[field])

    state = _root_local_state(home, data_root)
    assert data_root / "daemon.json" in state
    assert data_root / "daemon.pid" in state
    assert data_root / "daemon.port" in state
    assert data_root / "logs" / "daemon.log" in state
    assert data_root / "memory.db" in state
    return descriptor


def _assert_stopped_namespace(data_root: Path) -> None:
    """A graceful stop removes only ephemeral identity, not durable state."""
    for name in ("daemon.json", "daemon.pid", "daemon.port"):
        assert not (data_root / name).exists(), f"stale lifecycle state: {name}"
    assert (data_root / "memory.db").is_file()
    assert (data_root / "logs" / "daemon.log").is_file()


def test_installed_wheel_two_home_daemon_lifecycle_is_root_local(
    built_artifacts: BuiltArtifacts,
    tmp_path: Path,
) -> None:
    """Start, inspect, restart, and stop two wheel-installed daemons safely."""
    venv_root = tmp_path / "venv-wheel-daemon"
    python = create_venv(venv_root)
    work_dir = tmp_path / "outside-checkout"
    work_dir.mkdir()
    install_artifact(python, built_artifacts.wheel, work_dir)

    port_a = _random_private_port(set(_PUBLIC_PORTS))
    port_b = _random_private_port({*_PUBLIC_PORTS, port_a})
    home_a = tmp_path / "home-a"
    home_b = tmp_path / "home-b"
    root_a = home_a / "slm-data"
    root_b = home_b / "slm-data"
    env_a = _daemon_env(home_a, port_a)
    env_b = _daemon_env(home_b, port_b)
    owned: list[dict[str, Any]] = []

    try:
        start_a = _cli(python, work_dir, env_a, "serve", "start")
        assert "Daemon started" in start_a.stdout, start_a.stdout + start_a.stderr
        descriptor_a = _assert_running_namespace(
            home=home_a, data_root=root_a, port=port_a,
        )
        owned.append(descriptor_a)

        start_b = _cli(python, work_dir, env_b, "serve", "start")
        assert "Daemon started" in start_b.stdout, start_b.stdout + start_b.stderr
        descriptor_b = _assert_running_namespace(
            home=home_b, data_root=root_b, port=port_b,
        )
        owned.append(descriptor_b)

        assert descriptor_a["pid"] != descriptor_b["pid"]
        assert descriptor_a["instance_id"] != descriptor_b["instance_id"]
        assert descriptor_a["namespace_id"] != descriptor_b["namespace_id"]
        assert json.loads((root_a / "daemon.json").read_text())["port"] == port_a
        assert json.loads((root_b / "daemon.json").read_text())["port"] == port_b
        assert not [
            path
            for path in work_dir.rglob("*")
            if path.is_file()
            and (
                path.name in _LIFECYCLE_NAMES
                or path.suffix in {".db", ".sqlite", ".sqlite3"}
            )
        ], "daemon state escaped into the process working directory"

        status_a = _cli(python, work_dir, env_a, "serve", "status")
        status_b = _cli(python, work_dir, env_b, "serve", "status")
        assert f"PID {descriptor_a['pid']}" in status_a.stdout
        assert f"PID {descriptor_b['pid']}" in status_b.stdout

        restart_a = _cli(
            python, work_dir, env_a, "restart", "--json", timeout=150,
        )
        _wait_owned_exit(descriptor_a)
        restarted_a = _assert_running_namespace(
            home=home_a, data_root=root_a, port=port_a,
        )
        owned.append(restarted_a)
        assert restarted_a["pid"] != descriptor_a["pid"]
        assert restarted_a["instance_id"] != descriptor_a["instance_id"]
        assert '"success": true' in restart_a.stdout.lower(), restart_a.stdout

        # Restarting A must not disturb B's process or namespace.
        assert _read_descriptor(root_b) == descriptor_b
        assert _same_owned_process(descriptor_b) is not None
        status_b_after = _cli(python, work_dir, env_b, "serve", "status")
        assert f"PID {descriptor_b['pid']}" in status_b_after.stdout

        stop_a = _cli(python, work_dir, env_a, "serve", "stop")
        assert "Daemon stopped" in stop_a.stdout, stop_a.stdout + stop_a.stderr
        _wait_owned_exit(restarted_a)
        _assert_stopped_namespace(root_a)
        assert _same_owned_process(descriptor_b) is not None

        stop_b = _cli(python, work_dir, env_b, "serve", "stop")
        assert "Daemon stopped" in stop_b.stdout, stop_b.stdout + stop_b.stderr
        _wait_owned_exit(descriptor_b)
        _assert_stopped_namespace(root_b)
    finally:
        # Never inspect or kill by process name. Cleanup is restricted to the
        # exact PIDs and creation times published inside these fixture roots.
        for root in (root_a, root_b):
            try:
                current = _read_descriptor(root)
            except (OSError, ValueError, AssertionError, json.JSONDecodeError):
                continue
            owned.append(current)
        for descriptor in reversed(owned):
            _terminate_owned_tree(descriptor)
