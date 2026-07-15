"""Focused contracts for the release-artifact test harness."""

from __future__ import annotations

import subprocess
import sys

from . import _harness


def test_candidate_artifacts_are_selected_from_prebuilt_directory(tmp_path) -> None:
    dist = tmp_path / "python"
    dist.mkdir()
    wheel = dist / "superlocalmemory-3.7.0-py3-none-any.whl"
    sdist = dist / "superlocalmemory-3.7.0.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")

    artifacts = _harness.candidate_artifacts_from_directory(_harness.REPO_ROOT, dist)

    assert artifacts.wheel == wheel
    assert artifacts.sdist == sdist
    assert artifacts.output_dir == dist


def test_create_venv_stdlib_fallback_avoids_ensurepip(
    tmp_path, monkeypatch,
) -> None:
    """The stdlib fallback must not invoke the ensurepip bootstrap."""
    path = tmp_path / "release-venv"
    python = (
        path / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else path / "bin" / "python"
    )

    class GuardedBuilder:
        def __init__(self, *args, **kwargs):
            assert kwargs.get("with_pip") is False
            assert kwargs.get("system_site_packages") is True

        def create(self, _path):
            python.parent.mkdir(parents=True)
            python.touch()

    monkeypatch.setattr(_harness.venv, "EnvBuilder", GuardedBuilder)
    monkeypatch.setattr(_harness.shutil, "which", lambda _name: None)
    monkeypatch.setattr(_harness, "_venv_has_pip", lambda *_args: (True, "pip 1"))

    assert _harness.create_venv(path) == python


def test_create_venv_prefers_offline_uv_for_standalone_python(
    tmp_path, monkeypatch,
) -> None:
    """An installed uv owns venv creation and is forbidden from networking."""
    path = tmp_path / "release-venv"
    python = (
        path / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else path / "bin" / "python"
    )
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        python.parent.mkdir(parents=True)
        python.touch()
        return subprocess.CompletedProcess(command, 0, "created", "")

    monkeypatch.setattr(_harness.shutil, "which", lambda name: "/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(_harness.subprocess, "run", fake_run)
    monkeypatch.setattr(_harness, "_venv_has_pip", lambda *_args: (True, "pip 1"))

    assert _harness.create_venv(path) == python
    assert commands
    assert commands[0][0:2] == ["/bin/uv", "venv"]
    assert "--offline" in commands[0]
    assert "--system-site-packages" in commands[0]
    assert "--seed" not in commands[0]


def test_create_venv_uses_offline_virtualenv_fallback(
    tmp_path, monkeypatch,
) -> None:
    """A local virtualenv install may recover, but downloads stay disabled."""
    path = tmp_path / "release-venv"
    python = (
        path / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else path / "bin" / "python"
    )

    class NoSeedBuilder:
        def __init__(self, *args, **kwargs):
            assert kwargs.get("with_pip") is False

        def create(self, _path):
            python.parent.mkdir(parents=True)
            python.touch()

    probes = iter(((False, "shared pip unavailable"), (True, "pip 1")))
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "created", "")

    monkeypatch.setattr(_harness.venv, "EnvBuilder", NoSeedBuilder)
    monkeypatch.setattr(_harness.shutil, "which", lambda _name: None)
    monkeypatch.setattr(_harness, "_venv_has_pip", lambda *_args: next(probes))
    monkeypatch.setattr(_harness.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(_harness.subprocess, "run", fake_run)

    assert _harness.create_venv(path) == python
    assert commands
    assert "--no-download" in commands[0]
    assert "--system-site-packages" in commands[0]


def test_create_venv_reports_offline_attempts_when_no_pip_provider(
    tmp_path, monkeypatch,
) -> None:
    """An unsupported host fails diagnostically instead of invoking ensurepip."""

    class BrokenBuilder:
        def __init__(self, *args, **kwargs):
            assert kwargs.get("with_pip") is False

        def create(self, _path):
            raise OSError("venv unavailable")

    monkeypatch.setattr(_harness.venv, "EnvBuilder", BrokenBuilder)
    monkeypatch.setattr(_harness.shutil, "which", lambda _name: None)
    monkeypatch.setattr(_harness.importlib.util, "find_spec", lambda _name: None)

    try:
        _harness.create_venv(tmp_path / "release-venv")
    except AssertionError as exc:
        message = str(exc)
    else:  # pragma: no cover - guards the assertion itself
        raise AssertionError("create_venv unexpectedly succeeded")

    assert "stdlib venv: OSError: venv unavailable" in message
    assert "uv venv: executable not installed" in message
    assert "virtualenv: module not installed" in message


def test_venv_pip_probe_converts_launch_failure_to_diagnostic(
    tmp_path, monkeypatch,
) -> None:
    """A CI process-launch failure must leave the fallback path available."""

    def fail_run(*_args, **_kwargs):
        raise OSError("process unavailable")

    monkeypatch.setattr(_harness.subprocess, "run", fail_run)

    available, detail = _harness._venv_has_pip(
        tmp_path / "python", tmp_path / "home",
    )

    assert available is False
    assert detail == "OSError: process unavailable"


def test_create_venv_reports_virtualenv_launch_failure(
    tmp_path, monkeypatch,
) -> None:
    """A failed fallback process is folded into the final harness evidence."""
    path = tmp_path / "release-venv"
    python = (
        path / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else path / "bin" / "python"
    )

    class NoSeedBuilder:
        def __init__(self, *args, **kwargs):
            assert kwargs.get("with_pip") is False

        def create(self, _path):
            python.parent.mkdir(parents=True)
            python.touch()

    monkeypatch.setattr(_harness.venv, "EnvBuilder", NoSeedBuilder)
    monkeypatch.setattr(_harness.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        _harness, "_venv_has_pip", lambda *_args: (False, "no shared pip"),
    )
    monkeypatch.setattr(_harness.importlib.util, "find_spec", lambda _name: object())

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["virtualenv"], 180)

    monkeypatch.setattr(_harness.subprocess, "run", timeout)

    try:
        _harness.create_venv(path)
    except AssertionError as exc:
        message = str(exc)
    else:  # pragma: no cover - guards the assertion itself
        raise AssertionError("create_venv unexpectedly succeeded")

    assert "virtualenv: TimeoutExpired" in message
