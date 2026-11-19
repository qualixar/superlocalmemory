# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The two Windows-aware helpers in tests/_portable.py answer correctly."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests import _portable

ROOT = Path(__file__).resolve().parents[1]


def test_a_found_bash_really_runs_posix_shell():
    bash = _portable.posix_bash()
    if bash is None:
        pytest.skip("no bash on this machine")
    done = subprocess.run([bash, "-c", 'printf "%s" "$((6 * 7))"'],
                          capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.stdout == "42"


def test_the_wsl_launcher_under_the_windows_folder_is_refused(tmp_path, monkeypatch):
    windows = tmp_path / "Windows"
    (windows / "System32").mkdir(parents=True)
    launcher = windows / "System32" / "bash.exe"
    launcher.write_bytes(b"")
    monkeypatch.setenv("SystemRoot", str(windows))
    assert _portable._is_wsl_launcher(launcher)
    assert not _portable._is_wsl_launcher(tmp_path / "Git" / "bin" / "bash.exe")


@pytest.mark.parametrize("has_bit", [True, False], ids=["posix", "windows"])
def test_the_shipped_executable_bit_is_read_on_either_platform(monkeypatch, has_bit):
    if not (ROOT / ".git").exists():
        pytest.skip("needs a git checkout")
    monkeypatch.setattr(_portable, "HAS_EXECUTABLE_BIT", has_bit)
    if has_bit and _portable.os.name == "nt":
        pytest.skip("Windows files have no executable bit to read")
    assert _portable.committed_executable(ROOT / "plugin" / "scripts" / "slm-launch")
    assert not _portable.committed_executable(ROOT / "README.md")


def test_on_windows_the_bit_is_the_one_git_recorded(tmp_path, monkeypatch):
    """The file on disk says not executable; Git says it is. On Windows (no
    bit on disk) Git is the one to believe."""
    script = tmp_path / "launch"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    script.chmod(0o644)
    for args in (["init", "-q"], ["add", "launch"], ["update-index", "--chmod=+x", "launch"]):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.setattr(_portable, "HAS_EXECUTABLE_BIT", False)
    assert _portable.committed_executable(script)


def test_a_windows_child_environment_keeps_what_windows_needs(monkeypatch, tmp_path):
    monkeypatch.setattr(_portable.os, "name", "nt")
    monkeypatch.setenv("SYSTEMROOT", r"C:\Windows")
    monkeypatch.setenv("TEMP", r"C:\Temp")
    env = _portable.child_env_base(tmp_path)
    assert env["SYSTEMROOT"] == r"C:\Windows"  # Winsock needs it (WinError 10106)
    assert env["TEMP"] == r"C:\Temp"
    assert env["USERPROFILE"] == env["HOME"] == str(tmp_path)  # Path.home() there


def test_a_posix_child_environment_is_only_home_and_the_guards(monkeypatch, tmp_path):
    monkeypatch.setattr(_portable.os, "name", "posix")
    monkeypatch.setenv("SYSTEMROOT", r"C:\Windows")
    assert _portable.child_env_base(tmp_path) == {"HOME": str(tmp_path), **_portable.GUARD_ENV}
