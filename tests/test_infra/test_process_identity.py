# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Clock-independent process identity (issue #104).

The Linux/WSL2 scheme is exercised against a synthetic procfs so it is covered
on every platform, including the macOS and Windows CI runners that have no
``/proc`` at all.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from superlocalmemory.infra import process_identity as pi

BOOT_ID = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"


def _fake_procfs(tmp_path: Path, pid: int, *, comm: str, starttime: int) -> Path:
    """Build the two procfs files the Linux scheme reads."""
    root = tmp_path / "proc"
    (root / "sys" / "kernel" / "random").mkdir(parents=True, exist_ok=True)
    (root / "sys" / "kernel" / "random" / "boot_id").write_text(BOOT_ID + "\n")
    proc_dir = root / str(pid)
    proc_dir.mkdir(parents=True, exist_ok=True)
    # "man proc" field order; starttime is field 22 (index 19 after comm).
    fields = [str(n) for n in range(3, 53)]
    fields[22 - 3] = str(starttime)
    proc_dir.joinpath("stat").write_text(
        f"{pid} ({comm}) S " + " ".join(fields[1:]) + "\n",
    )
    return root


def _force_linux(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(pi, "_PROCFS", root)
    monkeypatch.setattr(sys, "platform", "linux")


def test_linux_token_is_boot_id_plus_start_ticks(tmp_path, monkeypatch) -> None:
    root = _fake_procfs(tmp_path, 612, comm="python3", starttime=44219)
    _force_linux(monkeypatch, root)

    assert pi.process_start_token_for(612) == f"lx1:{BOOT_ID}:44219"


def test_linux_token_parses_a_comm_containing_spaces_and_parens(
    tmp_path, monkeypatch,
) -> None:
    """The comm field is attacker-adjacent: a daemon can be renamed."""
    root = _fake_procfs(
        tmp_path, 613, comm="sl m (weird) proc", starttime=987654,
    )
    _force_linux(monkeypatch, root)

    assert pi.process_start_token_for(613) == f"lx1:{BOOT_ID}:987654"


def test_linux_token_does_not_move_when_the_wall_clock_moves(
    tmp_path, monkeypatch,
) -> None:
    """The whole point: nothing about the token is derived from btime."""
    root = _fake_procfs(tmp_path, 612, comm="python3", starttime=44219)
    _force_linux(monkeypatch, root)
    before = pi.process_start_token_for(612)

    # Simulate a WSL2 host resync: btime slides, /proc/<pid>/stat does not.
    (root / "stat").write_text("btime 1785680130\n")
    after = pi.process_start_token_for(612)
    (root / "stat").write_text("btime 1785680165\n")
    later = pi.process_start_token_for(612)

    assert before == after == later


def test_linux_token_changes_across_reboots(tmp_path, monkeypatch) -> None:
    root = _fake_procfs(tmp_path, 612, comm="python3", starttime=44219)
    _force_linux(monkeypatch, root)
    first = pi.process_start_token_for(612)

    (root / "sys" / "kernel" / "random" / "boot_id").write_text(
        "9c858901-8a57-4791-81fe-4c455b099bc9\n",
    )
    second = pi.process_start_token_for(612)

    assert first != second


def test_missing_boot_id_yields_no_token_rather_than_a_weak_one(
    tmp_path, monkeypatch,
) -> None:
    root = _fake_procfs(tmp_path, 612, comm="python3", starttime=44219)
    (root / "sys" / "kernel" / "random" / "boot_id").unlink()
    _force_linux(monkeypatch, root)
    monkeypatch.setattr(pi, "_monotonic_start_token", lambda pid: None)

    assert pi.process_start_token_for(612) is None


def test_unknown_pid_yields_no_token(tmp_path, monkeypatch) -> None:
    root = _fake_procfs(tmp_path, 612, comm="python3", starttime=44219)
    _force_linux(monkeypatch, root)
    monkeypatch.setattr(pi, "_monotonic_start_token", lambda pid: None)

    assert pi.process_start_token_for(999999) is None


def test_truncated_stat_line_yields_no_token(tmp_path, monkeypatch) -> None:
    root = _fake_procfs(tmp_path, 612, comm="python3", starttime=44219)
    (root / "612" / "stat").write_text("612 (python3) S 1 2 3\n")
    _force_linux(monkeypatch, root)
    monkeypatch.setattr(pi, "_monotonic_start_token", lambda pid: None)

    assert pi.process_start_token_for(612) is None


@pytest.mark.parametrize("pid", [0, -1, None, "not-a-pid"])
def test_invalid_pids_are_rejected(pid) -> None:
    assert pi.process_start_token_for(pid) is None


def test_this_platform_produces_a_usable_or_absent_token() -> None:
    """Whatever this OS supports, the contract holds: str or None."""
    token = pi.process_start_token_for(os.getpid())
    assert token is None or (
        isinstance(token, str) and token.split(":", 1)[0] in {"lx1", "mn1"}
    )
    if token is not None:
        # Stable across repeated observation of the same live process.
        assert pi.process_start_token_for(os.getpid()) == token


@pytest.mark.parametrize(
    ("recorded", "observed", "expected"),
    [
        ("lx1:b:1", "lx1:b:1", True),
        ("lx1:b:1", "lx1:b:2", False),
        ("lx1:b:1", "lx1:c:1", False),
        ("mn1:1.5", "mn1:1.5", True),
        ("mn1:1.5", "mn1:1.6", False),
        ("lx1:b:1", "mn1:1.5", None),
        (None, "mn1:1.5", None),
        ("mn1:1.5", None, None),
        (None, None, None),
        ("", "mn1:1.5", None),
        (123, "mn1:1.5", None),
    ],
)
def test_compare_start_tokens_is_tri_state(recorded, observed, expected) -> None:
    assert pi.compare_start_tokens(recorded, observed) is expected
