# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Stage 8 SB-5

"""Tests for the CLI escape-hatch commands (disable / enable / clear-cache /
reconfigure / benchmark).

These call the handler functions directly with a namespace — the full
argparse round-trip is covered by a separate smoke test that just
confirms the subparser exists.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from superlocalmemory.cli import escape_hatch as eh


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "slm_home"
    home.mkdir()
    monkeypatch.setenv("SLM_HOME", str(home))
    monkeypatch.delenv("SLM_DISABLE", raising=False)
    return home


def _ns(**kwargs: object) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def test_disable_writes_marker(
    tmp_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    # Stub out the daemon stop helper to avoid touching the real daemon.
    with mock.patch("superlocalmemory.cli.daemon.stop_daemon",
                    return_value=False):
        eh.cmd_disable(_ns(reason="testing"))
    assert (tmp_home / ".disabled").exists()
    captured = capsys.readouterr().out
    assert "SLM disabled" in captured


def test_enable_removes_marker(
    tmp_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_home / ".disabled").write_text("x", encoding="utf-8")
    eh.cmd_enable(_ns())
    assert not (tmp_home / ".disabled").exists()
    captured = capsys.readouterr().out
    assert "enabled" in captured.lower()


def test_enable_idempotent(
    tmp_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    eh.cmd_enable(_ns())
    captured = capsys.readouterr().out
    assert "already enabled" in captured.lower()


def test_clear_cache_removes_caches_only(
    tmp_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    for name in ("active_brain_cache.db", "context_cache.db",
                 "entity_trigram_cache.db"):
        (tmp_home / name).write_text("x", encoding="utf-8")
    # User memories MUST NOT be deleted.
    (tmp_home / "memory.db").write_text("precious", encoding="utf-8")
    (tmp_home / "learning.db").write_text("signals", encoding="utf-8")

    eh.cmd_clear_cache(_ns())

    assert not (tmp_home / "active_brain_cache.db").exists()
    assert not (tmp_home / "context_cache.db").exists()
    assert not (tmp_home / "entity_trigram_cache.db").exists()
    # Sacred:
    assert (tmp_home / "memory.db").read_text() == "precious"
    assert (tmp_home / "learning.db").read_text() == "signals"
    captured = capsys.readouterr().out
    assert "memory.db" in captured  # confirms preservation note printed


def test_clear_cache_noop_when_empty(
    tmp_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    eh.cmd_clear_cache(_ns())
    captured = capsys.readouterr().out
    assert "nothing to do" in captured.lower()


def test_benchmark_prints_mrr_and_lift(
    tmp_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    try:
        eh.cmd_benchmark(_ns(json=False))
    except SystemExit:
        # Acceptable if harness not bundled; test the json path below.
        pytest.skip("evo_memory harness not discoverable from test cwd")
    out = capsys.readouterr().out
    assert "MRR" in out or "mrr" in out
    assert "lift" in out


def test_benchmark_json_path(
    tmp_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    try:
        eh.cmd_benchmark(_ns(json=True))
    except SystemExit:
        pytest.skip("evo_memory harness not discoverable from test cwd")
    import json
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "comparison" in parsed


def test_subparsers_registered() -> None:
    """End-to-end: argparse accepts each escape-hatch subcommand."""
    result = subprocess.run(
        [sys.executable, "-m", "superlocalmemory.cli.main", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    help_text = result.stdout
    for word in ("disable", "enable", "clear-cache",
                 "reconfigure", "benchmark"):
        assert word in help_text, f"{word!r} missing from top-level help"
