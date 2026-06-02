# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for isolated LightGBM work (OpenMP-crash guard).

The unified daemon serves the HTTP API in-process with PyTorch's OpenMP
runtime already loaded and warm. Importing lightgbm in that same process loads
a second libomp and SIGSEGVs the whole daemon. ``run_retrain_isolated`` and
``run_consolidation_isolated`` train out-of-process so that can never happen.

These tests lock in the load-bearing properties of the fix without depending
on a real training run:
  * the spawn imports lightgbm BEFORE the ``superlocalmemory`` package
    (the import ordering that avoids the crash) and never uses ``-m``;
  * a non-zero / crashing child is reported as an error, never raised, so the
    daemon stays up;
  * a well-formed JSON verdict is parsed and returned verbatim;
  * each task forwards the right CLI flags.
"""

from __future__ import annotations

import json

from superlocalmemory.learning import lightgbm_subprocess as rs


class _FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _capture_cmd(monkeypatch, stdout):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc(stdout=stdout)

    monkeypatch.setattr(rs.subprocess, "run", fake_run)
    return captured


# ---------------------------------------------------------------------------
# Import ordering — the actual crash guard.
# ---------------------------------------------------------------------------

def test_bootstrap_imports_lightgbm_before_package(monkeypatch):
    """Child must import lightgbm first — torch-OMP-first crashes."""
    captured = _capture_cmd(monkeypatch, json.dumps({"trained": False, "error": None}))
    rs.run_retrain_isolated("/tmp/learning.db", "default")

    cmd = captured["cmd"]
    assert "-c" in cmd
    assert "-m" not in cmd  # -m would run the package __init__ (torch) first
    bootstrap = cmd[cmd.index("-c") + 1]
    assert bootstrap.index("import lightgbm") < bootstrap.index("superlocalmemory")


def test_consolidate_bootstrap_also_lightgbm_first(monkeypatch):
    captured = _capture_cmd(monkeypatch, json.dumps({"stats": {}, "error": None}))
    rs.run_consolidation_isolated("/tmp/mem.db", "/tmp/learning.db", "default")

    cmd = captured["cmd"]
    assert "-c" in cmd and "-m" not in cmd
    bootstrap = cmd[cmd.index("-c") + 1]
    assert bootstrap.index("import lightgbm") < bootstrap.index("superlocalmemory")


# ---------------------------------------------------------------------------
# Flag forwarding.
# ---------------------------------------------------------------------------

def test_retrain_flags_forwarded(monkeypatch):
    captured = _capture_cmd(monkeypatch, json.dumps({"trained": True, "error": None}))
    rs.run_retrain_isolated("/tmp/learning.db", "p1", include_synthetic=True)
    cmd = captured["cmd"]
    assert "--task" in cmd and cmd[cmd.index("--task") + 1] == "retrain"
    assert "--learning-db" in cmd and "/tmp/learning.db" in cmd
    assert "--profile" in cmd and "p1" in cmd
    assert "--include-synthetic" in cmd

    captured = _capture_cmd(monkeypatch, json.dumps({"trained": False, "error": None}))
    rs.run_retrain_isolated("/tmp/learning.db", "p1", include_synthetic=False)
    assert "--include-synthetic" not in captured["cmd"]


def test_consolidate_flags_forwarded(monkeypatch):
    captured = _capture_cmd(monkeypatch, json.dumps({"stats": {"retrained": True}, "error": None}))
    rs.run_consolidation_isolated("/tmp/mem.db", "/tmp/learning.db", "p2", dry_run=True)
    cmd = captured["cmd"]
    assert cmd[cmd.index("--task") + 1] == "consolidate"
    assert "--memory-db" in cmd and "/tmp/mem.db" in cmd
    assert "--learning-db" in cmd and "/tmp/learning.db" in cmd
    assert "--profile" in cmd and "p2" in cmd
    assert "--dry-run" in cmd


# ---------------------------------------------------------------------------
# Failure handling — never raise, always keep the daemon alive.
# ---------------------------------------------------------------------------

def test_native_crash_reported_not_raised(monkeypatch):
    """A SIGSEGV in the child (no JSON, negative exit) → error dict, no raise."""
    monkeypatch.setattr(
        rs.subprocess, "run",
        lambda cmd, **kw: _FakeProc(stdout="", stderr="boom", returncode=-11),
    )
    result = rs.run_retrain_isolated("/tmp/learning.db", "default")
    assert result["trained"] is False
    assert "no verdict" in result["error"] and "exit=-11" in result["error"]


def test_consolidate_native_crash_reported(monkeypatch):
    monkeypatch.setattr(
        rs.subprocess, "run",
        lambda cmd, **kw: _FakeProc(stdout="", stderr="segfault", returncode=-11),
    )
    result = rs.run_consolidation_isolated("/tmp/mem.db", "/tmp/learning.db", "default")
    assert result["stats"] is None
    assert "no verdict" in result["error"]


def test_timeout_reported_not_raised(monkeypatch):
    import subprocess as _sp

    def fake_run(cmd, **kwargs):
        raise _sp.TimeoutExpired(cmd, kwargs.get("timeout", 1))

    monkeypatch.setattr(rs.subprocess, "run", fake_run)
    result = rs.run_retrain_isolated("/tmp/learning.db", "default", timeout_sec=7)
    assert result["trained"] is False and "timed out" in result["error"]


def test_verdict_parsed_from_last_json_line(monkeypatch):
    """Deprecation warnings / log noise before the JSON verdict are ignored."""
    noisy = (
        "DeprecationWarning: ranker_retrain_legacy ...\n"
        "some other log line\n"
        + json.dumps({"trained": True, "error": None}) + "\n"
    )
    monkeypatch.setattr(rs.subprocess, "run", lambda cmd, **kw: _FakeProc(stdout=noisy))
    result = rs.run_retrain_isolated("/tmp/learning.db", "default")
    assert result == {"trained": True, "error": None}


# ---------------------------------------------------------------------------
# Child dispatch (main) — exercised in-process; safe because these tasks
# short-circuit before any heavy training on an empty/missing DB.
# ---------------------------------------------------------------------------

def test_main_consolidate_requires_memory_db(capsys):
    rc = rs.main(["--task", "consolidate", "--learning-db", "/tmp/x.db"])
    assert rc == 1
    verdict = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert "--memory-db" in verdict["error"]
