# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The JSON switch in a real ``slm`` process: stdout holds one document, nothing else.

A child process of its own (constructed environment, temporary HOME and data
folder), so the file-descriptor capture that catches a child program's output
is exercised too. Only commands that start nothing are run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests._portable import child_env_base

REPO = Path(__file__).resolve().parents[2]


def _slm(tmp_path: Path, *argv: str, **env: str) -> subprocess.CompletedProcess:
    child = {**os.environ, "PYTHONPATH": str(REPO / "src"), "SLM_DATA_DIR": str(tmp_path),
             **child_env_base(tmp_path / "home"), **env}
    child.pop("SLM_JSON", None)
    child.update(env)
    return subprocess.run([sys.executable, "-m", "superlocalmemory.cli.main", *argv],
                          env=child, capture_output=True, text=True, timeout=120)


def _one(stdout: str) -> dict:
    assert stdout.count("\n") == 1 and stdout.endswith("\n"), stdout
    doc = json.loads(stdout)
    assert {"success", "command", "version"} <= set(doc)
    return doc


def test_global_flag_in_a_real_process(tmp_path):
    r = _slm(tmp_path, "--json", "clear-cache")
    assert r.returncode == 0, r.stderr
    assert _one(r.stdout)["data"]["removed"] == []


def test_env_var_in_a_real_process(tmp_path):
    r = _slm(tmp_path, "help-optimize", SLM_JSON="1")
    assert r.returncode == 0, r.stderr
    assert "cache" in _one(r.stdout)["data"]["topics"]


def test_a_child_programs_output_is_captured_into_the_envelope(tmp_path):
    code = ("import subprocess, sys; from superlocalmemory.cli import json_mode; "
            "json_mode.run_json('demo', lambda: subprocess.run([sys.executable, '-c', "
            "'print(\"from the child\")'], check=True))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=60, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert r.returncode == 0, r.stderr
    assert _one(r.stdout)["data"]["output"] == "from the child"


def test_gdpr_export_in_a_real_process(tmp_path):
    """The export writes bytes to the real stdout buffer, inside its own envelope."""
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager

    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
               ("default", "Default Profile"))
    r = _slm(tmp_path, "--json", "gdpr", "export", "--profile", "default")
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["success"] is True and doc["data"]["profile_id"] == "default"


def _same_with_and_without_env(tmp_path, *argv: str, stdin: str = "{}") -> None:
    runs = []
    for env in ({}, {"SLM_JSON": "1"}):
        child = {**os.environ, "PYTHONPATH": str(REPO / "src"), "SLM_DATA_DIR": str(tmp_path),
                 **child_env_base(tmp_path / "home")}
        child.pop("SLM_JSON", None)
        child.update(env)
        runs.append(subprocess.run([sys.executable, "-m", "superlocalmemory.cli.main", *argv],
                                   env=child, input=stdin, capture_output=True, text=True,
                                   timeout=120))
    plain, with_env = runs
    assert with_env.returncode == plain.returncode
    assert with_env.stdout == plain.stdout, (plain.stdout, with_env.stdout)


def test_hook_start_is_byte_identical_under_the_env_var(tmp_path):
    _same_with_and_without_env(tmp_path, "hook", "start")


def test_hook_stop_is_byte_identical_under_the_env_var(tmp_path):
    _same_with_and_without_env(tmp_path, "hook", "stop")


def test_hook_session_close_is_byte_identical_under_the_env_var(tmp_path):
    """The Stop hook runs `slm session close 2>/dev/null || true` on the control channel."""
    _same_with_and_without_env(tmp_path, "session", "close")


def test_session_context_is_byte_identical_under_the_env_var(tmp_path):
    _same_with_and_without_env(tmp_path, "session-context")


def test_non_ascii_under_an_ascii_stdout():
    code = ("from superlocalmemory.cli import json_mode; "
            "json_mode.run_json('demo', lambda: print('caf\\u00e9 \\u2713'))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=60,
                       env={**os.environ, "PYTHONPATH": str(REPO / "src"),
                            "PYTHONIOENCODING": "ascii"})
    assert r.returncode == 0, r.stderr
    out = r.stdout.decode("ascii")
    assert _one(out)["data"]["output"] == "café ✓"


def test_a_tool_error_in_a_real_process(tmp_path):
    r = _slm(tmp_path, "--json", "tool", "no_such_tool_anywhere")
    assert r.returncode == 2
    doc = json.loads(r.stdout)
    assert doc["success"] is False and doc["code"] == "unknown_tool"
    assert doc["error"]["code"] == "unknown_tool"
