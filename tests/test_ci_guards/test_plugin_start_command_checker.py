# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The release contract checker understands the one-entry-per-platform plugin
start command from #139, and still rejects a malformed one.

Before this, the checker accepted only the old POSIX-only shape, so five
release-gate tests failed on every tree after #139 shipped.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import scripts.integration_compatibility as compat
from scripts.integration_compatibility import _plugin_package

ROOT = Path(__file__).resolve().parents[2]


def _repo(tmp_path: Path, block: dict) -> Path:
    (tmp_path / "plugin" / "scripts").mkdir(parents=True)
    (tmp_path / "plugin-src").mkdir()
    shutil.copy2(ROOT / "plugin" / "scripts" / "slm-launch",
                 tmp_path / "plugin" / "scripts" / "slm-launch")
    text = json.dumps({"mcpServers": {"superlocalmemory": block}})
    for rel in ("plugin/.mcp.json", "plugin-src/.mcp.json"):
        (tmp_path / rel).write_text(text, encoding="utf-8")
    if not compat.HAS_EXECUTABLE_BIT:
        _record_in_git(tmp_path, "plugin/scripts/slm-launch", executable=True)
    return tmp_path


def _record_in_git(repo: Path, rel: str, *, executable: bool) -> None:
    """Record ``rel`` in a git index, the only place Windows keeps its +x."""
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    git("init", "-q")
    git("add", "--", rel)
    git("update-index", f"--chmod={'+' if executable else '-'}x", "--", rel)


def _shipped() -> dict:
    data = json.loads((ROOT / "plugin" / ".mcp.json").read_text(encoding="utf-8"))
    return data["mcpServers"]["superlocalmemory"]


def test_the_shipped_entry_is_accepted(tmp_path) -> None:
    assert _plugin_package(_repo(tmp_path, _shipped()), "claude-code", tmp_path) \
        == "plugin/scripts/slm-launch"


@pytest.mark.parametrize("mutate", [
    lambda b: b["args"].__setitem__(3, "echo hello"),       # never starts slm mcp
    lambda b: b["args"].append("extra"),
    lambda b: b.__setitem__("args", []),                     # cmd.exe with nothing to run
    lambda b: b.__setitem__("command", "${ComSpec:-/usr/bin/evil}"),
])
def test_a_malformed_entry_is_rejected(tmp_path, mutate) -> None:
    block = _shipped()
    mutate(block)
    with pytest.raises(AssertionError, match="invalid Claude plugin start command"):
        _plugin_package(_repo(tmp_path, block), "claude-code", tmp_path)


@pytest.mark.parametrize("executable", [True, False])
def test_on_windows_the_bit_comes_from_git(tmp_path, monkeypatch, executable) -> None:
    """Windows has no executable bit, so the checker reads the mode Git
    recorded instead of rejecting every launcher (it did, on every run)."""
    monkeypatch.setattr(compat, "HAS_EXECUTABLE_BIT", False)
    repo = _repo(tmp_path, _shipped())
    _record_in_git(repo, "plugin/scripts/slm-launch", executable=executable)
    if executable:
        assert _plugin_package(repo, "claude-code", tmp_path) == "plugin/scripts/slm-launch"
    else:
        with pytest.raises(AssertionError, match="missing or not executable"):
            _plugin_package(repo, "claude-code", tmp_path)
