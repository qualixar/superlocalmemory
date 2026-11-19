# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""#139 — one `.mcp.json`, two platforms.

A plugin `.mcp.json` has no per-platform command, and on native Windows the
host spawns without a shell, so the extensionless launcher never reaches
`slm-launch.bat` (proven on the Windows runner by
test_windows_mcp_spawn_premise.py). The entry therefore picks the program with
``${ComSpec:-<launcher>}``: Windows always defines ComSpec (cmd.exe), POSIX
does not, so

* macOS / Linux: the bash launcher, exactly as before — it ignores its
  arguments, so the cmd.exe arguments change nothing (run below);
* Windows: cmd.exe runs the INSTALLED slm (pip/pipx put slm.exe on PATH), and
  says how to install it when it is missing.

These tests expand the entry the way Claude Code does (see _mcp_expand.py).
The Windows half is structural here; it is executed for real only on the
Windows CI runner (test_windows_mcp_starts_installed_slm.py).
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

import pytest

from tests._portable import GUARD_ENV
from tests.test_plugin._mcp_expand import WINDOWS_ENV, declared_server, expanded_argv

REPO = Path(__file__).resolve().parents[2]

ENTRIES = (
    ("plugin-src", REPO / "plugin-src" / ".mcp.json", REPO / "plugin"),
    ("Claude Code", REPO / "plugin" / ".mcp.json", REPO / "plugin"),
    ("Codex marketplace", REPO / "codex-plugin" / ".mcp.json", REPO / "codex-plugin"),
)
IDS = [e[0] for e in ENTRIES]


def _posix_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k.lower() != "comspec"}
    return env


@pytest.mark.parametrize("name, mcp_json, root", ENTRIES, ids=IDS)
def test_posix_still_runs_the_bash_launcher(name, mcp_json, root) -> None:
    argv = expanded_argv(mcp_json, root.as_posix(), _posix_env())
    assert argv[0] == f"{root.as_posix()}/scripts/slm-launch", (name, argv)


@pytest.mark.parametrize("name, mcp_json, root", ENTRIES, ids=IDS)
def test_windows_runs_cmd_with_the_installed_slm(name, mcp_json, root) -> None:
    argv = expanded_argv(mcp_json, "C:/Users/alice smith/.claude/plugins/cache/x", WINDOWS_ENV)
    assert argv[0] == WINDOWS_ENV["ComSpec"], (name, argv)
    assert argv[1:4] == ["/d", "/s", "/c"], (name, argv)
    assert len(argv) == 5, (name, argv)
    line = argv[4]
    # The plugin's own paths are never handed to cmd.exe: an unquoted
    # forward-slash path is read as switches there. Only the installed slm.
    assert "C:/" not in line and "plugins" not in line, line
    assert line.startswith("where.exe /q slm || ("), line
    assert line.endswith("& slm serve start 1>&2 & slm mcp"), line
    # Says how to install it. No literal `pipx install` here: the Claude plugin
    # directory reads that as an unpinned package launcher and blocks the plugin.
    assert "Install SuperLocalMemory first" in line and "superlocalmemory.com" in line, line
    assert "pipx install" not in line and "pip install" not in line, line
    # Characters that would break the cmd line or the host's quoting.
    assert '"' not in line and "%" not in line and "^" not in line, line
    message = line[line.index("(") + 1: line.rindex(")")]
    assert "(" not in message and ")" not in message, message


@pytest.mark.parametrize("name, mcp_json, root", ENTRIES, ids=IDS)
def test_windows_finds_comspec_whatever_its_case(name, mcp_json, root) -> None:
    """A copy of the Windows environment made by Python spells it ``COMSPEC``;
    the host still finds it, as Windows names are case-insensitive."""
    env = {"COMSPEC": WINDOWS_ENV["ComSpec"]}
    argv = expanded_argv(mcp_json, root.as_posix(), env, case_insensitive=True)
    assert argv[0] == WINDOWS_ENV["ComSpec"], (name, argv)


@pytest.mark.parametrize("name, mcp_json, root", ENTRIES, ids=IDS)
def test_the_entry_sets_only_the_agent_id(name, mcp_json, root) -> None:
    assert set(declared_server(mcp_json).get("env", {})) == {"SLM_AGENT_ID"}


# --- POSIX: run the declared entry for real -----------------------------------

posix_only = pytest.mark.skipif(platform.system() == "Windows", reason="bash launcher")


def _stub_slm(directory: Path, marker: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / "slm"
    binary.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        '  --version) echo "superlocalmemory 4.1.20" ;;\n'
        "  serve)     exit 0 ;;\n"
        f'  mcp)       echo "{marker}" ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)


def _run_declared(mcp_json: Path, root: Path, path_dir: Path, tmp_path: Path):
    env = _posix_env()
    env.update(GUARD_ENV, PATH=f"{path_dir}:/usr/bin:/bin", SLM_LAUNCHER="auto",
               SLM_DATA_DIR=str(tmp_path / "data"), HOME=str(tmp_path / "home"))
    env.pop("CLAUDE_PLUGIN_DATA", None)
    argv = expanded_argv(mcp_json, root.as_posix(), env)
    # No shell, as the host spawns it.
    return subprocess.run(argv, capture_output=True, text=True, env=env,
                          cwd=tmp_path, timeout=120)


@posix_only
@pytest.mark.parametrize("name, mcp_json, root", ENTRIES[1:], ids=IDS[1:])
def test_posix_declared_entry_uses_the_installed_slm(name, mcp_json, root, tmp_path) -> None:
    _stub_slm(tmp_path / "bin", "CHOSE: installed")
    result = _run_declared(mcp_json, root, tmp_path / "bin", tmp_path)
    assert result.returncode == 0, result.stderr[:400]
    assert result.stdout.strip() == "CHOSE: installed", (result.stdout, result.stderr[:400])


@posix_only
@pytest.mark.parametrize("name, mcp_json, root", ENTRIES[1:], ids=IDS[1:])
def test_posix_declared_entry_explains_a_missing_slm(name, mcp_json, root, tmp_path) -> None:
    (tmp_path / "bin").mkdir()
    result = _run_declared(mcp_json, root, tmp_path / "bin", tmp_path)
    assert result.returncode != 0
    assert "pipx install superlocalmemory" in result.stderr, result.stderr[:400]
