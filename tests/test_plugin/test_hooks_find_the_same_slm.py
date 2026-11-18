# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""#141 — every plugin hook runs the slm the MCP server runs, and says so when
it cannot.

Hooks used to call a bare `slm hook X 2>/dev/null || true`. On a machine where
SLM lives only in the plugin venv — the fallback the launcher documents — every
hook was "command not found", swallowed, and memory capture silently stopped.

Each hook now goes through scripts/slm-run, which resolves slm with the same
code as the launcher (slm-resolve.sh). It never blocks the host session: it
always exits 0. When it cannot run slm, or slm fails, it writes the reason to
<data dir>/logs/plugin-hooks.log and one line to stderr.

The commands tested are the ones in the BUILT plugin/hooks/hooks.json, expanded
the way Claude Code expands them and run through bash — the shell Claude Code
uses for command hooks.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import pytest

from tests.test_plugin._plugin_sandbox import (
    BUILT_PLUGIN,
    REPO,
    run,
    sandbox_env,
    stub_slm,
    system_path_has_slm,
)

pytestmark = [
    pytest.mark.skipif(platform.system() == "Windows", reason="POSIX shell hooks"),
    pytest.mark.skipif(system_path_has_slm(), reason="an slm in /usr/bin or /bin defeats the sandbox"),
]

HOOK_FILES = (
    REPO / "plugin-src" / "hooks" / "hooks.json",
    BUILT_PLUGIN / "hooks" / "hooks.json",
)
SLM_RUN = BUILT_PLUGIN / "scripts" / "slm-run"


def _commands(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        hook["command"]
        for groups in data["hooks"].values()
        for group in groups
        for hook in group.get("hooks", [])
        if hook.get("type") == "command"
    ]


def _slm_hook_commands(path: Path) -> list[str]:
    return [c for c in _commands(path) if "ensure-venv" not in c]


def _expand(command: str) -> str:
    return command.replace("${CLAUDE_PLUGIN_ROOT}", str(BUILT_PLUGIN))


def _hook_name(command: str) -> str:
    return command.split(" hook ", 1)[1].split()[0]


def _log(tmp_path: Path) -> Path:
    return tmp_path / "slm-data" / "logs" / "plugin-hooks.log"


def _venv_slm(tmp_path: Path) -> Path:
    return tmp_path / "plugin-data" / "venv" / "bin" / "slm"


# --------------------------------------------------------------------------
# Shape: no hook calls a bare slm any more, in source or in the build.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("path", HOOK_FILES, ids=["plugin-src", "plugin"])
def test_no_hook_calls_a_bare_slm(path):
    for command in _slm_hook_commands(path):
        assert command.lstrip().startswith('"${CLAUDE_PLUGIN_ROOT}/scripts/slm-run" hook '), command


@pytest.mark.parametrize("path", HOOK_FILES, ids=["plugin-src", "plugin"])
def test_no_hook_throws_its_errors_away(path):
    for command in _slm_hook_commands(path):
        assert "2>/dev/null" not in command, command
        assert command.rstrip().endswith("|| true"), command


def test_the_full_hook_suite_survives_the_rewrite():
    names = {_hook_name(c) for c in _slm_hook_commands(BUILT_PLUGIN / "hooks" / "hooks.json")}
    assert names == {
        "mandate", "start", "checkpoint", "post_tool_outcome", "user_prompt_rehash",
        "topic_shift", "stop", "stop_outcome", "before_web", "auto_recall", "cue",
    }


# --------------------------------------------------------------------------
# Behaviour: run every shipped hook command for real.
# --------------------------------------------------------------------------
def test_every_hook_reaches_the_plugin_venv_when_it_is_the_only_slm(tmp_path):
    """The #141 case: no slm on PATH, SLM only in the plugin venv."""
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    stub_slm(_venv_slm(tmp_path), "venv")
    env = sandbox_env(tmp_path, stubs)

    for command in _slm_hook_commands(BUILT_PLUGIN / "hooks" / "hooks.json"):
        result = run(["bash", "-c", _expand(command)], env, stdin='{"session_id":"s1"}')
        name = _hook_name(command)
        assert result.returncode == 0, (command, result.stderr)
        assert f"RAN:venv ARGS:hook {name} STDIN:" in result.stdout, (command, result.stdout)
        assert '{"session_id":"s1"}' in result.stdout, "hook payload on stdin was not passed through"
    assert not _log(tmp_path).exists(), "logged a failure when nothing failed"


def test_an_installed_slm_wins_over_the_venv_like_the_launcher(tmp_path):
    stubs = tmp_path / "stubs"
    stub_slm(stubs / "slm", "system")
    stub_slm(_venv_slm(tmp_path), "venv")

    result = run([str(SLM_RUN), "hook", "start"], sandbox_env(tmp_path, stubs))

    assert "RAN:system ARGS:hook start" in result.stdout


@pytest.mark.parametrize("launcher, expected", [("plugin", "venv"), ("system", "system")])
def test_slm_launcher_steers_hooks_too(tmp_path, launcher, expected):
    stubs = tmp_path / "stubs"
    stub_slm(stubs / "slm", "system")
    stub_slm(_venv_slm(tmp_path), "venv")

    result = run([str(SLM_RUN), "hook", "start"], sandbox_env(tmp_path, stubs, SLM_LAUNCHER=launcher))

    assert f"RAN:{expected} ARGS:hook start" in result.stdout


def test_an_explicit_binary_is_used(tmp_path):
    stubs = tmp_path / "stubs"
    stub_slm(stubs / "slm", "system")
    stub_slm(tmp_path / "home" / "opt" / "slm", "explicit")

    result = run([str(SLM_RUN), "hook", "stop"], sandbox_env(tmp_path, stubs, SLM_LAUNCHER="~/opt/slm"))

    assert "RAN:explicit ARGS:hook stop" in result.stdout


def test_nothing_installed_is_logged_not_swallowed(tmp_path):
    stubs = tmp_path / "stubs"
    stubs.mkdir()

    result = run([str(SLM_RUN), "hook", "start"], sandbox_env(tmp_path, stubs))

    assert result.returncode == 0, "a hook must never fail the host session"
    assert result.stdout == ""
    assert len(result.stderr.strip().splitlines()) == 1, result.stderr
    assert "plugin-hooks.log" in result.stderr
    log = _log(tmp_path).read_text(encoding="utf-8")
    assert "hook start" in log
    assert "'slm' on PATH" in log and str(_venv_slm(tmp_path)) in log, log


def test_a_failing_hook_is_logged_with_its_exit_code_and_error(tmp_path):
    stubs = tmp_path / "stubs"
    stub_slm(stubs / "slm", "system", exit_code=3, stderr="database is locked")

    result = run([str(SLM_RUN), "hook", "checkpoint"], sandbox_env(tmp_path, stubs))

    assert result.returncode == 0
    assert len(result.stderr.strip().splitlines()) == 1, result.stderr
    log = _log(tmp_path).read_text(encoding="utf-8")
    assert "hook checkpoint" in log and "exit 3" in log and "database is locked" in log, log


def test_exit_code_2_can_never_block_the_host(tmp_path):
    """Claude Code treats a hook's exit 2 as 'block'. A crashing slm must not."""
    stubs = tmp_path / "stubs"
    stub_slm(stubs / "slm", "system", exit_code=2)

    result = run([str(SLM_RUN), "hook", "before_web"], sandbox_env(tmp_path, stubs))

    assert result.returncode == 0


def test_successful_hook_stderr_stays_out_of_the_session_and_the_log(tmp_path):
    stubs = tmp_path / "stubs"
    stub_slm(stubs / "slm", "system", stderr="harmless warning")

    result = run([str(SLM_RUN), "hook", "start"], sandbox_env(tmp_path, stubs))

    assert result.returncode == 0
    assert result.stderr == ""
    assert not _log(tmp_path).exists()


def test_an_unwritable_log_still_never_fails_the_hook(tmp_path):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    blocker = tmp_path / "slm-data"
    blocker.write_text("a file where the data directory should be", encoding="utf-8")

    result = run([str(SLM_RUN), "hook", "start"], sandbox_env(tmp_path, stubs))

    assert result.returncode == 0
    assert result.stderr.strip(), "the failure must still be visible somewhere"


def test_the_log_does_not_grow_without_bound(tmp_path):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    log = _log(tmp_path)
    log.parent.mkdir(parents=True)
    log.write_text("x" * (1024 * 1024 + 10), encoding="utf-8")

    run([str(SLM_RUN), "hook", "start"], sandbox_env(tmp_path, stubs))

    assert log.stat().st_size < 1024 * 1024
    assert (log.parent / "plugin-hooks.log.1").exists()
