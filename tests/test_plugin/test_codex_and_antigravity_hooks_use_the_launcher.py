# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""F-12 (19 Nov 2026): the Codex and Antigravity hooks reach slm through the plugin's launcher.

They registered the 4.2.30 per-prompt recall and cue hooks as a bare ``slm hook ... 2>/dev/null ||
true``. Where slm exists only in the data folder's venv (the case ``scripts/slm-launch`` was written
for) the hook was "command not found", silenced: registered, and doing nothing. Now each goes through
``scripts/slm-run``, which picks slm like the launcher (an installed one, then the venv), never
fails the host, and says in a log why a hook did not run.

Codex names the plugin folder with ``${CLAUDE_PLUGIN_ROOT}`` (the Codex binary sets it, and the plugin's
own MCP entry already uses it). Antigravity runs ``./scripts/...`` from the plugin folder, as its
own ``hooks.json`` does.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import uuid
from pathlib import Path

import pytest

from tests.test_plugin._plugin_sandbox import (
    REPO,
    SYSTEM_PATH,
    run,
    stub_slm,
    system_path_has_slm,
)

pytestmark = [
    pytest.mark.skipif(platform.system() == "Windows", reason="POSIX shell hooks"),
    pytest.mark.skipif(system_path_has_slm(), reason="an slm in /usr/bin or /bin defeats the sandbox"),
]

CODEX = REPO / "codex-plugin"
ANTIGRAVITY = REPO / "antigravity-plugin"
CODEX_HOOKS = CODEX / "hooks" / "hooks.json"
ANTIGRAVITY_HOOKS = ANTIGRAVITY / "hooks" / "hooks.json"
ANTIGRAVITY_ROOT_HOOKS = ANTIGRAVITY / "hooks.json"
ADAPTER = ANTIGRAVITY / "scripts" / "antigravity_hook_adapter.py"
CODEX_RUN = '"${CLAUDE_PLUGIN_ROOT}/scripts/slm-run" hook '
ANTIGRAVITY_RUN = "bash ./scripts/slm-run hook "


def _commands(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [h["command"] for groups in data["hooks"].values() for g in groups
            for h in g.get("hooks", []) if h.get("type") == "command"]


def _name(command: str) -> str:
    return command.split(" hook ", 1)[1].split()[0]


def _env(tmp_path: Path, root: Path, *, extra_path: Path | None = None) -> dict[str, str]:
    from tests._portable import GUARD_ENV

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    path = f"{extra_path}:{SYSTEM_PATH}" if extra_path else SYSTEM_PATH
    return {"HOME": str(home), "PATH": path, "SLM_DATA_DIR": str(tmp_path / "slm-data"),
            "CLAUDE_PLUGIN_ROOT": str(root), **GUARD_ENV}


def _venv_slm(tmp_path: Path, tag="venv", **kw) -> Path:
    return stub_slm(tmp_path / "slm-data" / "venv" / "bin" / "slm", tag, **kw)


def _log(tmp_path: Path) -> Path:
    return tmp_path / "slm-data" / "logs" / "plugin-hooks.log"


# -- shape ------------------------------------------------------------------------------------------

def test_no_codex_hook_calls_a_bare_slm_and_the_new_hooks_are_there():
    commands = _commands(CODEX_HOOKS)
    assert commands and all(c.startswith(CODEX_RUN) for c in commands), commands
    assert all("2>/dev/null" not in c and c.rstrip().endswith("|| true") for c in commands)
    names = {_name(c) for c in commands}
    assert {"auto_recall", "cue", "start", "mandate", "before_web", "post_tool_outcome"} <= names


def test_antigravity_hooks_run_the_launcher_from_the_plugin_folder():
    commands = _commands(ANTIGRAVITY_HOOKS)
    assert commands and all(c.startswith(ANTIGRAVITY_RUN) for c in commands), commands
    assert {_name(c) for c in commands} == {_name(c) for c in _commands(CODEX_HOOKS)}
    assert (ANTIGRAVITY / "scripts" / "slm-run").read_text() == (CODEX / "scripts" / "slm-run").read_text()
    assert "slm-run" in ADAPTER.read_text(encoding="utf-8")
    # the root hooks.json (current Antigravity) goes through the adapter, which uses the launcher
    assert all("antigravity_hook_adapter.py" in c for c in _commands_root(ANTIGRAVITY_ROOT_HOOKS))


def _commands_root(path: Path) -> list[str]:
    out: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "command":
                out.append(node["command"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(json.loads(path.read_text(encoding="utf-8")))
    return out


def test_the_codex_launcher_ships_executable():
    assert (CODEX / "scripts" / "slm-run").stat().st_mode & 0o111


# -- behaviour: every shipped Codex hook, run for real ---------------------------------------------------

def test_every_codex_hook_reaches_the_venv_slm_when_it_is_the_only_one(tmp_path):
    _venv_slm(tmp_path)
    env = _env(tmp_path, CODEX)
    for command in _commands(CODEX_HOOKS):
        result = run(["bash", "-c", command.replace("${CLAUDE_PLUGIN_ROOT}", str(CODEX))], env,
                     stdin='{"session_id":"s1"}')
        assert result.returncode == 0, (command, result.stderr)
        assert f"RAN:venv ARGS:hook {_name(command)} STDIN:" in result.stdout, (command, result.stdout)
        assert '{"session_id":"s1"}' in result.stdout  # the payload reaches slm
    assert not _log(tmp_path).exists()


def test_an_installed_slm_wins_over_the_venv(tmp_path):
    stubs = tmp_path / "stubs"
    stub_slm(stubs / "slm", "installed")
    _venv_slm(tmp_path)
    env = _env(tmp_path, CODEX, extra_path=stubs)
    out = run([str(CODEX / "scripts" / "slm-run"), "hook", "cue"], env, stdin="{}")
    assert "RAN:installed ARGS:hook cue" in out.stdout and "RAN:venv" not in out.stdout


def test_with_no_slm_anywhere_the_hook_still_exits_zero_and_says_why(tmp_path):
    env = _env(tmp_path, CODEX)
    out = run([str(CODEX / "scripts" / "slm-run"), "hook", "auto_recall"], env, stdin="{}")
    assert out.returncode == 0 and out.stdout == ""
    assert "see " in out.stderr and "plugin-hooks.log" in out.stderr
    assert "no SuperLocalMemory found" in _log(tmp_path).read_text(encoding="utf-8")


def test_a_failing_slm_never_fails_the_host_and_is_logged(tmp_path):
    _venv_slm(tmp_path, exit_code=3, stderr="boom detail")
    out = run([str(CODEX / "scripts" / "slm-run"), "hook", "cue"], _env(tmp_path, CODEX), stdin="{}")
    assert out.returncode == 0
    logged = _log(tmp_path).read_text(encoding="utf-8")
    assert "exit 3" in logged and "boom detail" in logged


# -- Antigravity: the adapter and the legacy hook file ------------------------------------------------------

def test_every_antigravity_hook_file_command_reaches_the_venv_slm(tmp_path):
    _venv_slm(tmp_path)
    env = _env(tmp_path, ANTIGRAVITY)
    for command in _commands(ANTIGRAVITY_HOOKS):
        result = subprocess.run(["bash", "-c", command], env=env, input="{}", capture_output=True,
                                text=True, cwd=ANTIGRAVITY, timeout=30)
        assert result.returncode == 0, (command, result.stderr)
        assert f"RAN:venv ARGS:hook {_name(command)} STDIN:" in result.stdout, (command, result.stdout)


def test_the_antigravity_adapter_reaches_slm_through_the_launcher(tmp_path):
    _venv_slm(tmp_path)
    env = _env(tmp_path, ANTIGRAVITY)
    # the adapter keeps one marker per conversation in the temp folder: a fresh id each run
    out = run(["python3", str(ADAPTER), "pre-invocation"], env,
              stdin=json.dumps({"conversationId": f"conversation-{uuid.uuid4().hex}"}))
    assert out.returncode == 0, out.stderr
    context = json.loads(out.stdout)["injectSteps"][0]["ephemeralMessage"]
    assert "RAN:venv ARGS:hook mandate" in context and "RAN:venv ARGS:hook start" in context


def test_the_adapter_builds_the_launcher_command_and_falls_back_to_a_bare_slm(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("antigravity_hook_adapter_under_test", ADAPTER)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    cmd = adapter.command("before_web")
    assert cmd[0].endswith("bash") and cmd[1] == str(ANTIGRAVITY / "scripts" / "slm-run")
    assert cmd[2:] == ["hook", "before_web"]
    monkeypatch.setattr(adapter.shutil, "which", lambda _name: None)  # no bash: the last resort
    assert adapter.command("before_web") == ["slm", "hook", "before_web"]
