# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""The POSIX launcher on a host that expands nothing and runs no hooks.

Grok Bot's plugin computer hands ``${CLAUDE_PLUGIN_ROOT}`` and
``${CLAUDE_PLUGIN_DATA}`` through literally and never runs the SessionStart hook
that builds the plugin venv. The launcher must then:

* find its own directory without ``CLAUDE_PLUGIN_ROOT``;
* treat any value still containing ``${`` as unset;
* fall back to CURSOR_PLUGIN_DATA / PLUGIN_DATA / ~/.local/share for the venv;
* run this release through ``uvx`` (CPU torch on Linux), daemon first;
* build a venv inline only when asked (``SLM_LAUNCHER_BOOTSTRAP=1``).

Every run starts from an EMPTY environment (``env -i``): only PATH and HOME and
what each test sets. Every program it may call is a stub that logs its argv.
"""

from __future__ import annotations

import json

import os
import platform
import subprocess
from pathlib import Path

import pytest

from tests.test_plugin._plugin_sandbox import BUILT_PLUGIN, SYSTEM_PATH, system_path_has_slm, write_exe

pytestmark = [
    pytest.mark.skipif(platform.system() == "Windows", reason="POSIX launcher"),
    pytest.mark.skipif(system_path_has_slm(), reason="an slm in /usr/bin or /bin defeats the sandbox"),
]

LAUNCHER = BUILT_PLUGIN / "scripts" / "slm-launch"
#: The four arguments the shared .mcp.json passes for cmd.exe; POSIX ignores them.
CMD_ARGS = ["/d", "/s", "/c", "where.exe /q slm || (echo x 1>&2 & exit 1) & slm mcp"]


def _pin() -> str:
    line = (BUILT_PLUGIN / "requirements.txt").read_text(encoding="utf-8").strip()
    assert line.startswith("superlocalmemory=="), line
    return line


def _logging_stub(path: Path, tag: str, log: Path) -> Path:
    """A program that logs who it is, its argv and the env the launcher left it."""
    return write_exe(
        path,
        "#!/bin/sh\n"
        f'echo "{tag} $* | DATA=${{SLM_DATA_DIR-<unset>}} TORCH=${{UV_TORCH_BACKEND-<unset>}}" >> "{log}"\n'
        'case "$*" in *--version*) echo "superlocalmemory 9.9.9" ;; *"serve start"*) exit 0 ;; '
        f'*mcp*) echo "RAN:{tag}" ;; esac\n'
        "exit 0\n",
    )


def _run(tmp_path: Path, stub_dir: Path, extra: dict[str, str], *, launcher: Path = LAUNCHER):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {"PATH": f"{stub_dir}:{SYSTEM_PATH}", "HOME": str(home)}
    env.update(extra)
    return subprocess.run(
        [str(launcher), *CMD_ARGS], env=env, capture_output=True, text=True,
        cwd=tmp_path, timeout=60,
    )


def _log_lines(log: Path) -> list[str]:
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


LITERAL_HOST_ENV = {
    "CLAUDE_PLUGIN_ROOT": "${CLAUDE_PLUGIN_ROOT}",
    "CLAUDE_PLUGIN_DATA": "${CLAUDE_PLUGIN_DATA}",
    "SLM_DATA_DIR": "${CLAUDE_PLUGIN_DATA}",
}


def test_literal_placeholders_are_treated_as_unset(tmp_path) -> None:
    stubs, log = tmp_path / "bin", tmp_path / "calls.log"
    _logging_stub(stubs / "slm", "slm", log)
    result = _run(tmp_path, stubs, LITERAL_HOST_ENV)
    assert result.returncode == 0, result.stderr[-600:]
    assert result.stdout.strip() == "RAN:slm"
    # The store is SLM's own, never a directory literally named "${CLAUDE_PLUGIN_DATA}".
    assert all("DATA=<unset>" in line for line in _log_lines(log)), _log_lines(log)
    assert not list(tmp_path.glob("$*")), "a literal ${...} directory was created"


def test_no_slm_and_no_venv_runs_this_release_through_uvx_daemon_first(tmp_path) -> None:
    stubs, log = tmp_path / "bin", tmp_path / "calls.log"
    _logging_stub(stubs / "uvx", "uvx", log)
    result = _run(tmp_path, stubs, LITERAL_HOST_ENV)
    assert result.returncode == 0, result.stderr[-600:]
    assert result.stdout.strip() == "RAN:uvx"
    calls = [line.split(" |")[0] for line in _log_lines(log)]
    assert calls == [f"uvx --from {_pin()} slm serve start", f"uvx --from {_pin()} slm mcp"], calls


@pytest.mark.parametrize("os_name, expected", [("Linux", "cpu"), ("Darwin", "<unset>")])
def test_uvx_asks_for_cpu_torch_only_on_linux(tmp_path, os_name, expected) -> None:
    stubs, log = tmp_path / "bin", tmp_path / "calls.log"
    _logging_stub(stubs / "uvx", "uvx", log)
    write_exe(stubs / "uname", f"#!/bin/sh\necho {os_name}\n")
    result = _run(tmp_path, stubs, {})
    assert result.returncode == 0, result.stderr[-600:]
    assert all(f"TORCH={expected}" in line for line in _log_lines(log)), _log_lines(log)


def test_a_torch_backend_the_user_chose_is_kept(tmp_path) -> None:
    stubs, log = tmp_path / "bin", tmp_path / "calls.log"
    _logging_stub(stubs / "uvx", "uvx", log)
    write_exe(stubs / "uname", "#!/bin/sh\necho Linux\n")
    result = _run(tmp_path, stubs, {"UV_TORCH_BACKEND": "cu128"})
    assert result.returncode == 0, result.stderr[-600:]
    assert all("TORCH=cu128" in line for line in _log_lines(log)), _log_lines(log)


@pytest.mark.parametrize("var", ["CURSOR_PLUGIN_DATA", "PLUGIN_DATA"])
def test_a_venv_under_another_hosts_data_dir_is_used(tmp_path, var) -> None:
    stubs, log = tmp_path / "bin", tmp_path / "calls.log"
    stubs.mkdir()
    data = tmp_path / "host-data"
    _logging_stub(data / "venv" / "bin" / "slm", "venv", log)
    _logging_stub(stubs / "uvx", "uvx", log)
    result = _run(tmp_path, stubs, {**LITERAL_HOST_ENV, var: str(data)})
    assert result.returncode == 0, result.stderr[-600:]
    assert result.stdout.strip() == "RAN:venv"


def test_with_no_data_dir_the_venv_lives_under_local_share(tmp_path) -> None:
    stubs, log = tmp_path / "bin", tmp_path / "calls.log"
    stubs.mkdir()
    venv_slm = tmp_path / "home" / ".local" / "share" / "superlocalmemory-plugin" / "venv" / "bin" / "slm"
    _logging_stub(venv_slm, "venv", log)
    result = _run(tmp_path, stubs, {})
    assert result.returncode == 0, result.stderr[-600:]
    assert result.stdout.strip() == "RAN:venv"


def test_it_finds_itself_through_a_symlink(tmp_path) -> None:
    stubs, log = tmp_path / "bin", tmp_path / "calls.log"
    _logging_stub(stubs / "slm", "slm", log)
    link = tmp_path / "elsewhere" / "slm-launch"
    link.parent.mkdir()
    link.symlink_to(LAUNCHER)
    result = _run(tmp_path, stubs, LITERAL_HOST_ENV, launcher=link)
    assert result.returncode == 0, result.stderr[-600:]
    assert result.stdout.strip() == "RAN:slm"


def test_nothing_to_run_says_how_to_fix_it_and_builds_nothing(tmp_path) -> None:
    stubs, log = tmp_path / "bin", tmp_path / "calls.log"
    _logging_stub(stubs / "python3", "python3", log)
    result = _run(tmp_path, stubs, LITERAL_HOST_ENV)
    assert result.returncode == 1
    assert "pipx install superlocalmemory" in result.stderr
    # uv named, not linked: a remote URL beside TOKENIZERS_PARALLELISM in this
    # script is held by the Claude plugin directory as a credential leaving.
    assert "install uv" in result.stderr.lower()
    assert result.stdout == "", "stdout is the MCP channel"
    assert _log_lines(log) == [], "no venv is built unless asked"


def test_the_inline_venv_is_built_only_when_asked(tmp_path) -> None:
    stubs = tmp_path / "bin"
    write_exe(  # a python3 the bootstrap's own >= 3.12 guard rejects
        stubs / "python3",
        '#!/bin/sh\ncase "$*" in *--version*) echo "Python 3.11.9" ;; *) exit 1 ;; esac\n',
    )
    result = _run(tmp_path, stubs, {"SLM_LAUNCHER_BOOTSTRAP": "1"})
    assert result.returncode == 1
    assert "requires Python >= 3.12" in result.stderr, result.stderr[-600:]
    assert result.stdout == ""


def test_an_installed_slm_still_wins_over_uvx(tmp_path) -> None:
    """Claude Code's common case is unchanged: the user's slm, not a second copy."""
    stubs, log = tmp_path / "bin", tmp_path / "calls.log"
    _logging_stub(stubs / "slm", "slm", log)
    _logging_stub(stubs / "uvx", "uvx", log)
    result = _run(tmp_path, stubs, {"CLAUDE_PLUGIN_DATA": str(tmp_path / "pd")})
    assert result.returncode == 0, result.stderr[-600:]
    assert result.stdout.strip() == "RAN:slm"
    assert not any(line.startswith("uvx") for line in _log_lines(log))


def test_launcher_is_executable_and_identical_in_every_tree() -> None:
    repo = Path(__file__).resolve().parents[2]
    version = json.loads((repo / "plugin-src" / "manifest.json").read_text(encoding="utf-8"))["version"]
    src = (repo / "plugin-src" / "scripts" / "slm-launch").read_text(encoding="utf-8")
    # Identical except the release, which the build writes in literally.
    assert "__SLM_RELEASE__" in src
    assert LAUNCHER.read_text(encoding="utf-8") == src.replace("__SLM_RELEASE__", version)
    assert f'uvx --from "superlocalmemory=={version}" slm mcp' in LAUNCHER.read_text(encoding="utf-8")
    assert os.access(LAUNCHER, os.X_OK)


def test_launcher_exports_no_secret_like_names() -> None:
    # Anthropic's plugin directory holds a plugin for review ("forwards a
    # credential", MCP_FORWARDS_CREDENTIAL_ENV) when the MCP launch path names
    # anything token-, key- or secret-like. TOKENIZERS_PARALLELISM alone did it.
    import re

    src = Path(__file__).resolve().parents[2] / "plugin-src" / "scripts" / "slm-launch"
    text = src.read_text(encoding="utf-8")
    flagged = re.findall(r"(?i)\b\w*(token|secret|api_?key|password|credential)\w*\b", text)
    assert flagged == [], f"secret-like names in slm-launch: {flagged}"
