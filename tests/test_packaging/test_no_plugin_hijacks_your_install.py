# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Installing a plugin must not change what SLM you run or what it reads.

Four surfaces ship an MCP server definition — Claude Code, Codex, VS Code and
Antigravity. Three of them were quietly overriding the user's own setup:

* **A private data directory.** Claude Code's pointed at `${CLAUDE_PLUGIN_DATA}`.
  On a machine already using SLM that means the editor talks to a different,
  empty store: measured on the author's machine, that directory held 28 KB while
  the real store was 611 MB with 5,370 memories in it. Installing the plugin
  looked exactly like losing every memory.
* **A narrower tool set.** Three surfaces forced `SLM_MCP_PROFILE=code`, which is
  31 tools and drops the 8 mesh tools among others. A user running a wider
  profile deliberately had tools taken away by installing something.
* **A second installation.** Two launchers preferred a plugin-owned virtual
  environment over the `slm` already on PATH, so a machine that had once
  bootstrapped one kept using it forever — including after the user installed or
  upgraded SLM properly. Two copies reading one store, and which one served it
  depended on which started first.

The rule these tests hold: a plugin may say **who is calling** (`SLM_AGENT_ID`,
which is attribution) and nothing else about the environment. Everything else —
which binary, which store, which tools — belongs to whoever installed it, in
either install order.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

#: The env block each surface hands its MCP server, however that surface spells it.
def _claude_env():
    d = json.loads((REPO / "plugin/.mcp.json").read_text(encoding="utf-8"))
    return d["mcpServers"]["superlocalmemory"].get("env", {})


def _antigravity_env():
    d = json.loads((REPO / "antigravity-plugin/mcp_config.json").read_text(encoding="utf-8"))
    return d["mcpServers"]["superlocalmemory"].get("env", {})


def _vscode_env():
    d = json.loads((REPO / "copilot-plugin/.vscode/mcp.json").read_text(encoding="utf-8"))
    return d["servers"]["superlocalmemory"].get("env", {})


def _codex_plugin_mcp_env():
    """The Codex tree's own .mcp.json — what a marketplace install of
    ``superlocalmemory-codex`` actually runs. Added in 4.1.3: before it existed,
    the marketplace could only serve the Claude tree, so Codex ran with
    ``SLM_AGENT_ID=claude_code`` and filed every memory under the wrong host."""
    d = json.loads((REPO / "codex-plugin/.mcp.json").read_text(encoding="utf-8"))
    return d["mcpServers"]["superlocalmemory"].get("env", {})


def _codex_env():
    text = (REPO / "codex-plugin/.codex/config.toml").read_text(encoding="utf-8")
    match = re.search(r"^env = \{(.*)\}\s*$", text, re.MULTILINE)
    if not match:
        return {}
    out = {}
    for part in match.group(1).split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip().strip('"')
    return out


SURFACES = (
    ("Claude Code", _claude_env),
    ("Codex", _codex_env),
    ("VS Code", _vscode_env),
    ("Antigravity", _antigravity_env),
    ("Codex plugin tree", _codex_plugin_mcp_env),
)


class TestNoSurfaceNarrowsYourToolSet:
    @pytest.mark.parametrize("name, reader", SURFACES, ids=[s[0] for s in SURFACES])
    def test_it_does_not_force_a_profile(self, name, reader) -> None:
        env = reader()

        # The harm this guards is NARROWING: three surfaces once forced
        # SLM_MCP_PROFILE=code (31 tools, mesh tools dropped), silently
        # taking away tools the user configured. Forcing the widest
        # profile ('power', the full set) cannot narrow anyone and is the
        # standing product decision — so absent or 'power' passes, while
        # any narrower forced profile still fails.
        forced = env.get("SLM_MCP_PROFILE")
        assert forced is None or forced == "power", (
            f"{name} forces SLM_MCP_PROFILE={forced!r}. "
            f"Installing a plugin must not remove tools the user configured."
        )


class TestNoSurfaceRepointsYourStore:
    @pytest.mark.parametrize("name, reader", SURFACES, ids=[s[0] for s in SURFACES])
    def test_it_does_not_force_a_data_directory(self, name, reader) -> None:
        env = reader()

        assert "SLM_DATA_DIR" not in env, (
            f"{name} forces SLM_DATA_DIR={env.get('SLM_DATA_DIR')!r}. SLM resolves "
            f"its own store; pinning it here overrides a user who moved theirs, "
            f"and pointing at a private directory makes an existing install look "
            f"empty."
        )


class TestAttributionIsStillSet:
    """The control. Stripping the overrides must not strip the one legitimate
    thing — without it, memories written from every host are indistinguishable."""

    @pytest.mark.parametrize(
        "name, reader, expected",
        [
            ("Claude Code", _claude_env, "claude_code"),
            ("Codex", _codex_env, "codex"),
            ("Antigravity", _antigravity_env, "antigravity"),
            ("Codex plugin tree", _codex_plugin_mcp_env, "codex"),
        ],
    )
    def test_the_agent_id_identifies_the_host(self, name, reader, expected) -> None:
        assert reader().get("SLM_AGENT_ID") == expected


class TestLaunchersPreferWhatIsAlreadyInstalled:
    """Both install orders converge on one environment.

    Each launcher is RUN, against a stubbed slm that reports a version and
    names which copy was chosen. Nothing real is started.
    """

    LAUNCHERS = (
        ("Claude Code", "plugin/scripts/slm-launch"),
        ("Codex", "codex-plugin/scripts/slm-launch"),
    )

    def _stub_slm(self, directory, version, marker):
        """A stand-in for the slm binary that reports a version and says which
        copy was chosen, without starting anything."""
        directory.mkdir(parents=True, exist_ok=True)
        binary = directory / "slm"
        binary.write_text(
            "#!/bin/sh\n"
            'case "$1" in\n'
            f'  --version) echo "superlocalmemory {version}" ;;\n'
            "  serve)     exit 0 ;;\n"
            f'  mcp)       echo "{marker}" ;;\n'
            "esac\n",
            encoding="utf-8",
        )
        binary.chmod(0o755)
        return binary

    def _run(self, rel, *, path_dir, plugin_data, tmp_path):
        """Run the launcher for real. Its own resolution decides the output.

        Positional text checks were tried first and are the wrong instrument:
        both launchers ASSIGN their venv variable near the top, before the PATH
        check, so the venv string appears earlier in the file while being used
        later. Only running it answers the question.
        """
        import os
        import subprocess

        env = dict(
            os.environ,
            PATH=f"{path_dir}:/usr/bin:/bin",
            SLM_LAUNCHER="auto",
            SLM_DATA_DIR=str(tmp_path / "data"),
        )
        if plugin_data is None:
            env.pop("CLAUDE_PLUGIN_DATA", None)
        else:
            env["CLAUDE_PLUGIN_DATA"] = str(plugin_data)
        return subprocess.run(
            ["bash", str(REPO / rel)],
            capture_output=True, text=True, env=env, timeout=120,
        )

    @pytest.mark.parametrize("name, rel", LAUNCHERS, ids=[l[0] for l in LAUNCHERS])
    def test_an_installed_slm_is_used(self, name, rel, tmp_path) -> None:
        """Installed first, plugin second — nothing was ever forked."""
        path_dir = tmp_path / "bin"
        self._stub_slm(path_dir, "4.1.2", "CHOSE: installed")

        result = self._run(rel, path_dir=path_dir, plugin_data=None, tmp_path=tmp_path)

        assert "CHOSE: installed" in result.stdout, (
            f"{name}: did not use the slm on PATH.\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr[:300]!r}"
        )

    @pytest.mark.parametrize("name, rel", LAUNCHERS, ids=[l[0] for l in LAUNCHERS])
    def test_a_leftover_venv_loses_to_an_installed_slm(
        self, name, rel, tmp_path,
    ) -> None:
        """Plugin first, then installed. The venv is now the older copy and must
        stop being used — and the duplicate must be reported, not hidden."""
        path_dir = tmp_path / "bin"
        self._stub_slm(path_dir, "4.1.2", "CHOSE: installed")

        # Both launchers look for a venv, in the two places they each look.
        plugin_data = tmp_path / "plugin-data"
        self._stub_slm(plugin_data / "venv" / "bin", "4.0.8", "CHOSE: venv")
        self._stub_slm(tmp_path / "data" / "venv" / "bin", "4.0.8", "CHOSE: venv")

        result = self._run(
            rel, path_dir=path_dir, plugin_data=plugin_data, tmp_path=tmp_path,
        )

        assert "CHOSE: installed" in result.stdout, (
            f"{name}: used the leftover venv instead of the installed slm.\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr[:300]!r}"
        )
        assert "IGNORING" in result.stderr, (
            f"{name}: the duplicate install was not reported"
        )
        assert "4.0.8" in result.stderr, f"{name}: the stale version was not named"

    @pytest.mark.parametrize("name, rel", LAUNCHERS, ids=[l[0] for l in LAUNCHERS])
    def test_a_venv_is_used_when_nothing_is_installed(
        self, name, rel, tmp_path,
    ) -> None:
        """The fallback still works — a machine with no system install is not
        left with nothing."""
        empty = tmp_path / "bin"
        empty.mkdir(parents=True, exist_ok=True)
        plugin_data = tmp_path / "plugin-data"
        self._stub_slm(plugin_data / "venv" / "bin", "4.1.2", "CHOSE: venv")
        self._stub_slm(tmp_path / "data" / "venv" / "bin", "4.1.2", "CHOSE: venv")

        result = self._run(
            rel, path_dir=empty, plugin_data=plugin_data, tmp_path=tmp_path,
        )

        assert "CHOSE: venv" in result.stdout, (
            f"{name}: no system slm and the venv was not used either.\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr[:300]!r}"
        )

    @pytest.mark.parametrize("name, rel", LAUNCHERS, ids=[l[0] for l in LAUNCHERS])
    def test_it_says_something_useful_when_nothing_is_installed(
        self, name, rel,
    ) -> None:
        """A path that silently does not exist is the worst outcome — that is how
        the Codex server failed to start with no explanation at all."""
        text = (REPO / rel).read_text(encoding="utf-8")

        assert "pipx install superlocalmemory" in text, (
            f"{name}: no install advice for a machine that has no SLM"
        )

    @pytest.mark.parametrize("name, rel", LAUNCHERS, ids=[l[0] for l in LAUNCHERS])
    def test_a_duplicate_install_is_reported(self, name, rel) -> None:
        """One store served by two versions is the failure worth seeing."""
        text = (REPO / rel).read_text(encoding="utf-8")

        assert "IGNORING" in text, (
            f"{name}: a leftover venv beside a real install is not reported"
        )
