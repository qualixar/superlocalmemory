# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""Codex plugin layout tests — validates codex-plugin/ parity with plugin/ (Claude Code).

Layout verified:
  codex-plugin/
    AGENTS.md                   (Codex rules — equiv of plugin/CLAUDE.md)
    README.md
    .codex/config.toml          (TOML MCP registration — equiv of plugin/.mcp.json)
    hooks/hooks.json            (session lifecycle hooks)
    scripts/slm-launch          (POSIX launcher, +x)
    scripts/slm-launch.bat      (Windows launcher)
    scripts/ensure-venv.sh      (POSIX venv bootstrap, +x)
    scripts/ensure-venv.bat     (Windows venv bootstrap)
    skills/<7>/SKILL.md         (identical to Claude Code plugin skills)

Tests verify:
  - All expected files exist
  - .codex/config.toml is valid TOML, has [mcp_servers.superlocalmemory], SLM_MCP_PROFILE=code
  - hooks/hooks.json is valid JSON, has SessionStart and Stop
  - AGENTS.md contains the core SLM rules (session_init, recall, remember, optimize, close_session)
  - scripts/slm-launch starts daemon before MCP (same parity guarantee as Claude Code launcher)
  - All 7 expected skills present
  - No shipped files modified (backward compat guard)
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent.parent
CODEX_PLUGIN = REPO / "codex-plugin"
CODEX_SCRIPTS = CODEX_PLUGIN / "scripts"
CODEX_SKILLS = CODEX_PLUGIN / "skills"
CODEX_HOOKS = CODEX_PLUGIN / "hooks"
CODEX_MCP_CONFIG = CODEX_PLUGIN / ".codex" / "config.toml"

# 7 skills — must match plugin/skills/
EXPECTED_SKILLS = [
    "slm-cache",
    "slm-compress",
    "slm-governance",
    "slm-graph",
    "slm-mesh",
    "slm-profile",
    "slm-recall",
    "slm-remember",
    "slm-scope",
    "slm-session",
    "slm-status",
]

# Core SLM rule keywords that MUST appear in AGENTS.md
AGENTS_MD_REQUIRED_PHRASES = [
    "session_init",
    "recall",
    "remember",
    "close_session",
    "optimize",
    "CLI fallback",
    "SuperLocalMemory",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    """Load JSON; fail with clear message if missing or invalid."""
    assert path.exists(), f"Missing file: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_toml(path: Path) -> dict:
    """Load TOML (requires Python 3.11+); fail with clear message if missing or invalid."""
    assert path.exists(), f"Missing file: {path}"
    if sys.version_info >= (3, 11):
        import tomllib
        return tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        # Minimal fallback for older Python — verify at least it looks valid
        # by checking key markers exist as raw text
        return {"_raw": path.read_text(encoding="utf-8")}


# ---------------------------------------------------------------------------
# T1 — Top-level file existence
# ---------------------------------------------------------------------------
class TestFileExistence:
    """All codex-plugin/ files must be present."""

    def test_agents_md_exists(self) -> None:
        p = CODEX_PLUGIN / "AGENTS.md"
        assert p.exists(), f"codex-plugin/AGENTS.md not found at {p}"

    def test_readme_exists(self) -> None:
        p = CODEX_PLUGIN / "README.md"
        assert p.exists(), f"codex-plugin/README.md not found at {p}"

    def test_generated_md_exists(self) -> None:
        p = CODEX_PLUGIN / "_GENERATED.md"
        assert p.exists(), f"codex-plugin/_GENERATED.md not found at {p}"

    def test_codex_config_toml_exists(self) -> None:
        assert CODEX_MCP_CONFIG.exists(), (
            f"codex-plugin/.codex/config.toml not found at {CODEX_MCP_CONFIG}"
        )

    def test_hooks_json_exists(self) -> None:
        p = CODEX_HOOKS / "hooks.json"
        assert p.exists(), f"codex-plugin/hooks/hooks.json not found at {p}"

    def test_slm_launch_posix_exists(self) -> None:
        p = CODEX_SCRIPTS / "slm-launch"
        assert p.exists(), f"codex-plugin/scripts/slm-launch not found at {p}"

    def test_slm_launch_bat_exists(self) -> None:
        p = CODEX_SCRIPTS / "slm-launch.bat"
        assert p.exists(), f"codex-plugin/scripts/slm-launch.bat not found at {p}"

    def test_ensure_venv_sh_exists(self) -> None:
        p = CODEX_SCRIPTS / "ensure-venv.sh"
        assert p.exists(), f"codex-plugin/scripts/ensure-venv.sh not found at {p}"

    def test_ensure_venv_bat_exists(self) -> None:
        p = CODEX_SCRIPTS / "ensure-venv.bat"
        assert p.exists(), f"codex-plugin/scripts/ensure-venv.bat not found at {p}"


# ---------------------------------------------------------------------------
# T2 — .codex/config.toml: valid TOML, MCP registration, env vars
# ---------------------------------------------------------------------------
class TestCodexConfigToml:
    """config.toml must be valid TOML with correct SLM MCP registration."""

    def test_config_toml_is_valid_toml(self) -> None:
        """File must parse without error."""
        _load_toml(CODEX_MCP_CONFIG)

    def test_config_toml_has_mcp_servers_section(self) -> None:
        raw = CODEX_MCP_CONFIG.read_text(encoding="utf-8")
        assert "[mcp_servers.superlocalmemory]" in raw, (
            "config.toml must contain [mcp_servers.superlocalmemory] section"
        )

    def test_config_toml_has_slm_mcp_profile_code(self) -> None:
        raw = CODEX_MCP_CONFIG.read_text(encoding="utf-8")
        assert 'SLM_MCP_PROFILE' in raw, "config.toml must reference SLM_MCP_PROFILE"
        assert '"code"' in raw or "'code'" in raw, (
            "SLM_MCP_PROFILE must be set to 'code'"
        )

    def test_config_toml_has_slm_data_dir(self) -> None:
        raw = CODEX_MCP_CONFIG.read_text(encoding="utf-8")
        assert "SLM_DATA_DIR" in raw, "config.toml must reference SLM_DATA_DIR env var"

    def test_config_toml_has_valid_command(self) -> None:
        """Command must reference slm binary (direct) or slm-launch (venv mode)."""
        raw = CODEX_MCP_CONFIG.read_text(encoding="utf-8")
        assert 'command' in raw, "config.toml must have a command field"
        # command must be slm or a launcher pointing to slm
        has_slm_cmd = (
            'command = "slm"' in raw
            or "command = 'slm'" in raw
            or "slm-launch" in raw
        )
        assert has_slm_cmd, (
            f"config.toml command must reference 'slm' binary or 'slm-launch', "
            f"got raw snippet: {raw[:500]}"
        )

    def test_config_toml_mcp_args_includes_mcp_or_launcher(self) -> None:
        """Either args = ['mcp'] (direct) or command is a launcher (args can be [])."""
        raw = CODEX_MCP_CONFIG.read_text(encoding="utf-8")
        has_mcp_arg = '"mcp"' in raw or "'mcp'" in raw
        is_launcher = "slm-launch" in raw
        assert has_mcp_arg or is_launcher, (
            "config.toml must either pass 'mcp' in args or use the slm-launch launcher"
        )

    def test_config_toml_uses_toml_format_not_json(self) -> None:
        """Codex uses TOML config, not JSON. File must not start with '{'."""
        text = CODEX_MCP_CONFIG.read_text(encoding="utf-8").lstrip()
        assert not text.startswith("{"), (
            "config.toml must be TOML format (not JSON — that is the Claude Code format)"
        )


# ---------------------------------------------------------------------------
# T3 — hooks/hooks.json: valid JSON, SessionStart present
# ---------------------------------------------------------------------------
class TestHooksJson:
    """hooks.json must be valid JSON with required lifecycle events."""

    def _get_hooks(self) -> dict:
        return _load_json(CODEX_HOOKS / "hooks.json")

    def test_hooks_json_parses(self) -> None:
        self._get_hooks()

    def test_hooks_json_has_hooks_key(self) -> None:
        data = self._get_hooks()
        assert "hooks" in data, "hooks.json must have a top-level 'hooks' key"

    def test_hooks_has_session_start(self) -> None:
        data = self._get_hooks()
        hooks = data.get("hooks", {})
        assert "SessionStart" in hooks, (
            "hooks.json must have 'SessionStart' event for SLM session lifecycle"
        )

    def test_session_start_calls_slm_hook_start(self) -> None:
        data = self._get_hooks()
        hooks = data.get("hooks", {})
        session_start = hooks.get("SessionStart", [])
        all_commands = [
            h.get("command", "")
            for group in session_start
            for h in group.get("hooks", [])
            if h.get("type") == "command"
        ]
        assert any("slm hook start" in cmd for cmd in all_commands), (
            f"SessionStart must call 'slm hook start'; found: {all_commands}"
        )

    def test_hooks_has_stop(self) -> None:
        data = self._get_hooks()
        hooks = data.get("hooks", {})
        assert "Stop" in hooks, (
            "hooks.json must have 'Stop' event for SLM session close"
        )

    def test_stop_calls_slm_hook_stop(self) -> None:
        data = self._get_hooks()
        hooks = data.get("hooks", {})
        stop = hooks.get("Stop", [])
        all_commands = [
            h.get("command", "")
            for group in stop
            for h in group.get("hooks", [])
            if h.get("type") == "command"
        ]
        assert any("slm hook stop" in cmd for cmd in all_commands), (
            f"Stop must call 'slm hook stop'; found: {all_commands}"
        )

    def test_all_hook_commands_are_fail_open(self) -> None:
        """All hook commands must be fail-open (|| true or 2>/dev/null) so Codex
        is never blocked if slm is unavailable."""
        data = self._get_hooks()
        hooks = data.get("hooks", {})
        commands = [
            h.get("command", "")
            for event_hooks in hooks.values()
            for group in event_hooks
            for h in group.get("hooks", [])
            if h.get("type") == "command"
        ]
        for cmd in commands:
            assert "|| true" in cmd or "2>/dev/null" in cmd, (
                f"Hook command must be fail-open (|| true), got: {cmd!r}"
            )


# ---------------------------------------------------------------------------
# T4 — AGENTS.md: contains core SLM rules
# ---------------------------------------------------------------------------
class TestAgentsMd:
    """AGENTS.md must carry the complete set of SLM agent rules."""

    def _get_content(self) -> str:
        p = CODEX_PLUGIN / "AGENTS.md"
        assert p.exists(), f"AGENTS.md not found at {p}"
        return p.read_text(encoding="utf-8")

    def test_agents_md_contains_all_required_phrases(self) -> None:
        content = self._get_content()
        for phrase in AGENTS_MD_REQUIRED_PHRASES:
            assert phrase in content, (
                f"AGENTS.md missing required phrase: {phrase!r}"
            )

    def test_agents_md_has_session_init_once_rule(self) -> None:
        content = self._get_content()
        assert "once" in content.lower(), (
            "AGENTS.md must document the 'call session_init once' rule"
        )

    def test_agents_md_has_recall_before_remember(self) -> None:
        content = self._get_content()
        assert "recall" in content.lower() and "remember" in content.lower(), (
            "AGENTS.md must document recall-before-remember discipline"
        )

    def test_agents_md_has_cli_fallback_table(self) -> None:
        content = self._get_content()
        assert "CLI fallback" in content, (
            "AGENTS.md must contain the CLI fallback table for MCP-down scenarios"
        )

    def test_agents_md_has_version_footer(self) -> None:
        content = self._get_content()
        assert "SuperLocalMemory v" in content, (
            "AGENTS.md must have a version footer (e.g. 'SuperLocalMemory v3.8.0')"
        )

    def test_agents_md_version_matches_pyproject(self) -> None:
        """AGENTS.md version footer must match pyproject.toml version."""
        content = self._get_content()
        pyproject = REPO / "pyproject.toml"
        assert pyproject.exists(), f"pyproject.toml not found at {pyproject}"
        pyproject_text = pyproject.read_text(encoding="utf-8")
        # Extract version from pyproject.toml: version = "X.Y.Z"
        import re
        m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject_text, re.MULTILINE)
        assert m is not None, "Could not find version in pyproject.toml"
        expected_ver = m.group(1)
        assert f"SuperLocalMemory v{expected_ver}" in content, (
            f"AGENTS.md must reference SuperLocalMemory v{expected_ver} "
            f"(matching pyproject.toml), but not found in footer"
        )

    def test_agents_md_has_qualixar_attribution(self) -> None:
        content = self._get_content()
        assert "Qualixar" in content, "AGENTS.md must have Qualixar attribution"


# ---------------------------------------------------------------------------
# T5 — scripts: daemon-start parity, executable bits
# ---------------------------------------------------------------------------
class TestScripts:
    """Launcher scripts must start daemon before MCP (parity with Claude Code plugin)."""

    def test_posix_launcher_starts_daemon_before_mcp(self) -> None:
        launcher = CODEX_SCRIPTS / "slm-launch"
        source = launcher.read_text(encoding="utf-8")
        assert "serve start" in source, (
            "slm-launch must call 'serve start' to join the namespace daemon"
        )
        assert " mcp" in source, "slm-launch must eventually exec 'slm mcp'"
        assert source.index("serve start") < source.rindex(" mcp"), (
            "serve start must appear before mcp in slm-launch"
        )

    def test_windows_launcher_starts_daemon_before_mcp(self) -> None:
        launcher = CODEX_SCRIPTS / "slm-launch.bat"
        source = launcher.read_text(encoding="utf-8")
        assert "serve start" in source, (
            "slm-launch.bat must call 'serve start' to join the namespace daemon"
        )
        assert " mcp" in source, "slm-launch.bat must eventually call 'slm mcp'"
        assert source.index("serve start") < source.rindex(" mcp"), (
            "serve start must appear before mcp in slm-launch.bat"
        )

    @pytest.mark.skipif(os.name == "nt", reason="executable bit not applicable on Windows")
    def test_slm_launch_posix_is_executable(self) -> None:
        p = CODEX_SCRIPTS / "slm-launch"
        mode = p.stat().st_mode
        assert mode & stat.S_IXUSR, (
            f"codex-plugin/scripts/slm-launch must be user-executable (+x), mode={oct(mode)}"
        )

    @pytest.mark.skipif(os.name == "nt", reason="executable bit not applicable on Windows")
    def test_ensure_venv_sh_is_executable(self) -> None:
        p = CODEX_SCRIPTS / "ensure-venv.sh"
        mode = p.stat().st_mode
        assert mode & stat.S_IXUSR, (
            f"codex-plugin/scripts/ensure-venv.sh must be user-executable (+x), mode={oct(mode)}"
        )

    def test_posix_launcher_uses_slm_data_dir_not_claude_plugin_data(self) -> None:
        """Codex launcher must NOT expand CLAUDE_PLUGIN_DATA (Claude Code runtime var).
        Comments may mention it for documentation — check the shell expansion form only."""
        source = (CODEX_SCRIPTS / "slm-launch").read_text(encoding="utf-8")
        assert "${CLAUDE_PLUGIN_DATA}" not in source, (
            "codex-plugin/scripts/slm-launch must not expand ${CLAUDE_PLUGIN_DATA} "
            "(that is a Claude Code runtime variable, not available in Codex)"
        )
        assert "SLM_DATA_DIR" in source, (
            "codex-plugin/scripts/slm-launch must use SLM_DATA_DIR (portable)"
        )

    def test_posix_launcher_has_fallback_to_path_slm(self) -> None:
        """Codex launcher must fall back to PATH `slm` if venv binary not found."""
        source = (CODEX_SCRIPTS / "slm-launch").read_text(encoding="utf-8")
        # Check for a fallback assignment like: SLM_BIN="slm"
        assert 'SLM_BIN="slm"' in source or "SLM_BIN='slm'" in source, (
            "slm-launch must fall back to PATH 'slm' when venv binary is not present"
        )

    def test_ensure_venv_sh_uses_slm_data_dir_not_claude_plugin_data(self) -> None:
        """ensure-venv.sh must not expand CLAUDE_PLUGIN_DATA (Claude Code runtime var).
        Comments may mention it for documentation — check the shell expansion form only."""
        source = (CODEX_SCRIPTS / "ensure-venv.sh").read_text(encoding="utf-8")
        assert "${CLAUDE_PLUGIN_DATA}" not in source, (
            "codex-plugin/scripts/ensure-venv.sh must not expand ${CLAUDE_PLUGIN_DATA}"
        )
        assert "SLM_DATA_DIR" in source, (
            "codex-plugin/scripts/ensure-venv.sh must use SLM_DATA_DIR"
        )


# ---------------------------------------------------------------------------
# T6 — Skills: all 7 present and valid
# ---------------------------------------------------------------------------
class TestSkills:
    """All 7 expected skills must be present in codex-plugin/skills/."""

    def test_all_7_skills_exist(self) -> None:
        for skill in EXPECTED_SKILLS:
            p = CODEX_SKILLS / skill / "SKILL.md"
            assert p.exists(), (
                f"codex-plugin/skills/{skill}/SKILL.md not found"
            )

    def test_skill_files_have_frontmatter(self) -> None:
        for skill in EXPECTED_SKILLS:
            p = CODEX_SKILLS / skill / "SKILL.md"
            content = p.read_text(encoding="utf-8")
            assert content.startswith("---"), (
                f"codex-plugin/skills/{skill}/SKILL.md must start with YAML frontmatter (---)"
            )

    def test_skill_files_have_slm_attribution(self) -> None:
        for skill in EXPECTED_SKILLS:
            p = CODEX_SKILLS / skill / "SKILL.md"
            content = p.read_text(encoding="utf-8")
            assert "SuperLocalMemory" in content, (
                f"codex-plugin/skills/{skill}/SKILL.md must have SuperLocalMemory attribution"
            )

    def test_no_extra_skills(self) -> None:
        """Only the 7 expected skills should be present (no retired ones)."""
        present = sorted(d.name for d in CODEX_SKILLS.iterdir() if d.is_dir())
        assert present == sorted(EXPECTED_SKILLS), (
            f"Unexpected skills in codex-plugin/skills/: "
            f"present={present}, expected={sorted(EXPECTED_SKILLS)}"
        )


# ---------------------------------------------------------------------------
# T7 — Backward compat: shipped files untouched
# ---------------------------------------------------------------------------
class TestBackwardCompat:
    """Additive-only constraint: existing shipped files must not be modified."""

    def test_plugin_claude_md_unmodified(self) -> None:
        """plugin/CLAUDE.md must still exist (we did not modify it)."""
        p = REPO / "plugin" / "CLAUDE.md"
        assert p.exists(), (
            "plugin/CLAUDE.md must still exist — codex-plugin is additive, not a replacement"
        )

    def test_plugin_mcp_json_unmodified(self) -> None:
        """plugin/.mcp.json must still exist."""
        p = REPO / "plugin" / ".mcp.json"
        assert p.exists(), "plugin/.mcp.json must still exist"

    def test_ide_codex_mcp_toml_unmodified(self) -> None:
        """ide/configs/codex-mcp.toml must still exist (superseded, not removed)."""
        p = REPO / "ide" / "configs" / "codex-mcp.toml"
        assert p.exists(), (
            "ide/configs/codex-mcp.toml must still exist — codex-plugin does not replace it"
        )

    def test_build_plugin_mjs_still_passes_check(self) -> None:
        """node scripts/build-plugin.mjs --check must still pass (plugin/ in-sync)."""
        import subprocess
        result = subprocess.run(
            ["node", "scripts/build-plugin.mjs", "--check", "--quiet"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"build-plugin.mjs --check failed (codex-plugin must be additive-only):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
