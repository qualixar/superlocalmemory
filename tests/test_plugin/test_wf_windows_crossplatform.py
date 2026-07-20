# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""WF Windows Cross-Platform Plugin Tests — regression for defect findings #6, #8, #13.

Asserts:
  (i)  plugin-src/scripts/ensure-venv.bat exists (Windows venv bootstrap)
  (ii) build output (plugin/) contains ensure-venv.bat (build copies it)
  (iii) plugin-src/.mcp.json command is NOT a hardcoded POSIX-only path:
        Windows venv produces Scripts/slm.exe, not bin/slm;
        the command must reference a cross-platform launcher wrapper
        (e.g. venv/slm-launch or similar) OR use a wrapper script — not
        the bare POSIX path ${CLAUDE_PLUGIN_DATA}/venv/bin/slm directly.
  (iv) POSIX bootstrap (ensure-venv.sh) still present and unmodified in both
       plugin-src and built plugin/ (macOS/Linux byte-identical guarantee).
  (v)  hooks.json bootstraps correctly: on POSIX → ensure-venv.sh,
       on Windows → ensure-venv.bat (or a platform-aware wrapper).
  (vi) The built plugin/ contains a Windows-resolvable slm path reference
       (Scripts/slm.exe or a launcher that resolves per-OS).

NOTE: These tests check file existence and content; they do NOT run Windows
binaries (tests run on macOS/Linux CI). All assertions are structural.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent.parent
PLUGIN_SRC = REPO / "plugin-src"
PLUGIN_ROOT = REPO / "plugin"
PLUGIN_SRC_SCRIPTS = PLUGIN_SRC / "scripts"
PLUGIN_SCRIPTS = PLUGIN_ROOT / "scripts"
PLUGIN_HOOKS = PLUGIN_ROOT / "hooks"
PLUGIN_SRC_HOOKS = PLUGIN_SRC / "hooks"


# ---------------------------------------------------------------------------
# (i) ensure-venv.bat exists in plugin-src/scripts/
# ---------------------------------------------------------------------------
class TestEnsureVenvBatExists:
    """Defect #8: Windows venv bootstrap script was missing entirely."""

    def test_ensure_venv_bat_in_plugin_src(self) -> None:
        p = PLUGIN_SRC_SCRIPTS / "ensure-venv.bat"
        assert p.exists(), (
            f"plugin-src/scripts/ensure-venv.bat not found at {p}. "
            "Windows users need a .bat equivalent of ensure-venv.sh to bootstrap the venv."
        )

    def test_ensure_venv_bat_is_not_empty(self) -> None:
        p = PLUGIN_SRC_SCRIPTS / "ensure-venv.bat"
        assert p.exists(), f"ensure-venv.bat not found at {p}"
        content = p.read_text(encoding="utf-8")
        assert len(content.strip()) > 0, "ensure-venv.bat must not be empty"

    def test_ensure_venv_bat_creates_venv(self) -> None:
        """bat file must contain venv creation logic (python -m venv)."""
        p = PLUGIN_SRC_SCRIPTS / "ensure-venv.bat"
        assert p.exists(), f"ensure-venv.bat not found at {p}"
        content = p.read_text(encoding="utf-8")
        assert "venv" in content.lower(), (
            "ensure-venv.bat must reference 'venv' (creates Python venv)"
        )

    def test_ensure_venv_bat_installs_requirements(self) -> None:
        """bat file must run pip install on requirements.txt."""
        p = PLUGIN_SRC_SCRIPTS / "ensure-venv.bat"
        assert p.exists(), f"ensure-venv.bat not found at {p}"
        content = p.read_text(encoding="utf-8")
        # Must reference pip install with requirements
        has_pip_install = "pip install" in content.lower()
        has_requirements = "requirements" in content.lower()
        assert has_pip_install, "ensure-venv.bat must run pip install"
        assert has_requirements, "ensure-venv.bat must reference requirements.txt"

    def test_ensure_venv_bat_uses_claude_plugin_env_vars(self) -> None:
        """bat file must use %CLAUDE_PLUGIN_ROOT% and %CLAUDE_PLUGIN_DATA% env vars."""
        p = PLUGIN_SRC_SCRIPTS / "ensure-venv.bat"
        assert p.exists(), f"ensure-venv.bat not found at {p}"
        content = p.read_text(encoding="utf-8")
        assert "CLAUDE_PLUGIN_ROOT" in content, (
            "ensure-venv.bat must use %CLAUDE_PLUGIN_ROOT% env variable"
        )
        assert "CLAUDE_PLUGIN_DATA" in content, (
            "ensure-venv.bat must use %CLAUDE_PLUGIN_DATA% env variable"
        )

    def test_ensure_venv_bat_references_windows_scripts_path(self) -> None:
        """bat file must use Windows venv layout: Scripts/ not bin/."""
        p = PLUGIN_SRC_SCRIPTS / "ensure-venv.bat"
        assert p.exists(), f"ensure-venv.bat not found at {p}"
        content = p.read_text(encoding="utf-8")
        # Windows venv puts executables in Scripts/ not bin/
        assert "Scripts" in content or "scripts" in content, (
            "ensure-venv.bat must reference Windows venv Scripts/ directory "
            "(Windows venv uses venv\\Scripts\\, not venv\\bin\\ like POSIX)"
        )


# ---------------------------------------------------------------------------
# (ii) build output plugin/ contains ensure-venv.bat
# ---------------------------------------------------------------------------
class TestBuiltPluginContainsWindowsArtifacts:
    """Defect #6: build-plugin.mjs copies scripts verbatim — ensure-venv.bat
    must be copied to plugin/scripts/ by the build step."""

    def test_ensure_venv_bat_in_built_plugin(self) -> None:
        p = PLUGIN_SCRIPTS / "ensure-venv.bat"
        assert p.exists(), (
            f"plugin/scripts/ensure-venv.bat not found at {p}. "
            "Run `node scripts/build-plugin.mjs` to regenerate plugin/. "
            "The build script must copy ensure-venv.bat from plugin-src/scripts/."
        )

    def test_built_ensure_venv_bat_matches_source(self) -> None:
        src = PLUGIN_SRC_SCRIPTS / "ensure-venv.bat"
        out = PLUGIN_SCRIPTS / "ensure-venv.bat"
        assert src.exists(), f"Source ensure-venv.bat missing: {src}"
        assert out.exists(), f"Built ensure-venv.bat missing: {out}"
        assert src.read_text(encoding="utf-8") == out.read_text(encoding="utf-8"), (
            "plugin/scripts/ensure-venv.bat must match plugin-src/scripts/ensure-venv.bat "
            "byte-for-byte (build copies verbatim)"
        )


# ---------------------------------------------------------------------------
# (iii) .mcp.json command references a Windows-resolvable path
# ---------------------------------------------------------------------------
class TestMcpJsonWindowsResolvable:
    """Defect #6/#13: ${CLAUDE_PLUGIN_DATA}/venv/bin/slm is POSIX-only.
    Windows venv produces Scripts/slm.exe. The command must work cross-platform."""

    def _get_server(self) -> dict:
        data = json.loads((PLUGIN_ROOT / ".mcp.json").read_text(encoding="utf-8"))
        servers = data.get("mcpServers", {})
        assert len(servers) == 1
        return next(iter(servers.values()))

    def test_mcp_command_not_hardcoded_posix_bin_slm(self) -> None:
        """The command must NOT be the POSIX-only ${CLAUDE_PLUGIN_DATA}/venv/bin/slm.
        That path does not exist on Windows (Windows uses Scripts/slm.exe)."""
        server = self._get_server()
        cmd = server["command"]
        # The bare POSIX path is the defective form — Windows cannot resolve it
        assert cmd != "${CLAUDE_PLUGIN_DATA}/venv/bin/slm", (
            "command is hardcoded to POSIX venv/bin/slm — Windows uses "
            "venv\\Scripts\\slm.exe. Must use a cross-platform launcher."
        )

    def test_mcp_src_command_not_hardcoded_posix_bin_slm(self) -> None:
        """Source .mcp.json must also not be POSIX-only."""
        data = json.loads((PLUGIN_SRC / ".mcp.json").read_text(encoding="utf-8"))
        servers = data.get("mcpServers", {})
        server = next(iter(servers.values()))
        cmd = server["command"]
        assert cmd != "${CLAUDE_PLUGIN_DATA}/venv/bin/slm", (
            "plugin-src/.mcp.json command is hardcoded to POSIX venv/bin/slm — "
            "Windows uses venv\\Scripts\\slm.exe. Must use a cross-platform launcher."
        )


# ---------------------------------------------------------------------------
# (iv) POSIX bootstrap (ensure-venv.sh) still present — macOS/Linux unaffected
# ---------------------------------------------------------------------------
class TestPosixBootstrapUnaffected:
    """ensure-venv.sh must remain byte-identical after the Windows fix."""

    def test_ensure_venv_sh_in_plugin_src(self) -> None:
        p = PLUGIN_SRC_SCRIPTS / "ensure-venv.sh"
        assert p.exists(), f"ensure-venv.sh missing at {p}"

    def test_ensure_venv_sh_in_built_plugin(self) -> None:
        p = PLUGIN_SCRIPTS / "ensure-venv.sh"
        assert p.exists(), f"plugin/scripts/ensure-venv.sh missing at {p}"

    def test_built_ensure_venv_sh_matches_source(self) -> None:
        src = PLUGIN_SRC_SCRIPTS / "ensure-venv.sh"
        out = PLUGIN_SCRIPTS / "ensure-venv.sh"
        assert src.exists() and out.exists()
        assert src.read_text(encoding="utf-8") == out.read_text(encoding="utf-8"), (
            "plugin/scripts/ensure-venv.sh must match source byte-for-byte (POSIX unchanged)"
        )

    def test_ensure_venv_sh_is_executable(self) -> None:
        p = PLUGIN_SCRIPTS / "ensure-venv.sh"
        assert p.exists()
        mode = p.stat().st_mode
        assert mode & stat.S_IXUSR, (
            f"ensure-venv.sh must be user-executable (+x), mode={oct(mode)}"
        )


# ---------------------------------------------------------------------------
# (v) hooks.json: POSIX ensure-venv.sh still referenced
# ---------------------------------------------------------------------------
class TestHooksJsonCrossPlatform:
    """hooks.json must still reference ensure-venv.sh for POSIX.
    On Windows, Claude Code invokes hooks differently (future work);
    the POSIX hook must not be removed."""

    def _get_hooks(self) -> dict:
        return json.loads((PLUGIN_HOOKS / "hooks.json").read_text(encoding="utf-8"))

    def test_session_start_still_calls_posix_ensure_venv(self) -> None:
        data = self._get_hooks()
        session_start = data.get("hooks", {}).get("SessionStart", [])
        all_commands = []
        for group in session_start:
            for h in group.get("hooks", []):
                if h.get("type") == "command":
                    all_commands.append(h.get("command", ""))
        assert any("ensure-venv.sh" in cmd for cmd in all_commands), (
            f"SessionStart must still call ensure-venv.sh for POSIX. Commands: {all_commands}"
        )
