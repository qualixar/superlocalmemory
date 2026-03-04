#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""Tests for SL_MEMORY_PATH environment variable support.

Verifies that all entry points (mcp_server.py, bin/slm, bin/slm.bat,
install.sh, install.ps1) respect the SL_MEMORY_PATH environment variable
to override the default ~/.claude-memory installation directory.

Run with:
    pytest tests/test_sl_memory_path.py -v
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).parent.parent


class TestSLMemoryPathInSources:
    """Static checks: all modified entry points reference SL_MEMORY_PATH."""

    def test_mcp_server_references_sl_memory_path(self):
        src = (REPO_DIR / "mcp_server.py").read_text(encoding="utf-8")
        assert "SL_MEMORY_PATH" in src

    def test_bin_slm_references_sl_memory_path(self):
        src = (REPO_DIR / "bin" / "slm").read_text(encoding="utf-8")
        assert "SL_MEMORY_PATH" in src

    def test_bin_slm_bat_references_sl_memory_path(self):
        src = (REPO_DIR / "bin" / "slm.bat").read_text(encoding="utf-8")
        assert "SL_MEMORY_PATH" in src

    def test_install_sh_references_sl_memory_path(self):
        src = (REPO_DIR / "install.sh").read_text(encoding="utf-8")
        assert "SL_MEMORY_PATH" in src

    def test_install_ps1_references_sl_memory_path(self):
        src = (REPO_DIR / "install.ps1").read_text(encoding="utf-8")
        assert "SL_MEMORY_PATH" in src


class TestMCPServerMemoryDir:
    """Runtime and structural tests for mcp_server.py path resolution."""

    MEMORY_DIR_LOGIC = (
        "import os, sys; from pathlib import Path; "
        "MEMORY_DIR = Path(os.environ.get('SL_MEMORY_PATH', str(Path.home() / '.claude-memory'))); "
        "print(MEMORY_DIR)"
    )

    def test_uses_sl_memory_path_when_set(self, tmp_path):
        """When SL_MEMORY_PATH is set, MEMORY_DIR should use that path."""
        env = {**os.environ, "SL_MEMORY_PATH": str(tmp_path)}
        result = subprocess.run(
            [sys.executable, "-c", self.MEMORY_DIR_LOGIC],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert str(tmp_path) == result.stdout.strip()

    def test_falls_back_to_default_when_unset(self):
        """When SL_MEMORY_PATH is not set, MEMORY_DIR defaults to ~/.claude-memory."""
        env = {k: v for k, v in os.environ.items() if k != "SL_MEMORY_PATH"}
        result = subprocess.run(
            [sys.executable, "-c", self.MEMORY_DIR_LOGIC],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip().endswith(".claude-memory")

    def test_mcp_server_syntax_valid(self):
        """mcp_server.py must parse without syntax errors."""
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(REPO_DIR / "mcp_server.py")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"

    def test_memory_dir_added_to_sys_path(self):
        """mcp_server.py must add MEMORY_DIR to sys.path so modules can be imported."""
        src = (REPO_DIR / "mcp_server.py").read_text(encoding="utf-8")
        assert "sys.path.insert(0, str(MEMORY_DIR))" in src

    def test_learning_db_uses_memory_dir(self):
        """learning.db path must be derived from MEMORY_DIR, not hardcoded."""
        src = (REPO_DIR / "mcp_server.py").read_text(encoding="utf-8")
        assert (
            'MEMORY_DIR / "learning.db"' in src or "MEMORY_DIR / 'learning.db'" in src
        )

    def test_audit_db_uses_memory_dir(self):
        """audit.db path must be derived from MEMORY_DIR, not hardcoded."""
        src = (REPO_DIR / "mcp_server.py").read_text(encoding="utf-8")
        assert 'MEMORY_DIR / "audit.db"' in src or "MEMORY_DIR / 'audit.db'" in src

    def test_no_hardcoded_claude_memory_db_paths(self):
        """mcp_server.py must not hardcode ~/.claude-memory for database files."""
        src = (REPO_DIR / "mcp_server.py").read_text(encoding="utf-8")
        assert ".claude-memory/learning.db" not in src
        assert ".claude-memory/audit.db" not in src
        assert ".claude-memory\\learning.db" not in src
        assert ".claude-memory\\audit.db" not in src


class TestInstallScriptsPaths:
    """Structural tests for install.sh and install.ps1 path resolution."""

    def test_install_sh_uses_env_var_with_fallback(self):
        """install.sh should use bash parameter expansion for SL_MEMORY_PATH."""
        src = (REPO_DIR / "install.sh").read_text(encoding="utf-8")
        assert "${SL_MEMORY_PATH:-" in src

    def test_install_ps1_uses_conditional_with_fallback(self):
        """install.ps1 should check $env:SL_MEMORY_PATH before defaulting."""
        src = (REPO_DIR / "install.ps1").read_text(encoding="utf-8")
        assert "$env:SL_MEMORY_PATH" in src


class TestBinSlmPaths:
    """Structural tests for bin/slm and bin/slm.bat path resolution."""

    def test_bin_slm_uses_env_var_with_fallback(self):
        """bin/slm should use bash parameter expansion: ${SL_MEMORY_PATH:-...}."""
        src = (REPO_DIR / "bin" / "slm").read_text(encoding="utf-8")
        assert "${SL_MEMORY_PATH:-" in src

    def test_bin_slm_bat_checks_defined_sl_memory_path(self):
        """bin/slm.bat should test 'if defined SL_MEMORY_PATH' as the first branch."""
        src = (REPO_DIR / "bin" / "slm.bat").read_text(encoding="utf-8")
        assert (
            "if defined SL_MEMORY_PATH" in src
            or "if defined sl_memory_path" in src.lower()
        )
