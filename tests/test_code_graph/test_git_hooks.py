# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for git_hooks (Phase 6).

Tests: hook installation, idempotent install, uninstall,
append to existing, run_post_commit.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.code_graph.git_hooks import (
    _HOOK_MARKER,
    install_post_commit_hook,
    run_post_commit,
    uninstall_post_commit_hook,
)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a fake git repo structure."""
    git_dir = tmp_path / ".git" / "hooks"
    git_dir.mkdir(parents=True)
    return tmp_path


# ---------------------------------------------------------------------------
# Installation tests
# ---------------------------------------------------------------------------


class TestInstallHook:
    """Test install_post_commit_hook."""

    def test_install_new_hook(self, git_repo: Path) -> None:
        result = install_post_commit_hook(git_repo)
        assert result["success"] is True
        assert result["action"] == "installed"

        hook_path = git_repo / ".git" / "hooks" / "post-commit"
        assert hook_path.exists()
        content = hook_path.read_text(encoding="utf-8")
        assert _HOOK_MARKER in content
        assert "#!/bin/sh" in content

        # Verify executable. Windows files have no executable bit (Git for
        # Windows runs a hook by its #! line), so there is nothing to check.
        if os.name != "nt":
            assert hook_path.stat().st_mode & stat.S_IXUSR

    def test_install_idempotent(self, git_repo: Path) -> None:
        """Second install should detect existing and return already_present."""
        result1 = install_post_commit_hook(git_repo)
        assert result1["action"] == "installed"

        result2 = install_post_commit_hook(git_repo)
        assert result2["success"] is True
        assert result2["action"] == "already_present"

        # Content should not be duplicated
        content = (git_repo / ".git" / "hooks" / "post-commit").read_text(encoding="utf-8")
        assert content.count(_HOOK_MARKER) == 1

    def test_install_appends_to_existing(self, git_repo: Path) -> None:
        """If a hook already exists (without our marker), append."""
        hook_path = git_repo / ".git" / "hooks" / "post-commit"
        hook_path.write_text("#!/bin/sh\necho 'existing hook'\n", encoding="utf-8")
        hook_path.chmod(0o755)

        result = install_post_commit_hook(git_repo)
        assert result["success"] is True
        assert result["action"] == "appended"

        content = hook_path.read_text(encoding="utf-8")
        assert "existing hook" in content
        assert _HOOK_MARKER in content

    def test_install_creates_hooks_dir(self, tmp_path: Path) -> None:
        """Install should create .git/hooks/ if it doesn't exist."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        # hooks dir does NOT exist

        result = install_post_commit_hook(tmp_path)
        assert result["success"] is True
        assert (tmp_path / ".git" / "hooks" / "post-commit").exists()


# ---------------------------------------------------------------------------
# Uninstall tests
# ---------------------------------------------------------------------------


class TestUninstallHook:
    """Test uninstall_post_commit_hook."""

    def test_uninstall_removes_section(self, git_repo: Path) -> None:
        install_post_commit_hook(git_repo)
        result = uninstall_post_commit_hook(git_repo)
        assert result["success"] is True
        assert result["action"] == "removed"

        hook_path = git_repo / ".git" / "hooks" / "post-commit"
        # Hook file should be gone (was only our content)
        assert not hook_path.exists()

    def test_uninstall_preserves_other_content(self, git_repo: Path) -> None:
        """Uninstall should only remove our section, keep the rest."""
        hook_path = git_repo / ".git" / "hooks" / "post-commit"
        hook_path.write_text("#!/bin/sh\necho 'keep this'\n", encoding="utf-8")
        hook_path.chmod(0o755)

        # Install (append)
        install_post_commit_hook(git_repo)
        content_before = hook_path.read_text(encoding="utf-8")
        assert _HOOK_MARKER in content_before

        # Uninstall
        result = uninstall_post_commit_hook(git_repo)
        assert result["action"] == "removed"

        content_after = hook_path.read_text(encoding="utf-8")
        assert "keep this" in content_after
        assert _HOOK_MARKER not in content_after

    def test_uninstall_not_found(self, git_repo: Path) -> None:
        result = uninstall_post_commit_hook(git_repo)
        assert result["success"] is True
        assert result["action"] == "not_found"

    def test_uninstall_no_marker(self, git_repo: Path) -> None:
        """Hook exists but doesn't contain our marker."""
        hook_path = git_repo / ".git" / "hooks" / "post-commit"
        hook_path.write_text("#!/bin/sh\necho 'other hook'\n", encoding="utf-8")

        result = uninstall_post_commit_hook(git_repo)
        assert result["success"] is True
        assert result["action"] == "not_found"


# ---------------------------------------------------------------------------
# run_post_commit tests
# ---------------------------------------------------------------------------


class TestRunPostCommit:
    """Test run_post_commit."""

    def test_run_with_no_git(self, tmp_path: Path) -> None:
        """Should handle missing git gracefully."""
        with patch("superlocalmemory.code_graph.git_hooks.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            result = run_post_commit(tmp_path)
            assert result["success"] is False
            assert "git not found" in result["error"]

    def test_run_with_git_error(self, tmp_path: Path) -> None:
        """Should handle git errors gracefully."""
        with patch("superlocalmemory.code_graph.git_hooks.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 128
            mock_result.stderr = "fatal: not a git repository"
            mock_run.return_value = mock_result

            result = run_post_commit(tmp_path)
            assert result["success"] is False

    def test_run_with_changed_files(self, tmp_path: Path) -> None:
        """Should detect changed files and return count."""
        with patch("superlocalmemory.code_graph.git_hooks.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "src/auth/handler.py\nsrc/utils.py\nREADME.md\n"
            mock_run.return_value = mock_result

            result = run_post_commit(tmp_path)
            assert result["success"] is True
            assert result["files_updated"] == 2  # .py files only
            assert result["duration_ms"] >= 0

    def test_run_with_no_supported_files(self, tmp_path: Path) -> None:
        """No supported files changed should return 0."""
        with patch("superlocalmemory.code_graph.git_hooks.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "README.md\npackage.json\n"
            mock_run.return_value = mock_result

            result = run_post_commit(tmp_path)
            assert result["success"] is True
            assert result["files_updated"] == 0
