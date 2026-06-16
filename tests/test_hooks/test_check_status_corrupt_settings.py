# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""Regression tests: check_status() with corrupt vs. missing settings.json.

WP-J: check_status() must distinguish 'missing' from 'corrupt' settings.json.
- Corrupt (JSONDecodeError): installed=None, error field present with parse/JSON mention
- Missing (no file): installed=False, no error field (or error=None)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest import mock


def _patch_settings_path(tmp_path: str):
    """Context manager: redirect CLAUDE_SETTINGS to a temp file path."""
    return mock.patch(
        "superlocalmemory.hooks.claude_code_hooks.CLAUDE_SETTINGS",
        Path(tmp_path),
    )


class TestCheckStatusCorruptSettings:
    """check_status() behavior when settings.json is malformed."""

    def test_corrupt_json_returns_installed_none(self, tmp_path):
        """Corrupt settings.json must yield installed=None (indeterminate), not False."""
        corrupt_file = tmp_path / "settings.json"
        corrupt_file.write_text("{invalid json}")

        with _patch_settings_path(str(corrupt_file)):
            from superlocalmemory.hooks import claude_code_hooks
            # Also patch VERSION_FILE so version state doesn't interfere
            with mock.patch.object(
                claude_code_hooks, "VERSION_FILE",
                tmp_path / ".version",
            ):
                result = claude_code_hooks.check_status()

        assert result["installed"] is None, (
            f"Expected installed=None for corrupt settings, got {result['installed']!r}"
        )

    def test_corrupt_json_has_error_field_mentioning_json(self, tmp_path):
        """Corrupt settings.json must include an 'error' field mentioning parse/JSON."""
        corrupt_file = tmp_path / "settings.json"
        corrupt_file.write_text("{invalid json}")

        with _patch_settings_path(str(corrupt_file)):
            from superlocalmemory.hooks import claude_code_hooks
            with mock.patch.object(
                claude_code_hooks, "VERSION_FILE",
                tmp_path / ".version",
            ):
                result = claude_code_hooks.check_status()

        assert "error" in result, "Expected 'error' key in result for corrupt settings"
        error_msg = (result["error"] or "").lower()
        assert "json" in error_msg or "parse" in error_msg, (
            f"Expected error to mention 'json' or 'parse', got: {result['error']!r}"
        )

    def test_missing_settings_returns_installed_false(self, tmp_path):
        """Missing settings.json must yield installed=False (clean not-installed state)."""
        missing_path = tmp_path / "nonexistent_settings.json"
        # Ensure it truly doesn't exist
        assert not missing_path.exists()

        with _patch_settings_path(str(missing_path)):
            from superlocalmemory.hooks import claude_code_hooks
            with mock.patch.object(
                claude_code_hooks, "VERSION_FILE",
                tmp_path / ".version",
            ):
                result = claude_code_hooks.check_status()

        assert result["installed"] is False, (
            f"Expected installed=False for missing settings, got {result['installed']!r}"
        )

    def test_missing_settings_has_no_error(self, tmp_path):
        """Missing settings.json must NOT produce an error field (or error=None)."""
        missing_path = tmp_path / "nonexistent_settings.json"

        with _patch_settings_path(str(missing_path)):
            from superlocalmemory.hooks import claude_code_hooks
            with mock.patch.object(
                claude_code_hooks, "VERSION_FILE",
                tmp_path / ".version",
            ):
                result = claude_code_hooks.check_status()

        # Either 'error' key is absent or its value is None/empty
        error_val = result.get("error")
        assert not error_val, (
            f"Expected no error for missing settings, got: {error_val!r}"
        )

    def test_valid_settings_without_hooks_returns_installed_false(self, tmp_path):
        """Valid settings.json with no SLM hooks must still return installed=False."""
        valid_file = tmp_path / "settings.json"
        valid_file.write_text(json.dumps({"theme": "dark"}))

        with _patch_settings_path(str(valid_file)):
            from superlocalmemory.hooks import claude_code_hooks
            with mock.patch.object(
                claude_code_hooks, "VERSION_FILE",
                tmp_path / ".version",
            ):
                result = claude_code_hooks.check_status()

        assert result["installed"] is False, (
            f"Expected installed=False for valid-but-hookless settings, got {result['installed']!r}"
        )
        # No error either
        assert not result.get("error"), f"Unexpected error: {result.get('error')!r}"

    def test_corrupt_json_hook_types_empty(self, tmp_path):
        """Corrupt settings.json: hook_types should be empty (can't parse them)."""
        corrupt_file = tmp_path / "settings.json"
        corrupt_file.write_text('{"hooks": {bad}}')

        with _patch_settings_path(str(corrupt_file)):
            from superlocalmemory.hooks import claude_code_hooks
            with mock.patch.object(
                claude_code_hooks, "VERSION_FILE",
                tmp_path / ".version",
            ):
                result = claude_code_hooks.check_status()

        assert result.get("hook_types") == [], (
            f"Expected hook_types=[] for corrupt settings, got {result.get('hook_types')!r}"
        )
