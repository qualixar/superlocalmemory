# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for CLI --json agent-native output.

Verifies:
- JSON envelope structure (success, command, version, data/error)
- Backward compatibility (no --json = unchanged human output)
- next_actions HATEOAS guidance
- Error envelopes

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from superlocalmemory.cli.json_output import json_print
from superlocalmemory.cli.main import main

# -- JSON Envelope Helper Tests -------------------------------------------


class TestJsonEnvelope:
    """Test the shared json_print() helper."""

    def test_success_envelope_structure(self, capsys):
        json_print("test_cmd", data={"key": "value"})
        output = json.loads(capsys.readouterr().out)
        assert output["success"] is True
        assert output["command"] == "test_cmd"
        assert "version" in output
        assert output["data"]["key"] == "value"

    def test_output_is_compact_not_pretty_printed(self, capsys):
        """Q9 (2026-10-06): --json is documented as agent-native output for
        scripts/CI — a script or an agent capturing stdout pays per byte,
        not per readable line. Same data, no indentation whitespace."""
        json_print("test_cmd", data={"nested": {"a": [1, 2, 3]}})
        raw = capsys.readouterr().out
        assert raw.count("\n") == 1  # the envelope plus trailing newline only
        assert json.loads(raw) == {
            "success": True, "command": "test_cmd", "version": json.loads(raw)["version"],
            "data": {"nested": {"a": [1, 2, 3]}},
        }

    def test_error_envelope_structure(self, capsys):
        json_print("test_cmd", error={"code": "TEST_ERR", "message": "something failed"})
        output = json.loads(capsys.readouterr().out)
        assert output["success"] is False
        assert output["command"] == "test_cmd"
        assert output["error"]["code"] == "TEST_ERR"
        assert output["error"]["message"] == "something failed"
        assert "data" not in output

    def test_next_actions_included(self, capsys):
        json_print("test_cmd", data={}, next_actions=[
            {"command": "slm status --json", "description": "Check status"},
        ])
        output = json.loads(capsys.readouterr().out)
        assert "next_actions" in output
        assert len(output["next_actions"]) == 1
        assert output["next_actions"][0]["command"] == "slm status --json"

    def test_metadata_included(self, capsys):
        json_print("test_cmd", data={}, metadata={"execution_time_ms": 42})
        output = json.loads(capsys.readouterr().out)
        assert "metadata" in output
        assert output["metadata"]["execution_time_ms"] == 42

    def test_no_next_actions_means_no_key(self, capsys):
        json_print("test_cmd", data={"k": 1})
        output = json.loads(capsys.readouterr().out)
        assert "next_actions" not in output

    def test_no_metadata_means_no_key(self, capsys):
        json_print("test_cmd", data={"k": 1})
        output = json.loads(capsys.readouterr().out)
        assert "metadata" not in output

    def test_output_is_valid_json(self, capsys):
        json_print("test_cmd", data={"nested": {"a": [1, 2, 3]}})
        raw = capsys.readouterr().out
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_default_serializer_handles_non_json_types(self, capsys):
        """Ensure pathlib.Path, datetime, etc. don't crash json.dumps."""
        from pathlib import Path
        json_print("test_cmd", data={"path": Path("/tmp/test")})
        output = json.loads(capsys.readouterr().out)
        assert output["data"]["path"] == str(Path("/tmp/test"))


# -- CLI --json Integration Tests (no engine needed) ----------------------


class TestStatusJson:
    """Test slm status --json (no engine needed — only reads config)."""

    def test_status_json_valid(self, capsys):
        with patch("sys.argv", ["slm", "status", "--json"]):
            main()
        output = json.loads(capsys.readouterr().out)
        assert output["success"] is True
        assert output["command"] == "status"
        assert "mode" in output["data"]
        assert "provider" in output["data"]
        assert "next_actions" in output

    def test_status_json_has_version(self, capsys):
        with patch("sys.argv", ["slm", "status", "--json"]):
            main()
        output = json.loads(capsys.readouterr().out)
        assert "version" in output
        assert isinstance(output["version"], str)


class TestModeJson:
    """Test slm mode --json (no engine needed — only reads config)."""

    def test_mode_get_json(self, capsys):
        with patch("sys.argv", ["slm", "mode", "--json"]):
            main()
        output = json.loads(capsys.readouterr().out)
        assert output["success"] is True
        assert output["command"] == "mode"
        assert "current_mode" in output["data"]
        assert output["data"]["current_mode"] in ("A", "B", "C")

    def test_mode_json_has_next_actions(self, capsys):
        with patch("sys.argv", ["slm", "mode", "--json"]):
            main()
        output = json.loads(capsys.readouterr().out)
        assert "next_actions" in output
        assert len(output["next_actions"]) >= 1


# -- Backward Compatibility Tests -----------------------------------------


class TestBackwardCompatibility:
    """Verify --json flag doesn't break existing human-readable output.

    No-args exit code, the "SuperLocalMemory V4" banner on `status`, the
    "mode" text on `mode`, and dispatch's importability are already asserted
    without --json in tests/test_cli_core.py (test_main_no_args,
    test_main_status, test_main_mode_get, test_commands_dispatch_importable)
    — duplicating those assertions here added no coverage. What's left is
    the one thing specific to the --json feature: a human-mode invocation
    must NOT emit JSON, and --version must still work.
    """

    def test_status_human_output_is_not_json(self, capsys):
        """status without --json produces human text, not a JSON envelope."""
        with patch("sys.argv", ["slm", "status"]):
            main()
        captured = capsys.readouterr()
        with pytest.raises(json.JSONDecodeError):
            json.loads(captured.out)

    def test_version_flag_still_works(self):
        """--version flag still works."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["slm", "--version"]):
                main()
