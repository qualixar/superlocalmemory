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
import pytest
from unittest.mock import patch

from superlocalmemory.cli.main import main
from superlocalmemory.cli.json_output import json_print


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
    """Verify --json flag doesn't break existing human-readable output."""

    def test_no_args_still_exits_zero(self):
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["slm"]):
                main()
        assert exc_info.value.code == 0

    def test_status_human_unchanged(self, capsys):
        """status without --json still produces human text, not JSON."""
        with patch("sys.argv", ["slm", "status"]):
            main()
        captured = capsys.readouterr()
        assert "SuperLocalMemory V4" in captured.out
        # Must NOT be valid JSON (it's human-readable text)
        with pytest.raises(json.JSONDecodeError):
            json.loads(captured.out)

    def test_mode_human_unchanged(self, capsys):
        """mode without --json still produces human text."""
        with patch("sys.argv", ["slm", "mode"]):
            main()
        captured = capsys.readouterr()
        assert "mode" in captured.out.lower()

    def test_version_flag_still_works(self):
        """--version flag still works."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["slm", "--version"]):
                main()

    def test_dispatch_importable(self):
        """dispatch function unchanged."""
        from superlocalmemory.cli.commands import dispatch
        assert callable(dispatch)
