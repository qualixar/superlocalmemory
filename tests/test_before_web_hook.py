# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.43

"""Tests for before_web_hook.py — pre-web recall on WebSearch/WebFetch.

Covers:
  - Pure logic: _extract_query, _read_input
  - Subprocess: _run_recall happy path, timeout, non-zero exit, short output
  - Main entry: silent on missing input, silent on empty query, fires reminder
    when recall returns useful content
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.hooks import before_web_hook as bwh


# --------------------------------------------------------------------------
# _extract_query
# --------------------------------------------------------------------------

class TestExtractQuery:
    def test_websearch_query_field(self):
        assert bwh._extract_query({"tool_input": {"query": "hello world"}}) == "hello world"

    def test_webfetch_url_field(self):
        assert bwh._extract_query({"tool_input": {"url": "https://example.com"}}) == "https://example.com"

    def test_webfetch_prompt_field_priority(self):
        # prompt takes priority over url when both present
        ti = {"tool_input": {"url": "https://example.com", "prompt": "summarize this"}}
        assert bwh._extract_query(ti) == "summarize this"

    def test_query_takes_priority(self):
        ti = {"tool_input": {"query": "Q", "prompt": "P", "url": "U"}}
        assert bwh._extract_query(ti) == "Q"

    def test_missing_tool_input(self):
        assert bwh._extract_query({}) == ""

    def test_truncates_to_max(self):
        long = "a" * 1000
        out = bwh._extract_query({"tool_input": {"query": long}})
        assert len(out) == bwh._QUERY_TRUNCATE

    def test_non_string_value(self):
        # int, list, None should all gracefully return ""
        assert bwh._extract_query({"tool_input": {"query": 42}}) == ""
        assert bwh._extract_query({"tool_input": {"url": None}}) == ""

    def test_non_dict_tool_input(self):
        assert bwh._extract_query({"tool_input": "not a dict"}) == ""

    def test_strips_whitespace(self):
        assert bwh._extract_query({"tool_input": {"query": "  hello  "}}) == "hello"


# --------------------------------------------------------------------------
# _read_input
# --------------------------------------------------------------------------

class TestReadInput:
    def test_empty_stdin(self):
        with patch("sys.stdin", io.StringIO("")):
            assert bwh._read_input() == {}

    def test_invalid_json(self):
        with patch("sys.stdin", io.StringIO("not json {{{")):
            assert bwh._read_input() == {}

    def test_non_dict_json(self):
        with patch("sys.stdin", io.StringIO('["not", "dict"]')):
            assert bwh._read_input() == {}

    def test_valid_dict(self):
        with patch("sys.stdin", io.StringIO('{"tool_input": {"query": "hi"}}')):
            assert bwh._read_input() == {"tool_input": {"query": "hi"}}


# --------------------------------------------------------------------------
# _run_recall (subprocess)
# --------------------------------------------------------------------------

class TestRunRecall:
    def test_happy_path(self):
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "a" * 500  # 500 chars, above MIN_USEFUL
        with patch("subprocess.run", return_value=fake_result):
            out = bwh._run_recall("test query")
        assert len(out) == 500

    def test_truncates_to_max(self):
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "x" * 10000
        with patch("subprocess.run", return_value=fake_result):
            out = bwh._run_recall("q")
        assert len(out) == bwh._RECALLED_MAX_CHARS

    def test_non_zero_return_code_empty(self):
        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stdout = "error"
        with patch("subprocess.run", return_value=fake_result):
            assert bwh._run_recall("q") == ""

    def test_short_output_filtered(self):
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "tiny"  # below MIN_USEFUL=50
        with patch("subprocess.run", return_value=fake_result):
            assert bwh._run_recall("q") == ""

    def test_timeout_empty(self):
        with patch("subprocess.run",
                   side_effect=subprocess.TimeoutExpired("slm", 3)):
            assert bwh._run_recall("q") == ""

    def test_oserror_empty(self):
        with patch("subprocess.run", side_effect=OSError("slm not on PATH")):
            assert bwh._run_recall("q") == ""


# --------------------------------------------------------------------------
# main() — integration
# --------------------------------------------------------------------------

class TestMain:
    def _invoke(self, stdin_data: str, capsys, recall_output: str = ""):
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = recall_output

        with patch("sys.stdin", io.StringIO(stdin_data)), \
             patch("subprocess.run", return_value=fake_result):
            rc = bwh.main()
        out = capsys.readouterr().out
        return rc, out

    def test_empty_stdin_silent(self, capsys):
        rc, out = self._invoke("", capsys)
        assert rc == 0
        assert out == ""

    def test_empty_query_silent(self, capsys):
        rc, out = self._invoke(
            json.dumps({"tool_input": {"query": ""}}), capsys, recall_output="X" * 500
        )
        assert rc == 0
        assert out == ""

    def test_short_query_silent(self, capsys):
        rc, out = self._invoke(
            json.dumps({"tool_input": {"query": "hi"}}), capsys, recall_output="X" * 500
        )
        assert rc == 0
        assert out == ""

    def test_recall_empty_silent(self, capsys):
        rc, out = self._invoke(
            json.dumps({"tool_input": {"query": "valid query here"}}),
            capsys, recall_output="",  # MIN_USEFUL filter kicks in
        )
        assert rc == 0
        assert out == ""

    def test_fires_reminder_on_useful_recall(self, capsys):
        recalled = "MEMORY LINE 1\nMEMORY LINE 2\nMEMORY LINE 3\nMEMORY LINE 4 with content words"
        rc, out = self._invoke(
            json.dumps({"tool_input": {"query": "valid query about something"}}),
            capsys, recall_output=recalled,
        )
        assert rc == 0
        assert "PRE-WEB RECALL" in out
        assert "MEMORY LINE 1" in out
        assert "UNTRUSTED SLM EVIDENCE v1" in out

    def test_query_preview_in_reminder(self, capsys):
        recalled = "X" * 500
        rc, out = self._invoke(
            json.dumps({"tool_input": {"query": "specific search topic abc def"}}),
            capsys, recall_output=recalled,
        )
        assert "specific search topic" in out

    def test_quote_escaped_in_preview(self, capsys):
        """Ensure preview doesn't break the system-reminder block on quote chars."""
        recalled = "X" * 500
        rc, out = self._invoke(
            json.dumps({"tool_input": {"query": 'query with "quotes" inside it'}}),
            capsys, recall_output=recalled,
        )
        # Should not have unescaped double-quotes that would break the block
        assert rc == 0

    def test_fail_open_on_exception(self, capsys):
        """If anything raises, main() must still return 0."""
        with patch("sys.stdin", io.StringIO('{"tool_input": {"query": "ok"}}')), \
             patch("superlocalmemory.hooks.before_web_hook._run_recall",
                   side_effect=RuntimeError("simulated crash")):
            rc = bwh.main()
        assert rc == 0
