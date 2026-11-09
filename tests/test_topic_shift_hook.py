# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.43

"""Tests for topic_shift_hook.py — event-based topic-shift detection.

Covers:
  - Pure logic: extract_content_words, detect_shift, is_substantive
  - State file: save_state / load_state round-trip, corruption recovery, staleness
  - Main entry: silent on malformed input, fires on shift, updates window
  - Observability log: writes line per decision, respects SLM_TOPIC_SHIFT_LOG=0
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
from unittest.mock import patch

import pytest

from superlocalmemory.hooks import topic_shift_hook as tsh

# --------------------------------------------------------------------------
# Pure logic
# --------------------------------------------------------------------------

class TestExtractContentWords:
    def test_empty(self):
        assert tsh.extract_content_words("") == []

    def test_stopwords_filtered(self):
        words = tsh.extract_content_words("This is a test of the system architecture")
        assert "the" not in words
        assert "is" not in words
        assert "test" in words
        assert "architecture" in words

    def test_short_words_filtered(self):
        words = tsh.extract_content_words("a is at by I we go")
        # All <3 chars or stopwords — should be empty
        assert words == []

    def test_hyphenated_splits(self):
        """varunpratap-website should split into 2 tokens."""
        words = tsh.extract_content_words("varunpratap-website redesign")
        assert "varunpratap" in words
        assert "website" in words
        # NOT the joined form
        assert "varunpratap-website" not in words

    def test_truncation_at_max(self):
        """Inputs longer than MAX_PROMPT_CHARS are truncated before regex."""
        long_input = "alpha " * 2000  # ~12000 chars
        words = tsh.extract_content_words(long_input)
        # Should still produce at least one match (regex bounded but works)
        assert "alpha" in words

    def test_generic_temporal_filtered(self):
        """'next', 'week', 'time' etc. should be in stopword list."""
        words = tsh.extract_content_words("Tell me next week about the time stuff")
        assert "next" not in words
        assert "week" not in words
        assert "time" not in words
        assert "stuff" not in words


class TestIsSubstantive:
    @pytest.mark.parametrize("text", ["ok", "yes", "no", "thanks", "go", "y", "n"])
    def test_acks_not_substantive(self, text):
        assert not tsh.is_substantive(text)

    @pytest.mark.parametrize("text", ["yes go", "okay thanks", "thanks done"])
    def test_compound_acks_not_substantive(self, text):
        assert not tsh.is_substantive(text)

    @pytest.mark.parametrize("text", ["", "hi", "short"])
    def test_too_short_not_substantive(self, text):
        assert not tsh.is_substantive(text)

    def test_real_prompt_substantive(self):
        assert tsh.is_substantive("Let's start building the homepage component now")


class TestDetectShift:
    def test_empty_window_no_shift(self):
        words = ["alpha", "beta", "gamma", "delta", "epsilon"]
        fired, max_ov = tsh.detect_shift(words, [])
        assert fired is False
        assert max_ov == -1

    def test_short_current_no_shift(self):
        fired, max_ov = tsh.detect_shift(
            ["alpha"],
            [["beta", "gamma"], ["delta", "epsilon"], ["zeta", "eta"]],
        )
        assert fired is False
        assert max_ov == -1

    def test_single_window_entry_no_shift(self):
        fired, max_ov = tsh.detect_shift(
            ["alpha", "beta", "gamma", "delta", "epsilon"],
            [["zeta", "eta", "theta"]],
        )
        # only 1 window entry < MIN_WINDOW_ENTRIES (2)
        assert fired is False

    def test_genuine_pivot_fires(self):
        window = [
            tsh.extract_content_words("Agent Amplifier dogfood window phase 13"),
            tsh.extract_content_words("AAA stage 14 launch prep github push"),
            tsh.extract_content_words("Agent Amplifier mkdocs site publish PyPI"),
        ]
        current = tsh.extract_content_words(
            "Show me the kitchen recipe garam masala spice ratio coriander cumin")
        fired, max_ov = tsh.detect_shift(current, window)
        assert fired is True
        assert max_ov == 0

    def test_same_topic_follow_up_silent(self):
        window = [
            tsh.extract_content_words("varunpratap.com homepage redesign bento layout"),
            tsh.extract_content_words("homepage hero section copy tuning fork"),
        ]
        current = tsh.extract_content_words(
            "Update the homepage tuning fork copy with new wording across pages")
        fired, _ = tsh.detect_shift(current, window)
        assert fired is False


# --------------------------------------------------------------------------
# State file
# --------------------------------------------------------------------------

class TestStateFile:
    def test_path_deterministic(self):
        assert tsh.state_path("abc") == tsh.state_path("abc")
        assert tsh.state_path("abc") != tsh.state_path("xyz")

    def test_path_under_tmp(self):
        assert tsh.state_path("anything").startswith(tempfile.gettempdir())

    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "state.json")
        win = [["alpha", "beta"], ["gamma"]]
        tsh.save_state(path, win)
        loaded = tsh.load_state(path)
        assert loaded == win

    def test_missing_file_returns_empty(self, tmp_path):
        path = str(tmp_path / "missing.json")
        assert tsh.load_state(path) == []

    def test_corrupt_file_returns_empty(self, tmp_path):
        path = str(tmp_path / "corrupt.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("not valid JSON {{{")
        assert tsh.load_state(path) == []

    def test_wrong_version_returns_empty(self, tmp_path):
        path = str(tmp_path / "wrong-version.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 99, "window": [["alpha"]]}, f)
        assert tsh.load_state(path) == []

    def test_wrong_shape_returns_empty(self, tmp_path):
        path = str(tmp_path / "wrong-shape.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "window": "not a list"}, f)
        assert tsh.load_state(path) == []

    def test_stale_file_ignored(self, tmp_path):
        path = str(tmp_path / "stale.json")
        tsh.save_state(path, [["alpha"]])
        # Backdate mtime 25h
        old = time.time() - (25 * 3600)
        os.utime(path, (old, old))
        assert tsh.load_state(path) == []

    def test_window_trimmed_on_save(self, tmp_path):
        path = str(tmp_path / "trim.json")
        big = [[f"word{i}"] for i in range(20)]
        tsh.save_state(path, big)
        loaded = tsh.load_state(path)
        assert len(loaded) == tsh._WINDOW_SIZE

    def test_save_silent_on_io_error(self, tmp_path):
        # Path containing a non-existent directory — save should swallow error
        bad_path = str(tmp_path / "nonexistent" / "subdir" / "state.json")
        # Should not raise
        tsh.save_state(bad_path, [["x"]])


# --------------------------------------------------------------------------
# main() — stdin/stdout integration
# --------------------------------------------------------------------------

class TestMain:
    def _invoke(self, stdin_data: str, capsys):
        with patch("sys.stdin", io.StringIO(stdin_data)):
            rc = tsh.main()
        out = capsys.readouterr().out
        return rc, out

    def test_empty_stdin_silent(self, capsys):
        rc, out = self._invoke("", capsys)
        assert rc == 0
        assert out == ""

    def test_malformed_json_silent(self, capsys):
        rc, out = self._invoke("not json {{{", capsys)
        assert rc == 0
        assert out == ""

    def test_missing_session_id_silent(self, capsys):
        rc, out = self._invoke(json.dumps({"prompt": "something"}), capsys)
        assert rc == 0
        assert out == ""

    def test_missing_prompt_silent(self, capsys):
        rc, out = self._invoke(json.dumps({"session_id": "abc"}), capsys)
        assert rc == 0
        assert out == ""

    def test_short_prompt_silent(self, capsys):
        rc, out = self._invoke(
            json.dumps({"session_id": "abc", "prompt": "ok"}), capsys
        )
        assert rc == 0
        assert out == ""

    def test_genuine_pivot_fires(self, capsys, tmp_path, monkeypatch):
        # Use a unique session_id + clean state by patching state_path
        sid = "test-pivot-session"
        original_path = tsh.state_path(sid)
        # Pre-seed window with 5 prompts on Topic A
        seed = [
            tsh.extract_content_words(f"Agent Amplifier stage {i} launch prep")
            for i in range(5)
        ]
        tsh.save_state(original_path, seed)
        try:
            rc, out = self._invoke(
                json.dumps({
                    "session_id": sid,
                    "prompt": "Tell me the recipe for garam masala coriander cumin cardamom ratios",
                }),
                capsys,
            )
            assert rc == 0
            assert "Topic shift detected" in out
        finally:
            try:
                os.remove(original_path)
            except OSError:
                pass

    def test_same_topic_silent(self, capsys):
        sid = "test-same-topic-session"
        path = tsh.state_path(sid)
        seed = [
            tsh.extract_content_words("Agent Amplifier dogfood window"),
            tsh.extract_content_words("Agent Amplifier stage 14 launch"),
        ]
        tsh.save_state(path, seed)
        try:
            rc, out = self._invoke(
                json.dumps({
                    "session_id": sid,
                    "prompt": "Agent Amplifier mkdocs site publish PyPI npm release",
                }),
                capsys,
            )
            assert rc == 0
            assert "Topic shift detected" not in out
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def test_window_updated_on_each_call(self, tmp_path, capsys):
        sid = "test-window-grow-session"
        path = tsh.state_path(sid)
        try:
            os.remove(path)
        except OSError:
            pass
        try:
            for i in range(3):
                self._invoke(
                    json.dumps({
                        "session_id": sid,
                        "prompt": f"This is iteration {i} of the test with sufficient content words alpha beta",
                    }),
                    capsys,
                )
            loaded = tsh.load_state(path)
            assert len(loaded) == 3
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


class TestLogging:
    def test_log_disabled_via_env(self, monkeypatch, tmp_path, capsys):
        log_path = str(tmp_path / "topic-shift.log")
        monkeypatch.setattr(tsh, "_LOG_PATH", log_path)
        monkeypatch.setattr(tsh, "_LOG_ENABLED", False)

        with patch("sys.stdin", io.StringIO(json.dumps({
            "session_id": "test-no-log",
            "prompt": "A substantive prompt with lots of content words for testing logger off",
        }))):
            tsh.main()
        assert not os.path.exists(log_path)

    def test_log_written_when_enabled(self, monkeypatch, tmp_path, capsys):
        log_dir = str(tmp_path / "logs")
        log_path = str(tmp_path / "logs" / "topic-shift.log")
        monkeypatch.setattr(tsh, "_LOG_DIR", log_dir)
        monkeypatch.setattr(tsh, "_LOG_PATH", log_path)
        monkeypatch.setattr(tsh, "_LOG_ENABLED", True)

        sid = "test-log-enabled"
        path = tsh.state_path(sid)
        try:
            os.remove(path)
        except OSError:
            pass
        try:
            with patch("sys.stdin", io.StringIO(json.dumps({
                "session_id": sid,
                "prompt": "A substantive test prompt with several content words alpha beta gamma",
            }))):
                tsh.main()
            assert os.path.exists(log_path)
            with open(log_path, encoding="utf-8") as f:
                content = f.read()
            # TSV format: contains timestamp, sess_hash, etc.
            assert "\t" in content
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


def test_the_log_does_not_keep_the_prompt_unless_asked(monkeypatch, tmp_path):
    """The log names no profile, so an erasure cannot reach it: it must not hold the text."""
    log_path = str(tmp_path / "topic-shift.log")
    monkeypatch.setattr(tsh, "_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(tsh, "_LOG_PATH", log_path)
    monkeypatch.setattr(tsh, "_LOG_ENABLED", True)
    monkeypatch.setattr(tsh, "_LOG_PROMPT_TEXT", False)

    tsh._log_decision("s", ["alpha"], [], 0, False, "my secret prompt about Alice Zebrowski")

    text = open(log_path, encoding="utf-8").read()
    assert "Zebrowski" not in text and "[38 chars]" in text

    monkeypatch.setattr(tsh, "_LOG_PROMPT_TEXT", True)
    tsh._log_decision("s", ["alpha"], [], 0, False, "opted in to the text")
    assert "opted in to the text" in open(log_path, encoding="utf-8").read()
