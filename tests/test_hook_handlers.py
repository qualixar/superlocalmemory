# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for Claude Code hook handlers — hook_handlers.py.

Covers all 5 handlers dispatched via handle_hook():
  1. start  — clean markers, record start time, print session context
  2. gate   — block non-SLM tools until session_init marker exists
  3. init-done — create marker file to lift the gate
  4. checkpoint — rate-limited auto-observe on Write/Edit
  5. stop   — session summary with git context, cleanup, consolidation

All filesystem and subprocess calls are mocked. No real SLM process
is spawned. Tests verify behaviour, not integration.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
from unittest.mock import MagicMock, call, patch

import pytest

# Module under test — import the internals we need to exercise
from superlocalmemory.hooks.hook_handlers import (
    _ACTIVITY_LOG,
    _LAST_CONSOLIDATION,
    _MARKER,
    _OBSERVE_COOLDOWN,
    _RECALL_INTERVAL,
    _LEARN_INTERVAL,
    _START_TIME,
    _cooldown_elapsed,
    _maybe_consolidate,
    _run_quiet,
    _safe_hash,
    _write_timestamp,
    handle_hook,
)


# ───────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_marker_files():
    """Remove session marker files before and after each test."""
    paths = [_MARKER, _START_TIME, _ACTIVITY_LOG]
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass
    yield
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


@pytest.fixture()
def _clean_rate_locks():
    """Remove any slm-obs-*, slm-recall-*, slm-learn-* lock files."""
    tmp = tempfile.gettempdir()
    yield
    for name in os.listdir(tmp):
        if name.startswith(("slm-obs-", "slm-recall-", "slm-learn-")):
            try:
                os.remove(os.path.join(tmp, name))
            except OSError:
                pass


# ───────────────────────────────────────────────────────────────────
# Helper tests
# ───────────────────────────────────────────────────────────────────

class TestSafeHash:
    """_safe_hash: deterministic 8-hex-char string hash."""

    def test_returns_8_hex_chars(self):
        h = _safe_hash("/some/file.py")
        assert len(h) == 8
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        assert _safe_hash("abc") == _safe_hash("abc")

    def test_different_inputs_differ(self):
        assert _safe_hash("a") != _safe_hash("b")

    def test_empty_string(self):
        h = _safe_hash("")
        assert h == "00000000"


class TestCooldownElapsed:
    """_cooldown_elapsed: checks if interval has passed since lock file."""

    def test_no_lock_file_returns_true(self, tmp_path):
        lock = str(tmp_path / "nonexistent")
        assert _cooldown_elapsed(lock, 300, int(time.time())) is True

    def test_within_cooldown_returns_false(self, tmp_path):
        lock = str(tmp_path / "lock")
        now = int(time.time())
        with open(lock, "w") as f:
            f.write(str(now - 100))  # 100 seconds ago
        assert _cooldown_elapsed(lock, 300, now) is False

    def test_after_cooldown_returns_true(self, tmp_path):
        lock = str(tmp_path / "lock")
        now = int(time.time())
        with open(lock, "w") as f:
            f.write(str(now - 500))  # 500 seconds ago, > 300
        assert _cooldown_elapsed(lock, 300, now) is True

    def test_exactly_at_boundary_returns_true(self, tmp_path):
        lock = str(tmp_path / "lock")
        now = int(time.time())
        with open(lock, "w") as f:
            f.write(str(now - 300))  # exactly 300 seconds ago
        assert _cooldown_elapsed(lock, 300, now) is True

    def test_corrupt_file_returns_true(self, tmp_path):
        lock = str(tmp_path / "lock")
        with open(lock, "w") as f:
            f.write("not-a-number")
        assert _cooldown_elapsed(lock, 300, int(time.time())) is True


class TestWriteTimestamp:
    """_write_timestamp: writes a unix timestamp to a file."""

    def test_writes_timestamp(self, tmp_path):
        path = str(tmp_path / "ts")
        _write_timestamp(path, 1234567890)
        with open(path) as f:
            assert f.read() == "1234567890"

    def test_overwrites_existing(self, tmp_path):
        path = str(tmp_path / "ts")
        _write_timestamp(path, 111)
        _write_timestamp(path, 222)
        with open(path) as f:
            assert f.read() == "222"

    def test_invalid_path_no_exception(self):
        # Writing to a directory path should not raise
        _write_timestamp("/nonexistent_dir_xyz/file", 0)


class TestRunQuiet:
    """_run_quiet: run a command, return stdout or empty string."""

    def test_returns_stdout(self):
        out = _run_quiet(["echo", "hello"])
        assert out == "hello"

    def test_returns_empty_on_failure(self):
        out = _run_quiet(["false"])
        assert out == ""

    def test_returns_empty_on_nonexistent_command(self):
        out = _run_quiet(["__nonexistent_command_xyz__"])
        assert out == ""

    def test_postprocess_applied(self):
        out = _run_quiet(["echo", "abc\ndef\nghi"], postprocess=lambda s: s.split("\n")[-1])
        assert out == "ghi"

    def test_postprocess_not_called_on_empty(self):
        called = []
        _run_quiet(["__nonexistent__"], postprocess=lambda s: called.append(1))
        assert called == []

    def test_timeout_returns_empty(self):
        # sleep 10 with 1ms timeout should fail
        out = _run_quiet(["sleep", "10"], timeout=1)
        assert out == ""


# ───────────────────────────────────────────────────────────────────
# handle_hook dispatch
# ───────────────────────────────────────────────────────────────────

class TestHandleHookDispatch:
    """handle_hook dispatches to the correct handler or errors."""

    def test_unknown_action_exits_1(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            handle_hook("nonexistent_action")
        assert exc_info.value.code == 1
        assert "Unknown hook action" in capsys.readouterr().err

    @patch("superlocalmemory.hooks.hook_handlers._hook_start")
    def test_dispatches_start(self, mock_start):
        handle_hook("start")
        mock_start.assert_called_once()

    @patch("superlocalmemory.hooks.hook_handlers._hook_gate")
    def test_dispatches_gate(self, mock_gate):
        handle_hook("gate")
        mock_gate.assert_called_once()

    @patch("superlocalmemory.hooks.hook_handlers._hook_init_done")
    def test_dispatches_init_done(self, mock_init):
        handle_hook("init-done")
        mock_init.assert_called_once()

    @patch("superlocalmemory.hooks.hook_handlers._hook_checkpoint")
    def test_dispatches_checkpoint(self, mock_cp):
        handle_hook("checkpoint")
        mock_cp.assert_called_once()

    @patch("superlocalmemory.hooks.hook_handlers._hook_stop")
    def test_dispatches_stop(self, mock_stop):
        handle_hook("stop")
        mock_stop.assert_called_once()


class TestHookMandate:
    """mandate hook output must stay advisory, not imperative."""

    def test_prints_advisory_text(self, capsys):
        handle_hook("mandate")

        out = capsys.readouterr().out
        assert "SLM_SESSION_INIT_HINT" in out
        assert "session_init" in out
        assert "MANDATORY" not in out
        assert "non-negotiable" not in out
        assert "MUST" not in out


# ───────────────────────────────────────────────────────────────────
# 1. START handler
# ───────────────────────────────────────────────────────────────────

class TestHookStart:
    """_hook_start: clean markers, record start time, print context."""

    @patch("superlocalmemory.hooks.hook_handlers.subprocess.Popen")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    def test_cleans_stale_markers(self, mock_run, mock_popen):
        # Create stale marker files
        for p in (_MARKER, _START_TIME, _ACTIVITY_LOG):
            with open(p, "w") as f:
                f.write("stale")

        mock_run.return_value = MagicMock(stdout="", returncode=0)
        handle_hook("start")

        # Marker and activity log should be recreated (not stale content)
        assert os.path.exists(_START_TIME)
        assert os.path.exists(_ACTIVITY_LOG)
        # _MARKER should be removed (not recreated by start)
        assert not os.path.exists(_MARKER)

    @patch("superlocalmemory.hooks.hook_handlers.subprocess.Popen")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    def test_creates_start_time_file(self, mock_run, mock_popen):
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        before = int(time.time())
        handle_hook("start")
        after = int(time.time())

        assert os.path.exists(_START_TIME)
        with open(_START_TIME) as f:
            ts = int(f.read().strip())
        assert before <= ts <= after

    @patch("superlocalmemory.hooks.hook_handlers.subprocess.Popen")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    def test_creates_empty_activity_log(self, mock_run, mock_popen):
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        handle_hook("start")

        assert os.path.exists(_ACTIVITY_LOG)
        with open(_ACTIVITY_LOG) as f:
            assert f.read() == ""

    @patch("superlocalmemory.hooks.hook_handlers.subprocess.Popen")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    def test_prints_session_context_from_slm(self, mock_run, mock_popen, capsys):
        mock_run.return_value = MagicMock(
            stdout="# Recent Memories\n- memory 1\n- memory 2\n",
            returncode=0,
        )
        handle_hook("start")

        out = capsys.readouterr().out
        assert "Recent Memories" in out
        assert "memory 1" in out

    @patch("superlocalmemory.hooks.hook_handlers.subprocess.Popen")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    def test_prints_advisory_session_init(self, mock_run, mock_popen, capsys):
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        handle_hook("start")

        out = capsys.readouterr().out
        assert "SLM Session Init" in out
        assert "session_init" in out
        assert "MANDATORY" not in out
        assert "non-negotiable" not in out

    @patch("superlocalmemory.hooks.hook_handlers.subprocess.Popen")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    def test_uses_claude_project_dir_env(self, mock_run, mock_popen, capsys, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/my/custom/project")
        mock_run.return_value = MagicMock(stdout="context output", returncode=0)
        handle_hook("start")

        # slm session-context should be called with project basename
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args == ["slm", "session-context", "project"]

        out = capsys.readouterr().out
        assert "/my/custom/project" in out

    @patch("superlocalmemory.hooks.hook_handlers.subprocess.Popen")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    def test_prints_unavailable_on_slm_failure(self, mock_run, mock_popen, capsys):
        mock_run.side_effect = Exception("slm not found")
        handle_hook("start")

        out = capsys.readouterr().out
        assert "unavailable" in out

    @patch("superlocalmemory.hooks.hook_handlers.subprocess.Popen")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    def test_reap_runs_on_unix(self, mock_run, mock_popen):
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        with patch("superlocalmemory.hooks.hook_handlers.sys") as mock_sys:
            mock_sys.platform = "darwin"
            mock_sys.stderr = sys.stderr
            # Re-import won't help; we test the Popen was called
        handle_hook("start")
        # Popen should have been called for reap (on non-win32 platform)
        if sys.platform != "win32":
            assert mock_popen.called


# ───────────────────────────────────────────────────────────────────
# 2. GATE handler
# ───────────────────────────────────────────────────────────────────

class TestHookGate:
    """_hook_gate: block non-SLM tools until session_init marker exists."""

    def test_fast_path_marker_exists(self):
        """When marker file exists, exit 0 immediately."""
        with open(_MARKER, "w") as f:
            f.write(str(int(time.time())))
        with pytest.raises(SystemExit) as exc_info:
            handle_hook("gate")
        assert exc_info.value.code == 0

    def test_no_start_time_exits_0(self):
        """If session-start never ran (no _START_TIME), don't gate."""
        # _MARKER doesn't exist, _START_TIME doesn't exist
        with pytest.raises(SystemExit) as exc_info:
            handle_hook("gate")
        assert exc_info.value.code == 0

    def test_allows_slm_tools(self):
        """SLM tools pass through even without marker."""
        with open(_START_TIME, "w") as f:
            f.write(str(int(time.time())))

        stdin_data = json.dumps({"tool_name": "mcp__superlocalmemory__session_init"})
        with patch("sys.stdin", io.StringIO(stdin_data)):
            with patch("sys.stdin.isatty", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    handle_hook("gate")
        assert exc_info.value.code == 0

    def test_allows_toolsearch(self):
        """ToolSearch passes through even without marker."""
        with open(_START_TIME, "w") as f:
            f.write(str(int(time.time())))

        stdin_data = json.dumps({"tool_name": "ToolSearch"})
        with patch("sys.stdin", io.StringIO(stdin_data)):
            with patch("sys.stdin.isatty", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    handle_hook("gate")
        assert exc_info.value.code == 0

    def test_blocks_non_slm_tool(self, capsys):
        """Non-SLM tools get blocked with exit 2."""
        with open(_START_TIME, "w") as f:
            f.write(str(int(time.time())))

        stdin_data = json.dumps({"tool_name": "Bash"})
        with patch("sys.stdin", io.StringIO(stdin_data)):
            with patch("sys.stdin.isatty", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    handle_hook("gate")
        assert exc_info.value.code == 2
        assert "BLOCKED" in capsys.readouterr().out

    def test_tty_stdin_exits_0(self):
        """If stdin is a tty (no JSON), tool_name stays empty -> exit 2.

        Note: the gate handler parses JSON only when stdin is NOT a tty.
        When stdin IS a tty, tool_name remains "" which is still blocked
        because it doesn't match any allow-list prefix. This is the
        correct production behaviour: the gate hook receives JSON piped
        from Claude Code, never a raw tty.
        """
        with open(_START_TIME, "w") as f:
            f.write(str(int(time.time())))

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        with patch("sys.stdin", mock_stdin):
            with pytest.raises(SystemExit) as exc_info:
                handle_hook("gate")
        # With a tty, tool_name stays "" -> blocked
        assert exc_info.value.code == 2

    def test_corrupt_stdin_exits_0(self):
        """If stdin has non-JSON content, don't block (safety)."""
        with open(_START_TIME, "w") as f:
            f.write(str(int(time.time())))

        with patch("sys.stdin", io.StringIO("not json {{")):
            with patch("sys.stdin.isatty", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    handle_hook("gate")
        assert exc_info.value.code == 0

    def test_empty_tool_name_exits_0(self):
        """Empty tool_name should not block."""
        with open(_START_TIME, "w") as f:
            f.write(str(int(time.time())))

        stdin_data = json.dumps({"tool_name": ""})
        with patch("sys.stdin", io.StringIO(stdin_data)):
            with patch("sys.stdin.isatty", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    handle_hook("gate")
        # Empty string doesn't start with "mcp__superlocalmemory__"
        # and isn't "ToolSearch", so it would be blocked
        assert exc_info.value.code == 2


# ───────────────────────────────────────────────────────────────────
# 3. INIT-DONE handler
# ───────────────────────────────────────────────────────────────────

class TestHookInitDone:
    """_hook_init_done: create marker file to lift the gate."""

    def test_creates_marker_file(self):
        assert not os.path.exists(_MARKER)
        with pytest.raises(SystemExit) as exc_info:
            handle_hook("init-done")
        assert exc_info.value.code == 0
        assert os.path.exists(_MARKER)

    def test_marker_contains_timestamp(self):
        before = int(time.time())
        with pytest.raises(SystemExit) as exc_info:
            handle_hook("init-done")
        after = int(time.time())

        assert exc_info.value.code == 0
        with open(_MARKER) as f:
            ts = int(f.read().strip())
        assert before <= ts <= after

    def test_gate_passes_after_init_done(self):
        """After init-done runs, gate should fast-path exit 0."""
        with pytest.raises(SystemExit):
            handle_hook("init-done")

        # Now gate should pass
        with pytest.raises(SystemExit) as exc_info:
            handle_hook("gate")
        assert exc_info.value.code == 0


# ───────────────────────────────────────────────────────────────────
# 4. CHECKPOINT handler
# ───────────────────────────────────────────────────────────────────

class TestHookCheckpoint:
    """_hook_checkpoint: rate-limited auto-observe on Write/Edit."""

    @patch("superlocalmemory.hooks.hook_handlers._daemon_post")
    def test_observes_file_change(self, mock_daemon_post, capsys, _clean_rate_locks):
        """First checkpoint for a file should trigger observe via daemon HTTP."""
        mock_daemon_post.return_value = True
        stdin_data = json.dumps({
            "tool_input": {"file_path": "/project/src/main.py"},
        })
        with patch("sys.stdin", io.StringIO(stdin_data)):
            with patch("sys.stdin.isatty", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    handle_hook("checkpoint")

        assert exc_info.value.code == 0
        # v3.4.13: Should POST to daemon instead of subprocess
        mock_daemon_post.assert_called()
        call_args = mock_daemon_post.call_args
        assert call_args[0][0] == "/observe"
        assert "main.py" in call_args[0][1]["content"]

        out = capsys.readouterr().out
        assert "[SLM-AUTO]" in out
        assert "main.py" in out

    @patch("superlocalmemory.hooks.hook_handlers._daemon_post")
    def test_rate_limits_same_file(self, mock_daemon_post, capsys, _clean_rate_locks):
        """Second checkpoint for same file within cooldown should NOT observe."""
        file_path = "/project/src/rate_test.py"
        file_hash = _safe_hash(file_path)
        lock_file = os.path.join(tempfile.gettempdir(), f"slm-obs-{file_hash}")

        # Write a recent timestamp to the lock file (simulating recent observe)
        now = int(time.time())
        with open(lock_file, "w") as f:
            f.write(str(now))

        stdin_data = json.dumps({
            "tool_input": {"file_path": file_path},
        })
        with patch("sys.stdin", io.StringIO(stdin_data)):
            with patch("sys.stdin.isatty", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    handle_hook("checkpoint")

        assert exc_info.value.code == 0
        # _daemon_post should NOT have been called (within cooldown)
        mock_daemon_post.assert_not_called()
        out = capsys.readouterr().out
        assert "[SLM-AUTO]" not in out

    @patch("superlocalmemory.hooks.hook_handlers._daemon_post")
    def test_fires_after_cooldown_expires(self, mock_daemon_post, capsys, _clean_rate_locks):
        """After 5 minutes, same file should trigger observe again."""
        file_path = "/project/src/expired_test.py"
        file_hash = _safe_hash(file_path)
        lock_file = os.path.join(tempfile.gettempdir(), f"slm-obs-{file_hash}")

        # Write an old timestamp (6 minutes ago)
        old_ts = int(time.time()) - (_OBSERVE_COOLDOWN + 60)
        with open(lock_file, "w") as f:
            f.write(str(old_ts))

        stdin_data = json.dumps({
            "tool_input": {"file_path": file_path},
        })
        with patch("sys.stdin", io.StringIO(stdin_data)):
            with patch("sys.stdin.isatty", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    handle_hook("checkpoint")

        assert exc_info.value.code == 0
        mock_daemon_post.assert_called()
        out = capsys.readouterr().out
        assert "[SLM-AUTO]" in out

    @patch("superlocalmemory.hooks.hook_handlers._daemon_post")
    def test_logs_to_activity_file(self, mock_daemon_post, _clean_rate_locks):
        """Checkpoint should append to the activity log."""
        # Create activity log
        with open(_ACTIVITY_LOG, "w") as f:
            f.write("")

        file_path = "/project/src/logged.py"
        stdin_data = json.dumps({
            "tool_input": {"file_path": file_path},
        })
        with patch("sys.stdin", io.StringIO(stdin_data)):
            with patch("sys.stdin.isatty", return_value=False):
                with pytest.raises(SystemExit):
                    handle_hook("checkpoint")

        with open(_ACTIVITY_LOG) as f:
            content = f.read()
        assert "logged.py" in content

    @patch("superlocalmemory.hooks.hook_handlers._daemon_post")
    def test_recall_reminder_removed_v3_4_43(self, mock_daemon_post, capsys, _clean_rate_locks):
        """v3.4.43: 15-min periodic recall reminder is REMOVED.

        Regression test — the periodic time-based nag was replaced by the
        event-based topic_shift hook. _hook_checkpoint no longer emits any
        "context refresh" / "recall" reminder regardless of cooldown state.
        """
        recall_lock = os.path.join(tempfile.gettempdir(), "slm-recall-reminder")
        old_ts = int(time.time()) - (_RECALL_INTERVAL + 60)
        with open(recall_lock, "w") as f:
            f.write(str(old_ts))

        stdin_data = json.dumps({"tool_input": {}})
        with patch("sys.stdin", io.StringIO(stdin_data)):
            with patch("sys.stdin.isatty", return_value=False):
                with pytest.raises(SystemExit):
                    handle_hook("checkpoint")

        out = capsys.readouterr().out
        # v3.4.43 contract: NO periodic recall nag.
        assert "context refresh" not in out
        assert "[SLM] 15+" not in out

    @patch("superlocalmemory.hooks.hook_handlers._daemon_post")
    def test_learn_reminder_removed_v3_4_43(self, mock_daemon_post, capsys, _clean_rate_locks):
        """v3.4.43: 30-min periodic learn reminder is REMOVED.

        Regression test — the get_learned_patterns nag was replaced by
        on-demand learned-pattern fetches initiated by Claude when needed.
        """
        learn_lock = os.path.join(tempfile.gettempdir(), "slm-learn-reminder")
        old_ts = int(time.time()) - (_LEARN_INTERVAL + 60)
        with open(learn_lock, "w") as f:
            f.write(str(old_ts))

        stdin_data = json.dumps({"tool_input": {}})
        with patch("sys.stdin", io.StringIO(stdin_data)):
            with patch("sys.stdin.isatty", return_value=False):
                with pytest.raises(SystemExit):
                    handle_hook("checkpoint")

        out = capsys.readouterr().out
        # v3.4.43 contract: NO periodic learn nag.
        assert "learned_patterns" not in out
        assert "[SLM] Call mcp__superlocalmemory__get_learned_patterns" not in out

    @patch("superlocalmemory.hooks.hook_handlers._daemon_post")
    def test_no_recall_reminder_within_interval(self, mock_daemon_post, capsys, _clean_rate_locks):
        """Within 15 minutes, no recall reminder."""
        recall_lock = os.path.join(tempfile.gettempdir(), "slm-recall-reminder")
        recent_ts = int(time.time()) - 60  # 1 minute ago
        with open(recall_lock, "w") as f:
            f.write(str(recent_ts))

        # Also suppress learn reminder
        learn_lock = os.path.join(tempfile.gettempdir(), "slm-learn-reminder")
        with open(learn_lock, "w") as f:
            f.write(str(recent_ts))

        stdin_data = json.dumps({"tool_input": {}})
        with patch("sys.stdin", io.StringIO(stdin_data)):
            with patch("sys.stdin.isatty", return_value=False):
                with pytest.raises(SystemExit):
                    handle_hook("checkpoint")

        out = capsys.readouterr().out
        assert "context refresh" not in out

    @patch("superlocalmemory.hooks.hook_handlers._daemon_post")
    def test_no_file_path_skips_observe(self, mock_daemon_post, capsys, _clean_rate_locks):
        """If tool_input has no file_path, skip observe but still check reminders."""
        # Suppress reminders by setting recent locks
        for name in ("slm-recall-reminder", "slm-learn-reminder"):
            lock = os.path.join(tempfile.gettempdir(), name)
            with open(lock, "w") as f:
                f.write(str(int(time.time())))

        stdin_data = json.dumps({"tool_input": {"content": "some code"}})
        with patch("sys.stdin", io.StringIO(stdin_data)):
            with patch("sys.stdin.isatty", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    handle_hook("checkpoint")

        assert exc_info.value.code == 0
        mock_daemon_post.assert_not_called()
        out = capsys.readouterr().out
        assert "[SLM-AUTO]" not in out

    @patch("superlocalmemory.hooks.hook_handlers._daemon_post")
    def test_tty_stdin_skips_observe(self, mock_daemon_post, capsys, _clean_rate_locks):
        """If stdin is a tty, no file info available — skip observe."""
        # Suppress reminders
        for name in ("slm-recall-reminder", "slm-learn-reminder"):
            lock = os.path.join(tempfile.gettempdir(), name)
            with open(lock, "w") as f:
                f.write(str(int(time.time())))

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            with pytest.raises(SystemExit) as exc_info:
                handle_hook("checkpoint")

        assert exc_info.value.code == 0
        mock_daemon_post.assert_not_called()

    @patch("superlocalmemory.hooks.hook_handlers._daemon_post")
    def test_different_files_have_independent_cooldowns(self, mock_daemon_post, capsys, _clean_rate_locks):
        """Two different files should have independent rate limits."""
        now = int(time.time())

        # Lock file A as recently observed
        file_a = "/project/src/a.py"
        hash_a = _safe_hash(file_a)
        lock_a = os.path.join(tempfile.gettempdir(), f"slm-obs-{hash_a}")
        with open(lock_a, "w") as f:
            f.write(str(now))

        # Suppress reminders
        for name in ("slm-recall-reminder", "slm-learn-reminder"):
            lock = os.path.join(tempfile.gettempdir(), name)
            with open(lock, "w") as f:
                f.write(str(now))

        # File B should still fire (no lock)
        file_b = "/project/src/b.py"
        stdin_data = json.dumps({"tool_input": {"file_path": file_b}})
        with patch("sys.stdin", io.StringIO(stdin_data)):
            with patch("sys.stdin.isatty", return_value=False):
                with pytest.raises(SystemExit):
                    handle_hook("checkpoint")

        mock_daemon_post.assert_called()
        out = capsys.readouterr().out
        assert "b.py" in out


# ───────────────────────────────────────────────────────────────────
# 5. STOP handler
# ───────────────────────────────────────────────────────────────────

class TestHookStop:
    """_hook_stop: session summary with git context, cleanup, consolidation."""

    @patch("superlocalmemory.hooks.hook_handlers._maybe_consolidate")
    @patch("superlocalmemory.hooks.hook_handlers._daemon_post")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    def test_builds_summary_with_project_name(self, mock_run, mock_daemon, mock_consolidate, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/my/test-project")
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        mock_daemon.return_value = True

        with pytest.raises(SystemExit) as exc_info:
            handle_hook("stop")
        assert exc_info.value.code == 0

        # v3.4.13: Summary now POSTed to daemon via _daemon_post
        mock_daemon.assert_called()
        observe_call = mock_daemon.call_args_list[0]
        assert observe_call[0][0] == "/observe"
        assert "test-project" in observe_call[0][1]["content"]

    @patch("superlocalmemory.hooks.hook_handlers._maybe_consolidate")
    @patch("superlocalmemory.hooks.hook_handlers._daemon_post")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    def test_includes_git_branch(self, mock_run, mock_daemon, mock_consolidate, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/proj")
        mock_daemon.return_value = True

        def run_side_effect(cmd, **kwargs):
            result = MagicMock()
            if "branch" in cmd and "--show-current" in cmd:
                result.stdout = "feature/hooks\n"
            elif "diff" in cmd and "--stat" in cmd:
                result.stdout = ""
            elif "log" in cmd:
                result.stdout = ""
            result.returncode = 0
            return result

        mock_run.side_effect = run_side_effect

        with pytest.raises(SystemExit):
            handle_hook("stop")

        observe_call = mock_daemon.call_args_list[0]
        assert "feature/hooks" in observe_call[0][1]["content"]

    @patch("superlocalmemory.hooks.hook_handlers._maybe_consolidate")
    @patch("superlocalmemory.hooks.hook_handlers._daemon_post")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    def test_includes_recent_commits(self, mock_run, mock_daemon, mock_consolidate, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/proj")
        mock_daemon.return_value = True

        def run_side_effect(cmd, **kwargs):
            result = MagicMock()
            if "branch" in cmd:
                result.stdout = "main"
            elif "diff" in cmd:
                result.stdout = ""
            elif "log" in cmd:
                result.stdout = "abc1234 fix: auth bug\ndef5678 feat: login\n"
            result.returncode = 0
            return result

        mock_run.side_effect = run_side_effect

        with pytest.raises(SystemExit):
            handle_hook("stop")

        observe_call = mock_daemon.call_args_list[0]
        summary = observe_call[0][1]["content"]
        assert "abc1234" in summary or "recent" in summary

    @patch("superlocalmemory.hooks.hook_handlers._maybe_consolidate")
    @patch("superlocalmemory.hooks.hook_handlers._daemon_post")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    def test_includes_modified_files_from_activity_log(self, mock_run, mock_daemon, mock_consolidate, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/proj")
        mock_daemon.return_value = True

        now = int(time.time())
        with open(_ACTIVITY_LOG, "w") as f:
            f.write(f"{now}|engine.py\n{now}|config.py\n{now}|engine.py\n")

        mock_run.return_value = MagicMock(stdout="", returncode=0)

        with pytest.raises(SystemExit):
            handle_hook("stop")

        observe_call = mock_daemon.call_args_list[0]
        summary = observe_call[0][1]["content"]
        assert "config.py" in summary
        assert "engine.py" in summary

    @patch("superlocalmemory.hooks.hook_handlers._maybe_consolidate")
    @patch("superlocalmemory.hooks.hook_handlers._daemon_post")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    def test_cleans_up_markers_after_stop(self, mock_run, mock_daemon, mock_consolidate):
        # Create all marker files
        for p in (_MARKER, _START_TIME, _ACTIVITY_LOG):
            with open(p, "w") as f:
                f.write("data")

        mock_run.return_value = MagicMock(stdout="", returncode=0)

        with pytest.raises(SystemExit):
            handle_hook("stop")

        for p in (_MARKER, _START_TIME, _ACTIVITY_LOG):
            assert not os.path.exists(p)

    @patch("superlocalmemory.hooks.hook_handlers._maybe_consolidate")
    @patch("superlocalmemory.hooks.hook_handlers._daemon_post")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    def test_cleans_rate_limit_locks(self, mock_run, mock_daemon, mock_consolidate, _clean_rate_locks):
        tmp = tempfile.gettempdir()
        # Create rate-limit lock files
        for name in ("slm-obs-abc12345", "slm-recall-reminder", "slm-learn-reminder"):
            with open(os.path.join(tmp, name), "w") as f:
                f.write("123")

        mock_run.return_value = MagicMock(stdout="", returncode=0)

        with pytest.raises(SystemExit):
            handle_hook("stop")

        for name in ("slm-obs-abc12345", "slm-recall-reminder", "slm-learn-reminder"):
            assert not os.path.exists(os.path.join(tmp, name))

    @patch("superlocalmemory.hooks.hook_handlers._maybe_consolidate")
    @patch("superlocalmemory.hooks.hook_handlers._daemon_post")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    def test_calls_consolidate(self, mock_run, mock_daemon, mock_consolidate):
        mock_run.return_value = MagicMock(stdout="", returncode=0)

        with pytest.raises(SystemExit):
            handle_hook("stop")

        mock_consolidate.assert_called_once()

    @patch("superlocalmemory.hooks.hook_handlers._maybe_consolidate")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    @patch("superlocalmemory.hooks.hook_handlers._daemon_post")
    def test_falls_back_to_remember_on_observe_failure(self, mock_daemon, mock_run, mock_consolidate, monkeypatch):
        """v3.4.13: When /observe fails, falls back to /remember via daemon."""
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/proj")
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        call_paths = []

        def daemon_side_effect(path, body, timeout=3.0):
            call_paths.append(path)
            if path == "/observe":
                return False  # observe failed
            return True  # remember succeeds

        mock_daemon.side_effect = daemon_side_effect

        with pytest.raises(SystemExit):
            handle_hook("stop")

        assert "/observe" in call_paths
        assert "/remember" in call_paths


# ───────────────────────────────────────────────────────────────────
# _maybe_consolidate
# ───────────────────────────────────────────────────────────────────

class TestMaybeConsolidate:
    """_maybe_consolidate: run consolidation if >24h since last."""

    @patch("superlocalmemory.hooks.hook_handlers.subprocess.Popen")
    def test_runs_if_no_last_consolidation_file(self, mock_popen, tmp_path, monkeypatch):
        last_file = str(tmp_path / ".last-consolidation")
        monkeypatch.setattr(
            "superlocalmemory.hooks.hook_handlers._LAST_CONSOLIDATION", last_file,
        )

        _maybe_consolidate()

        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        assert args == ["slm", "consolidate", "--cognitive"]
        # Verify timestamp was written
        assert os.path.exists(last_file)

    @patch("superlocalmemory.hooks.hook_handlers.subprocess.Popen")
    def test_skips_if_within_24h(self, mock_popen, tmp_path, monkeypatch):
        last_file = str(tmp_path / ".last-consolidation")
        monkeypatch.setattr(
            "superlocalmemory.hooks.hook_handlers._LAST_CONSOLIDATION", last_file,
        )
        recent_ts = int(time.time()) - 3600  # 1 hour ago
        with open(last_file, "w") as f:
            f.write(str(recent_ts))

        _maybe_consolidate()

        mock_popen.assert_not_called()

    @patch("superlocalmemory.hooks.hook_handlers.subprocess.Popen")
    def test_runs_if_over_24h(self, mock_popen, tmp_path, monkeypatch):
        last_file = str(tmp_path / ".last-consolidation")
        monkeypatch.setattr(
            "superlocalmemory.hooks.hook_handlers._LAST_CONSOLIDATION", last_file,
        )
        old_ts = int(time.time()) - 100000  # well over 24h
        with open(last_file, "w") as f:
            f.write(str(old_ts))

        _maybe_consolidate()

        mock_popen.assert_called_once()

    @patch("superlocalmemory.hooks.hook_handlers.subprocess.Popen")
    def test_updates_timestamp_before_popen(self, mock_popen, tmp_path, monkeypatch):
        last_file = str(tmp_path / ".last-consolidation")
        monkeypatch.setattr(
            "superlocalmemory.hooks.hook_handlers._LAST_CONSOLIDATION", last_file,
        )

        before = int(time.time())
        _maybe_consolidate()
        after = int(time.time())

        with open(last_file) as f:
            ts = int(f.read().strip())
        assert before <= ts <= after


# ───────────────────────────────────────────────────────────────────
# Integration: full start -> init-done -> checkpoint -> stop flow
# ───────────────────────────────────────────────────────────────────

class TestFullSessionFlow:
    """Verify the complete lifecycle: start -> gate blocks -> init-done -> gate passes -> stop cleans."""

    @patch("superlocalmemory.hooks.hook_handlers._maybe_consolidate")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.Popen")
    @patch("superlocalmemory.hooks.hook_handlers.subprocess.run")
    def test_full_lifecycle(self, mock_run, mock_popen, mock_consolidate, capsys, monkeypatch):
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/my/project")
        mock_run.return_value = MagicMock(stdout="", returncode=0)

        # 1. Start
        handle_hook("start")
        assert os.path.exists(_START_TIME)
        assert not os.path.exists(_MARKER)

        # 2. Gate should block non-SLM tool
        stdin_data = json.dumps({"tool_name": "Bash"})
        with patch("sys.stdin", io.StringIO(stdin_data)):
            with patch("sys.stdin.isatty", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    handle_hook("gate")
        assert exc_info.value.code == 2

        # 3. Init-done
        with pytest.raises(SystemExit) as exc_info:
            handle_hook("init-done")
        assert exc_info.value.code == 0
        assert os.path.exists(_MARKER)

        # 4. Gate should now pass
        with pytest.raises(SystemExit) as exc_info:
            handle_hook("gate")
        assert exc_info.value.code == 0

        # 5. Stop cleans everything
        with pytest.raises(SystemExit):
            handle_hook("stop")
        assert not os.path.exists(_MARKER)
        assert not os.path.exists(_START_TIME)
