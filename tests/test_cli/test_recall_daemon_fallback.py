# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""S02: Daemon fallback in cmd_recall must log at WARNING, not swallow silently.

When the daemon path in cmd_recall raises an exception, the code must:
  1. Log the exception at WARNING level (not pass silently).
  2. Continue with the direct-engine fallback (behaviour unchanged).

Note: is_daemon_running / daemon_request / ensure_daemon are imported
*inside* the try block in cmd_recall, so we patch at the daemon module level.
"""

from __future__ import annotations

import logging
import sys
from argparse import Namespace
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest


def _minimal_args(**kwargs):
    defaults = dict(
        query="test query",
        limit=10,
        json=False,
        fast=False,
        include_global=None,
        include_shared=None,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


def test_daemon_fallback_logs_warning_on_exception(caplog) -> None:
    """S02: An exception in the daemon path must produce a WARNING log entry.

    Before the fix, the except block was `except Exception: pass` which
    silently discarded the error.  After the fix it must log at WARNING.
    """
    from superlocalmemory.cli.commands import cmd_recall

    boom = RuntimeError("daemon exploded")

    with (
        patch(
            "superlocalmemory.cli.daemon.is_daemon_running",
            return_value=True,
        ),
        patch(
            "superlocalmemory.cli.daemon.ensure_daemon",
            return_value=True,
        ),
        patch(
            "superlocalmemory.cli.daemon.daemon_request",
            side_effect=boom,
        ),
        patch(
            "superlocalmemory.core.config.SLMConfig",
        ) as mock_cfg,
        patch(
            "superlocalmemory.core.engine.MemoryEngine",
        ) as mock_eng,
        caplog.at_level(logging.WARNING, logger="superlocalmemory.cli.commands"),
    ):
        # Make direct-engine fallback succeed minimally
        mock_cfg.load.return_value = MagicMock()
        engine_instance = MagicMock()
        mock_eng.return_value = engine_instance

        with patch(
            "superlocalmemory.server.recall_serializer.recall_response_metadata",
            return_value={"results": [], "no_confident_match": True},
        ):
            with pytest.raises(SystemExit):
                cmd_recall(_minimal_args())

    warning_records = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING
        and ("fallback" in r.message.lower() or "falling back" in r.message.lower() or "daemon" in r.message.lower())
    ]
    assert warning_records, (
        "Expected at least one WARNING log about daemon fallback, "
        f"but got: {[(r.levelname, r.message) for r in caplog.records]}"
    )


def test_daemon_fallback_includes_exception_text(caplog) -> None:
    """S02: The fallback WARNING must include the exception text so operators
    can diagnose why the daemon path failed."""
    from superlocalmemory.cli.commands import cmd_recall

    sentinel = "unique-error-text-7a3f9"
    boom = ConnectionRefusedError(sentinel)

    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch("superlocalmemory.cli.daemon.ensure_daemon", return_value=True),
        patch("superlocalmemory.cli.daemon.daemon_request", side_effect=boom),
        patch("superlocalmemory.core.config.SLMConfig") as mock_cfg,
        patch("superlocalmemory.core.engine.MemoryEngine") as mock_eng,
        caplog.at_level(logging.WARNING, logger="superlocalmemory.cli.commands"),
    ):
        mock_cfg.load.return_value = MagicMock()
        engine_instance = MagicMock()
        mock_eng.return_value = engine_instance

        with patch(
            "superlocalmemory.server.recall_serializer.recall_response_metadata",
            return_value={"results": [], "no_confident_match": True},
        ):
            with pytest.raises(SystemExit):
                cmd_recall(_minimal_args())

    all_messages = " ".join(r.message for r in caplog.records)
    assert sentinel in all_messages, (
        f"Exception text '{sentinel}' missing from log output: {all_messages!r}"
    )


def test_cmd_recall_handles_null_retrieval_time_ms(capsys) -> None:
    """Regression: result with retrieval_time_ms=None must not raise
    TypeError: unsupported format string passed to NoneType.__format__.

    dict.get(key, default) falls back to default only when the key is absent.
    When the key is present with value None, the default is ignored and
    f"{None:.0f}" raises the TypeError.  The fix uses `or 0` to guard
    both the absent-key and the None-value cases.
    """
    from superlocalmemory.cli.commands import cmd_recall

    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch("superlocalmemory.cli.daemon.ensure_daemon", return_value=True),
        patch(
            "superlocalmemory.cli.daemon.daemon_request",
            return_value={
                "results": [
                    {"score": 0.91, "content": "memory content here"},
                ],
                "retrieval_time_ms": None,
            },
        ),
    ):
        # Must not raise TypeError — the old code raised:
        #   TypeError: unsupported format string passed to NoneType.__format__
        cmd_recall(_minimal_args())

    captured = capsys.readouterr()
    # Verify the success line was printed despite retrieval_time_ms being null
    assert "SpreadingActivation.search completed via daemon" in captured.out
    assert "SpreadingActivation.search completed via daemon (0ms)" in captured.out


def test_cmd_recall_via_keyword_fallback_has_retrieval_time(capsys) -> None:
    """Keyword fallback path (budget exceeded) always includes retrieval_time_ms.

    When semantic recall exceeds its budget, _recall_keyword_fallback() serves
    a degraded lexical response.  The return dict must include retrieval_time_ms
    so CLI formatting (f"{time:.0f}ms") does not crash on a missing key.
    """
    from superlocalmemory.cli.commands import cmd_recall

    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch("superlocalmemory.cli.daemon.ensure_daemon", return_value=True),
        patch(
            "superlocalmemory.cli.daemon.daemon_request",
            return_value={
                "results": [
                    {"score": None, "content": "fallback result"},
                ],
                "query_type": "text_search",
                "retrieval_mode": "degraded_lexical",
                "degraded_reason": "recall_budget_exceeded",
                # Note: no retrieval_time_ms key — simulating the old bug
            },
        ),
    ):
        # Must not raise KeyError on missing retrieval_time_ms
        cmd_recall(_minimal_args())

    captured = capsys.readouterr()
    assert "SpreadingActivation.search completed via daemon" in captured.out
    assert "SpreadingActivation.search completed via daemon (0ms)" in captured.out
