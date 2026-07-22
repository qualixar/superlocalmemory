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
from argparse import Namespace
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
            try:
                cmd_recall(_minimal_args())
            except Exception:
                pass  # We only care about the log, not the exit path

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
            try:
                cmd_recall(_minimal_args())
            except Exception:
                pass

    all_messages = " ".join(r.message for r in caplog.records)
    assert sentinel in all_messages, (
        f"Exception text '{sentinel}' missing from log output: {all_messages!r}"
    )
