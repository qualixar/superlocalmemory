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


# ---------------------------------------------------------------------------
# PR #101 — null / missing retrieval_time_ms
#
# The original PR shipped a CLI guard and a daemon-side contract fix, but only
# covered the CLI half: reverting the daemon change left every test green. The
# daemon contract now has a DIRECT test that calls _recall_keyword_fallback
# rather than mocking daemon_request over the top of it.
# ---------------------------------------------------------------------------


def test_keyword_fallback_response_includes_retrieval_time_ms() -> None:
    """The degraded lexical path must satisfy the recall response contract.

    This calls _recall_keyword_fallback DIRECTLY. Mocking ``daemon_request``
    instead (as the original PR did) never reaches this function, so the
    contract fix it was meant to protect had no coverage at all.
    """
    from superlocalmemory.server.unified_daemon import _recall_keyword_fallback

    class _Fact:
        fact_id = "f1"
        content = "fallback content"
        confidence = 0.5
        created_at = "2026-01-01T00:00:00Z"

    class _Engine:
        def search_facts(self, *a, **kw):
            return [_Fact()]

        def __getattr__(self, name):
            def _any(*a, **kw):
                return []
            return _any

    result = _recall_keyword_fallback(_Engine(), "query", 5)

    assert "retrieval_time_ms" in result, (
        "keyword fallback must return retrieval_time_ms — every other recall "
        "path does, and clients format it unconditionally"
    )
    # Must be formattable with a numeric format spec (the original crash).
    assert f"{result['retrieval_time_ms']:.0f}" is not None


def test_cmd_recall_survives_present_but_null_retrieval_time() -> None:
    """A PRESENT key whose value is None must not crash the formatter.

    dict.get(key, default) ignores the default when the key exists with value
    None, so f"{None:.0f}" raises TypeError. Guarded with ``or 0``.
    """
    from superlocalmemory.cli.commands import cmd_recall

    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch("superlocalmemory.cli.daemon.ensure_daemon", return_value=True),
        patch(
            "superlocalmemory.cli.daemon.daemon_request",
            return_value={
                "results": [{"score": None, "content": "c"}],
                "retrieval_time_ms": None,
            },
        ),
    ):
        cmd_recall(_minimal_args())  # must not raise TypeError
