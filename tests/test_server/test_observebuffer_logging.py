"""Regression test for ObserveBuffer._flush silent-failure bug.

Work package E: inner/outer bare except-pass swallows capture errors and reports
false success.  This test pins the corrected behaviour:

- When AutoCapture.capture() raises, a WARNING is logged (not silently dropped).
- The summary log reports failed >= 1, not processed == len(batch) with no failures.
"""

from __future__ import annotations

import logging
import types
import unittest.mock as mock

import pytest

from superlocalmemory.server.unified_daemon import ObserveBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeDecision:
    """Minimal stand-in for AutoCapture's CaptureDecision."""
    capture = True
    category = "test"


class _BrokenAutoCapture:
    """AutoCapture whose capture() always raises to simulate embedder/DB failure."""

    def __init__(self, engine):  # noqa: ARG002
        pass

    def evaluate(self, content: str) -> _FakeDecision:  # noqa: ARG002
        return _FakeDecision()

    def capture(self, content: str, *, category: str) -> None:  # noqa: ARG002
        raise RuntimeError("embedder down – simulated failure")


# ---------------------------------------------------------------------------
# Test: inner capture() failure must emit WARNING, not be swallowed silently
# ---------------------------------------------------------------------------

def test_flush_logs_warning_on_capture_failure(caplog):
    """_flush() must log WARNING when AutoCapture.capture() raises."""
    buf = ObserveBuffer(debounce_sec=999)  # timer won't fire; we flush manually
    buf.set_engine(object())  # any non-None sentinel

    buf._buffer.append("test observation content")
    buf._seen.add("any-hash")

    # Patch the module-level import inside _flush so it returns _BrokenAutoCapture.
    fake_module = types.ModuleType("superlocalmemory.hooks.auto_capture")
    fake_module.AutoCapture = _BrokenAutoCapture

    with mock.patch.dict(
        "sys.modules",
        {"superlocalmemory.hooks.auto_capture": fake_module},
    ):
        with caplog.at_level(logging.WARNING, logger="superlocalmemory.unified_daemon"):
            buf._flush()

    # At least one WARNING must be present about the capture failure.
    warning_records = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "ObserveBuffer" in r.getMessage()
    ]
    assert warning_records, (
        "Expected at least one WARNING from ObserveBuffer when capture() raises, "
        f"but caplog only contains: {[r.getMessage() for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Test: summary log must report failed >= 1, not silent success
# ---------------------------------------------------------------------------

def test_flush_summary_reports_failed_count(caplog):
    """_flush() summary log must include failed>=1 when captures fail."""
    buf = ObserveBuffer(debounce_sec=999)
    buf.set_engine(object())

    buf._buffer.append("another observation")
    buf._seen.add("any-hash")

    fake_module = types.ModuleType("superlocalmemory.hooks.auto_capture")
    fake_module.AutoCapture = _BrokenAutoCapture

    with mock.patch.dict(
        "sys.modules",
        {"superlocalmemory.hooks.auto_capture": fake_module},
    ):
        with caplog.at_level(logging.DEBUG, logger="superlocalmemory.unified_daemon"):
            buf._flush()

    # The summary INFO line must mention "failed" with a non-zero count.
    # Accept formats like "failed=1", "failed: 1", "1 failed", etc.
    summary_records = [
        r for r in caplog.records
        if r.levelno == logging.INFO and "observe debounce" in r.getMessage().lower()
    ]
    assert summary_records, (
        "Expected a summary INFO log with 'Observe debounce', "
        f"caplog messages: {[r.getMessage() for r in caplog.records]}"
    )
    summary_text = summary_records[0].getMessage()
    # Must NOT claim 1 processed with 0 failed when capture raised.
    # The fixed code tracks successful vs failed separately.
    assert "failed" in summary_text.lower(), (
        f"Summary log does not mention 'failed': {summary_text!r}"
    )
    # The failed count must be >= 1.
    import re
    match = re.search(r"failed[=:\s]+(\d+)", summary_text, re.IGNORECASE)
    assert match and int(match.group(1)) >= 1, (
        f"Failed count not >= 1 in summary: {summary_text!r}"
    )


# ---------------------------------------------------------------------------
# Test: successful captures still log correctly (no regression)
# ---------------------------------------------------------------------------

class _GoodAutoCapture:
    """AutoCapture that always succeeds."""

    def __init__(self, engine):  # noqa: ARG002
        pass

    def evaluate(self, content: str) -> _FakeDecision:  # noqa: ARG002
        return _FakeDecision()

    def capture(self, content: str, *, category: str) -> None:  # noqa: ARG002
        pass  # success


def test_flush_summary_reports_zero_failed_on_success(caplog):
    """When all captures succeed, summary must report processed=N, failed=0."""
    buf = ObserveBuffer(debounce_sec=999)
    buf.set_engine(object())

    buf._buffer.append("good observation")
    buf._seen.add("any-hash")

    fake_module = types.ModuleType("superlocalmemory.hooks.auto_capture")
    fake_module.AutoCapture = _GoodAutoCapture

    with mock.patch.dict(
        "sys.modules",
        {"superlocalmemory.hooks.auto_capture": fake_module},
    ):
        with caplog.at_level(logging.DEBUG, logger="superlocalmemory.unified_daemon"):
            buf._flush()

    summary_records = [
        r for r in caplog.records
        if r.levelno == logging.INFO and "observe debounce" in r.getMessage().lower()
    ]
    assert summary_records, (
        f"Expected summary INFO log. Got: {[r.getMessage() for r in caplog.records]}"
    )
    import re
    summary_text = summary_records[0].getMessage()
    match = re.search(r"failed[=:\s]+(\d+)", summary_text, re.IGNORECASE)
    # If "failed" appears in the summary, its count must be 0.
    if match:
        assert int(match.group(1)) == 0, (
            f"Expected failed=0 on clean run, got: {summary_text!r}"
        )
