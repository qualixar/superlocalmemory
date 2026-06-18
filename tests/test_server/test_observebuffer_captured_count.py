"""Stage-9 regression: ObserveBuffer must report what was actually WRITTEN to
memory (captured), not count skipped (capture=False) items as successes. The
WP-E fix introduced a 'processed N' counter that incremented on every
non-exception item regardless of whether anything was stored.
"""
import logging
import time
import unittest.mock as mock

from superlocalmemory.server.unified_daemon import ObserveBuffer


def _flush_with_capture_decision(capture_value, caplog_handler):
    buf = ObserveBuffer(debounce_sec=0.01)
    with mock.patch("superlocalmemory.hooks.auto_capture.AutoCapture") as M:
        decision = mock.MagicMock()
        decision.capture = capture_value
        decision.category = "note"
        M.return_value.evaluate.return_value = decision
        buf.set_engine(object())
        buf.enqueue("content 1")
        buf.enqueue("content 2")
        time.sleep(0.1)
    return [r.getMessage() for r in caplog_handler.records]


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, r):
        self.records.append(r)


def _logger():
    lg = logging.getLogger("superlocalmemory.unified_daemon")
    h = _Capture()
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    return lg, h


def test_skipped_items_not_counted_as_captured():
    lg, h = _logger()
    try:
        msgs = _flush_with_capture_decision(False, h)
    finally:
        lg.removeHandler(h)
    summary = [m for m in msgs if "Observe debounce" in m]
    assert summary, f"no summary line: {msgs}"
    # capture=False on all items -> captured=0, evaluated=2
    assert "captured=0" in summary[0]
    assert "evaluated=2" in summary[0]


def test_captured_items_counted():
    lg, h = _logger()
    try:
        msgs = _flush_with_capture_decision(True, h)
    finally:
        lg.removeHandler(h)
    summary = [m for m in msgs if "Observe debounce" in m]
    assert summary
    assert "captured=2" in summary[0]
    assert "failed=0" in summary[0]
