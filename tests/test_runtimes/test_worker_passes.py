"""The worker embeds in small passes, so one request's peak memory stays bounded."""

from __future__ import annotations

import importlib.util
from pathlib import Path

WORKER = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory" / "runtimes" / "multimodal_worker.py"


def _worker():
    spec = importlib.util.spec_from_file_location("mm_worker_under_test", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_long_texts_go_one_or_two_at_a_time_and_short_ones_together():
    w = _worker()
    long_pass = w._passes([7900] * 16, w.TEXT_PASS_CHARS, w.TEXT_PASS_MAX)
    assert all(end - start <= 1 for start, end in long_pass) and long_pass[-1][1] == 16
    short_pass = w._passes([200] * 64, w.TEXT_PASS_CHARS, w.TEXT_PASS_MAX)
    assert len(short_pass) == 64 // w.TEXT_PASS_MAX
    assert [i for s, e in short_pass for i in range(s, e)] == list(range(64))


def test_pictures_go_a_few_at_a_time():
    w = _worker()
    groups = w._passes([1] * 16, w.IMAGE_PASS, w.IMAGE_PASS)
    assert all(e - s <= w.IMAGE_PASS for s, e in groups) and groups[-1][1] == 16


def test_an_empty_request_has_no_passes():
    assert _worker()._passes([], 10, 2) == []
