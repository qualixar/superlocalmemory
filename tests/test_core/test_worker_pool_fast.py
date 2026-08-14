# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""WorkerPool recall request shaping."""

from __future__ import annotations


def test_worker_pool_recall_forwards_fast_flag(monkeypatch):
    from superlocalmemory.core.worker_pool import WorkerPool

    pool = WorkerPool()
    sent = {}

    def _fake_send(payload):
        sent.update(payload)
        return {"ok": True}

    monkeypatch.setattr(pool, "_send", _fake_send)

    assert pool.recall("q", limit=3, session_id="s-1", fast=True) == {"ok": True}
    assert sent == {
        "cmd": "recall",
        "query": "q",
        "limit": 3,
        "session_id": "s-1",
        "fast": True,
    }


def test_worker_pool_recall_forwards_two_clock_boundaries(monkeypatch):
    from superlocalmemory.core.worker_pool import WorkerPool

    pool = WorkerPool()
    sent = {}
    monkeypatch.setattr(pool, "_send", lambda payload: sent.update(payload) or {"ok": True})

    pool.recall(
        "q", known_as_of="2026-01-01T00:00:00+00:00",
        valid_at="2025-01-01T00:00:00+00:00", include_unknown=True,
    )

    assert sent["known_as_of"] == "2026-01-01T00:00:00+00:00"
    assert sent["valid_at"] == "2025-01-01T00:00:00+00:00"
    assert sent["include_unknown"] is True
