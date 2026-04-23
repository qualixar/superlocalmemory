# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""v3.4.32: Recall-in-flight counter used to give /search priority over the
pending materializer.

Every recall handler calls ``begin_recall()`` on entry and ``end_recall()``
in a finally block. The pending-memory materializer thread polls
``in_flight()`` and sleeps while any recall is active, so the shared
embedder worker never serves a materialization ahead of a user-initiated
recall.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_active = 0


def begin_recall() -> None:
    global _active
    with _lock:
        _active += 1


def end_recall() -> None:
    global _active
    with _lock:
        _active = max(0, _active - 1)


def in_flight() -> int:
    with _lock:
        return _active
