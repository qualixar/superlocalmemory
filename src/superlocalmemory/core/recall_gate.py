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
from contextlib import contextmanager
from typing import Iterator

_condition = threading.Condition(threading.Lock())
_active = 0
_work_context = threading.local()


def begin_recall() -> None:
    global _active
    with _condition:
        _active += 1


def end_recall() -> None:
    global _active
    with _condition:
        _active = max(0, _active - 1)
        if _active == 0:
            _condition.notify_all()


def in_flight() -> int:
    with _condition:
        return _active


@contextmanager
def background_work() -> Iterator[None]:
    """Mark best-effort work that must yield shared inference to recall.

    The marker is thread-local because materialization, health probes, and
    interactive handlers all share one resident engine and one embedder.
    Nested callers restore the previous marker on exit.
    """
    previous = bool(getattr(_work_context, "background", False))
    _work_context.background = True
    try:
        yield
    finally:
        _work_context.background = previous


def is_background_work() -> bool:
    """Return whether the current thread is running best-effort work."""
    return bool(getattr(_work_context, "background", False))


def wait_for_foreground_idle() -> None:
    """Block background inference while an interactive recall is active."""
    if not is_background_work():
        return
    with _condition:
        while _active > 0:
            _condition.wait(timeout=0.1)
