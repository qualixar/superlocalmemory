# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The client's absolute deadline, carried to the code that writes the delivery receipt.

A hook asks the daemon for memories and waits a fixed time. When it gives up, the agent got
nothing; the recall may still be running. Whatever finishes after the deadline was never
delivered, so nothing may record it as delivered (D-47, Fable audit A-14).

``scope(deadline_at)`` marks the current context with an absolute ``time.time()`` value;
``expired()`` says whether it has passed. A ``ContextVar``, like ``answer_check_scope``: the
queue consumer runs the daemon's adapter on its own thread, inside the block, so the marker
reaches it. No deadline (``None``) means none applies.
"""

from __future__ import annotations

import contextlib
import contextvars
import time
from collections.abc import Iterator

_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "slm_recall_client_deadline", default=None)


@contextlib.contextmanager
def scope(deadline_at: float | None) -> Iterator[None]:
    """Within this block the answer is wanted only until ``deadline_at`` (epoch seconds)."""
    token = _DEADLINE.set(deadline_at)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def expired() -> bool:
    """True when a deadline is set and it has passed."""
    deadline = _DEADLINE.get()
    return deadline is not None and time.time() >= deadline


__all__ = ["expired", "scope"]
