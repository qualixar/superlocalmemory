# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""One retrieval channel switched off for one fetch, at random and on record (W3-3, D-44).

The channel arm of ``outcomes.randomiser`` (design ``one-out-v2``) drops one
channel from the fetch behind a treated session start, so the value of each
channel can be estimated from logged propensities (M-12). The drop travels
like the other per-call scopes (``answer_check_scope``): ``dropping(channel)``
marks the context, and ``run_recall`` adds ``current()`` to the channels the
retrieval engine skips for that one call. A context variable cannot leak into
another recall, and a channel outside ``DROPPABLE_CHANNELS`` is never dropped.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator

_DROPPED: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "slm_dropped_channels", default=frozenset())


def droppable(channel: str | None) -> bool:
    from superlocalmemory.outcomes.randomiser import DROPPABLE_CHANNELS

    return bool(channel) and channel in DROPPABLE_CHANNELS


@contextlib.contextmanager
def dropping(channel: str | None) -> Iterator[None]:
    """Skip ``channel`` for recalls in this context; anything not droppable is ignored."""
    if not droppable(channel):
        yield
        return
    token = _DROPPED.set(_DROPPED.get() | {channel})
    try:
        yield
    finally:
        _DROPPED.reset(token)


def current() -> frozenset[str]:
    return _DROPPED.get()


__all__ = ["current", "droppable", "dropping"]
