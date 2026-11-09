# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Recalls that are not questions skip the answer check.

Loading context — at session start, for an auto-injected resource, for a
per-prompt hook, for the daemon's own warm-up — runs a recall to fetch
memories, not to answer a question. Judging it would ask whether memories
answer a "question" nobody asked, could abstain on it, and, with the online
check on, would send those memories off the machine and bill the user's key.

Results are unaffected: a skipped check only means the recall reports no
verdict, exactly as with the check off.

``skip_answer_check()`` marks the current context. It is a ``ContextVar``, so
``asyncio.to_thread`` and ``contextvars.copy_context().run`` carry it; a plain
thread or another process does not — callers on the far side of HTTP send
``ANSWER_CHECK_PARAM=ANSWER_CHECK_SKIP`` and the route re-enters the marker.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator

#: Query parameter (and value) a caller sends so the daemon skips the check.
ANSWER_CHECK_PARAM = "answer_check"
ANSWER_CHECK_SKIP = "skip"
#: 4.1.25: a context load (session start, ``slm://context``, a context file):
#: the check is skipped AND the answer-support gate does not run.
ANSWER_CHECK_CONTEXT = "context"

_SKIPPED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "slm_answer_check_skipped", default=False,
)


@contextlib.contextmanager
def skip_answer_check() -> Iterator[None]:
    """Within this block, recalls report no verdict and send nothing to be judged."""
    token = _SKIPPED.set(True)
    try:
        yield
    finally:
        _SKIPPED.reset(token)


_CONTEXT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "slm_recall_context_load", default=False,
)


@contextlib.contextmanager
def context_load() -> Iterator[None]:
    """Within this block a recall loads context, it does not ask a question: no
    answer check, and no answer-support gate (the surface caps the tokens)."""
    token = _CONTEXT.set(True)
    try:
        with skip_answer_check():
            yield
    finally:
        _CONTEXT.reset(token)


_NOT_A_READ: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "slm_recall_not_a_read", default=False,
)


@contextlib.contextmanager
def not_a_read() -> Iterator[None]:
    """Within this block a recall is SLM's own (the daemon's warm-up): nobody
    reads its results, so they are not logged as recalled (4.2.17)."""
    token = _NOT_A_READ.set(True)
    try:
        yield
    finally:
        _NOT_A_READ.reset(token)


def is_a_read() -> bool:
    """Whether the current recall's results reach a person or an agent."""
    return not _NOT_A_READ.get()


def answer_check_skipped() -> bool:
    """Whether the current context asked for the answer check to be skipped."""
    return _SKIPPED.get()


def in_context_load() -> bool:
    """Whether the current recall loads context rather than asking a question."""
    return _CONTEXT.get()


def wants_skip(value: object) -> bool:
    """Whether a wire value (the query parameter) asks for the skip."""
    return isinstance(value, str) and value.strip().lower() in (
        ANSWER_CHECK_SKIP, ANSWER_CHECK_CONTEXT)


def wants_context(value: object) -> bool:
    """Whether a wire value marks a context load."""
    return isinstance(value, str) and value.strip().lower() == ANSWER_CHECK_CONTEXT


__all__ = [
    "is_a_read",
    "not_a_read",
    "ANSWER_CHECK_CONTEXT",
    "ANSWER_CHECK_PARAM",
    "ANSWER_CHECK_SKIP",
    "answer_check_skipped",
    "context_load",
    "in_context_load",
    "wants_context",
    "skip_answer_check",
    "wants_skip",
]
