# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""One lock per source, shared by the scan service and the calls that remove a source."""

from __future__ import annotations

import threading

_guard = threading.Lock()
_locks: dict[str, threading.Lock] = {}
_removing: set[str] = set()


def source_lock(source_id: str) -> threading.Lock:
    """The lock a scan holds while it works on the source, and a removal holds while it clears it."""
    with _guard:
        return _locks.setdefault(source_id, threading.Lock())


def mark_removing(source_id: str) -> None:
    """Tell a running scan of this source to stop at its next file."""
    with _guard:
        _removing.add(source_id)


def clear_removing(source_id: str) -> None:
    with _guard:
        _removing.discard(source_id)


def is_removing(source_id: str) -> bool:
    with _guard:
        return source_id in _removing


__all__ = ["clear_removing", "is_removing", "mark_removing", "source_lock"]
