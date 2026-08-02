# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import threading
import time

_TTL_SECONDS = 300.0

_lock = threading.RLock()
_marks: dict[tuple[str, str], float] = {}


def mark_erasing(profile_id: str, fact_id: str) -> None:
    now = time.time()
    with _lock:
        _marks[(profile_id, fact_id)] = now
        _prune(now)


def clear_erasing(profile_id: str, fact_id: str) -> None:
    with _lock:
        _marks.pop((profile_id, fact_id), None)


def is_erasing(profile_id: str, fact_id: str) -> bool:
    now = time.time()
    with _lock:
        ts = _marks.get((profile_id, fact_id))
        if ts is None:
            return False
        if now - ts > _TTL_SECONDS:
            _marks.pop((profile_id, fact_id), None)
            return False
        return True


def _prune(now: float) -> None:
    expired = [key for key, ts in _marks.items() if now - ts > _TTL_SECONDS]
    for key in expired:
        _marks.pop(key, None)


__all__ = ["mark_erasing", "clear_erasing", "is_erasing"]
