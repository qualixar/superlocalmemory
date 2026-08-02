# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import threading
import time

_TTL_SECONDS = 300.0

_lock = threading.RLock()
_epochs: dict[tuple[str, str], tuple[int, float]] = {}


def record_admission_epoch(profile_id: str, idempotency_key: str, epoch: int) -> None:
    if not idempotency_key:
        return
    now = time.time()
    with _lock:
        _prune(now)
        # First-writer-wins while an admit for this key is in flight: a
        # concurrent admit captured after a binding transition must not be able
        # to relax the epoch a prior in-flight admit will be checked against.
        _epochs.setdefault((profile_id, idempotency_key), (epoch, now))


def admitted_epoch(profile_id: str, idempotency_key: str) -> int | None:
    if not idempotency_key:
        return None
    now = time.time()
    with _lock:
        entry = _epochs.get((profile_id, idempotency_key))
        if entry is None:
            return None
        epoch, ts = entry
        if now - ts > _TTL_SECONDS:
            _epochs.pop((profile_id, idempotency_key), None)
            return None
        return epoch


def clear_admission_epoch(profile_id: str, idempotency_key: str) -> None:
    if not idempotency_key:
        return
    with _lock:
        _epochs.pop((profile_id, idempotency_key), None)


def _prune(now: float) -> None:
    expired = [key for key, (_, ts) in _epochs.items() if now - ts > _TTL_SECONDS]
    for key in expired:
        _epochs.pop(key, None)


__all__ = ["record_admission_epoch", "admitted_epoch", "clear_admission_epoch"]
