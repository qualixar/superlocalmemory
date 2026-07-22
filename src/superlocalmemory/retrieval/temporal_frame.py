# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Temporal-context injection helpers (Phase 4, T-inject).

LLMs have no innate sense of time — a recalled fact reads the same whether it
was stored an hour ago or two years ago. These pure helpers give every recalled
memory a human-relative age label and give the whole result set a "temporal
frame" header anchoring it to *now*, so the model can weigh recency and treat
aged facts as possibly stale.

Reuses ``time_window.parse_timestamp`` for tolerant timestamp parsing (SQLite
space form, ISO ``T``/``Z``, date-only) — comparisons are always on parsed
datetimes, never strings.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from superlocalmemory.retrieval.time_window import parse_timestamp

__all__ = ["relative_age", "temporal_frame"]


def _plural(n: int, unit: str) -> str:
    return f"{n} {unit}" + ("" if n == 1 else "s")


def relative_age(timestamp: str | None, now: datetime | None = None) -> str:
    """Human-relative age of ``timestamp`` vs now, e.g. "3 days ago".

    Returns "" when the timestamp is missing/unparseable. Future timestamps
    (e.g. a referenced event date ahead of now) read as "in N …".
    """
    dt = parse_timestamp(timestamp)
    if dt is None:
        return ""
    _now = now or datetime.now(timezone.utc)
    if _now.tzinfo is None:
        _now = _now.replace(tzinfo=timezone.utc)
    secs = (_now - dt).total_seconds()
    future = secs < 0
    secs = abs(secs)

    if secs < 45:
        return "just now"
    minutes = secs / 60.0
    hours = minutes / 60.0
    days = hours / 24.0
    if minutes < 45:
        phrase = _plural(round(minutes), "minute")
    elif hours < 24:
        phrase = _plural(round(hours), "hour")
    elif days < 14:
        phrase = _plural(round(days), "day")
    elif days < 60:
        phrase = _plural(round(days / 7.0), "week")
    elif days < 365:
        phrase = _plural(round(days / 30.0), "month")
    else:
        phrase = _plural(round(days / 365.0), "year")
    return f"in {phrase}" if future else f"{phrase} ago"


def temporal_frame(
    timestamps: Iterable[str | None],
    now: datetime | None = None,
) -> str:
    """A one-line "now" anchor + age span for a set of recalled timestamps.

    Example: ``"Now: 2026-07-22T12:00:00+00:00. Recalled memories span 2 years
    ago → just now. Treat undated or aged facts as possibly stale."``

    With no dated timestamps, returns just the now-anchor + an undated note.
    """
    _now = now or datetime.now(timezone.utc)
    if _now.tzinfo is None:
        _now = _now.replace(tzinfo=timezone.utc)
    now_iso = _now.replace(microsecond=0).isoformat()

    dts = [d for d in (parse_timestamp(t) for t in timestamps) if d is not None]
    if not dts:
        return f"Now: {now_iso}. Recalled memories are undated."

    oldest = min(dts)
    newest = max(dts)
    span = (
        relative_age(oldest.isoformat(), _now)
        if oldest == newest
        else f"{relative_age(oldest.isoformat(), _now)} → "
             f"{relative_age(newest.isoformat(), _now)}"
    )
    return (
        f"Now: {now_iso}. Recalled memories span {span}. "
        f"Treat undated or aged facts as possibly stale."
    )
