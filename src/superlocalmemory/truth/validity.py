# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""When a superseded memory stopped being true (its ``valid_until``).

A correction that retires a predecessor records *when SLM learned* it was
replaced (``system_expired_at``) but, before 4.2.30, never *when it stopped
being true in the world* (``valid_until``): on one real store 340 applied
corrections had none. Without it "what was true in March" cannot be answered
from valid time.

The best evidence for that moment, strongest first: a time the reviewer gave
(``event_valid_until``); the successor's own validity start; the successor's
valid-from mirror; when the successor was saved; when the correction was
applied. Evidence that would end the interval at or before the predecessor
began is skipped (typically a date-only value on the same day, coarser than
the timestamps after it); when every candidate would, the interval stays open,
because an inverted interval is worse than an open one.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime

_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalise(value: object) -> str | None:
    """An RFC 3339 UTC timestamp, or None for empty / unreadable values."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if _DATE_ONLY.match(text):
        text += "T00:00:00+00:00"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def closing_time(
    *,
    event_valid_until: object = None,
    successor_validity_from: object = None,
    successor_fact_valid_from: object = None,
    successor_created_at: object = None,
    applied_at: object = None,
    predecessor_valid_from: object = None,
) -> str | None:
    """The predecessor's ``valid_until``, or None when no safe value exists."""
    start = normalise(predecessor_valid_from)
    for candidate in (event_valid_until, successor_validity_from, successor_fact_valid_from,
                      successor_created_at, applied_at):
        close = normalise(candidate)
        if close is None:
            continue
        if start is not None and _instant(close) <= _instant(start):
            # Often a date-only value on the predecessor's own day ("2026-10-06"
            # vs "2026-10-06"): coarser than the timestamps after it, so try those.
            continue
        return close
    return None


def _instant(value: str) -> datetime:
    return datetime.fromisoformat(value)


def earlier(a: str | None, b: str | None) -> bool:
    """True when both are readable and ``a`` is strictly before ``b``."""
    first, second = normalise(a), normalise(b)
    return first is not None and second is not None and _instant(first) < _instant(second)


def successor_takeover(
    conn: sqlite3.Connection, successor_fact_id: str, *,
    predecessor_valid_from: object, occurred_at: str,
) -> str | None:
    """When ``successor_fact_id`` became the true value; None when no safe time exists."""
    row_tv = conn.execute("SELECT valid_from FROM fact_temporal_validity WHERE fact_id=?",
                          (successor_fact_id,)).fetchone()
    try:
        row = conn.execute("SELECT valid_from, created_at FROM atomic_facts WHERE fact_id=?",
                           (successor_fact_id,)).fetchone()
    except sqlite3.OperationalError:  # a minimal fact table without the time columns
        row = None
    return closing_time(
        successor_validity_from=row_tv[0] if row_tv is not None else None,
        successor_fact_valid_from=row[0] if row is not None else None,
        successor_created_at=row[1] if row is not None else None,
        applied_at=occurred_at,
        predecessor_valid_from=predecessor_valid_from,
    )


__all__ = ["closing_time", "earlier", "normalise", "successor_takeover"]
