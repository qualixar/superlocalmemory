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
from datetime import UTC, datetime, timedelta

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


#: Where a ``valid_until`` came from (I-12, R-11), stored as
#: ``fact_temporal_validity.valid_until_source`` (M074). ``stated``: the memory or
#: its writer gave the end. ``reviewer``: a person gave it. ``successor_validity`` /
#: ``successor_fact``: when the replacing memory became true. ``successor_saved`` /
#: ``applied``: system time only (when SLM learned), so an UPPER BOUND on when the
#: old value stopped being true, never the time itself (M-10). ``recorded``: an end
#: that was already there, whose evidence is not known.
SOURCES = ("stated", "reviewer", "successor_validity", "successor_fact", "successor_saved",
           "applied", "recorded")
UPPER_BOUND_SOURCES = frozenset({"successor_saved", "applied"})


def closing_time_sourced(
    *,
    event_valid_until: object = None,
    successor_validity_from: object = None,
    successor_fact_valid_from: object = None,
    successor_created_at: object = None,
    applied_at: object = None,
    predecessor_valid_from: object = None,
) -> tuple[str | None, str | None]:
    """(the predecessor's ``valid_until``, its source), or (None, None)."""
    start = normalise(predecessor_valid_from)
    for source, candidate in (("reviewer", event_valid_until),
                              ("successor_validity", successor_validity_from),
                              ("successor_fact", successor_fact_valid_from),
                              ("successor_saved", successor_created_at),
                              ("applied", applied_at)):
        close = normalise(candidate)
        if close is None:
            continue
        if start is not None and _instant(close) <= _instant(start):
            # Often a date-only value on the predecessor's own day ("2026-10-06"
            # vs "2026-10-06"): coarser than the timestamps after it, so try those.
            continue
        return close, source
    return None, None


def closing_time(**kwargs: object) -> str | None:
    """The predecessor's ``valid_until``, or None when no safe value exists."""
    return closing_time_sourced(**kwargs)[0]


def _instant(value: str) -> datetime:
    return datetime.fromisoformat(value)


#: A date with no time zone is every instant that is that date somewhere on
#: Earth: from 00:00 at UTC+14 to 24:00 at UTC-12.
_EARLIEST_ZONE = timedelta(hours=14)
_LATEST_ZONE = timedelta(hours=12)

BEFORE, AFTER, SAME, INCOMPARABLE = "before", "after", "same", "incomparable"


def span(value: object) -> tuple[datetime, datetime] | None:
    """The instants ``value`` may denote: a point for a timestamp, ~2 days for a bare date."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if _DATE_ONLY.match(text):
        try:
            day = datetime.fromisoformat(text + "T00:00:00+00:00")
        except ValueError:
            return None
        return day - _EARLIEST_ZONE, day + timedelta(days=1) + _LATEST_ZONE
    point = normalise(text)
    if point is None:
        return None
    instant = _instant(point)
    return instant, instant


def order(a: object, b: object) -> str | None:
    """Interval order: ``a`` before ``b`` only if every instant of ``a`` precedes ``b``.

    Overlapping spans (a bare date and a timestamp on that day, two different
    rewordings of one day) are ``incomparable``: no conclusion about which came
    first, so callers must not turn the pair into history or a supersession.
    None when either side is unreadable.
    """
    sa, sb = span(a), span(b)
    if sa is None or sb is None:
        return None
    if sa == sb:
        return SAME if sa[0] == sa[1] else INCOMPARABLE
    if sa[1] <= sb[0]:
        return BEFORE
    if sb[1] <= sa[0]:
        return AFTER
    return INCOMPARABLE


def earlier(a: str | None, b: str | None) -> bool:
    """True when ``a`` is certainly before ``b`` (interval order)."""
    return order(a, b) == BEFORE


def retire_until(*, reviewer: object, existing: object, derived: object,
                 predecessor_valid_from: object) -> str | None:
    """``retire_until_sourced`` without the source."""
    return retire_until_sourced(reviewer=reviewer, existing=existing, derived=derived,
                                predecessor_valid_from=predecessor_valid_from)[0]


def retire_until_sourced(*, reviewer: object, existing: object, derived: object,
                         predecessor_valid_from: object, existing_source: str | None = None,
                         derived_source: str | None = None) -> tuple[str | None, str | None]:
    """The ``valid_until`` to store when a memory is retired.

    A reviewer's time wins, and is refused (``ValueError``) when it is not after
    the memory began: a person's input is never silently replaced. Otherwise
    the earlier of an end already recorded and the successor's takeover: a
    future expiry set before must not outlive the supersession that ended it
    sooner. When the two cannot be ordered, the recorded one is kept.
    """
    if reviewer is not None:
        when = normalise(reviewer)
        if when is None:
            raise ValueError("event_valid_until is not a readable time")
        if predecessor_valid_from is not None and order(when, predecessor_valid_from) in (
                BEFORE, SAME):
            raise ValueError("event_valid_until is not after the memory began")
        return when, "reviewer"
    kept = (str(existing), existing_source or "recorded") if existing is not None else None
    if not isinstance(derived, str):
        return kept if kept is not None else (None, None)
    if kept is None or order(derived, existing) == BEFORE:
        return derived, derived_source
    return kept


def successor_takeover(
    conn: sqlite3.Connection, successor_fact_id: str, *,
    predecessor_valid_from: object, occurred_at: str,
) -> str | None:
    """When ``successor_fact_id`` became the true value; None when no safe time exists."""
    return successor_takeover_sourced(conn, successor_fact_id,
                                      predecessor_valid_from=predecessor_valid_from,
                                      occurred_at=occurred_at)[0]


def successor_takeover_sourced(
    conn: sqlite3.Connection, successor_fact_id: str, *,
    predecessor_valid_from: object, occurred_at: str,
) -> tuple[str | None, str | None]:
    """``successor_takeover`` with the source of the time it found."""
    row_tv = conn.execute("SELECT valid_from FROM fact_temporal_validity WHERE fact_id=?",
                          (successor_fact_id,)).fetchone()
    try:
        row = conn.execute("SELECT valid_from, created_at FROM atomic_facts WHERE fact_id=?",
                           (successor_fact_id,)).fetchone()
    except sqlite3.OperationalError:  # a minimal fact table without the time columns
        row = None
    return closing_time_sourced(
        successor_validity_from=row_tv[0] if row_tv is not None else None,
        successor_fact_valid_from=row[0] if row is not None else None,
        successor_created_at=row[1] if row is not None else None,
        applied_at=occurred_at,
        predecessor_valid_from=predecessor_valid_from,
    )


__all__ = ["AFTER", "BEFORE", "INCOMPARABLE", "SAME", "SOURCES", "UPPER_BOUND_SOURCES",
           "closing_time", "closing_time_sourced", "earlier", "normalise", "order", "retire_until",
           "retire_until_sourced", "span", "successor_takeover", "successor_takeover_sourced"]
