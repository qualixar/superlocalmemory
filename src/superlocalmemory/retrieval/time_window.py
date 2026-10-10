# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Time-window parsing + membership for recall (Phase 4, T-window).

Pure, dependency-free helpers so recall() can prune candidates to an
event-time range. Two window forms are accepted:

  * relative string — ``"1h" | "24h" | "7d" | "30d" | "90d" | "1y"`` etc.
    (``<int><unit>`` where unit is h/d/w/m/y; m≈30d, y≈365d) → ``[now-Δ, now]``.
  * explicit range — ``(start_iso, end_iso)`` two-tuple of timestamps.

Timestamps are parsed tolerantly: SQLite ``datetime('now')`` form
(``"YYYY-MM-DD HH:MM:SS"``), ISO-8601 with ``T`` and/or trailing ``Z``, and
date-only ``"YYYY-MM-DD"`` are all accepted. Comparing the *strings* would be
wrong (a space sorts before ``T``), so everything is parsed to tz-aware UTC
datetimes before comparison — never lexicographic.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

__all__ = [
    "parse_timestamp",
    "parse_window",
    "in_window",
    "infer_window_from_query",
    "has_primary_evidence",
    "windowed_candidates",
]

_REL = re.compile(r"^\s*(\d+)\s*([hdwmy])\s*$", re.IGNORECASE)
_DATE_ONLY = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s*$")

# Hours per unit. Month and year are documented approximations.
_UNIT_HOURS: dict[str, int] = {
    "h": 1,
    "d": 24,
    "w": 24 * 7,
    "m": 24 * 30,
    "y": 24 * 365,
}


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse a stored timestamp into a tz-aware UTC datetime, or None.

    Accepts SQLite ``datetime('now')`` (space-separated), ISO-8601 (``T`` and
    optional ``Z``), and date-only strings. Naive values are assumed UTC.
    """
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    dt: datetime | None = None
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        # Fall back to a date-only prefix (e.g. "2026-03-15 ...").
        try:
            dt = datetime.fromisoformat(text[:10])
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_window(
    window: str | tuple[str, str] | list | None,
    now: datetime | None = None,
) -> tuple[datetime, datetime] | None:
    """Resolve a window spec to a ``(start, end)`` UTC datetime pair, or None.

    Returns None when the spec is None or unparseable (caller then applies no
    time filter — additive, fail-open).
    """
    if window is None:
        return None
    _now = now or datetime.now(timezone.utc)
    if _now.tzinfo is None:
        _now = _now.replace(tzinfo=timezone.utc)

    # Explicit (start, end) range.
    if isinstance(window, (tuple, list)):
        if len(window) != 2:
            return None
        return _explicit_range(window[0], window[1])

    # String forms: relative "<int><unit>" or an explicit range written as
    # "start..end" / "start,end" (so it survives URL params, JSON, and CLI args
    # as a single value — no tuple serialization needed on the wire).
    if isinstance(window, str):
        m = _REL.match(window)
        if m:
            n = int(m.group(1))
            unit = m.group(2).lower()
            hours = n * _UNIT_HOURS[unit]
            return (_now - timedelta(hours=hours), _now)
        for sep in ("..", ","):
            if sep in window:
                left, _, right = window.partition(sep)
                return _explicit_range(left, right)

    return None


def _explicit_range(left: object, right: object) -> tuple[datetime, datetime] | None:
    """Order the two ends; a date-only END covers that whole day.

    "2026-07-01..2026-07-31" means all of July. Read as instants, its end was
    31 July 00:00 and the last day fell outside its own range. The end of a
    date-only bound is the last microsecond of that day — the same set as an
    exclusive end at the next midnight, at datetime's full resolution. An end
    written with a time is taken exactly as written.
    """
    ends = []
    for raw in (left, right):
        parsed = parse_timestamp(raw) if isinstance(raw, str) else None
        if parsed is None:
            return None
        ends.append((parsed, bool(_DATE_ONLY.match(raw))))
    (start, _), (end, end_is_date) = sorted(ends, key=lambda item: (item[0], item[1]))
    if end_is_date:
        end = end + timedelta(days=1) - timedelta(microseconds=1)
    return (start, end)


# Natural-language temporal-scope patterns → relative window spec. Ordered:
# more specific ("last 3 weeks") is matched before generic ("last week").
_UNIT_TO_SPEC = {"day": "d", "week": "w", "month": "m", "year": "y"}
_QUERY_N_UNIT = re.compile(
    r"\b(?:last|past|previous|prior)\s+(\d{1,3})\s+(day|week|month|year)s?\b",
    re.IGNORECASE,
)
_QUERY_PHRASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\btoday\b", re.IGNORECASE), "1d"),
    (re.compile(r"\byesterday\b", re.IGNORECASE), "2d"),
    (re.compile(r"\b(?:this|last|past|previous)\s+week\b", re.IGNORECASE), "7d"),
    (re.compile(r"\b(?:this|last|past|previous)\s+month\b", re.IGNORECASE), "30d"),
    (re.compile(r"\b(?:this|last|past|previous)\s+year\b", re.IGNORECASE), "1y"),
    (re.compile(r"\b(?:recent|recently|lately)\b", re.IGNORECASE), "30d"),
)


def infer_window_from_query(query: str | None) -> str | None:
    """Infer a relative time window from natural-language scope in a query.

    Recognises a small, unambiguous set of temporal-scope phrases ("yesterday",
    "last week", "past 3 months", "recently") and maps them to a relative
    window spec ("2d", "7d", "3m", …) that ``parse_window`` understands. Returns
    None when no clear temporal scope is present, so recall applies no window.

    Deliberately conservative — only fires on explicit scope words, never on
    bare content — so it augments, never surprises. Callers use it only when the
    user did not pass an explicit window.
    """
    if not query or not isinstance(query, str):
        return None
    m = _QUERY_N_UNIT.search(query)
    if m:
        n = int(m.group(1))
        unit = _UNIT_TO_SPEC.get(m.group(2).lower())
        if unit and n > 0:
            return f"{n}{unit}"
    for pattern, spec in _QUERY_PHRASES:
        if pattern.search(query):
            return spec
    return None


def in_window(
    event_time: str | None,
    bounds: tuple[datetime, datetime] | None,
) -> bool:
    """True if ``event_time`` falls within ``bounds`` (inclusive).

    No bounds → always True (no filtering). An unparseable/missing event time
    is excluded (conservative: a windowed query returns only datable facts in
    range).
    """
    if bounds is None:
        return True
    dt = parse_timestamp(event_time)
    if dt is None:
        return False
    start, end = bounds
    return start <= dt <= end


def has_primary_evidence(channel_scores: Mapping[str, float] | None,
                         min_semantic: float, min_media: float = 0.30) -> bool:
    """The evidence floor's test (``RetrievalEngine._apply_evidence_floor``):
    meaning at or above ``min_semantic``, or any keyword, entity or time
    evidence, or a picture match at or above ``min_media``. Associative channels
    (Hopfield, spreading activation) do not count.
    """
    cs = channel_scores or {}
    return (
        cs.get("semantic", 0.0) >= min_semantic
        or cs.get("bm25", 0.0) > 0.0
        or cs.get("entity_graph", 0.0) > 0.0
        or cs.get("temporal", 0.0) > 0.0
        or ("media" in cs and cs["media"] >= min_media)
    )


def windowed_candidates(fused: Sequence[Any], inside: Callable[[str], bool], *,
                        explicit: bool, min_semantic: float,
                        min_media: float = 0.30) -> list[Any]:
    """The candidates a time window keeps.

    An explicit window is the caller's instruction and is honoured even when it
    keeps nothing. A window inferred from the question's wording ("recently",
    "last week") must never make recall worse: it narrows only when some
    candidate inside it has primary evidence. Otherwise every candidate stays.
    Without that, a question carrying such a word about an older memory kept
    only unrelated recent facts, which the evidence floor then removed, and
    recall answered with nothing (measured: 36 questions made of a stored
    memory's own words, on a copy of a 22k-fact store).
    """
    kept = [fr for fr in fused if inside(fr.fact_id)]
    if explicit:
        return kept
    if any(has_primary_evidence(getattr(fr, "channel_scores", None), min_semantic, min_media)
           for fr in kept):
        return kept
    return list(fused)
