# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Temporal Parser (3-Date Model).

Parses and enriches temporal information for every memory:
  1. observation_date  — when the conversation happened
  2. referenced_date   — date mentioned in content ("last Tuesday")
  3. temporal_interval  — [start, end] for duration events

Mastra achieved 95.5% temporal reasoning with this pattern.
V1 stored session_date as raw strings and ignored content dates entirely.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from dateutil.parser import parse as dateutil_parse, ParserError
from dateutil.relativedelta import relativedelta

from superlocalmemory.storage.models import AtomicFact, TemporalEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex patterns (compiled once, reused)
# ---------------------------------------------------------------------------

_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

_US_DATE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")

_WRITTEN_DATE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},?\s*\d{4}\b",
    re.IGNORECASE,
)

_WRITTEN_DATE_DMY = re.compile(
    r"\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r",?\s*\d{4}\b",
    re.IGNORECASE,
)

# "January 2023", "March 2024" — month + year without day
_MONTH_YEAR = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{4}\b",
    re.IGNORECASE,
)

_WEEKDAYS = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
_TIME_UNITS = r"(?:days?|weeks?|months?|years?)"

_RELATIVE = re.compile(
    rf"\b(?:last|next|this)\s+(?:{_WEEKDAYS}|week|month|year|spring|summer|autumn|fall|winter)\b",
    re.IGNORECASE,
)

_AGO = re.compile(
    rf"\b(\d+)\s+{_TIME_UNITS}\s+ago\b",
    re.IGNORECASE,
)

_IN_FUTURE = re.compile(
    rf"\bin\s+(\d+)\s+{_TIME_UNITS}\b",
    re.IGNORECASE,
)

_DURATION = re.compile(
    r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:\.|,|;|$)",
    re.IGNORECASE,
)

_FOR_DURATION = re.compile(
    rf"\bfor\s+(\d+)\s+{_TIME_UNITS}\b",
    re.IGNORECASE,
)

_VAGUE_TERMS: dict[str, int] = {
    "yesterday": -1,
    "today": 0,
    "tomorrow": 1,
    "recently": -7,
    "a while ago": -30,
    "soon": 7,
    "the other day": -3,
    "last night": -1,
}

_SEASON_MONTHS: dict[str, tuple[int, int]] = {
    "spring": (3, 5),
    "summer": (6, 8),
    "autumn": (9, 11),
    "fall": (9, 11),
    "winter": (12, 2),
}

_WEEKDAY_MAP: dict[str, int] = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _safe_iso(dt: datetime | None) -> str | None:
    """Convert datetime to ISO-8601 string, or None."""
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _extract_unit(text: str) -> str:
    """Extract time unit from a matched span (days, weeks, months, years)."""
    lower = text.lower()
    for unit in ("year", "month", "week", "day"):
        if unit in lower:
            return unit
    return "day"


class TemporalParser:
    """3-date temporal parser for memory enrichment.

    Extracts observation_date, referenced_date, and temporal intervals
    from session dates and fact content. Handles absolute, relative,
    duration, and vague temporal expressions.
    """

    def __init__(self, reference_date: str | None = None) -> None:
        """Initialize with an optional reference date for relative computation.

        Args:
            reference_date: ISO-8601 string used as "today" for relative dates.
                           Defaults to current UTC time if None.
        """
        if reference_date is not None:
            try:
                parsed = dateutil_parse(reference_date)
                self._ref = parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed
            except (ParserError, ValueError):
                logger.warning("Unparseable reference_date %r, using UTC now", reference_date)
                self._ref = datetime.now(UTC)
        else:
            self._ref = datetime.now(UTC)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_session_date(self, raw_date: str) -> str | None:
        """Parse LoCoMo-style session dates into ISO-8601.

        Handles formats like:
          - "1:56 pm on 8 May, 2023"
          - "May 8, 2023"
          - "2023-05-08"
          - "8:56 pm on 20 July, 2023"

        Returns:
            ISO-8601 string or None if unparseable.
        """
        if not raw_date or not raw_date.strip():
            return None
        try:
            dt = dateutil_parse(raw_date, fuzzy=True)
            return _safe_iso(dt)
        except (ParserError, ValueError, OverflowError):
            logger.debug("Could not parse session_date: %r", raw_date)
            return None

    def extract_dates_from_text(self, text: str) -> dict[str, str | None]:
        """Extract dates mentioned in fact content.

        Strategy:
          1. Regex pass for structured date patterns
          2. Relative date resolution against reference_date
          3. Vague term mapping to approximate offsets
          4. Duration extraction for intervals

        Returns:
            dict with keys: referenced_date, interval_start, interval_end
            All values are ISO-8601 strings or None.
        """
        if not text:
            return {"referenced_date": None, "interval_start": None, "interval_end": None}

        dates_found: list[datetime] = []
        interval_start: datetime | None = None
        interval_end: datetime | None = None

        # --- Pass 1: Duration expressions ("from X to Y") ---
        dur_match = _DURATION.search(text)
        if dur_match:
            start_dt = self._try_parse(dur_match.group(1).strip())
            end_dt = self._try_parse(dur_match.group(2).strip())
            if start_dt is not None and end_dt is not None:
                interval_start = start_dt
                interval_end = end_dt

        # --- Pass 2: "for N units" duration ---
        if interval_start is None:
            for_match = _FOR_DURATION.search(text)
            if for_match:
                count = int(for_match.group(1))
                unit = _extract_unit(for_match.group(0))
                delta = self._unit_delta(count, unit)
                interval_start = self._ref
                interval_end = self._ref + delta

        # --- Pass 3: Absolute date patterns ---
        for pattern in (_ISO_DATE, _WRITTEN_DATE, _WRITTEN_DATE_DMY, _MONTH_YEAR, _US_DATE):
            for match in pattern.finditer(text):
                dt = self._try_parse(match.group(0))
                if dt is not None:
                    dates_found.append(dt)

        # --- Pass 4: Relative expressions ---
        for match in _RELATIVE.finditer(text):
            dt = self._resolve_relative(match.group(0))
            if dt is not None:
                dates_found.append(dt)

        # --- Pass 5: "N units ago" ---
        for match in _AGO.finditer(text):
            count = int(match.group(1))
            unit = _extract_unit(match.group(0))
            dt = self._ref - self._unit_delta(count, unit)
            dates_found.append(dt)

        # --- Pass 6: "in N units" ---
        for match in _IN_FUTURE.finditer(text):
            count = int(match.group(1))
            unit = _extract_unit(match.group(0))
            dt = self._ref + self._unit_delta(count, unit)
            dates_found.append(dt)

        # --- Pass 7: Vague terms ---
        text_lower = text.lower()
        for term, offset_days in _VAGUE_TERMS.items():
            if term in text_lower:
                dates_found.append(self._ref + timedelta(days=offset_days))
                break  # Take first vague match only

        # --- Assemble result ---
        referenced: datetime | None = None
        if dates_found:
            referenced = dates_found[0]
            # Two distinct dates without explicit "from...to" -> interval
            if len(dates_found) >= 2 and interval_start is None:
                # Normalize tz-awareness before comparing
                _normed = [d.replace(tzinfo=UTC) if d.tzinfo is None else d for d in dates_found[:2]]
                sorted_dates = sorted(_normed)
                interval_start = sorted_dates[0]
                interval_end = sorted_dates[1]

        return {
            "referenced_date": _safe_iso(referenced),
            "interval_start": _safe_iso(interval_start),
            "interval_end": _safe_iso(interval_end),
        }

    def build_temporal_event(
        self,
        fact: AtomicFact,
        session_date: str | None,
        entity_id: str,
    ) -> TemporalEvent | None:
        """Create a TemporalEvent for a fact-entity pair.

        Combines the parsed session_date (observation) with dates
        extracted from fact content (referenced + interval).

        Returns:
            TemporalEvent if any temporal info exists, else None.
        """
        obs_date = self.parse_session_date(session_date) if session_date else None
        content_dates = self.extract_dates_from_text(fact.content)

        ref_date = content_dates["referenced_date"]
        int_start = content_dates["interval_start"]
        int_end = content_dates["interval_end"]

        # Use fact-level fields if already populated (from upstream)
        if fact.observation_date and not obs_date:
            obs_date = fact.observation_date
        if fact.referenced_date and not ref_date:
            ref_date = fact.referenced_date
        if fact.interval_start and not int_start:
            int_start = fact.interval_start
        if fact.interval_end and not int_end:
            int_end = fact.interval_end

        # Only create event if we have at least one temporal anchor
        if not any([obs_date, ref_date, int_start, int_end]):
            return None

        return TemporalEvent(
            profile_id=fact.profile_id,
            scope=fact.scope,
            shared_with=fact.shared_with,
            entity_id=entity_id,
            fact_id=fact.fact_id,
            observation_date=obs_date,
            referenced_date=ref_date,
            interval_start=int_start,
            interval_end=int_end,
            description=fact.content[:200],
        )

    def build_entity_timeline(
        self,
        entity_id: str,
        facts: list[AtomicFact],
        session_date: str | None = None,
    ) -> list[TemporalEvent]:
        """Build a chronological timeline for an entity from its facts.

        Returns:
            List of TemporalEvents sorted by earliest available date.
        """
        events: list[TemporalEvent] = []
        for fact in facts:
            s_date = session_date or fact.observation_date
            event = self.build_temporal_event(fact, s_date, entity_id)
            if event is not None:
                events.append(event)

        # Sort by the earliest available date
        def _sort_key(ev: TemporalEvent) -> str:
            for field in (ev.referenced_date, ev.observation_date, ev.interval_start):
                if field:
                    return field
            return "9999-12-31T23:59:59"

        return sorted(events, key=_sort_key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_parse(self, text: str) -> datetime | None:
        """Attempt to parse a text fragment as a date."""
        if not text or len(text.strip()) < 3:
            return None
        try:
            return dateutil_parse(text, fuzzy=True)
        except (ParserError, ValueError, OverflowError):
            return None

    def _resolve_relative(self, expr: str) -> datetime | None:
        """Resolve a relative expression like 'last Tuesday' or 'next month'."""
        lower = expr.lower().strip()
        parts = lower.split()
        if len(parts) < 2:
            return None

        modifier = parts[0]  # last / next / this
        target = parts[1]

        # Weekday resolution
        if target in _WEEKDAY_MAP:
            target_day = _WEEKDAY_MAP[target]
            current_day = self._ref.weekday()
            if modifier == "last":
                diff = (current_day - target_day) % 7
                diff = diff if diff > 0 else 7
                return self._ref - timedelta(days=diff)
            elif modifier == "next":
                diff = (target_day - current_day) % 7
                diff = diff if diff > 0 else 7
                return self._ref + timedelta(days=diff)
            else:  # this
                diff = (target_day - current_day) % 7
                return self._ref + timedelta(days=diff)

        # Week / month / year
        if target == "week":
            offset = {"last": -7, "next": 7, "this": 0}
            return self._ref + timedelta(days=offset.get(modifier, 0))
        if target == "month":
            offset = {"last": -1, "next": 1, "this": 0}
            return self._ref + relativedelta(months=offset.get(modifier, 0))
        if target == "year":
            offset = {"last": -1, "next": 1, "this": 0}
            return self._ref + relativedelta(years=offset.get(modifier, 0))

        # Seasons
        if target in _SEASON_MONTHS:
            start_m, end_m = _SEASON_MONTHS[target]
            year = self._ref.year
            if modifier == "last":
                year -= 1
            elif modifier == "next":
                year += 1
            # Return midpoint of season
            if start_m <= end_m:
                mid_m = (start_m + end_m) // 2
            else:
                mid_m = 1  # winter spans Dec-Feb, midpoint ~ Jan
            try:
                return datetime(year, mid_m, 15)
            except ValueError:
                return None

        return None

    @staticmethod
    def _unit_delta(count: int, unit: str) -> timedelta | relativedelta:
        """Build a timedelta/relativedelta from count + unit string."""
        if unit == "day":
            return timedelta(days=count)
        if unit == "week":
            return timedelta(weeks=count)
        if unit == "month":
            return relativedelta(months=count)
        if unit == "year":
            return relativedelta(years=count)
        return timedelta(days=count)
