# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.encoding.temporal_parser.

Covers:
  - parse_session_date() for LoCoMo-style dates
  - extract_dates_from_text() — absolute, relative, vague, duration, interval
  - build_temporal_event() and build_entity_timeline()
  - Edge cases: empty text, bad dates, overflow
"""

from __future__ import annotations

import pytest

from superlocalmemory.encoding.temporal_parser import TemporalParser
from superlocalmemory.storage.models import AtomicFact, FactType, TemporalEvent


# ---------------------------------------------------------------------------
# parse_session_date
# ---------------------------------------------------------------------------

class TestParseSessionDate:
    def test_locomo_format(self) -> None:
        parser = TemporalParser()
        result = parser.parse_session_date("1:56 pm on 8 May, 2023")
        assert result is not None
        assert "2023" in result
        assert "05" in result

    def test_standard_date(self) -> None:
        parser = TemporalParser()
        result = parser.parse_session_date("May 8, 2023")
        assert result is not None

    def test_iso_date(self) -> None:
        parser = TemporalParser()
        result = parser.parse_session_date("2023-05-08")
        assert result is not None
        assert "2023-05-08" in result

    def test_empty_string(self) -> None:
        parser = TemporalParser()
        assert parser.parse_session_date("") is None
        assert parser.parse_session_date("   ") is None

    def test_unparseable(self) -> None:
        parser = TemporalParser()
        assert parser.parse_session_date("xyzzy not a date") is None


# ---------------------------------------------------------------------------
# extract_dates_from_text — absolute dates
# ---------------------------------------------------------------------------

class TestExtractDatesAbsolute:
    def test_iso_date(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")
        result = parser.extract_dates_from_text("Meeting on 2026-04-15")
        assert result["referenced_date"] is not None
        assert "2026-04-15" in result["referenced_date"]

    def test_written_date(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")
        result = parser.extract_dates_from_text("Born on March 15, 2000")
        assert result["referenced_date"] is not None

    def test_us_date(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")
        result = parser.extract_dates_from_text("Deadline is 3/15/2026")
        assert result["referenced_date"] is not None

    def test_no_dates(self) -> None:
        parser = TemporalParser()
        result = parser.extract_dates_from_text("No dates in this text at all")
        assert result["referenced_date"] is None
        assert result["interval_start"] is None
        assert result["interval_end"] is None

    def test_empty_text(self) -> None:
        parser = TemporalParser()
        result = parser.extract_dates_from_text("")
        assert result["referenced_date"] is None


# ---------------------------------------------------------------------------
# extract_dates_from_text — relative dates
# ---------------------------------------------------------------------------

class TestExtractDatesRelative:
    def test_last_weekday(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")  # Wednesday
        result = parser.extract_dates_from_text("I went there last Monday")
        assert result["referenced_date"] is not None

    def test_next_week(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")
        result = parser.extract_dates_from_text("Meeting next week")
        assert result["referenced_date"] is not None

    def test_next_month(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")
        result = parser.extract_dates_from_text("Starting next month")
        assert result["referenced_date"] is not None

    def test_ago(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")
        result = parser.extract_dates_from_text("That happened 3 days ago")
        assert result["referenced_date"] is not None

    def test_in_future(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")
        result = parser.extract_dates_from_text("Will be ready in 2 weeks")
        assert result["referenced_date"] is not None


# ---------------------------------------------------------------------------
# extract_dates_from_text — vague terms
# ---------------------------------------------------------------------------

class TestExtractDatesVague:
    def test_yesterday(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")
        result = parser.extract_dates_from_text("I saw her yesterday")
        assert result["referenced_date"] is not None

    def test_tomorrow(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")
        result = parser.extract_dates_from_text("Meeting tomorrow at 3pm")
        assert result["referenced_date"] is not None

    def test_today(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")
        result = parser.extract_dates_from_text("It happened today")
        assert result["referenced_date"] is not None

    def test_recently(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")
        result = parser.extract_dates_from_text("She recently joined the company")
        assert result["referenced_date"] is not None


# ---------------------------------------------------------------------------
# extract_dates_from_text — duration
# ---------------------------------------------------------------------------

class TestExtractDatesDuration:
    def test_from_to(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")
        result = parser.extract_dates_from_text(
            "From January 1, 2026 to March 1, 2026 we traveled."
        )
        assert result["interval_start"] is not None
        assert result["interval_end"] is not None

    def test_for_duration(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")
        result = parser.extract_dates_from_text("I worked there for 3 months")
        assert result["interval_start"] is not None
        assert result["interval_end"] is not None

    def test_two_distinct_dates_become_interval(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")
        result = parser.extract_dates_from_text(
            "Events on 2026-01-01 and 2026-06-01 were important"
        )
        assert result["interval_start"] is not None
        assert result["interval_end"] is not None


# ---------------------------------------------------------------------------
# build_temporal_event
# ---------------------------------------------------------------------------

class TestBuildTemporalEvent:
    def test_creates_event(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")
        fact = AtomicFact(
            fact_id="f1", content="Alice started a job yesterday",
            profile_id="default",
        )
        event = parser.build_temporal_event(fact, "2026-03-11", "ent_alice")
        assert event is not None
        assert event.entity_id == "ent_alice"
        assert event.fact_id == "f1"
        assert event.observation_date is not None

    def test_returns_none_no_temporal(self) -> None:
        parser = TemporalParser()
        fact = AtomicFact(
            fact_id="f1", content="Alice likes cats",
            profile_id="default",
        )
        event = parser.build_temporal_event(fact, None, "ent_alice")
        assert event is None

    def test_uses_fact_level_dates(self) -> None:
        parser = TemporalParser()
        fact = AtomicFact(
            fact_id="f1", content="Plain text no date mentions",
            profile_id="default",
            observation_date="2026-03-10",
            referenced_date="2026-02-15",
        )
        event = parser.build_temporal_event(fact, None, "ent_alice")
        assert event is not None
        assert event.observation_date == "2026-03-10"
        assert event.referenced_date == "2026-02-15"

    def test_event_inherits_fact_scope_authorization(self) -> None:
        parser = TemporalParser()
        fact = AtomicFact(
            fact_id="f-global", content="Launch on 2026-03-11",
            profile_id="publisher", scope="shared",
            shared_with=["requester"],
        )

        event = parser.build_temporal_event(fact, None, "ent_launch")

        assert event is not None
        assert event.scope == "shared"
        assert event.shared_with == ["requester"]


# ---------------------------------------------------------------------------
# build_entity_timeline
# ---------------------------------------------------------------------------

class TestBuildEntityTimeline:
    def test_timeline_sorted(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")
        facts = [
            AtomicFact(
                fact_id="f1", content="Alice started in 2025-06-01",
                profile_id="default",
            ),
            AtomicFact(
                fact_id="f2", content="Alice promoted in 2026-01-15",
                profile_id="default",
            ),
        ]
        events = parser.build_entity_timeline("ent_alice", facts, "2026-03-11")
        assert len(events) >= 1
        # Events should be sorted chronologically
        dates = [
            e.referenced_date or e.observation_date or "" for e in events
        ]
        assert dates == sorted(dates)

    def test_empty_facts(self) -> None:
        parser = TemporalParser()
        events = parser.build_entity_timeline("ent_alice", [])
        assert events == []


# ---------------------------------------------------------------------------
# Resolve relative
# ---------------------------------------------------------------------------

class TestResolveRelative:
    def test_last_year(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")
        result = parser.extract_dates_from_text("last year was different")
        assert result["referenced_date"] is not None
        assert "2025" in result["referenced_date"]

    def test_this_week(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")
        result = parser.extract_dates_from_text("I'll finish this week")
        assert result["referenced_date"] is not None

    def test_season(self) -> None:
        parser = TemporalParser(reference_date="2026-03-11")
        result = parser.extract_dates_from_text("We met last summer")
        assert result["referenced_date"] is not None

    def test_reference_date_invalid(self) -> None:
        # Should fall back to utcnow
        parser = TemporalParser(reference_date="not-a-date")
        result = parser.parse_session_date("2026-03-11")
        assert result is not None


# ---------------------------------------------------------------------------
# Unit delta
# ---------------------------------------------------------------------------

class TestUnitDelta:
    def test_days(self) -> None:
        from datetime import timedelta
        delta = TemporalParser._unit_delta(5, "day")
        assert delta == timedelta(days=5)

    def test_weeks(self) -> None:
        from datetime import timedelta
        delta = TemporalParser._unit_delta(2, "week")
        assert delta == timedelta(weeks=2)

    def test_months(self) -> None:
        from dateutil.relativedelta import relativedelta
        delta = TemporalParser._unit_delta(3, "month")
        assert delta == relativedelta(months=3)

    def test_years(self) -> None:
        from dateutil.relativedelta import relativedelta
        delta = TemporalParser._unit_delta(1, "year")
        assert delta == relativedelta(years=1)

    def test_unknown_defaults_to_days(self) -> None:
        from datetime import timedelta
        delta = TemporalParser._unit_delta(7, "unknown")
        assert delta == timedelta(days=7)
