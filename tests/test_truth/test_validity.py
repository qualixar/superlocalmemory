# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
from __future__ import annotations

from superlocalmemory.truth.validity import closing_time, normalise


def test_normalise() -> None:
    assert normalise("2026-02-01") == "2026-02-01T00:00:00+00:00"
    assert normalise("2026-02-01T10:00:00Z") == "2026-02-01T10:00:00+00:00"
    assert normalise("2026-02-01T10:00:00") == "2026-02-01T10:00:00+00:00"
    assert normalise("") is None and normalise(None) is None and normalise("nope") is None


def test_first_readable_evidence_wins_and_inversion_refuses() -> None:
    assert closing_time(event_valid_until="", successor_validity_from="junk",
                        successor_created_at="2026-05-01") == "2026-05-01T00:00:00+00:00"
    assert closing_time(applied_at="2026-01-01", predecessor_valid_from="2026-01-01") is None
    assert closing_time() is None


def test_a_same_day_date_falls_through_to_the_precise_timestamp() -> None:
    assert closing_time(successor_validity_from="2026-10-06",
                        successor_created_at="2026-10-06T11:26:14+00:00",
                        predecessor_valid_from="2026-10-06") == "2026-10-06T11:26:14+00:00"
    assert closing_time(event_valid_until="2025-01-01", applied_at="2026-10-07",
                        predecessor_valid_from="2026-10-06") == "2026-10-07T00:00:00+00:00"


def test_a_bare_date_is_an_interval_not_midnight_utc() -> None:
    from superlocalmemory.truth.validity import AFTER, BEFORE, INCOMPARABLE, SAME, earlier, order

    assert order("2026-11-15", "2026-11-15T08:00:00+00:00") == INCOMPARABLE  # Fable B3
    assert not earlier("2026-11-15", "2026-11-15T08:00:00+00:00")
    assert order("2026-11-13", "2026-11-15T08:00:00+00:00") == BEFORE
    assert order("2026-11-14", "2026-11-15") == INCOMPARABLE  # zones overlap across a day
    assert order("2026-11-12", "2026-11-15") == BEFORE
    assert order("2026-11-15T10:00:00Z", "2026-11-15T09:00:00+00:00") == AFTER
    assert order("2026-11-15T10:00:00Z", "2026-11-15T10:00:00+00:00") == SAME
    assert order("2026-11-15", "2026-11-15") == INCOMPARABLE
    assert order("garbage", "2026-11-15") is None
