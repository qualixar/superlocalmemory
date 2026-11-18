# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W0-23 (M-10): a date-only end is the interval [day 00:00, next day 00:00), stored as its
lower bound; two ends are ordered only when one interval is wholly before the other."""

from __future__ import annotations

import pytest

from superlocalmemory.truth.validity import (
    closing_time_interval,
    closing_time_sourced,
    retire_until_sourced,
    successor_takeover_interval,
)

DAY = "2026-10-06"
LOWER = "2026-10-06T00:00:00+00:00"
UPPER = "2026-10-07T00:00:00+00:00"


def test_a_date_only_end_reports_the_end_of_its_day_as_its_upper_bound() -> None:
    assert closing_time_interval(successor_validity_from=DAY) == (LOWER, "successor_validity", UPPER)
    assert closing_time_interval(event_valid_until=DAY) == (LOWER, "reviewer", UPPER)


def test_a_timestamp_end_is_a_point() -> None:
    stamp = "2026-10-06T11:26:14+00:00"
    assert closing_time_interval(successor_created_at=stamp) == (stamp, "successor_saved", stamp)
    assert closing_time_interval() == (None, None, None)


def test_the_stored_end_and_source_are_unchanged() -> None:
    assert closing_time_sourced(successor_validity_from=DAY) == (LOWER, "successor_validity")


def test_a_candidate_whose_lower_bound_is_not_after_the_start_is_skipped_as_before() -> None:
    # Stored as its lower bound, a date-only end on the start's own day would invert the span.
    assert closing_time_interval(successor_validity_from=DAY, applied_at="2026-10-08",
                                 predecessor_valid_from="2026-10-06T09:00:00+00:00") == (
        "2026-10-08T00:00:00+00:00", "applied", "2026-10-09T00:00:00+00:00")


def _retire(derived_raw: str, existing: str, *, source="successor_validity"):
    derived, src, upper = closing_time_interval(successor_validity_from=derived_raw)
    return retire_until_sourced(reviewer=None, existing=existing, derived=derived,
                                derived_upper=upper, predecessor_valid_from="2026-01-01",
                                existing_source="recorded", derived_source=src)


def test_an_existing_end_inside_the_day_cannot_be_ordered_so_it_is_kept() -> None:
    inside = "2026-10-06T10:00:00+00:00"
    assert _retire(DAY, inside) == (inside, "recorded")  # the old point rule took LOWER


def test_an_existing_end_wholly_after_the_day_is_replaced_by_the_earlier_date() -> None:
    after = "2026-10-07T00:00:00+00:00"
    assert _retire(DAY, after) == (LOWER, "successor_validity")
    assert _retire(DAY, "2026-10-09T08:00:00+00:00") == (LOWER, "successor_validity")


def test_an_existing_end_wholly_before_the_day_is_kept() -> None:
    before = "2026-10-05T23:00:00+00:00"
    assert _retire(DAY, before) == (before, "recorded")


def test_without_an_upper_bound_a_derived_timestamp_orders_as_a_point() -> None:
    assert retire_until_sourced(reviewer=None, existing="2026-10-06T10:00:00+00:00",
                                derived="2026-10-06T09:00:00+00:00", derived_source="applied",
                                predecessor_valid_from=None) == (
        "2026-10-06T09:00:00+00:00", "applied")


def test_the_takeover_carries_the_interval_too() -> None:
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE fact_temporal_validity (fact_id TEXT, valid_from TEXT)")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT, valid_from TEXT, created_at TEXT)")
    conn.execute("INSERT INTO fact_temporal_validity VALUES ('s', ?)", (DAY,))
    got = successor_takeover_interval(conn, "s", predecessor_valid_from="2026-01-01",
                                      occurred_at="2026-11-01T00:00:00+00:00")
    assert got == (LOWER, "successor_validity", UPPER)


@pytest.mark.parametrize("text", ["M074", "two new nullable columns"])
def test_the_stale_comments_are_gone(text) -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src/superlocalmemory"
    for rel in ("truth/validity.py", "storage/migrations/M073_valid_until_source.py"):
        assert text not in (root / rel).read_text(encoding="utf-8")


def test_a_correction_apply_orders_a_date_only_takeover_by_its_upper_bound() -> None:
    import sqlite3
    from types import SimpleNamespace

    from superlocalmemory.storage.correction_temporal import _retirement_end

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE fact_temporal_validity (fact_id TEXT, valid_from TEXT)")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT, valid_from TEXT, created_at TEXT)")
    conn.execute("INSERT INTO fact_temporal_validity VALUES ('succ', ?)", (DAY,))
    case = SimpleNamespace(successor_fact_id="succ", predecessor_fact_id="old",
                           profile_id="default", event_valid_until=None)
    inside = "2026-10-06T10:00:00+00:00"
    got = _retirement_end(conn, case, ("2026-01-01T00:00:00+00:00", inside),
                          occurred_at="2026-11-01T00:00:00+00:00", sourced=False)
    assert got == (inside, "recorded")  # the takeover day contains the recorded end
    later = "2026-10-20T00:00:00+00:00"
    got = _retirement_end(conn, case, ("2026-01-01T00:00:00+00:00", later),
                          occurred_at="2026-11-01T00:00:00+00:00", sourced=False)
    assert got == (LOWER, "successor_validity")
