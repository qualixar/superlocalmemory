# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for retrieval.time_window — window parsing + membership (T-window)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from superlocalmemory.retrieval.time_window import (
    in_window,
    infer_window_from_query,
    parse_timestamp,
    parse_window,
)

NOW = datetime(2026, 7, 22, 12, 0, 0, tzinfo=timezone.utc)


# ---- parse_timestamp: tolerant across formats ----

@pytest.mark.parametrize("raw", [
    "2026-07-22 12:00:00",       # SQLite datetime('now') form
    "2026-07-22T12:00:00",       # ISO with T
    "2026-07-22T12:00:00Z",      # ISO with Z
    "2026-07-22T12:00:00+00:00",  # ISO with offset
    "2026-07-22",                # date only
])
def test_parse_timestamp_formats(raw: str) -> None:
    dt = parse_timestamp(raw)
    assert dt is not None
    assert dt.tzinfo is not None            # always tz-aware
    assert dt.year == 2026 and dt.month == 7 and dt.day == 22


def test_parse_timestamp_bad_input() -> None:
    assert parse_timestamp(None) is None
    assert parse_timestamp("") is None
    assert parse_timestamp("not-a-date") is None


def test_space_vs_T_compare_equal_not_lexicographic() -> None:
    # The core trap: "…22 12:00" vs "…22T12:00" differ as strings but are the
    # same instant. parse_timestamp must make them equal.
    a = parse_timestamp("2026-07-22 12:00:00")
    b = parse_timestamp("2026-07-22T12:00:00Z")
    assert a == b


# ---- parse_window: relative strings ----

def test_parse_window_relative_units() -> None:
    assert parse_window("24h", now=NOW) == (NOW - timedelta(hours=24), NOW)
    assert parse_window("7d", now=NOW) == (NOW - timedelta(days=7), NOW)
    assert parse_window("2w", now=NOW) == (NOW - timedelta(weeks=2), NOW)
    assert parse_window("1m", now=NOW) == (NOW - timedelta(days=30), NOW)
    assert parse_window("1y", now=NOW) == (NOW - timedelta(days=365), NOW)


def test_parse_window_none_and_bad() -> None:
    assert parse_window(None, now=NOW) is None
    assert parse_window("yesterday", now=NOW) is None
    assert parse_window("abc", now=NOW) is None
    assert parse_window(("only-one",), now=NOW) is None


# ---- parse_window: explicit ranges ----

def test_parse_window_explicit_range() -> None:
    bounds = parse_window(("2026-07-01", "2026-07-15"), now=NOW)
    assert bounds is not None
    start, end = bounds
    assert start < end
    assert start.day == 1 and end.day == 15


def test_parse_window_explicit_range_swapped_is_normalized() -> None:
    bounds = parse_window(("2026-07-15", "2026-07-01"), now=NOW)
    assert bounds is not None
    start, end = bounds
    assert start < end   # swapped inputs normalized


def test_parse_window_string_range_forms() -> None:
    # Wire-safe single-string range forms (URL param / JSON / CLI arg).
    for spec in ("2026-07-01..2026-07-15", "2026-07-01,2026-07-15"):
        bounds = parse_window(spec, now=NOW)
        assert bounds is not None, spec
        start, end = bounds
        assert start.day == 1 and end.day == 15


def test_parse_window_string_range_bad_side() -> None:
    assert parse_window("2026-07-01..nonsense", now=NOW) is None


# ---- in_window ----

def test_in_window_none_bounds_is_true() -> None:
    assert in_window("2026-01-01", None) is True


def test_in_window_membership() -> None:
    bounds = parse_window("7d", now=NOW)   # [2026-07-15 12:00, 2026-07-22 12:00]
    assert in_window("2026-07-20 09:00:00", bounds) is True
    assert in_window("2026-07-10 09:00:00", bounds) is False   # too old
    assert in_window("2026-07-22T12:00:00Z", bounds) is True   # inclusive end


def test_in_window_unparseable_excluded() -> None:
    bounds = parse_window("7d", now=NOW)
    assert in_window(None, bounds) is False
    assert in_window("garbage", bounds) is False


# ---- infer_window_from_query (T3 time-aware query) ----

@pytest.mark.parametrize("query,expected", [
    ("what did I do today", "1d"),
    ("notes from yesterday", "2d"),
    ("decisions last week", "7d"),
    ("what happened this month", "30d"),
    ("summary of the past year", "1y"),
    ("what have I worked on recently", "30d"),
    ("changes in the last 3 weeks", "3w"),
    ("meetings in the last 5 days", "5d"),
    ("past 2 months of work", "2m"),
])
def test_infer_window_from_query_hits(query, expected) -> None:
    assert infer_window_from_query(query) == expected


@pytest.mark.parametrize("query", [
    "notes about cars",
    "what is the architecture of the retrieval engine",
    "",
    None,
    "weekly report format",   # "week" without a scope word must NOT fire
])
def test_infer_window_from_query_misses(query) -> None:
    assert infer_window_from_query(query) is None


def test_infer_n_unit_takes_precedence_over_generic() -> None:
    # "last 3 weeks" must map to 3w, not the generic "last week" -> 7d.
    assert infer_window_from_query("the last 3 weeks were busy") == "3w"
