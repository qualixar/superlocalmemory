# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for retrieval.temporal_frame — relative age + frame header (T-inject)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from superlocalmemory.retrieval.temporal_frame import relative_age, temporal_frame

NOW = datetime(2026, 7, 22, 12, 0, 0, tzinfo=timezone.utc)


def _ago(**kw) -> str:
    return (NOW - timedelta(**kw)).isoformat()


# ---- relative_age ----

def test_relative_age_just_now() -> None:
    assert relative_age(_ago(seconds=5), NOW) == "just now"


@pytest.mark.parametrize("kw,expected", [
    ({"minutes": 1}, "1 minute ago"),
    ({"minutes": 5}, "5 minutes ago"),
    ({"hours": 1}, "1 hour ago"),
    ({"hours": 3}, "3 hours ago"),
    ({"days": 1}, "1 day ago"),
    ({"days": 3}, "3 days ago"),
    ({"days": 21}, "3 weeks ago"),
    ({"days": 60}, "2 months ago"),
    ({"days": 365}, "1 year ago"),
    ({"days": 730}, "2 years ago"),
])
def test_relative_age_units(kw, expected) -> None:
    assert relative_age(_ago(**kw), NOW) == expected


def test_relative_age_future() -> None:
    future = (NOW + timedelta(days=3)).isoformat()
    assert relative_age(future, NOW) == "in 3 days"


def test_relative_age_bad_input() -> None:
    assert relative_age(None, NOW) == ""
    assert relative_age("", NOW) == ""
    assert relative_age("not-a-date", NOW) == ""


def test_relative_age_sqlite_space_format() -> None:
    # SQLite datetime('now') form must parse the same as ISO.
    assert relative_age("2026-07-21 12:00:00", NOW) == "1 day ago"


# ---- temporal_frame ----

def test_temporal_frame_spans_oldest_to_newest() -> None:
    frame = temporal_frame([_ago(days=730), _ago(days=1), _ago(seconds=5)], NOW)
    assert frame.startswith("Now: 2026-07-22T12:00:00+00:00.")
    assert "2 years ago" in frame     # oldest
    assert "just now" in frame        # newest
    assert "stale" in frame           # guidance clause


def test_temporal_frame_single_timestamp() -> None:
    frame = temporal_frame([_ago(days=3)], NOW)
    assert "3 days ago" in frame
    assert "→" not in frame           # single point, no span arrow


def test_temporal_frame_no_dated() -> None:
    frame = temporal_frame([None, "", "garbage"], NOW)
    assert frame.startswith("Now: 2026-07-22T12:00:00+00:00.")
    assert "undated" in frame


def test_temporal_frame_empty() -> None:
    frame = temporal_frame([], NOW)
    assert "undated" in frame
