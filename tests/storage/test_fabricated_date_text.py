# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The clause that carries a fabricated date is removed; the sentence still reads.

Every "before" here is a real fact shape from a store that ran a build before
4.1.6, when the extraction prompt's worked example ("session date 2024-01-15")
leaked into facts as their date.
"""

from __future__ import annotations

import pytest

from superlocalmemory.storage.fabricated_dates import (
    contains_fabricated_date,
    is_fabricated_date_value,
    remove_date_clauses,
    source_names_the_day,
)


@pytest.mark.parametrize(("before", "after"), [
    ("The file CHANGELOG.md was modified on 2024-01-15",
     "The file CHANGELOG.md was modified"),
    ("The file ISSUE-010-LINKEDIN-POST.md was modified on 2024-01-15.",
     "The file ISSUE-010-LINKEDIN-POST.md was modified."),
    ("settings.json was last modified on 2024-01-15", "settings.json was last modified"),
    ("The file test_remote_reranker.py was modified on 2024-01-15, but the modification "
     "status of test_reranker_load_errors.py is not specified.",
     "The file test_remote_reranker.py was modified, but the modification "
     "status of test_reranker_load_errors.py is not specified."),
    ("Wide tables and long paths no longer overflow their pane as of 2024-01-15",
     "Wide tables and long paths no longer overflow their pane"),
    ("2024-01-15 File engine.py was modified", "File engine.py was modified"),
    ("2024-01-15 file commands.py was modified", "File commands.py was modified"),
    ("2024-01-15 coding started", "Coding started"),
    ("2024-01-15 the modification was made by an unknown user",
     "The modification was made by an unknown user"),
    ("2024-01-15 varunpratapbhardwaj reported that pytest -q cwd must agree",
     "varunpratapbhardwaj reported that pytest -q cwd must agree"),
    ("2024-01-15 /Users/x/LLD-L01.md has 859 lines", "/Users/x/LLD-L01.md has 859 lines"),
    ("2024-01-15 14:30:00, Qualixar AI YouTube upload failed due to invalid OAuth token",
     "Qualixar AI YouTube upload failed due to invalid OAuth token"),
    ("2024-01-15 is the date when the cost was approved", "The cost was approved"),
    ("2024-01-15 was the day when the HTTP server was restarted",
     "The HTTP server was restarted"),
    ("2024-01-15 was the day on which the speaker took the subscription of Hermes",
     "The speaker took the subscription of Hermes"),
    ("2024-01-15 is a date when Mac was taking my camera off",
     "Mac was taking my camera off"),
    ("The session started at 2024-01-15T00:00:00Z and lasted for 53.73 seconds",
     "The session started and lasted for 53.73 seconds"),
    ("Agent Amplifier posts should be made in the evening around 2024-01-15 20:00:00 "
     "to increase agent power", "Agent Amplifier posts should be made in the evening "
     "to increase agent power"),
    ("The MCP servers were updated. As of 2026-01-15, in production-v5.2 one object "
     "remains.", "The MCP servers were updated. In production-v5.2 one object remains."),
    ("Mel was seen on January 15, 2024", "Mel was seen"),
    ("The conversation took place on January 15, 2024, at 10:43 AM.",
     "The conversation took place."),
    ("The release (2024-01-15) fixed the bug", "The release fixed the bug"),
    # The prompt's own "yesterday" answer leaked too.
    ("The file slm-370-release-audit.md was last modified on 2024-01-14",
     "The file slm-370-release-audit.md was last modified"),
    ("2024-01-14 File engine.py was modified", "File engine.py was modified"),
    ("The test passed on January 14, 2024.", "The test passed."),
])
def test_the_date_clause_goes_and_the_sentence_reads(before: str, after: str) -> None:
    assert remove_date_clauses(before) == after


@pytest.mark.parametrize("text", [
    "2024-01-15",
    "2024-01-15 14:30:00",
    "2024-01-15 was a Tuesday",
    "2024-01-15 is today",
    "2024-01-15 is the current date",
    "2024-01-15 is the date of the file change",
    "2024-01-15 has IP leaks in the dashboard videos",
    "2024-01-15 23:30:00 is the generation time for classroom LUTQTO0N0Q",
    "The current date is January 15, 2024",
    "The plan runs from 2024-01-15 to 2024-01-20",
    "Nothing about any date here",
])
def test_a_statement_that_is_about_the_date_is_left_as_written(text: str) -> None:
    """When the date is the subject or the point, no clause can go cleanly."""
    assert remove_date_clauses(text) is None


def test_detection_covers_both_dates_and_the_spelled_form() -> None:
    assert contains_fabricated_date("modified on 2024-01-15")
    assert contains_fabricated_date("As of 2026-01-15, x")
    assert contains_fabricated_date("seen on January 15, 2024")
    assert contains_fabricated_date("modified on 2024-01-14")
    assert contains_fabricated_date("on January 14, 2024")
    assert not contains_fabricated_date("on 2026-01-14 and 2025-01-14")
    assert not contains_fabricated_date("released 2024-01-16 and 2025-01-15")
    assert not contains_fabricated_date("version 12024-01-155")


def test_a_date_column_value_matches_only_the_fabricated_day() -> None:
    assert is_fabricated_date_value("2024-01-15")
    assert is_fabricated_date_value("2026-01-15T00:00:00Z")
    assert is_fabricated_date_value("2024-01-14")
    assert not is_fabricated_date_value("2026-01-14")
    assert not is_fabricated_date_value("2024-01-16")
    assert not is_fabricated_date_value(None)
    assert not is_fabricated_date_value("")


@pytest.mark.parametrize("source", [
    "paid on 2024-01-15", "As of 2026-01-15 we ship", "seen on January 15, 2024",
    "seen on Jan 15", "meeting jan. 15th", "15 Jan 2024", "the 15th of January",
    "15 January", "due 01/15/2024", "due 1/15/26", "le 15/01/2024", "am 15.01.2024",
    "on 2024-01-14", "Jan 14th", "14 January 2024", "01/14/2024",
])
def test_a_source_names_the_day_in_any_written_form(source: str) -> None:
    assert source_names_the_day(source)


@pytest.mark.parametrize("source", [
    "", None, "2024-01-16", "January 25, 2024", "Jan 1, 2024", "150 January", "v2024.01.15x",
    "June 15, 2024",
])
def test_other_days_are_not_the_day(source) -> None:
    assert not source_names_the_day(source)
