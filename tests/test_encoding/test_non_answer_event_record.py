# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""is_event_record: hook stubs plus the extraction model's rewordings of a session end."""

from __future__ import annotations

import pytest

from superlocalmemory.encoding.non_answer import is_event_record, is_stub


@pytest.mark.parametrize("text", [
    "The conversation ended at 2026-09-04 13:32",
    "2026-08-16 is the end date of Alex's session",
    "The session ended at 18:47 on October 8, 2026",
    "2026-10-07 is the date when Alex's session ended",
    "The session ended on October 10, 2026, at 10:38",
    "session ended 2026-11-10 16:20",
    "File changed: assemble.py",
])
def test_records_of_moments(text) -> None:
    assert is_event_record(text)


@pytest.mark.parametrize("text", [
    "", "The release cadence of SLM is monthly",
    "The session ended badly because the API key was revoked and nobody noticed for a week",
    "Sessions end when the user closes the tab",
    "Call stage_wait (timeout_s 180, detail true).",
])
def test_not_records_of_moments(text) -> None:
    assert not is_event_record(text)


def test_is_stub_is_unchanged_for_the_rewordings() -> None:
    assert not is_stub("The conversation ended at 2026-09-04 13:32")
