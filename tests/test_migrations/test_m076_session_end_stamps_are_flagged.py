# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M076 / R-31: a reworded session end that is only a stamp gets the flag; one
that says anything else stays an answer. Shapes measured on one real store
(353 stamps, 52 rewordings with content). Runs on a temporary store."""

from __future__ import annotations

import pytest

from superlocalmemory.encoding.non_answer import is_event_record, is_stub
from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS
from superlocalmemory.storage.migrations import M060_memory_description
from superlocalmemory.storage.migrations import M062_hook_bookkeeping_is_flagged as M062
from superlocalmemory.storage.migrations import M076_session_end_stamps_are_flagged as M076
from tests.test_migrations.test_m068_newer_bookkeeping_shapes_are_flagged import (
    _add,
    _flags,
    conn,  # noqa: F401 - the fixture
)

STAMPS = (
    "The conversation ended at 2026-08-01 22:48",
    "The conversation ended at 22:48",
    "The session ended at 9:39 PM",
    "The session ended at 21:33 on August 7, 2026",
    "The session ended on October 8, 2026, at 18:47",
    "The session ended on a Thursday",
    "The session ended at 05:11 UTC",
    "The session ended on the 30th of July 2026",
    "The conversation ended manually at 01:31",
    "The conversation ended at an unspecified time",
    "2026-07-19 is the end date of Alex's session",
    "2026-03-30 is the date when [Agentic_official] ended their session",
    "2026-04-11 22:20 is the end date of an official session",
)
STILL_ANSWERS = (
    "The session ended after the execution of the api-test-suite.json file",
    "The session ended with files INDEX.md and executive-producer.md",
    "The session ended in AgentAssert",
    "The session ended with the e2e protocol test",
    "The session ended last week",
    "The session ended at the office",
    "The session ended after 22 minutes",
    "The session ended at 13:28 yesterday",
    "The session ended in config at 19:42",
)


@pytest.mark.parametrize("text", STAMPS)
def test_a_session_end_stamp_is_a_stub(text) -> None:
    assert is_stub(text) and is_event_record(text)


@pytest.mark.parametrize("text", STILL_ANSWERS)
def test_a_session_end_that_says_more_stays_an_answer(text) -> None:
    assert not is_stub(text)
    assert is_event_record(text)  # the judge still treats it as a moment (separate events)


def test_registered_after_m062_with_a_floor_and_no_replay():
    assert _MODULES[M076.NAME] is M076
    migration = next(m for m in DEFERRED_MIGRATIONS if m.name == M076.NAME)
    assert {M060_memory_description.NAME, M062.NAME} <= set(migration.dependencies)
    assert M076.DB_TARGET == "memory" and M076.DOWNGRADE_FLOOR == 51
    assert M076.REPAIR_NOT_APPLICABLE.strip()


def test_stored_stamps_are_flagged_and_answers_are_not(conn):  # noqa: F811
    for i, text in enumerate(STAMPS):
        _add(conn, f"s{i}", text)
    for i, text in enumerate(STILL_ANSWERS):
        _add(conn, f"a{i}", text)
    _add(conn, "decided", STAMPS[0], non_answer=0, user=0)
    M076.apply(conn)
    flags = _flags(conn)
    assert all(flags[f"s{i}"] == (1, None) for i in range(len(STAMPS))), flags
    assert all(flags[f"a{i}"] == (0, None) for i in range(len(STILL_ANSWERS))), flags
    assert flags["decided"] == (0, 0)  # the user's decision stands
    before = conn.total_changes
    M076.apply(conn)
    assert conn.total_changes == before
