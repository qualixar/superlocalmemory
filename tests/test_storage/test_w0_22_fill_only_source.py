# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W0-22: ``valid_until_source`` is stamped only when this call wrote the end (I-12)."""

from __future__ import annotations

import pytest

from tests.test_core.test_coverage_backfill import _db
from tests.test_truth.test_slot_pass import _fact


@pytest.fixture
def db(tmp_path):
    return _db(tmp_path / "memory.db")


def _end(db, fid):
    row = db.execute("SELECT valid_until, valid_until_source FROM fact_temporal_validity "
                     "WHERE fact_id = ?", (fid,))[0]
    return row[0], row[1]


def _with_end(db, source="reviewer"):
    fid = _fact(db, "The standup is at nine", "2026-03-01T00:00:00+00:00")
    db.execute("UPDATE fact_temporal_validity SET valid_until = '2026-05-01T00:00:00+00:00', "
               "valid_until_source = ? WHERE fact_id = ?", (source, fid))
    return fid


def test_fill_only_leaves_an_existing_end_and_its_source_alone(db) -> None:
    fid = _with_end(db)
    db.store_temporal_validity(fid, "default", valid_from="2026-03-01", valid_until="2026-09-09",
                              fill_only=True)
    assert _end(db, fid) == ("2026-05-01T00:00:00+00:00", "reviewer")


def test_without_fill_only_the_given_end_is_written_and_stated(db) -> None:
    fid = _with_end(db)
    db.store_temporal_validity(fid, "default", valid_until="2026-09-09T00:00:00+00:00")
    assert _end(db, fid) == ("2026-09-09T00:00:00+00:00", "stated")


def test_a_new_record_with_an_end_is_stated_even_with_fill_only(db) -> None:
    fid = _fact(db, "A fixed term contract", "2026-03-01T00:00:00+00:00")
    db.execute("DELETE FROM fact_temporal_validity WHERE fact_id = ?", (fid,))
    db.store_temporal_validity(fid, "default", valid_from="2026-03-01",
                              valid_until="2026-12-31T00:00:00+00:00", fill_only=True)
    assert _end(db, fid) == ("2026-12-31T00:00:00+00:00", "stated")


def test_fill_only_with_an_end_on_a_record_that_has_none_writes_and_stamps_nothing(db) -> None:
    fid = _fact(db, "An open ended fact", "2026-03-01T00:00:00+00:00")
    assert _end(db, fid)[0] is None
    db.store_temporal_validity(fid, "default", valid_from="2026-03-01",
                              valid_until="2026-09-09T00:00:00+00:00", fill_only=True)
    assert _end(db, fid) == (None, None)  # the end was not written, so no source either
