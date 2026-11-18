# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W0-17b (M-12): randomised deliveries before the first pull delivery are pilot data.

Until every recall path carried receipts the re-ask outcome could not be seen, so the
deliveries recorded then are not evidence. Both the effect estimators and the harm gate
read only deliveries delivered at or after the profile's first pull delivery.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from superlocalmemory.core import harm_gate as hg
from superlocalmemory.outcomes import effects as fx
from superlocalmemory.outcomes import memory_report as mr
from superlocalmemory.storage.migrations import M075_injection_receipts as M075
from tests.test_outcomes.test_effects import LATER, T0, _delivery


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    M075.apply(c)
    yield c
    c.close()


def _ids(conn, outcome=fx.USED):
    return sorted(u.delivery_id for u in fx.load_units(conn, "default", outcome, as_of=LATER))


def test_deliveries_before_the_first_pull_are_not_evidence(conn) -> None:
    _delivery(conn, "pilot-1", at=T0 - timedelta(hours=5), session="a")
    _delivery(conn, "pilot-2", at=T0 - timedelta(hours=1), session="b", path="session_context")
    _delivery(conn, "first-pull", path="mcp_recall", u=None, at=T0, session="c")
    _delivery(conn, "same-instant", at=T0, session="d")
    _delivery(conn, "after", at=T0 + timedelta(hours=2), session="e")
    assert _ids(conn) == ["after", "same-instant"]
    assert _ids(conn, fx.REASK) == ["after", "same-instant"]


def test_with_no_pull_delivery_nothing_is_evidence(conn) -> None:
    _delivery(conn, "a")
    _delivery(conn, "b", path="session_context", session="x")
    assert _ids(conn) == []


def test_every_pull_path_starts_the_evidence_but_a_push_does_not(conn) -> None:
    from superlocalmemory.outcomes import randomiser as rz

    for path in sorted(rz.PULL_PATHS):
        c = sqlite3.connect(":memory:")
        M075.apply(c)
        _delivery(c, "early", at=T0 - timedelta(hours=2), session="a")
        _delivery(c, "pull", path=path, u=None, at=T0 - timedelta(hours=1), session="b")
        _delivery(c, "late", at=T0, session="c")
        assert _ids(c) == ["late"], path
        c.close()


def test_another_profiles_pull_does_not_start_this_profiles_evidence(conn) -> None:
    _delivery(conn, "mine", at=T0, session="a")
    _delivery(conn, "their-pull", path="mcp_recall", u=None, at=T0 - timedelta(hours=1),
              session="b", profile="other")
    assert _ids(conn) == []


def test_the_count_of_pilot_deliveries_is_exact(conn) -> None:
    for i in range(3):
        _delivery(conn, f"p{i}", at=T0 - timedelta(hours=3 - i), session=f"s{i}")
    _delivery(conn, "pull", path="cli_recall", u=None, at=T0, session="z")
    _delivery(conn, "e", at=T0 + timedelta(hours=1), session="e")
    _delivery(conn, "elsewhere", at=T0 - timedelta(hours=9), session="o", profile="other")
    assert fx.pilot_deliveries(conn, "default") == 3
    assert fx.pilot_deliveries(conn, "other") == 1  # it has no pull yet
    assert fx.pilot_deliveries(sqlite3.connect(":memory:"), "default") == 0


def test_the_memory_health_report_says_how_many_were_excluded(conn, tmp_path) -> None:
    for i in range(4):
        _delivery(conn, f"p{i}", at=T0 - timedelta(hours=3), session=f"s{i}")
    report = mr.build(conn, "default", as_of=LATER, slm_home=tmp_path)
    assert report["pilot_deliveries_excluded"] == 4
    assert "pilot deliveries excluded: 4" in "\n".join(mr.describe(report))


# -- the harm gate reads the same deliveries ----------------------------------------------------

def _receipt_delivery(c, did, at, fact="f1"):
    c.execute(
        "INSERT INTO injection_deliveries (delivery_id, profile_id, session_id, client, path,"
        " design, holdout_state, epsilon, eligible_count, withhold_draw, pick_draw, key_id,"
        " candidates, delivered_at) VALUES (?, 'default', ?, 'hook', 'session_context', 'one-out-v2',"
        " 'on', 0.1, 4, 0.5, 0.5, 'k1', 4, ?)", (did, did, at.isoformat()))
    c.execute(
        "INSERT INTO injection_receipts (delivery_id, fact_id, profile_id, role, position,"
        " score, eligible, exclusion, propensity, included, delivered) VALUES"
        " (?, ?, 'default', 'recall', 1, 0.5, 1, NULL, 0.5, 1, 1)", (did, fact))


def test_the_harm_gate_skips_pilot_deliveries_too(conn) -> None:
    now = datetime(2026, 11, 18, 12, tzinfo=UTC)
    _receipt_delivery(conn, "pilot", now - timedelta(days=5))
    _receipt_delivery(conn, "pilot2", now - timedelta(days=4))
    assert hg._units(conn, "default", fx.USED, now) == {}  # no pull yet: nothing is evidence
    _delivery(conn, "pull", path="mcp_recall", u=None, at=now - timedelta(days=3, hours=12),
              session="p")
    _receipt_delivery(conn, "real", now - timedelta(days=3))
    units = hg._units(conn, "default", fx.USED, now)
    assert [len(v) for v in units.values()] == [1]  # only the delivery after the first pull
