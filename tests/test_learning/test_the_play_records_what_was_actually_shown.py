# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A play's evidence has to describe the answer the user saw.

A play is settled later by asking whether anything downstream referenced one of
the memories this query surfaced. That question is answered against a stored
list of fact ids — so if the list describes a different answer than the one
returned, an outcome citing a memory the user actually saw settles nothing, and
the arm never learns from it.

The list was written by the ranking pass. The continuity bias runs after that
pass and can lift a memory into the answer, so the two disagreed exactly when
continuity did something — which is the case the bias exists for.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import pytest

from superlocalmemory.core import working_memory as wm
from superlocalmemory.core.recall_pipeline import _resettle_shown_after_bias


@pytest.fixture(autouse=True)
def _clean_registry():
    wm._REGISTRY.clear()
    yield
    wm._REGISTRY.clear()


@dataclass
class _Fact:
    fact_id: str


@dataclass
class _Result:
    fact: _Fact
    score: float = 0.5
    ranking_score: float | None = None


@pytest.fixture
def learning_db(tmp_path):
    """A learning store with one recorded play, as the ranking pass leaves it.

    The bandit tables come from M005 and the column that holds the shown set
    from M046's predecessor, M044. Without them ``choose`` returns no play id
    and there is nothing to correct — which would make every test below pass
    for the wrong reason.
    """
    from superlocalmemory.learning.bandit import ContextualBandit
    from superlocalmemory.storage.migrations import (
        M005_bandit_tables as _M005,
    )
    from superlocalmemory.storage.migrations import (
        M044_play_carries_its_own_evidence as _M044,
    )

    path = tmp_path / "learning.db"
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_M005.DDL)
        _M044.apply(conn)
        conn.commit()
    finally:
        conn.close()

    bandit = ContextualBandit(path, "default")
    choice = bandit.choose({"query_type": "factual", "entity_count": 0}, "q1")
    assert choice.play_id, "no play recorded, so there is nothing to correct"
    assert bandit.record_shown(choice.play_id, ["a", "b", "c"])
    return path, choice.play_id


def _stored_shown(path, play_id) -> list[str]:
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute(
            "SELECT shown_fact_ids FROM bandit_plays WHERE play_id = ?",
            (play_id,),
        ).fetchone()
    finally:
        conn.close()
    if not row or not row[0]:
        return []
    return json.loads(row[0])


def test_the_record_is_corrected_when_the_bias_changed_the_answer(learning_db):
    """The case the bias exists for is the case the record got wrong."""
    path, play_id = learning_db
    assert _stored_shown(path, play_id) == ["a", "b", "c"]

    sink = {"play_id": play_id, "learning_db": str(path)}
    # The bias lifted "z" into the answer and pushed "c" out of it.
    after_bias = [_Result(_Fact(x)) for x in ("z", "a", "b")]
    _resettle_shown_after_bias(sink, "default", after_bias, ["a", "b", "c"])

    assert set(_stored_shown(path, play_id)) == {"z", "a", "b"}, (
        "an outcome citing 'z' would have settled nothing: the play still said "
        "it showed a, b and c"
    )


def test_an_unchanged_answer_is_not_rewritten(learning_db):
    """The common case must cost nothing.

    A cold session, or a warm one whose held memories were already on top,
    should not trigger a second write on every recall.
    """
    path, play_id = learning_db
    sink = {"play_id": play_id, "learning_db": str(path)}

    calls = []
    from superlocalmemory.learning import bandit as bandit_mod

    real = bandit_mod.ContextualBandit.record_shown

    def counting(self, *args, **kwargs):
        calls.append(args)
        return real(self, *args, **kwargs)

    bandit_mod.ContextualBandit.record_shown = counting
    try:
        same = [_Result(_Fact(x)) for x in ("a", "b", "c")]
        _resettle_shown_after_bias(sink, "default", same, ["a", "b", "c"])
    finally:
        bandit_mod.ContextualBandit.record_shown = real

    assert calls == [], "the record was rewritten although nothing moved"


def test_a_reorder_without_a_membership_change_is_not_rewritten(learning_db):
    """Settlement matches on membership, so order alone is not a difference."""
    path, play_id = learning_db
    sink = {"play_id": play_id, "learning_db": str(path)}
    reordered = [_Result(_Fact(x)) for x in ("c", "a", "b")]
    _resettle_shown_after_bias(sink, "default", reordered, ["a", "b", "c"])
    assert set(_stored_shown(path, play_id)) == {"a", "b", "c"}


def test_no_play_means_nothing_to_correct(tmp_path):
    """A read-only recall records no play, and this must be a no-op."""
    _resettle_shown_after_bias({}, "default", [], [])
    _resettle_shown_after_bias(
        {"play_id": None, "learning_db": str(tmp_path / "x.db")},
        "default", [_Result(_Fact("a"))], [],
    )


def test_a_broken_store_does_not_break_the_recall(learning_db):
    """Correcting the evidence is advisory; failing it must not fail the answer."""
    path, play_id = learning_db
    sink = {"play_id": play_id, "learning_db": "/nonexistent/dir/learning.db"}
    _resettle_shown_after_bias(
        sink, "default", [_Result(_Fact("z"))], ["a"],
    )  # must not raise
