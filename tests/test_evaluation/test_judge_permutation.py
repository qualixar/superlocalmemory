# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The batch permutation-invariance check (R-23, Fable review item 23)."""

from __future__ import annotations

from superlocalmemory.core.judge_ladder import Item, TierAnswer
from superlocalmemory.core.judge_questions import SLOT_RELATION
from superlocalmemory.evaluation import judge_permutation as perm


def _items(n: int) -> list[Item]:
    return [Item("seed", f"s{i}", "default", {"older": f"o{i}", "newer": f"n{i}"})
            for i in range(n)]


def _steady(question, items):
    """Answers depend on the item alone: invariant by construction."""
    return [TierAnswer("same", 0.5 + int(it.subject_id[1:]) % 5 / 10) for it in items]


def test_an_item_only_judge_is_invariant() -> None:
    report = perm.check(_steady, SLOT_RELATION, _items(perm.N_MIN), draws=3, batch=16)
    assert report.items == perm.N_MIN and report.draws == 3
    assert (report.flips, report.crossings, report.missing) == (0, 0, 0)
    assert report.max_range == 0.0 and report.invariant
    assert report.requests == 3 * 4  # 59 items in batches of 16, per draw


def test_too_few_items_cannot_show_invariance() -> None:
    report = perm.check(_steady, SLOT_RELATION, _items(perm.N_MIN - 1), draws=3)
    assert report.flips == 0 and not report.invariant


def test_a_score_that_depends_on_position_is_caught() -> None:
    def by_position(question, items):
        return [TierAnswer("same", 0.95 - 0.04 * pos) for pos, _ in enumerate(items)]

    report = perm.check(by_position, SLOT_RELATION, _items(64), draws=3, batch=16)
    assert report.max_range > 0.3 and not report.invariant
    assert report.crossings > 0  # some items fall below the routing floor in one draw only


def test_a_verdict_that_depends_on_its_partners_is_a_flip() -> None:
    def by_partner(question, items):
        first = items[0].subject_id
        return [TierAnswer("same" if first < "s3" else "both_true", 0.9) for _ in items]

    report = perm.check(by_partner, SLOT_RELATION, _items(64), draws=3, batch=16)
    assert report.flips > 0 and not report.invariant
    assert report.flip_upper > report.flips / report.items


def test_a_missing_answer_counts_against_invariance() -> None:
    calls = {"n": 0}

    def flaky(question, items):
        calls["n"] += 1
        return None if calls["n"] == 2 else _steady(question, items)

    report = perm.check(flaky, SLOT_RELATION, _items(64), draws=3, batch=16)
    assert report.missing == 16 and not report.invariant


def test_a_crashing_or_malformed_ask_is_missing_not_fatal() -> None:
    def broken(question, items):
        if len(items) == 16:
            raise RuntimeError("boom")
        return [TierAnswer("same", 0.9)]  # wrong length for any batch over one

    report = perm.check(broken, SLOT_RELATION, _items(64), draws=2, batch=16)
    assert report.missing == 64 and report.failures == 8


def test_answers_are_mapped_back_to_their_own_item() -> None:
    report = perm.check(_steady, SLOT_RELATION, _items(20), draws=3, batch=7, seed=11)
    by_id = {s.subject_id: s for s in report.spreads}
    assert by_id["s3"].scores == (0.8, 0.8, 0.8)
    assert by_id["s4"].scores == (0.9, 0.9, 0.9)


def test_draws_change_order_and_partners() -> None:
    seen: list[tuple[str, ...]] = []

    def record(question, items):
        seen.append(tuple(it.subject_id for it in items))
        return _steady(question, items)

    perm.check(record, SLOT_RELATION, _items(32), draws=3, batch=16, seed=5)
    firsts = [seen[0], seen[2], seen[4]]
    assert len(set(firsts)) == 3  # a new order each draw
    assert len({frozenset(b) for b in firsts}) > 1  # and new partners


def test_the_report_is_plain_data() -> None:
    data = perm.check(_steady, SLOT_RELATION, _items(10), draws=2).as_dict()
    assert data["items"] == 10 and "spreads" not in data and data["invariant"] is False


def test_an_item_never_answered_is_not_invariant() -> None:
    def skips_one(question, items):
        return [None if it.subject_id == "s0" else TierAnswer("same", 0.9) for it in items]

    report = perm.check(skips_one, SLOT_RELATION, _items(64), draws=3, batch=16)
    assert report.flips == 0 and report.missing == 1 and not report.invariant


def test_the_control_sends_the_same_batches_every_draw() -> None:
    seen: list[tuple[str, ...]] = []

    def record(question, items):
        seen.append(tuple(it.subject_id for it in items))
        return _steady(question, items)

    report = perm.check(record, SLOT_RELATION, _items(32), draws=3, batch=16, seed=5,
                        reorder=False)
    assert seen[0] == seen[2] == seen[4] and seen[1] == seen[3] == seen[5]
    assert report.reorder is False and report.as_dict()["reorder"] is False
