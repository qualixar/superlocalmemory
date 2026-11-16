# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W2 innovation 4: Tier 0.5, trained on audit labels only and asked in shadow only."""

from __future__ import annotations

import json
import sqlite3

import pytest

from superlocalmemory.core import relation_scorer as rs
from superlocalmemory.core.judge_ladder import SHADOW_SUFFIX, Context, JudgeLadder, TierAnswer
from superlocalmemory.core.judge_questions import SLOT_RELATION
from superlocalmemory.storage.migrations import M072_judge_receipts as M072
from tests.test_core.test_audit_sampler import KEY, _items


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    M072.apply(c)
    c.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, content TEXT)")
    return c


def _pair(conn, i, older, newer, *labels, source="audit", blind=1):
    a, b = f"a{i}", f"b{i}"
    conn.execute("INSERT INTO atomic_facts VALUES (?, ?), (?, ?)", (a, older, b, newer))
    for j, label in enumerate(labels):
        rid = f"r{i}-{j}"
        conn.execute(
            "INSERT INTO judge_receipts (receipt_id, profile_id, question_id, question_version, "
            "subject_kind, subject_id, population, fact_ids_json, tier, model, verdict, "
            "confidence, threshold, accepted, calibration_id, inputs_hash, decided_at, "
            "audit_selected, audit_probability, batching_regime, inbox_selected) VALUES "
            "(?, 'p', 'SLOT_RELATION', ?, 's', ?, 'slot_dispute', ?, 'laya', 'm', 'same', 0.9, "
            "0.7, 1, 'c', ?, '2026-11-16', 1, 0.1, 'single', 0)",
            (rid, SLOT_RELATION.version, rid, json.dumps([a, b]), f"h{i}"))
        conn.execute("INSERT INTO judge_labels (receipt_id, label, source, labeller, blind, "
                     "labelled_at) VALUES (?, ?, ?, 'person', ?, '2026-11-16')",
                     (rid, label, source, blind))


def _seed(conn, n=rs.MIN_PER_CLASS, start=0):
    for i in range(start, start + n):
        _pair(conn, i, f"The deploy window for service {i} is at {i % 9 + 1}:00",
              f"The deploy window for service {i} is at {i % 9 + 2}:30", "replaces")
    for i in range(start + 1000, start + 1000 + n):
        _pair(conn, i, f"Team {i} had a planning meeting about the roadmap",
              f"The cafe near office {i} serves good espresso", "unrelated")


def test_features_see_changed_numbers_negation_and_subject() -> None:
    f = dict(zip(rs.FEATURES, rs.features("The rate limit is 600 per minute",
                                          "The rate limit is 60 per minute")))
    assert f["numbers_differ"] == 1.0 and f["same_opening"] == 1.0 and f["overlap"] > 0.6
    g = dict(zip(rs.FEATURES, rs.features("Alex uses Vim", "Alex no longer uses Vim")))
    assert g["negation_flip"] == 1.0
    assert rs.features("x", "x") == rs.features("x", "x")


def test_no_model_until_each_class_has_enough_audited_pairs(conn) -> None:
    _seed(conn, n=rs.MIN_PER_CLASS - 1)
    assert rs.train(conn) is None


def test_it_learns_from_audit_labels_and_scores_a_new_pair(conn) -> None:
    _seed(conn)
    model = rs.train(conn)
    assert model is not None and (model.positives, model.negatives) == (20, 20)
    hi = model.probability(rs.features("The deploy window for billing is at 4:00",
                                       "The deploy window for billing is at 5:30"))
    lo = model.probability(rs.features("Team 7 discussed hiring plans",
                                       "The office plants need water on Fridays"))
    assert hi > 0.8 and lo < 0.2


def test_only_unanimous_blind_audit_labels_train_it(conn) -> None:
    _seed(conn)
    base = rs.train(conn)
    _pair(conn, 5000, "The standup is at 9", "The standup is at 10", "replaces", "same")
    _pair(conn, 5001, "Backups run nightly", "Backups run weekly", "cannot_tell")
    _pair(conn, 5002, "Sam likes tea", "Sam likes coffee", "replaces", source="review")
    _pair(conn, 5003, "Alex is in Paris", "Alex is in Rome", "replaces", blind=0)
    again = rs.train(conn)
    assert (again.positives, again.negatives) == (base.positives, base.negatives)
    assert again.name == base.name


def test_new_labels_make_a_new_model_name(conn) -> None:
    _seed(conn)
    first = rs.train(conn).name
    _pair(conn, 6000, "The fee is 10 dollars", "The fee is 12 dollars", "replaces")
    assert rs.train(conn).name != first and first.startswith("tier05-lr-")


def test_an_erased_memory_takes_its_label_out_of_training(conn) -> None:
    _seed(conn)
    assert rs.train(conn) is not None
    conn.execute("DELETE FROM atomic_facts WHERE fact_id = 'a0'")
    assert rs.train(conn) is None  # 19 replaced pairs left: below the minimum again


class Decider:
    name = "laya"
    per_item = True

    def answer(self, question, items):
        return [TierAnswer("same", 0.95, model="m")] * len(items)


def test_the_ladder_asks_it_in_shadow_and_never_stops_on_it(conn) -> None:
    _seed(conn)
    tier = rs.RelationScorerTier(rs.train(conn))
    assert tier.shadow_only and tier.per_item
    ladder = JudgeLadder([tier, Decider()])
    decisions, rows = ladder.decide(SLOT_RELATION, _items(3),
                                    Context(population="slot_dispute", hash_key=KEY))
    assert [d.tier for d in decisions] == ["laya"] * 3
    shadow = [r for r in rows if r.tier == "tier05"]
    assert len(shadow) == 3 and not any(r.accepted for r in shadow)
    assert {r.population for r in shadow} == {"slot_dispute" + SHADOW_SUFFIX}
    assert {r.verdict for r in shadow} == {"replaces"}
    assert "tier05" not in ladder.available_model_tiers(SLOT_RELATION)


def test_without_a_model_it_is_never_asked() -> None:
    tier = rs.RelationScorerTier(None)
    assert tier.available(SLOT_RELATION) is False and tier.answer(SLOT_RELATION, _items(1)) is None
    _d, rows = JudgeLadder([tier, Decider()]).decide(SLOT_RELATION, _items(2),
                                                     Context(hash_key=KEY))
    assert not any(r.tier == "tier05" for r in rows)


def test_even_a_confident_shadow_answer_is_never_accepted(conn) -> None:
    from superlocalmemory.core.judge_ladder import Item

    _seed(conn)
    tier = rs.RelationScorerTier(rs.train(conn))
    items = [Item("slot_dispute", "d0", "p", {"older": "The deploy window for api is at 3:00",
                                              "newer": "The deploy window for api is at 4:30"})]
    (answer,) = tier.answer(SLOT_RELATION, items)
    assert answer.confidence >= SLOT_RELATION.threshold("replaces")  # sure of itself
    _d, rows = JudgeLadder([tier, Decider()]).decide(SLOT_RELATION, items, Context(hash_key=KEY))
    (shadow,) = [r for r in rows if r.tier == "tier05"]
    assert shadow.accepted is False
