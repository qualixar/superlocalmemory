# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-43: Jev asks a consequential question one pair per request; Laya keeps its batches.

Measured on 128 real pairs: Jev's batched answer moved with order and partners
(F-42); Laya's score was identical alone and in any batch (F-43).
"""

from __future__ import annotations

from dataclasses import replace

from superlocalmemory.core.judge_ladder import (
    MAX_BATCH,
    Context,
    JudgeLadder,
    TierAnswer,
    batch_limit,
    batching_regime,
)
from superlocalmemory.core.judge_questions import SLOT_RELATION
from superlocalmemory.core.judge_tiers import JevTier, LayaTier
from tests.test_core.test_audit_sampler import KEY, _items

CTX = Context(population="slot_dispute", hash_key=KEY)
QUIET = replace(SLOT_RELATION, consequential=False)


class Recorder:
    name = "jev"

    def __init__(self, limit=None, per_item=None):
        self.sizes: list[int] = []
        if limit is not None:
            self.batch_limit = limit
        if per_item is not None:
            self.per_item = per_item

    def answer(self, question, items):
        self.sizes.append(len(items))
        return [TierAnswer("same", 0.95, model="m")] * len(items)


def test_slot_relation_is_consequential() -> None:
    assert SLOT_RELATION.consequential is True


def test_jev_asks_a_consequential_question_one_pair_at_a_time() -> None:
    assert JevTier.batch_limit(SLOT_RELATION) == 1
    assert JevTier.batch_limit(QUIET) == MAX_BATCH


def test_the_ladder_honours_a_tier_s_limit_and_receipts_single() -> None:
    tier = Recorder(limit=JevTier.batch_limit)
    _decisions, rows = JudgeLadder([tier]).decide(SLOT_RELATION, _items(5), CTX)
    assert tier.sizes == [1] * 5
    assert {r.batching_regime for r in rows} == {"single"}
    assert {r.batch_size for r in rows} == {1}


def test_a_tier_without_a_limit_keeps_full_batches() -> None:
    tier = Recorder()
    JudgeLadder([tier]).decide(SLOT_RELATION, _items(MAX_BATCH + 3), CTX)
    assert tier.sizes == [10, 9]  # balanced (W0-25), none above MAX_BATCH, no lone remainder


def test_laya_keeps_its_batches_and_its_receipts_are_single() -> None:
    assert LayaTier.per_item is True and not hasattr(LayaTier, "batch_limit")
    tier = Recorder(per_item=True)
    _decisions, rows = JudgeLadder([tier]).decide(SLOT_RELATION, _items(MAX_BATCH), CTX)
    assert tier.sizes == [MAX_BATCH]
    assert {r.batching_regime for r in rows} == {"single"}
    assert {r.batch_size for r in rows} == {MAX_BATCH}


def test_per_item_needs_a_literal_true() -> None:
    tier = Recorder(per_item="yes")
    _decisions, rows = JudgeLadder([tier]).decide(SLOT_RELATION, _items(3), CTX)
    assert {r.batching_regime for r in rows} == {"batched"}


def test_a_model_without_per_item_in_a_batch_is_still_batched() -> None:
    assert batching_regime(16, model=True) == "batched"
    assert batching_regime(16, model=True, per_item=True) == "single"
    assert batching_regime(16, model=False, per_item=True) == "unbatched"


def test_an_unreadable_limit_falls_back_to_one_item() -> None:
    def broken(question):
        raise RuntimeError("boom")

    assert batch_limit(Recorder(limit=broken), SLOT_RELATION) == 1
    assert batch_limit(Recorder(limit=lambda q: 0), SLOT_RELATION) == 1
    assert batch_limit(Recorder(limit=lambda q: 999), SLOT_RELATION) == MAX_BATCH
