# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The model tiers of the judge ladder, over the judge the answer check runs.

The ladder never starts, warms or swaps a judge: it borrows the one recall
already has (``memory_kind_wiring.live_judge``). On a Mac that runs Laya, the
Laya tier answers; on Windows and Linux, where the answer check is Jev, the Jev
tier answers, and only when the user switched ``judge.ladder.hosted`` on,
because the ladder sends pairs of memories that no earlier consent covered.
A machine with neither still works: Tier 0 answers what rules can, the rest
abstains, and nothing is retired.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from superlocalmemory.core.judge_ladder import Item, JudgeLadder, RulesTier, TierAnswer
from superlocalmemory.core.judge_questions import LOCAL_ONLY, Question

JudgeSupplier = Callable[[], Any | None]


def _model_name(judge: Any, default: str) -> str:
    return str(getattr(judge, "_model", "") or getattr(judge, "model", "") or default)


class LayaTier:
    """Laya on a Mac: for ``SLOT_RELATION`` it settles only "same".

    Measured on labelled pairs (``retrieval/judge_ask.SAME_STATEMENT``), the
    shipped model tells a paraphrase from anything else well and cannot tell a
    replaced value from a compatible one. It therefore answers "same" when it
    is sure and passes every other pair up the ladder unanswered. Questions it
    has not been measured on are not asked of it at all.
    """

    name = "laya"
    #: Positive pairs scored 0.66-0.95 and negatives at most 0.09 in that run.
    SAME_FLOOR = 0.8

    def __init__(self, supplier: JudgeSupplier) -> None:
        self._supplier = supplier

    def answer(self, question: Question, items: Sequence[Item]) -> list[TierAnswer | None] | None:
        if question.question_id != "SLOT_RELATION":
            return None
        judge = self._supplier()
        if judge is None or not hasattr(judge, "_proc"):  # the Jev judge has no worker
            return None
        from superlocalmemory.retrieval.judge_ask import SAME_STATEMENT, ask_laya_statement

        scores = ask_laya_statement(judge, SAME_STATEMENT, question.fields, items)
        if scores is None:
            return None
        model = _model_name(judge, "laya")
        return [TierAnswer("same", s, {"same": s}, model=model,
                           calibration_id=f"laya:{model}:same-v1")
                if s >= self.SAME_FLOOR else None for s in scores]


class JevTier:
    name = "jev"

    def __init__(self, supplier: JudgeSupplier, *, consent: Callable[[], bool]) -> None:
        self._supplier = supplier
        self._consent = consent

    def answer(self, question: Question, items: Sequence[Item]) -> list[TierAnswer | None] | None:
        if question.privacy == LOCAL_ONLY:
            return None
        judge = self._supplier()
        ask = getattr(judge, "ask_ladder", None)
        if not callable(ask):
            return None
        return ask(question, items, consent=self._consent() is True)


def build_ladder(supplier: JudgeSupplier | None, *, hosted_consent: Callable[[], bool]) -> JudgeLadder:
    """Rules first, then whichever model judge this machine runs."""
    tiers: list[Any] = [RulesTier()]
    if supplier is not None:
        tiers += [LayaTier(supplier), JevTier(supplier, consent=hosted_consent)]
    return JudgeLadder(tiers)


__all__ = ["JevTier", "LayaTier", "build_ladder"]
