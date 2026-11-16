# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The model tiers of the judge ladder, over the judge the answer check runs.

The ladder never starts, warms or swaps a judge: it borrows the one recall
already has (``memory_kind_wiring.live_judge``). On a Mac that runs Laya, the
Laya tier answers; on Windows and Linux, where the answer check is Jev, the Jev
tier answers, and only when the user gave the "judge_ladder_hosted" consent (``core/consent_store``),
because the ladder sends pairs of memories that no earlier consent covered.
A machine with neither still works: Tier 0 answers what rules can, the rest
abstains, and nothing is retired.

The hosted tier also needs ``judge.ladder`` to be ``on``. In shadow the ladder
records only in its own tables and sends nothing off the machine (blueprint
D-27, I-15), so a shadow ladder is rules and Laya only. The flag is checked when
the ladder is built and again before every hosted ask, so turning the flag back
to shadow stops the next request, not the next pass. A flag that cannot be read
counts as shadow.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable, Sequence

from superlocalmemory.core.judge_ladder import (
    MAX_BATCH,
    Item,
    JudgeLadder,
    RulesTier,
    TierAnswer,
    TierFailure,
)
from superlocalmemory.core.judge_questions import LOCAL_ONLY, Question

JudgeSupplier = Callable[[], Any | None]

logger = logging.getLogger(__name__)


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
    #: Routing only: an answer at or above this stops escalation. Whether it may
    #: change memory is decided by ``judge_calibration.ActingGate`` from labels.
    #: (25 labelled pairs: positives 0.66-0.95, negatives at most 0.09.)
    floors = {"same": 0.8}
    #: The worker scores each pair as its own item: on 128 real pairs the score
    #: was identical alone, reshuffled and in the same batch (F-43). Its receipts
    #: are ``single`` at any batch size, so batches of 16 stay and its cells can certify.
    per_item = True

    def __init__(self, supplier: JudgeSupplier) -> None:
        self._supplier = supplier

    @staticmethod
    def _usable(judge: Any) -> bool:
        # The Jev judge has no worker; a Laya worker that is not ready declines.
        return judge is not None and hasattr(judge, "_proc") and getattr(judge, "ready", False) is True

    def available(self, question: Question) -> bool:
        return question.question_id == "SLOT_RELATION" and self._usable(self._supplier())

    def answer(self, question: Question, items: Sequence[Item]) -> list[TierAnswer | None] | None:
        if question.question_id != "SLOT_RELATION":
            return None
        judge = self._supplier()
        if not self._usable(judge):
            return None
        from superlocalmemory.retrieval.judge_ask import SAME_STATEMENT, ask_laya_statement

        scores = ask_laya_statement(judge, SAME_STATEMENT, question.fields, items)
        if scores is None:
            return None
        model = _model_name(judge, "laya")
        # Every score is returned, below the floor too: a receipt for each ask is
        # what lets the floor be calibrated later instead of guessed.
        return [TierAnswer("same", s, {"same": s}, model=model,
                           calibration_id=f"laya:{model}:same-v1") for s in scores]


def _literally_true(check: Callable[[], object]) -> bool:
    """Only a literal ``True`` opens the hosted path; any failure keeps it shut."""
    try:
        return check() is True
    except Exception as exc:  # noqa: BLE001 - fail closed: an unreadable flag is shadow
        logger.info("judge ladder: hosted check failed (%s); treated as shadow", type(exc).__name__)
        return False


def _consent_ref(check: Callable[[], object]) -> str | None:
    """The consent record's reference, or None (no consent, or unreadable)."""
    try:
        ref = check()
    except Exception as exc:  # noqa: BLE001 - fail closed: an unreadable consent is none
        logger.info("judge ladder: consent check failed (%s); treated as none", type(exc).__name__)
        return None
    return ref if isinstance(ref, str) and ref else None


class JevTier:
    """Jev, hosted: asked only while ``judge.ladder`` is on and the user consented.

    ``consent_ref`` returns the consent record's reference
    (``consent_store.reference``) or None. Every receipt of a hosted ask carries
    it (I-15), failures included, so a receipt names the record it relied on.
    """

    name = "jev"

    def __init__(self, supplier: JudgeSupplier, *, consent_ref: Callable[[], str | None],
                 live: Callable[[], bool]) -> None:
        self._supplier = supplier
        self._consent_ref = consent_ref
        self._live = live

    def _permit(self, question: Question) -> str | None:
        # Checked before the judge is even fetched: in shadow, or without the
        # user's consent, nothing leaves (D-27, D-20). ``ask_ladder`` checks consent
        # again at the point of sending.
        if question.privacy == LOCAL_ONLY or not _literally_true(self._live):
            return None
        return _consent_ref(self._consent_ref)

    @staticmethod
    def batch_limit(question: Question) -> int:
        """One pair per request for a consequential question (D-43): batched, its
        answer moved with order and partners at about three times its own noise
        (F-42). Background only, so the extra requests never slow a recall."""
        return 1 if question.consequential else MAX_BATCH

    def available(self, question: Question) -> bool:
        if self._permit(question) is None:
            return False
        judge = self._supplier()
        return callable(getattr(judge, "ask_ladder", None)) and getattr(judge, "ready", True) is not False

    def answer(self, question: Question, items: Sequence[Item]) -> list[TierAnswer | None] | None:
        ref = self._permit(question)
        if ref is None:
            return None
        judge = self._supplier()
        ask = getattr(judge, "ask_ladder", None)
        if not callable(ask):
            return None
        try:
            answers = ask(question, items, consent=True)
        except TierFailure as exc:
            raise TierFailure(exc.kind, consent_ref=ref) from exc
        except Exception as exc:  # noqa: BLE001 - the request may have gone out: receipt it
            raise TierFailure("error", consent_ref=ref) from exc
        if not isinstance(answers, list):
            return answers
        return [replace(a, consent_ref=ref) if isinstance(a, TierAnswer) else a for a in answers]


def build_ladder(supplier: JudgeSupplier | None, *, hosted_consent_ref: Callable[[], str | None],
                 hosted_live: Callable[[], bool], scorer: Any = None) -> JudgeLadder:
    """Rules first, then whichever model judge this machine runs.

    ``hosted_live`` says whether ``judge.ladder`` is on (not merely observing).
    It has no default: every caller decides, and a shadow ladder has no hosted tier.
    ``scorer`` is the Tier 0.5 model (``core/relation_scorer``), asked in shadow only.
    """
    tiers: list[Any] = [RulesTier()]
    if scorer is not None:
        from superlocalmemory.core.relation_scorer import RelationScorerTier

        tiers.append(RelationScorerTier(scorer))
    if supplier is not None:
        tiers.append(LayaTier(supplier))
        if _literally_true(hosted_live):
            tiers.append(JevTier(supplier, consent_ref=hosted_consent_ref, live=hosted_live))
    return JudgeLadder(tiers)


__all__ = ["JevTier", "LayaTier", "build_ladder"]
