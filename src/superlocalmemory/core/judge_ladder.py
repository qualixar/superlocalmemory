# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The judge ladder: one contract for typed memory questions, routed by difficulty.

Tiers are tried in order -- Tier 0 rules, then Laya (Apple Silicon, local), then
Jev (any platform, hosted, opt-in) -- and the first answer confident enough for
its label wins. A tier that cannot answer (not installed, not ready, no
consent, a malformed reply) is skipped; a tier that answers below threshold
hands the question up. When no tier is sure, the ladder abstains, and
abstaining is an answer: the caller leaves things as they are.

Every answer any tier gives is written as a receipt (``judge_receipts``, M072):
ids, labels, numbers and a hash of the inputs, never the text. Receipts make a
decision auditable and are what personal calibration learns from.

Items go in batches (a tier sees up to ``MAX_BATCH`` at once); a tier answers
the whole batch or nothing, so an answer can never pair with the wrong item.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Mapping, Protocol, Sequence

from superlocalmemory.core.judge_questions import Question

logger = logging.getLogger(__name__)

MAX_BATCH = 16
ABSTAIN = "abstain"


@dataclass(frozen=True)
class Item:
    """One thing to judge: what it is about, and its named texts."""
    subject_kind: str
    subject_id: str
    profile_id: str
    texts: Mapping[str, str]


@dataclass(frozen=True)
class TierAnswer:
    verdict: str
    confidence: float
    probabilities: Mapping[str, float] = field(default_factory=dict)
    model: str = ""
    calibration_id: str = "uncalibrated"


NO_ANSWER = "no_answer"


@dataclass(frozen=True)
class Decision:
    verdict: str
    confidence: float
    tier: str
    receipts: tuple[str, ...]
    model: str = ""

    @property
    def abstained(self) -> bool:
        return self.verdict == ABSTAIN


@dataclass(frozen=True)
class ReceiptRow:
    receipt_id: str
    profile_id: str
    question_id: str
    question_version: str
    subject_kind: str
    subject_id: str
    tier: str
    model: str
    verdict: str
    confidence: float
    threshold: float
    accepted: bool
    probabilities_json: str
    calibration_id: str
    inputs_hash: str
    decided_at: str


class Tier(Protocol):
    name: str

    def answer(self, question: Question,
               items: Sequence[Item]) -> list[TierAnswer | None] | None: ...


def inputs_hash(question: Question, item: Item) -> str:
    raw = json.dumps({"q": question.question_id, "v": question.version,
                      "t": {k: item.texts.get(k, "") for k in question.fields}},
                     sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _row(question: Question, item: Item, tier: str, answer: TierAnswer, *, threshold: float,
         accepted: bool) -> ReceiptRow:
    return ReceiptRow(
        receipt_id="jr_" + uuid.uuid4().hex, profile_id=item.profile_id,
        question_id=question.question_id, question_version=question.version,
        subject_kind=item.subject_kind, subject_id=item.subject_id, tier=tier,
        model=answer.model or tier, verdict=answer.verdict, confidence=float(answer.confidence),
        threshold=threshold, accepted=accepted,
        probabilities_json=json.dumps(dict(answer.probabilities), sort_keys=True),
        calibration_id=answer.calibration_id, inputs_hash=inputs_hash(question, item),
        decided_at=_now())


def write_receipts(conn: sqlite3.Connection, rows: Sequence[ReceiptRow]) -> None:
    conn.executemany(
        "INSERT INTO judge_receipts (receipt_id, profile_id, question_id, question_version, "
        "subject_kind, subject_id, tier, model, verdict, confidence, threshold, accepted, "
        "probabilities_json, calibration_id, inputs_hash, decided_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(r.receipt_id, r.profile_id, r.question_id, r.question_version, r.subject_kind,
          r.subject_id, r.tier, r.model, r.verdict, r.confidence, r.threshold, int(r.accepted),
          r.probabilities_json, r.calibration_id, r.inputs_hash, r.decided_at) for r in rows])


def _valid(question: Question, answer: object) -> TierAnswer | None:
    if not isinstance(answer, TierAnswer) or answer.verdict not in question.criteria:
        return None
    c = answer.confidence
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not 0.0 <= c <= 1.0:
        return None
    return answer


def _route_threshold(question: Question, tier: Tier, verdict: str) -> float:
    """When an answer is sure enough to stop escalating (never when to act: see
    ``judge_calibration.ActingGate``)."""
    floors = getattr(tier, "floors", {}) or {}
    return max(question.threshold(verdict), float(floors.get(verdict, 0.0)))


class JudgeLadder:
    """Asks tiers in order. Touches no database: receipts come back as rows,
    so the caller can judge outside any write lock and record afterwards."""

    def __init__(self, tiers: Sequence[Tier]) -> None:
        self._tiers = tuple(tiers)

    @property
    def tier_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self._tiers)

    def decide(self, question: Question,
               items: Sequence[Item]) -> tuple[list[Decision], list[ReceiptRow]]:
        pending = list(range(len(items)))
        receipts: dict[int, list[ReceiptRow]] = {i: [] for i in pending}
        decided: dict[int, Decision] = {}
        for tier in self._tiers:
            if not pending:
                break
            pending = self._ask_tier(tier, question, items, pending, receipts, decided)
        for i in pending:
            decided[i] = Decision(ABSTAIN, 0.0, "none", tuple(r.receipt_id for r in receipts[i]))
        rows = [row for i in range(len(items)) for row in receipts[i]]
        return [decided[i] for i in range(len(items))], rows

    def decide_and_record(self, conn: sqlite3.Connection, question: Question,
                          items: Sequence[Item]) -> list[Decision]:
        decisions, rows = self.decide(question, items)
        write_receipts(conn, rows)
        return decisions

    def _ask_tier(self, tier, question, items, pending, receipts, decided) -> list[int]:
        still: list[int] = []
        log_misses = getattr(tier, "log_abstentions", True)
        for start in range(0, len(pending), MAX_BATCH):
            batch = pending[start:start + MAX_BATCH]
            answers = self._answers(tier, question, [items[i] for i in batch])
            asked = answers is not None
            for pos, index in enumerate(batch):
                answer = answers[pos] if asked else None
                if answer is None:
                    if asked and log_misses:  # censoring would bias any later calibration
                        receipts[index].append(_row(
                            question, items[index], tier.name,
                            TierAnswer(NO_ANSWER, 0.0, model=tier.name), threshold=1.0,
                            accepted=False))
                    still.append(index)
                    continue
                threshold = _route_threshold(question, tier, answer.verdict)
                sure = answer.confidence >= threshold
                row = _row(question, items[index], tier.name, answer, threshold=threshold,
                           accepted=sure)
                receipts[index].append(row)
                if sure:
                    decided[index] = Decision(answer.verdict, answer.confidence, tier.name,
                                              tuple(r.receipt_id for r in receipts[index]),
                                              model=row.model)
                else:
                    still.append(index)
        return still

    @staticmethod
    def _answers(tier: Tier, question: Question,
                 items: list[Item]) -> list[TierAnswer | None] | None:
        """Per-item answers, or None when the tier could not be asked at all."""
        try:
            raw = tier.answer(question, items)
        except Exception as exc:  # noqa: BLE001 - a tier failure is a skipped tier
            logger.info("judge ladder: tier %s failed (%s)", tier.name, type(exc).__name__)
            return None
        if not isinstance(raw, list) or len(raw) != len(items):
            return None
        return [_valid(question, a) for a in raw]


_WORD = re.compile(r"[a-z0-9]+")
#: Words whose presence or absence never changes what a statement asserts.
#: Negators ("not", "no", "never"), numbers and units are deliberately absent:
#: a difference in any of them is a different statement.
_FUNCTION_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of", "in", "on", "at",
    "for", "and", "or", "it", "its", "this", "that", "these", "those", "with", "as", "by",
    "we", "our", "s", "so", "then", "also", "just", "very", "really", "please"})


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class RulesTier:
    """Tier 0: answers only what plain rules can be sure of; abstains otherwise.

    For ``SLOT_RELATION``: a record of a moment (a session ending, a file
    changing) never replaces another one -- on the maintainer's backlog such
    records were 29 of the 32 pairs a model judge called replacements -- and two
    texts that are identical once case, punctuation and function words are set
    aside say the same thing. A difference in any number, negator or content
    word is never "same" here (a changed port, an inserted "not"). Nothing else
    is decided: telling a changed value from a paraphrase is the models' job.
    Rule abstentions are not receipted (``log_abstentions``): rules are
    deterministic, so a miss carries no calibration information.
    """

    name = "rules"
    log_abstentions = False

    def answer(self, question: Question, items: Sequence[Item]) -> list[TierAnswer | None]:
        if question.question_id != "SLOT_RELATION":
            return [None] * len(items)
        return [self._one(item) for item in items]

    def _one(self, item: Item) -> TierAnswer | None:
        from superlocalmemory.encoding.non_answer import is_event_record

        older, newer = item.texts.get("older", ""), item.texts.get("newer", "")
        if is_event_record(older) or is_event_record(newer):
            return TierAnswer("separate_events", 1.0, {"separate_events": 1.0},
                              model="rules-v3", calibration_id="rules-v3")
        a, b = _tokens(older), _tokens(newer)
        if not a or not b:
            return None
        content_a = [w for w in a if w not in _FUNCTION_WORDS]
        if content_a and content_a == [w for w in b if w not in _FUNCTION_WORDS]:
            return TierAnswer("same", 1.0, {"same": 1.0}, model="rules-v3",
                              calibration_id="rules-v3")
        return None


__all__ = ["ABSTAIN", "Decision", "Item", "JudgeLadder", "MAX_BATCH", "NO_ANSWER", "ReceiptRow",
           "RulesTier", "Tier", "TierAnswer", "inputs_hash", "write_receipts"]
