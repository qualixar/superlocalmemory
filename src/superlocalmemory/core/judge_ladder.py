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


@dataclass(frozen=True)
class Decision:
    verdict: str
    confidence: float
    tier: str
    receipts: tuple[str, ...]

    @property
    def abstained(self) -> bool:
        return self.verdict == ABSTAIN


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


def write_receipt(conn: sqlite3.Connection, question: Question, item: Item, tier: str,
                  answer: TierAnswer, *, accepted: bool) -> str:
    receipt_id = "jr_" + uuid.uuid4().hex
    conn.execute(
        "INSERT INTO judge_receipts (receipt_id, profile_id, question_id, question_version, "
        "subject_kind, subject_id, tier, model, verdict, confidence, threshold, accepted, "
        "probabilities_json, calibration_id, inputs_hash, decided_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (receipt_id, item.profile_id, question.question_id, question.version,
         item.subject_kind, item.subject_id, tier, answer.model or tier, answer.verdict,
         float(answer.confidence), question.threshold(answer.verdict), int(accepted),
         json.dumps(dict(answer.probabilities), sort_keys=True), answer.calibration_id,
         inputs_hash(question, item), _now()))
    return receipt_id


def _valid(question: Question, answer: object) -> TierAnswer | None:
    if not isinstance(answer, TierAnswer) or answer.verdict not in question.criteria:
        return None
    c = answer.confidence
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not 0.0 <= c <= 1.0:
        return None
    return answer


class JudgeLadder:
    def __init__(self, tiers: Sequence[Tier]) -> None:
        self._tiers = tuple(tiers)

    @property
    def tier_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self._tiers)

    def decide(self, conn: sqlite3.Connection, question: Question,
               items: Sequence[Item]) -> list[Decision]:
        """One decision per item, in order; receipts written on ``conn``."""
        pending = list(range(len(items)))
        receipts: dict[int, list[str]] = {i: [] for i in pending}
        decided: dict[int, Decision] = {}
        for tier in self._tiers:
            if not pending:
                break
            pending = self._ask_tier(conn, tier, question, items, pending, receipts, decided)
        for i in pending:
            decided[i] = Decision(ABSTAIN, 0.0, "none", tuple(receipts[i]))
        return [decided[i] for i in range(len(items))]

    def _ask_tier(self, conn, tier, question, items, pending, receipts, decided) -> list[int]:
        still: list[int] = []
        for start in range(0, len(pending), MAX_BATCH):
            batch = pending[start:start + MAX_BATCH]
            answers = self._answers(tier, question, [items[i] for i in batch])
            for index, answer in zip(batch, answers):
                if answer is None:
                    still.append(index)
                    continue
                sure = answer.confidence >= question.threshold(answer.verdict)
                receipts[index].append(write_receipt(conn, question, items[index], tier.name,
                                                     answer, accepted=sure))
                if sure:
                    decided[index] = Decision(answer.verdict, answer.confidence, tier.name,
                                              tuple(receipts[index]))
                else:
                    still.append(index)
        return still

    @staticmethod
    def _answers(tier: Tier, question: Question,
                 items: list[Item]) -> list[TierAnswer | None]:
        try:
            raw = tier.answer(question, items)
        except Exception as exc:  # noqa: BLE001 - a tier failure is a skipped tier
            logger.info("judge ladder: tier %s failed (%s)", tier.name, type(exc).__name__)
            raw = None
        if not isinstance(raw, list) or len(raw) != len(items):
            return [None] * len(items)
        return [_valid(question, a) for a in raw]


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


class RulesTier:
    """Tier 0: answers only what plain rules can be sure of; abstains otherwise.

    For ``SLOT_RELATION``: a record of a moment (a session ending, a file
    changing) never replaces another one -- on the maintainer's backlog such
    records were 29 of the 32 pairs a model judge called replacements -- and two
    texts whose words nearly coincide say the same thing. Nothing else is
    decided here: telling a changed value from a paraphrase is the models' job.
    """

    name = "rules"
    SAME_OVERLAP = 0.85

    def answer(self, question: Question, items: Sequence[Item]) -> list[TierAnswer | None]:
        if question.question_id != "SLOT_RELATION":
            return [None] * len(items)
        return [self._one(item) for item in items]

    def _one(self, item: Item) -> TierAnswer | None:
        from superlocalmemory.encoding.non_answer import is_event_record

        older, newer = item.texts.get("older", ""), item.texts.get("newer", "")
        if is_event_record(older) or is_event_record(newer):
            return TierAnswer("separate_events", 0.99, {"separate_events": 1.0},
                              model="rules-v2", calibration_id="rules-v2")
        a, b = _tokens(older), _tokens(newer)
        overlap = len(a & b) / len(a | b) if a and b else 0.0
        if overlap >= self.SAME_OVERLAP:
            return TierAnswer("same", 0.95, {"same": overlap}, model="rules-v2",
                              calibration_id="rules-v2")
        return None


__all__ = ["ABSTAIN", "Decision", "Item", "JudgeLadder", "MAX_BATCH", "RulesTier", "Tier",
           "TierAnswer", "inputs_hash", "write_receipt"]
