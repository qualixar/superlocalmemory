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

A tier that cannot be asked here (not installed, a question it was never
measured on, shadow, no consent) declines by returning None: nothing was asked,
so nothing is receipted. A tier that was asked and failed as a whole (a timeout,
an HTTP error, a malformed reply, a crash) is receipted for every item of the
batch as ``tier_failed`` with its ``failure_kind`` (I-09), so availability has
a denominator and a failed ask is never mistaken for a judged one.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import sqlite3
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Mapping, Protocol, Sequence

from superlocalmemory.core.judge_questions import Question

#: Where the judge workers cut each text; the hash covers what they saw.
MAX_TEXT_CHARS = 1800

logger = logging.getLogger(__name__)

MAX_BATCH = 16
ABSTAIN = "abstain"


@dataclass(frozen=True)
class Item:
    """One thing to judge: what it is about, its named texts, and the facts it names."""
    subject_kind: str
    subject_id: str
    profile_id: str
    texts: Mapping[str, str]
    fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TierAnswer:
    verdict: str
    confidence: float
    probabilities: Mapping[str, float] = field(default_factory=dict)
    model: str = ""
    calibration_id: str = "uncalibrated"
    #: The consent record a hosted answer relied on (``consent_store.reference``).
    consent_ref: str | None = None


NO_ANSWER = "no_answer"
TIER_FAILED = "tier_failed"
#: How a tier failed as a whole. Closed: anything else is recorded as "error".
FAILURE_KINDS = frozenset({"timeout", "transport", "http", "malformed", "shutdown",
                           "preempted", "worker", "error"})


class TierFailure(Exception):
    """Raised by a tier that was asked and could not answer the batch."""

    def __init__(self, kind: str, *, consent_ref: str | None = None) -> None:
        self.kind = kind if kind in FAILURE_KINDS else "error"
        self.consent_ref = consent_ref
        super().__init__(self.kind)


@dataclass(frozen=True)
class Decision:
    verdict: str
    confidence: float
    tier: str
    receipts: tuple[str, ...]
    model: str = ""
    #: The batching regime it was decided in (M-01); ``unknown`` certifies nothing.
    regime: str = "unknown"
    #: The routing threshold the answer passed (W0-10: the gate checks it is
    #: inside the certified chain); None where unknown.
    routing_threshold: float | None = None

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
    population: str
    fact_ids_json: str
    tier: str
    tier_index: int
    model: str
    verdict: str
    #: None for a deterministic tier: a rule has no score (R-36).
    confidence: float | None
    raw_score: float | None
    threshold: float
    accepted: bool
    probabilities_json: str
    calibration_id: str
    inputs_hash: str
    batch_id: str
    batch_size: int
    batch_position: int
    peers_hash: str
    flag_state: str
    decided_at: str
    failure_kind: str | None = None
    consent_ref: str | None = None
    audit_selected: bool = False
    audit_probability: float | None = None
    batching_regime: str = "unknown"
    inbox_selected: bool = False


@dataclass(frozen=True)
class Context:
    """What a run of the ladder records about itself on every receipt."""
    population: str = "unknown"
    flag_state: str = "unknown"
    #: HMAC key for ``inputs_hash`` (``core/receipt_key``). None: an unkeyed
    #: digest, for tests and store copies only.
    hash_key: bytes | None = None
    #: The audit draw probability (``core/audit_sampler``, M-04); None: no sampling.
    audit_probability: float | None = None


class Tier(Protocol):
    name: str

    def answer(self, question: Question,
               items: Sequence[Item]) -> list[TierAnswer | None] | None: ...


def is_model_tier(tier: object) -> bool:
    """Rules are deterministic and never receipt a miss; every other tier is a model."""
    return getattr(tier, "is_model", True) is not False


#: How a decision was asked (M-01): alone, among peers, or by a rule, which
#: answers each item on its own whatever the batch. Errors measured in one
#: regime say nothing about another (R-23), so each is its own certificate cell.
BATCHING_REGIMES = ("single", "batched", "unbatched")


def batching_regime(batch_size: int, *, model: bool) -> str:
    if not model:
        return "unbatched"
    return "single" if batch_size == 1 else "batched"


def tier_available(tier: object, question: Question) -> bool:
    """Whether ``tier`` could be asked ``question`` here, now, without asking it."""
    probe = getattr(tier, "available", None)
    if not callable(probe):
        return True
    try:
        return probe(question) is True
    except Exception as exc:  # noqa: BLE001 - a probe that fails means not available
        logger.info("judge ladder: availability of %s unknown (%s)", getattr(tier, "name", "?"),
                    type(exc).__name__)
        return False


def inputs_hash(question: Question, item: Item, key: bytes | None = None) -> str:
    """The pair as judges see it (each text cut where the worker cuts it), keyed."""
    raw = json.dumps({"q": question.question_id, "v": question.version,
                      "t": {k: item.texts.get(k, "")[:MAX_TEXT_CHARS] for k in question.fields}},
                     sort_keys=True, ensure_ascii=False).encode("utf-8")
    if key is None:
        return hashlib.sha256(raw).hexdigest()
    return hmac.new(key, raw, hashlib.sha256).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class _Batch:
    batch_id: str
    size: int
    tier_index: int
    peers_hash: str
    consent_ref: str | None = None
    regime: str = "unknown"


def _row(question: Question, item: Item, tier: str, answer: TierAnswer, *, threshold: float,
         accepted: bool, ctx: Context, batch: _Batch, position: int,
         failure_kind: str | None = None, scored: bool = True) -> ReceiptRow:
    from superlocalmemory.core.audit_sampler import NOT_ANSWERS, inbox_select, select
    from superlocalmemory.core.risk_policy import audit_probability_for

    receipt_id = "jr_" + uuid.uuid4().hex
    selected, probability = select(ctx.hash_key, receipt_id, answer.verdict,
                                   audit_probability_for(answer.verdict, ctx.audit_probability))
    pair = inputs_hash(question, item, ctx.hash_key)
    in_inbox = (answer.verdict not in NOT_ANSWERS
                and inbox_select(ctx.hash_key, pair, ctx.audit_probability))
    # A hosted ask's consent applies to every row of its batch, misses included.
    return ReceiptRow(
        receipt_id=receipt_id, profile_id=item.profile_id,
        question_id=question.question_id, question_version=question.version,
        subject_kind=item.subject_kind, subject_id=item.subject_id, population=ctx.population,
        fact_ids_json=json.dumps(sorted(item.fact_ids)), tier=tier, tier_index=batch.tier_index,
        model=answer.model or tier, verdict=answer.verdict,
        confidence=float(answer.confidence) if scored else None,
        raw_score=float(answer.confidence) if scored else None, threshold=threshold, accepted=accepted,
        probabilities_json=json.dumps(dict(answer.probabilities), sort_keys=True),
        calibration_id=answer.calibration_id,
        inputs_hash=pair, batch_id=batch.batch_id,
        batch_size=batch.size, batch_position=position, peers_hash=batch.peers_hash,
        flag_state=ctx.flag_state, decided_at=_now(), failure_kind=failure_kind,
        consent_ref=answer.consent_ref or batch.consent_ref, audit_selected=selected,
        audit_probability=probability, batching_regime=batch.regime, inbox_selected=in_inbox)


_RECEIPT_COLUMNS = (
    "receipt_id", "profile_id", "question_id", "question_version", "subject_kind", "subject_id",
    "population", "fact_ids_json", "tier", "tier_index", "model", "verdict", "confidence",
    "raw_score", "threshold", "accepted", "probabilities_json", "calibration_id", "inputs_hash",
    "batch_id", "batch_size", "batch_position", "peers_hash", "flag_state", "decided_at",
    "failure_kind", "consent_ref", "audit_selected", "audit_probability", "batching_regime",
    "inbox_selected")


def write_receipts(conn: sqlite3.Connection, rows: Sequence[ReceiptRow]) -> None:
    sql = (f"INSERT INTO judge_receipts ({', '.join(_RECEIPT_COLUMNS)}) "  # noqa: S608
           f"VALUES ({', '.join('?' * len(_RECEIPT_COLUMNS))})")
    conn.executemany(sql, [tuple(int(v) if isinstance(v, bool) else v
                                 for v in (getattr(r, c) for c in _RECEIPT_COLUMNS))
                           for r in rows])
    # The erase trigger finds a fact's receipts through this index (I-11, W0-09).
    conn.executemany("INSERT OR IGNORE INTO judge_receipt_facts (receipt_id, fact_id) "
                     "VALUES (?, ?)", [(r.receipt_id, f) for r in rows
                                       for f in json.loads(r.fact_ids_json)])


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

    def available_model_tiers(self, question: Question) -> frozenset[str]:
        """The model tiers that could be asked ``question`` here now (nothing is asked)."""
        return frozenset(t.name for t in self._tiers
                         if is_model_tier(t) and tier_available(t, question))

    def decide(self, question: Question, items: Sequence[Item],
               ctx: Context = Context()) -> tuple[list[Decision], list[ReceiptRow]]:
        pending = list(range(len(items)))
        receipts: dict[int, list[ReceiptRow]] = {i: [] for i in pending}
        decided: dict[int, Decision] = {}
        for index, tier in enumerate(self._tiers):
            if not pending:
                break
            pending = self._ask_tier(tier, index, question, items, pending, receipts, decided,
                                     ctx)
        for i in pending:
            decided[i] = Decision(ABSTAIN, 0.0, "none", tuple(r.receipt_id for r in receipts[i]))
        rows = [row for i in range(len(items)) for row in receipts[i]]
        return [decided[i] for i in range(len(items))], rows

    def decide_and_record(self, conn: sqlite3.Connection, question: Question,
                          items: Sequence[Item], ctx: Context = Context()) -> list[Decision]:
        decisions, rows = self.decide(question, items, ctx)
        write_receipts(conn, rows)
        return decisions

    def _ask_tier(self, tier, tier_index, question, items, pending, receipts, decided,
                  ctx) -> list[int]:
        still: list[int] = []
        log_misses = getattr(tier, "log_abstentions", True)
        for start in range(0, len(pending), MAX_BATCH):
            batch = pending[start:start + MAX_BATCH]
            peers = hashlib.sha256("\0".join(sorted(items[i].subject_id for i in batch))
                                   .encode()).hexdigest()[:32]
            answers, failure, ref = self._answers(tier, question, [items[i] for i in batch])
            meta = _Batch("jb_" + uuid.uuid4().hex, len(batch), tier_index, peers, ref,
                          batching_regime(len(batch), model=is_model_tier(tier)))
            asked = answers is not None
            for pos, index in enumerate(batch):
                if failure is not None:  # asked and failed as a whole: receipted (I-09)
                    receipts[index].append(_row(
                        question, items[index], tier.name,
                        TierAnswer(TIER_FAILED, 0.0, model=tier.name), threshold=1.0,
                        accepted=False, ctx=ctx, batch=meta, position=pos,
                        failure_kind=failure))
                    still.append(index)
                    continue
                answer = answers[pos] if asked else None
                if answer is None:
                    if asked and log_misses:  # censoring would bias any later calibration
                        receipts[index].append(_row(
                            question, items[index], tier.name,
                            TierAnswer(NO_ANSWER, 0.0, model=tier.name), threshold=1.0,
                            accepted=False, ctx=ctx, batch=meta, position=pos))
                    still.append(index)
                    continue
                threshold = _route_threshold(question, tier, answer.verdict)
                sure = answer.confidence >= threshold
                row = _row(question, items[index], tier.name, answer, threshold=threshold,
                           accepted=sure, ctx=ctx, batch=meta, position=pos,
                           scored=is_model_tier(tier))
                receipts[index].append(row)
                if sure:
                    decided[index] = Decision(answer.verdict, answer.confidence, tier.name,
                                              tuple(r.receipt_id for r in receipts[index]),
                                              model=row.model, regime=row.batching_regime,
                                              routing_threshold=threshold)
                else:
                    still.append(index)
        return still

    @staticmethod
    def _answers(tier: Tier, question: Question, items: list[Item]
                 ) -> tuple[list[TierAnswer | None] | None, str | None, str | None]:
        """(answers, failure kind, consent ref). Answers None and no failure: the
        tier declined, so nothing was asked. Answers None with a failure: it was
        asked and failed. The consent ref is taken before answers are validated,
        so a hosted batch keeps it even when every answer is unusable."""
        try:
            raw = tier.answer(question, items)
        except TierFailure as exc:
            logger.info("judge ladder: tier %s failed (%s)", tier.name, exc.kind)
            return None, exc.kind, exc.consent_ref
        except Exception as exc:  # noqa: BLE001 - a crashing tier is a failed ask, receipted
            logger.warning("judge ladder: tier %s crashed (%s)", tier.name, type(exc).__name__)
            return None, "error", None
        if raw is None:
            return None, None, None
        ref = next((a.consent_ref for a in raw if isinstance(a, TierAnswer) and a.consent_ref),
                   None) if isinstance(raw, list) else None
        if not isinstance(raw, list) or len(raw) != len(items):
            logger.warning("judge ladder: tier %s returned a malformed batch", tier.name)
            return None, "malformed", ref
        return [_valid(question, a) for a in raw], None, ref


#: Words and single symbols: "-5" vs "5", ">" vs "<", "$" vs "€" differ.
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
#: Only sentence-final punctuation is ignored; nothing else is assumed harmless.
_TRAILING = ".;:!"


def _tokens(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKC", text).casefold().strip().rstrip(_TRAILING + " ")
    return _TOKEN.findall(folded)


class RulesTier:
    """Tier 0: answers only what plain rules can be sure of; abstains otherwise.

    For ``SLOT_RELATION``: a record of a moment (a session ending, a file
    changing) never replaces another one -- on the maintainer's backlog such
    records were 29 of the 32 pairs a model judge called replacements -- and two
    texts identical after Unicode normalisation and case folding (sentence-final
    punctuation aside) say the same thing. Any other difference, a sign, a
    symbol, a tense, a preposition, is never "same" here. Nothing else is
    decided: telling a changed value from a paraphrase is the models' job.
    Rule abstentions are not receipted (``log_abstentions``): rules are
    deterministic, so a miss carries no calibration information.
    """

    name = "rules"
    log_abstentions = False
    is_model = False

    def answer(self, question: Question, items: Sequence[Item]) -> list[TierAnswer | None]:
        if question.question_id != "SLOT_RELATION":
            return [None] * len(items)
        return [self._one(item) for item in items]

    def _one(self, item: Item) -> TierAnswer | None:
        from superlocalmemory.encoding.non_answer import is_event_record

        older, newer = item.texts.get("older", ""), item.texts.get("newer", "")
        if is_event_record(older) or is_event_record(newer):
            return TierAnswer("separate_events", 1.0, {"separate_events": 1.0},
                              model="rules-v4", calibration_id="rules-v4")
        a = _tokens(older)
        if a and a == _tokens(newer):
            return TierAnswer("same", 1.0, {"same": 1.0}, model="rules-v3",
                              calibration_id="rules-v3")
        return None


__all__ = ["ABSTAIN", "BATCHING_REGIMES", "batching_regime",
           "Context", "Decision", "FAILURE_KINDS", "Item", "JudgeLadder", "MAX_BATCH",
           "MAX_TEXT_CHARS", "NO_ANSWER", "ReceiptRow", "RulesTier", "TIER_FAILED", "Tier",
           "TierAnswer", "TierFailure", "inputs_hash", "is_model_tier", "tier_available",
           "write_receipts"]
