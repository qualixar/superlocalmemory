# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Does a batched judge answer an item the same whatever its order and partners? (R-23)

A model asked about 16 pairs in one request may score a pair differently
depending on where it sits and what sits next to it. Then its score is not a
function of the item, and labels gathered in one batching do not describe
another. This check asks the same items in ``draws`` fresh shuffles (new order
and new partners each time, cut into batches of ``batch``) and reports, per
item, whether the verdict changed (a flip), whether the score landed on both
sides of the question's routing threshold (a crossing), and how far the score
moved. A score only matters through the thresholds it is compared with: the
routing threshold and the certificate grid (``judge_calibration.GRID``). So a
crossing is a score on both sides of any of them; jitter between two grid
points changes nothing that is decided.

The check passes only with zero flips, crossings and missing answers over at
least ``N_MIN`` items: with 0 of 59 the exact one-sided 95% upper bound on the
flip rate is below 5% (``clean_decisions_needed(0.05, 0.05)``), the same bar a
certificate needs. A pass is evidence for this model, question version and
batch size only. Asking costs ``draws`` x ceil(items / batch) requests, so it is
a measurement run on a store copy, never on the recall path.

``reorder=False`` is the control: every draw sends the same batches in the same
order, so whatever still moves is the model's own run-to-run noise, not the
batching. Order effects are the flips seen with reordering beyond that.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from superlocalmemory.core.judge_calibration import GRID, clean_decisions_needed, clopper_pearson_upper
from superlocalmemory.core.judge_ladder import MAX_BATCH, Item, TierAnswer
from superlocalmemory.core.judge_questions import Question

logger = logging.getLogger(__name__)

#: Bound on the flip rate a pass must show, and the confidence it is shown at.
ALPHA = 0.05
DELTA = 0.05
N_MIN = clean_decisions_needed(ALPHA, DELTA)

Ask = Callable[[Question, Sequence[Item]], "list[TierAnswer | None] | None"]


@dataclass(frozen=True)
class Spread:
    """One item across the draws (``None``: no usable answer in that draw)."""
    subject_id: str
    verdicts: tuple[str | None, ...]
    scores: tuple[float | None, ...]
    flipped: bool
    crossed: bool
    missing: bool
    score_range: float
    score_sd: float


@dataclass(frozen=True)
class Report:
    question_id: str
    version: str
    draws: int
    reorder: bool
    batch: int
    seed: int
    items: int
    requests: int
    failures: int
    flips: int
    crossings: int
    missing: int
    max_range: float
    mean_sd: float
    flip_upper: float
    invariant: bool
    spreads: tuple[Spread, ...]

    def as_dict(self) -> dict:
        data = asdict(self)
        data.pop("spreads")
        return data


def _ask_safely(ask: Ask, question: Question, items: list[Item]) -> tuple[list, bool]:
    """(answers aligned to ``items``, failed). A crash or a wrong-length reply is a failure."""
    try:
        raw = ask(question, items)
    except Exception as exc:  # noqa: BLE001 - a failed request is missing data, receipted as such
        logger.info("permutation check: ask failed (%s)", type(exc).__name__)
        return [None] * len(items), True
    if not isinstance(raw, list) or len(raw) != len(items):
        return [None] * len(items), raw is not None
    return [a if isinstance(a, TierAnswer) else None for a in raw], False


def _spread(question: Question, subject_id: str, answers: list[TierAnswer | None]) -> Spread:
    verdicts = tuple(a.verdict if a else None for a in answers)
    scores = tuple(float(a.confidence) if a else None for a in answers)
    present = [s for s in scores if s is not None]
    cuts = {question.threshold(v) for v in verdicts if v is not None} | set(GRID)
    crossed = any(len({s >= t for s in present}) > 1 for t in cuts)
    mean = sum(present) / len(present) if present else 0.0
    sd = math.sqrt(sum((s - mean) ** 2 for s in present) / (len(present) - 1)) \
        if len(present) > 1 else 0.0
    return Spread(subject_id, verdicts, scores, flipped=len(set(verdicts)) > 1,
                  crossed=crossed, missing=None in verdicts,
                  score_range=(max(present) - min(present)) if present else 0.0, score_sd=sd)


def check(ask: Ask, question: Question, items: Sequence[Item], *, draws: int = 3,
          batch: int = MAX_BATCH, seed: int = 0, reorder: bool = True) -> Report:
    """Ask ``items`` in ``draws`` shuffles and report how much the answers move."""
    if draws < 2 or batch < 1:
        raise ValueError("need at least two draws and a batch of at least one")
    n = len(items)
    per_item: list[list[TierAnswer | None]] = [[] for _ in range(n)]
    requests = failures = 0
    for d in range(draws):
        order = list(range(n))
        random.Random(seed * 1_000_003 + (d if reorder else 0)).shuffle(order)
        for start in range(0, n, batch):
            chunk = order[start:start + batch]
            answers, failed = _ask_safely(ask, question, [items[i] for i in chunk])
            requests += 1
            failures += failed
            for i, answer in zip(chunk, answers):
                per_item[i].append(answer)
    spreads = tuple(_spread(question, items[i].subject_id, per_item[i]) for i in range(n))
    flips = sum(s.flipped for s in spreads)
    crossings = sum(s.crossed for s in spreads)
    missing = sum(s.missing for s in spreads)
    invariant = n >= N_MIN and flips == crossings == missing == 0
    return Report(
        question.question_id, question.version, draws, reorder, batch, seed, n, requests, failures,
        flips, crossings, missing,
        max_range=max((s.score_range for s in spreads), default=0.0),
        mean_sd=sum(s.score_sd for s in spreads) / n if n else 0.0,
        flip_upper=clopper_pearson_upper(flips, n, DELTA), invariant=invariant, spreads=spreads)


__all__ = ["ALPHA", "DELTA", "N_MIN", "Report", "Spread", "check"]
