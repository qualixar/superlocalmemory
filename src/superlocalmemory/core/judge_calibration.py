# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Thresholds the judge ladder may act on, chosen with a stated guarantee.

A judge's confidence is a model score, not a probability that it is right. So
no threshold is hand-picked: given labelled decisions (a person's review, a
rollback, a careful labelling run), ``select_threshold`` returns the most
permissive threshold for which

    P( error rate among the items the threshold would act on  >  alpha )  <=  delta

holds, or None when the labels cannot support any threshold (then nothing acts).

Method: fixed-sequence testing (Learn-then-Test; Angelopoulos, Bates, Candes,
Jordan and Lei, 2021). Candidate thresholds are tested from strictest to
loosest. At each, H0 "the error rate among the items with score >= t exceeds
alpha" is rejected when the exact one-sided Clopper-Pearson upper bound on that
rate, at level delta, is <= alpha. Testing stops at the first threshold that
cannot be rejected; because the order is fixed in advance, the family-wise
error is <= delta with no multiplicity correction. Which items a threshold
selects depends only on the scores, never on the labels, so given the
selection the error count is exactly binomial and the bound is exact.

There is no minimum-label rule: with few labels no bound gets below alpha, and
the function returns None. With zero errors among n acted-on items the bound is
1 - delta**(1/n), so alpha = 0.05 at delta = 0.05 needs n >= 59 clean decisions.

Pure Python (``math`` only): the daemon does not need SciPy for this.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), computed in log space."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
             + i * log_p + (n - i) * log_q for i in range(k + 1)]
    top = max(terms)
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """Exact one-sided upper bound U: P(Bin(n, U) <= k) = delta (1.0 when k == n)."""
    if n <= 0:
        return 1.0
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    if k == n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(80):  # bisection to well below 1e-12
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def clopper_pearson_lower(k: int, n: int, delta: float) -> float:
    """Exact one-sided lower bound on a success rate (k successes in n)."""
    return 1.0 - clopper_pearson_upper(n - k, n, delta) if n > 0 else 0.0


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Agreement between two labellers beyond chance (1 = perfect, 0 = chance)."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal, non-empty label sequences")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    expected = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    return 1.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)


@dataclass(frozen=True)
class ThresholdTest:
    threshold: float
    acted: int
    errors: int
    upper_error: float
    rejected: bool


@dataclass(frozen=True)
class Selection:
    threshold: float | None
    alpha: float
    delta: float
    trail: tuple[ThresholdTest, ...]

    @property
    def acts(self) -> bool:
        return self.threshold is not None


def select_threshold(scores: Sequence[float], wrong: Sequence[bool], *, alpha: float,
                     delta: float, grid: Sequence[float] | None = None) -> Selection:
    """The loosest threshold whose acted-on error rate is <= alpha w.p. >= 1 - delta.

    ``scores`` are the judge's confidences on items it would act on at some
    threshold; ``wrong[i]`` says acting on item i would have been an error.
    ``grid`` defaults to every distinct score, strictest first.
    """
    if len(scores) != len(wrong):
        raise ValueError("scores and wrong must align")
    if not (0.0 < alpha < 1.0 and 0.0 < delta < 1.0):
        raise ValueError("alpha and delta must be in (0, 1)")
    if grid is None:
        # Start at the n_min-th largest score: a threshold acting on fewer items can
        # never reach alpha, and testing it first would stop the sequence at once.
        # Chosen from the scores alone, so the test stays valid.
        ranked = sorted(scores, reverse=True)
        need = clean_decisions_needed(alpha, delta)
        grid = ranked[need - 1:] if len(ranked) >= need else []
    points = sorted(set(grid), reverse=True)
    trail: list[ThresholdTest] = []
    chosen: float | None = None
    for t in points:
        acted = [w for s, w in zip(scores, wrong) if s >= t]
        errors = sum(bool(w) for w in acted)
        upper = clopper_pearson_upper(errors, len(acted), delta)
        rejected = len(acted) > 0 and upper <= alpha
        trail.append(ThresholdTest(t, len(acted), errors, upper, rejected))
        if not rejected:
            break
        chosen = t
    return Selection(chosen, alpha, delta, tuple(trail))


def clean_decisions_needed(alpha: float, delta: float) -> int:
    """Smallest n with zero errors whose upper bound is <= alpha."""
    return math.ceil(math.log(delta) / math.log1p(-alpha))


_LABELLED = """
SELECT r.confidence, l.label FROM judge_receipts r
  JOIN judge_labels l ON l.receipt_id = r.receipt_id
 WHERE r.profile_id = ? AND r.question_id = ? AND r.question_version = ?
   AND r.tier = ? AND r.model = ? AND r.verdict = ?
 ORDER BY l.labelled_at, r.receipt_id
"""

#: The one verdict that retires a memory.
RETIRING = "replaces"


def checkpoints(alpha: float, delta: float, upto: int) -> list[int]:
    """Label counts at which a gate may be re-evaluated (geometric from n_min)."""
    first = clean_decisions_needed(alpha, delta)
    out, n = [], float(first)
    while math.ceil(n) <= upto:
        out.append(math.ceil(n))
        n *= 1.25
    return out


def is_wrong(verdict: str, label: str) -> bool:
    """The costly error for each verdict.

    Acting on ``replaces`` is wrong when the truth is anything else (a false
    retirement). Acting on any other verdict closes the question without
    retiring, so it is wrong only when the truth was a replacement (a false
    keep: the stale memory stays). Confusing "same" with "both true" costs
    nothing here and is not counted.
    """
    return label != RETIRING if verdict == RETIRING else label == RETIRING


class ActingGate:
    """Which (tier, model, verdict, confidence) may change memory, from the store's labels.

    For each verdict, the labelled receipts of that profile, question version,
    tier and model are read in the order they were labelled. The gate is
    evaluated only at scheduled label counts n_1 < n_2 < ... (``checkpoints``),
    on the first n_t labels, at level delta_t = delta * 6 / (pi^2 t^2): the
    deltas sum to at most delta, so re-checking as labels arrive stays valid
    (optional stopping). A verdict acts when the threshold chosen at the latest
    checkpoint exists and the confidence reaches it. No labels: nothing acts.
    """

    def __init__(self, conn: Any, *, profile_id: str, question_id: str, version: str,
                 alpha: float = 0.05, delta: float = 0.05) -> None:
        self._conn, self._profile = conn, profile_id
        self._question, self._version = question_id, version
        self._alpha, self._delta = alpha, delta
        self._cache: dict[tuple[str, str, str], Selection | None] = {}

    def selection(self, tier: str, model: str, verdict: str) -> Selection | None:
        key = (tier, model, verdict)
        if key not in self._cache:
            rows = self._conn.execute(_LABELLED, (self._profile, self._question, self._version,
                                                  tier, model, verdict)).fetchall()
            marks = checkpoints(self._alpha, self._delta, len(rows))
            if not marks:
                self._cache[key] = None
            else:
                t, n = len(marks), marks[-1]
                delta_t = self._delta * 6.0 / (math.pi ** 2 * t * t)
                head = rows[:n]
                self._cache[key] = select_threshold(
                    [float(r[0]) for r in head], [is_wrong(verdict, r[1]) for r in head],
                    alpha=self._alpha, delta=delta_t)
        return self._cache[key]

    def allows(self, tier: str, model: str, verdict: str, confidence: float) -> bool:
        chosen = self.selection(tier, model, verdict)
        return chosen is not None and chosen.threshold is not None and confidence >= chosen.threshold


__all__ = ["ActingGate", "RETIRING", "Selection", "ThresholdTest", "binom_cdf", "checkpoints",
           "clean_decisions_needed", "clopper_pearson_lower", "clopper_pearson_upper", "cohen_kappa",
           "is_wrong", "select_threshold"]
