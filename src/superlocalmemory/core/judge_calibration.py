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
from dataclasses import dataclass, replace
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
    ``grid`` defaults to every distinct score, strictest first. This is the
    fixed-n test, for evaluation studies; acting uses ``eprocess_certificate``.
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


#: Thresholds tested strictest first, fixed in advance (not chosen from labels).
GRID = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50)

#: The one verdict that retires a memory.
RETIRING = "replaces"


def is_wrong(verdict: str, label: str) -> bool:
    """The costly error for each verdict.

    Acting on ``replaces`` is wrong when the truth is anything else (a false
    retirement). Acting on any other verdict closes the question without
    retiring, so it is wrong only when the truth was a replacement (a false
    keep). Confusing "same" with "both true" costs nothing here.
    """
    return label != RETIRING if verdict == RETIRING else label == RETIRING


@dataclass(frozen=True)
class Certificate:
    threshold: float | None
    alpha: float
    delta: float
    certified_at: int | None   # labels needed before E_n first reached 1/delta
    revoked: bool
    tested: tuple[tuple[float, int, int, bool], ...]  # (threshold, n, errors, certified)
    #: Closing verdicts only (M-02): U(false keep | closed) and L(replacement rate
    #: of the tier's inbox), each at delta/2; None where not computed.
    false_keep_upper: float | None = None
    base_rate_lower: float | None = None

    @property
    def beats_base_rate(self) -> bool:
        if self.false_keep_upper is None and self.base_rate_lower is None:
            return True  # a retiring verdict: the condition does not apply
        return (self.false_keep_upper is not None and self.base_rate_lower is not None
                and self.false_keep_upper < self.base_rate_lower)

    @property
    def acts(self) -> bool:
        return self.threshold is not None and self.beats_base_rate


def _crossing(errors: Sequence[bool], log_error: float, log_clean: float,
              goal: float) -> int | None:
    log_e = 0.0
    for i, wrong in enumerate(errors, 1):
        log_e += log_error if wrong else log_clean
        if log_e >= goal:
            return i
    return None


def eprocess_certificate(scores: Sequence[float], wrong: Sequence[bool], *, alpha: float,
                         delta: float, grid: Sequence[float] = GRID) -> Certificate:
    """Anytime-valid certificate that the error rate at a threshold is below alpha.

    ``scores``/``wrong`` must be in the order the labels arrived. For each grid
    threshold (strictest first), over the items scoring >= it:

        E_n = prod_i (p1/alpha)^X_i * ((1-p1)/(1-alpha))^(1-X_i),   p1 = alpha/5

    is a non-negative supermartingale whenever the true error rate is >= alpha,
    so by Ville's inequality P(sup_n E_n >= 1/delta) <= delta: certify the first
    time E_n >= 1/delta, with no penalty for looking after every label. At
    alpha = delta = 0.05 that is 73 clean labels, 112 with one error, 151 with
    two. Fixed-sequence testing across the grid keeps the family-wise error at
    delta. Revocation: on the labels after certification, a second e-process for
    "error rate <= alpha" against 2*alpha; reaching 1/delta revokes it.
    """
    if len(scores) != len(wrong):
        raise ValueError("scores and wrong must align")
    if not (0.0 < alpha < 0.5 and 0.0 < delta < 1.0):
        raise ValueError("need 0 < alpha < 0.5 and 0 < delta < 1")
    p1 = alpha / 5.0
    goal = math.log(1.0 / delta)
    up = (math.log(p1 / alpha), math.log((1 - p1) / (1 - alpha)))
    down = (math.log(2.0), math.log((1 - 2 * alpha) / (1 - alpha)))
    # A threshold selecting fewer items than a clean run needs can never certify;
    # testing it first would stop the sequence at once. Skipping it depends on the
    # scores only, never the labels, so the test stays valid.
    floor = math.ceil(goal / up[1])
    chosen, at, revoked, tested = None, None, False, []
    for t in sorted(set(grid), reverse=True):
        seq = [bool(w) for s, w in zip(scores, wrong) if s >= t]
        if len(seq) < floor and chosen is None:
            continue
        cross = _crossing(seq, up[0], up[1], goal)
        if cross is not None and _crossing(seq[cross:], down[0], down[1], goal) is not None:
            revoked, cross = True, None
        tested.append((t, len(seq), sum(seq), cross is not None))
        if cross is None:
            break
        chosen, at = t, cross
    return Certificate(chosen, alpha, delta, at, revoked, tuple(tested))


#: The tier's inbox: every drawn, blindly audited pair that reached this tier
#: (a receipt at its index or above), once per pair hash. Used for the base rate.
_INBOX = """
SELECT r.inputs_hash, l.label FROM judge_receipts r
  JOIN judge_labels l ON l.receipt_id = r.receipt_id
 WHERE r.profile_id = ? AND r.question_id = ? AND r.question_version = ? AND r.population = ?
   AND r.tier_index >= (SELECT COALESCE(MIN(t.tier_index), 0) FROM judge_receipts t
                         WHERE t.profile_id = r.profile_id AND t.tier = ? AND t.model = ?)
   AND r.audit_selected = 1 AND l.source = 'audit' AND l.blind = 1
"""

_AUDITED = """
SELECT r.receipt_id, r.confidence, l.label, l.labelled_at FROM judge_receipts r
  JOIN judge_labels l ON l.receipt_id = r.receipt_id
 WHERE r.profile_id = ? AND r.question_id = ? AND r.question_version = ?
   AND r.tier = ? AND r.model = ? AND r.verdict = ? AND r.population = ?
   AND r.audit_selected = 1 AND l.source = 'audit' AND l.blind = 1
 ORDER BY l.labelled_at, r.receipt_id
"""


def _score(confidence: float | None) -> float:
    """A deterministic tier's decisions (no score) sit at the upper bound."""
    return 1.0 if confidence is None else float(confidence)


class ActingGate:
    """Which decisions may change memory, from this store's blind audit labels.

    Key: (profile, question version, tier, model, verdict, population). Only
    labels with ``source = 'audit'`` and ``blind = 1`` count; rollbacks and
    reviews raise alarms but never certify (they are not missing at random).
    A receipt counts once, at its first label time; when its audit labels
    disagree it counts as an error. A verdict acts only when
    ``eprocess_certificate`` holds and the confidence reaches its threshold.
    There is no shipped or pooled prior: every store starts from zero (D-09).
    """

    def __init__(self, conn: Any, *, profile_id: str, question_id: str, version: str,
                 alpha_for: Any = None, delta: float | None = None) -> None:
        from superlocalmemory.core import risk_policy

        self._conn, self._profile = conn, profile_id
        self._question, self._version = question_id, version
        self._alpha_for = alpha_for or risk_policy.alpha_for
        self._delta = risk_policy.DELTA if delta is None else delta
        self._cache: dict[tuple[str, str, str, str], Certificate] = {}

    def certificate(self, tier: str, model: str, verdict: str, population: str) -> Certificate:
        key = (tier, model, verdict, population)
        if key not in self._cache:
            rows = self._conn.execute(_AUDITED, (self._profile, self._question, self._version,
                                                 tier, model, verdict, population)).fetchall()
            per: dict[str, tuple[float, set[str], str]] = {}
            for rid, conf, label, at in rows:
                # A rule has no score (NULL, R-36): all its decisions are one class,
                # certified together at the top of the grid.
                score, labels, first = per.get(rid, (_score(conf), set(), at))
                per[rid] = (score, labels | {label}, min(first, at))
            ordered = sorted(per.values(), key=lambda v: v[2])
            scores = [v[0] for v in ordered]
            wrong = [len(v[1]) > 1 or is_wrong(verdict, next(iter(v[1]))) for v in ordered]
            if verdict == RETIRING:
                cert = eprocess_certificate(scores, wrong, alpha=self._alpha_for(verdict),
                                            delta=self._delta)
            else:
                cert = self._closing(scores, wrong, verdict, tier, model, population)
            self._cache[key] = cert
        return self._cache[key]

    def _closing(self, scores: list[float], wrong: list[bool], verdict: str, tier: str,
                 model: str, population: str) -> Certificate:
        """M-02: a closing verdict must also keep fewer replacements than closing
        everything would. The e-process and the two bounds each get delta/2."""
        half = self._delta / 2
        cert = eprocess_certificate(scores, wrong, alpha=self._alpha_for(verdict), delta=half)
        if cert.threshold is None:
            return cert
        picked = [w for s, w in zip(scores, wrong) if s >= cert.threshold]
        upper = clopper_pearson_upper(sum(picked), len(picked), half)
        return replace(cert, false_keep_upper=upper,
                       base_rate_lower=self._base_rate_lower(tier, model, population, half))

    def _base_rate_lower(self, tier: str, model: str, population: str, delta: float) -> float:
        labels: dict[str, set[str]] = {}
        for h, label in self._conn.execute(_INBOX, (self._profile, self._question,
                                                    self._version, population, tier, model)):
            labels.setdefault(h, set()).add(label)
        # Only a unanimous "replaces" counts: the lower bound stays conservative.
        replaced = sum(1 for got in labels.values() if got == {RETIRING})
        return clopper_pearson_lower(replaced, len(labels), delta)

    def allows(self, tier: str, model: str, verdict: str, confidence: float | None, *,
               population: str) -> bool:
        cert = self.certificate(tier, model, verdict, population)
        return cert.acts and _score(confidence) >= cert.threshold


__all__ = ["ActingGate", "Certificate", "GRID", "RETIRING", "Selection", "ThresholdTest", "binom_cdf",
           "clean_decisions_needed", "clopper_pearson_lower", "clopper_pearson_upper", "cohen_kappa",
           "eprocess_certificate", "is_wrong", "select_threshold"]
