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
    #: True only for a model tier's closing verdict (M-02). A retiring verdict and a
    #: deterministic rule's closing verdict (D-35) do not need it; a rule's miss
    #: rate is still reported in false_keep_upper.
    base_rate_required: bool = False
    #: The grid threshold the fixed sequence began at (W0-11); None when none was testable.
    start: float | None = None

    @property
    def beats_base_rate(self) -> bool:
        if not self.base_rate_required:
            return True
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


def _holds_now(errors: Sequence[bool], log_error: float, log_clean: float, goal: float) -> bool:
    """Whether E_n is at or above 1/delta after the last label (W0-12).

    {E_n >= 1/delta now} is inside {sup E_n >= 1/delta}, so acting only while it
    holds keeps Ville's bound, and stops as soon as new errors pull E_n back.
    """
    return sum(log_error if w else log_clean for w in errors) >= goal


def first_testable_start(scores: Sequence[float], *, alpha: float, delta: float,
                         grid: Sequence[float] = GRID) -> float | None:
    """The strictest grid threshold that selects enough items for a clean run to certify.

    A threshold selecting fewer items than a clean run needs can never certify, and
    testing it first would stop the sequence at once. The choice depends on the scores
    only, never the labels. The gate stores it at the cell's first test (W0-11).
    """
    p1 = alpha / 5.0
    floor = math.ceil(math.log(1.0 / delta) / math.log((1 - p1) / (1 - alpha)))
    for t in sorted(set(grid), reverse=True):
        if sum(1 for s in scores if s >= t) >= floor:
            return t
    return None


def eprocess_certificate(scores: Sequence[float], wrong: Sequence[bool], *, alpha: float,
                         delta: float, grid: Sequence[float] = GRID,
                         start: float | None = None) -> Certificate:
    """Anytime-valid certificate that the error rate at a threshold is below alpha.

    ``scores``/``wrong`` must be in the order the labels arrived. For each grid
    threshold (strictest first), over the items scoring >= it:

        E_n = prod_i (p1/alpha)^X_i * ((1-p1)/(1-alpha))^(1-X_i),   p1 = alpha/5

    is a non-negative supermartingale whenever the true error rate is >= alpha,
    so by Ville's inequality P(sup_n E_n >= 1/delta) <= delta. A threshold is
    certified while E_n stands at or above 1/delta after the latest label (not
    "ever reached it", W0-12), with no penalty for looking after every label. At
    alpha = delta = 0.05 a clean run needs 73 labels, 112 with one early error,
    151 with two. Fixed-sequence testing across the grid: the sequence begins at
    ``start`` and goes to ever looser thresholds, stopping at the first that is not
    certified. ``start`` is fixed once per cell, from the scores at its first test
    (``first_testable_start``; None here means "choose it now"), so the first test
    never moves as counts grow and the family-wise error stays at delta (W0-11); a
    stricter threshold that becomes testable later is never added in front, so a
    certificate at a looser one does not flap. Revocation: on the labels after the
    first crossing, a second e-process for "error rate <= alpha" against 2*alpha;
    reaching 1/delta revokes it for good.
    """
    if len(scores) != len(wrong):
        raise ValueError("scores and wrong must align")
    if not (0.0 < alpha < 0.5 and 0.0 < delta < 1.0):
        raise ValueError("need 0 < alpha < 0.5 and 0 < delta < 1")
    p1 = alpha / 5.0
    goal = math.log(1.0 / delta)
    up = (math.log(p1 / alpha), math.log((1 - p1) / (1 - alpha)))
    down = (math.log(2.0), math.log((1 - 2 * alpha) / (1 - alpha)))
    if start is None:
        start = first_testable_start(scores, alpha=alpha, delta=delta, grid=grid)
    if start is None:
        return Certificate(None, alpha, delta, None, False, ())
    chosen, at, revoked, tested = None, None, False, []
    for t in (g for g in sorted(set(grid), reverse=True) if g <= start):
        seq = [bool(w) for s, w in zip(scores, wrong) if s >= t]
        cross = _crossing(seq, up[0], up[1], goal)
        if cross is not None and _crossing(seq[cross:], down[0], down[1], goal) is not None:
            revoked, cross = True, None
        if cross is not None and not _holds_now(seq, up[0], up[1], goal):
            cross = None  # it crossed once, but the labels since have pulled it back
        tested.append((t, len(seq), sum(seq), cross is not None))
        if cross is None:
            break
        chosen, at = t, cross
    return Certificate(chosen, alpha, delta, at, revoked, tuple(tested), start=start)


#: The tier's inbox: every pair drawn for the inbox sample (one keyed draw per
#: pair, W0-05: a per-receipt draw over-samples pairs several tiers answered)
#: that reached this tier, once per pair hash, blindly audited. The base rate.
_INBOX = """
SELECT r.inputs_hash, l.label FROM judge_receipts r
  JOIN judge_labels l ON l.receipt_id = r.receipt_id
 WHERE r.profile_id = ? AND r.question_id = ? AND r.question_version = ? AND r.population = ?
   AND r.tier_index >= (SELECT COALESCE(MIN(t.tier_index), 0) FROM judge_receipts t
                         WHERE t.profile_id = r.profile_id AND t.tier = ? AND t.model = ?)
   AND r.inbox_selected = 1 AND l.source = 'audit' AND l.blind = 1
"""

#: Every drawn receipt of the cell, in the order it was decided, with its blind
#: audit labels if any (W0-03). The labeller's order and choice never matter:
#: the gate reads the drawn sequence and stops at the first unlabelled receipt.
_DRAWN = """
SELECT r.receipt_id, r.confidence, l.label, r.inputs_hash FROM judge_receipts r
  LEFT JOIN judge_labels l ON l.receipt_id = r.receipt_id
       AND l.source = 'audit' AND l.blind = 1
 WHERE r.profile_id = ? AND r.question_id = ? AND r.question_version = ?
   AND r.tier = ? AND r.model = ? AND r.verdict = ? AND r.population = ?
   AND r.batching_regime = ? AND r.audit_selected = 1 AND r.decided_at > ?
 ORDER BY r.decided_at, r.receipt_id
"""

#: A labeller who cannot decide says so; it counts as an error, never as a skip.
CANNOT_TELL = "cannot_tell"
#: Regimes whose labels are exchangeable within the cell (W0-04): a batch can be
#: wrong as a whole, so its items are not independent trials.
CERTIFIABLE_REGIMES = frozenset({"single", "unbatched"})


def _first_per_pair(rows: list[tuple]) -> list[tuple]:
    """One drawn receipt per pair in the cell: the first (W0-26). A pair re-asked because a
    tier became available gets new receipts at tiers that already answered; each is a fresh
    audit candidate, but the pair is one trial, so only its first drawn receipt counts."""
    first: dict[str, str] = {}
    return [row for row in rows if first.setdefault(row[3], row[0]) == row[0]]


def _drawn_prefix(rows: list[tuple]) -> list[tuple[float, frozenset[str]]]:
    """(score, labels) per drawn receipt, in decision order, up to the first
    receipt with no label yet."""
    out: list[tuple[float, set[str]]] = []
    index: dict[str, int] = {}
    for rid, conf, label, _pair in rows:
        if rid not in index:
            index[rid] = len(out)
            out.append((_score(conf), set()))
        if label is not None:
            out[index[rid]][1].add(label)
    prefix = []
    for score, labels in out:
        if not labels:
            break
        prefix.append((score, frozenset(labels)))
    return prefix


def _score(confidence: float | None) -> float:
    """A deterministic tier's decisions (no score) sit at the upper bound."""
    return 1.0 if confidence is None else float(confidence)


@dataclass(frozen=True)
class Permit:
    """A decision the gate lets act, and the stored certificate it relied on."""

    cert_id: str


class ActingGate:
    """Which decisions may change memory, from this store's blind audit labels.

    Key: (profile, question version, tier, model, verdict, population, batching
    regime). Only
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
        self._cache: dict[tuple[str, str, str, str, str], Certificate] = {}
        self._drawn_ids: dict[tuple[str, str, str, str, str], list[str]] = {}
        self._recorded: set[tuple[str, str, str, str, str]] = set()

    def certificate(self, tier: str, model: str, verdict: str, population: str,
                    regime: str) -> Certificate:
        key = (tier, model, verdict, population, regime)
        if key not in self._cache:
            from superlocalmemory.core import judge_certificates as records

            # R-26: after a drift void only decisions made since then count.
            since = records.labels_since(self._conn, self._cell(*key))
            rows = self._conn.execute(_DRAWN, (self._profile, self._question, self._version,
                                               tier, model, verdict, population,
                                               regime, since)).fetchall()
            rows = _first_per_pair(rows)  # W0-26
            drawn = _drawn_prefix(rows)
            self._drawn_ids[key] = list(dict.fromkeys(r[0] for r in rows))[:len(drawn)]
            scores = [score for score, _ in drawn]
            # Disagreeing labellers or "cannot tell" count as an error (conservative).
            wrong = [len(labels) > 1 or CANNOT_TELL in labels
                     or is_wrong(verdict, next(iter(labels))) for _, labels in drawn]
            start = records.start_of(self._conn, self._cell(*key))  # W0-11
            if verdict == RETIRING:
                cert = eprocess_certificate(scores, wrong, alpha=self._alpha_for(verdict),
                                            delta=self._delta, start=start)
            elif regime == "unbatched":
                cert = self._rule_closing(scores, wrong, verdict, start)
            else:
                cert = self._closing(scores, wrong, verdict, tier, model, population, start)
            self._cache[key] = cert
        return self._cache[key]

    def _cell(self, tier: str, model: str, verdict: str, population: str, regime: str):
        from superlocalmemory.core.judge_certificates import Cell

        return Cell(self._profile, self._question, self._version, tier, model, verdict,
                    population, regime)

    @staticmethod
    def _method(verdict: str, regime: str) -> str:
        if verdict == RETIRING:
            return "eprocess"
        return "rule-closing-D35" if regime == "unbatched" else "eprocess+base-rate-M02"

    def _record(self, tier: str, model: str, verdict: str, population: str,
                regime: str) -> Certificate:
        """I-14 / R-26: store the certificate's issue or void, and void it on drift
        (M-07); returns the certificate that holds after that."""
        from superlocalmemory.core import judge_certificates as records

        key = (tier, model, verdict, population, regime)
        cert = self.certificate(*key)
        if key in self._recorded:
            return cert
        self._recorded.add(key)
        cell = self._cell(*key)
        if cert.start is not None and records.fix_start(self._conn, cell, cert.start) != cert.start:
            self._cache.pop(key, None)  # another writer fixed the start first: it stands
            cert = self.certificate(*key)
        records.reconcile(self._conn, cell, cert, method=self._method(verdict, regime),
                          receipt_ids=self._drawn_ids.get(key, []))
        if cert.acts:
            audited = [_score(c) for (c,) in self._conn.execute(
                "SELECT confidence FROM judge_receipts WHERE receipt_id IN (%s)"
                % ",".join("?" * len(self._drawn_ids.get(key, []))),
                tuple(self._drawn_ids.get(key, [])))] if self._drawn_ids.get(key) else []
            drift = records.check_drift(self._conn, cell, audited, score_of=_score)
            if drift.shifted and records.void_for_drift(self._conn, cell):
                self._cache.pop(key, None)
                cert = self.certificate(*key)  # only labels since the void count now
        return cert

    def _rule_closing(self, scores: list[float], wrong: list[bool], verdict: str,
                      start: float | None = None) -> Certificate:
        """D-35: a deterministic rule closes on the e-process at alpha_close over its
        own drawn decisions; its miss rate is reported beside it, not compared."""
        cert = eprocess_certificate(scores, wrong, alpha=self._alpha_for(verdict),
                                    delta=self._delta, start=start)
        upper = (clopper_pearson_upper(sum(wrong), len(wrong), self._delta)
                 if wrong else None)
        return replace(cert, false_keep_upper=upper)

    def _closing(self, scores: list[float], wrong: list[bool], verdict: str, tier: str,
                 model: str, population: str, start: float | None = None) -> Certificate:
        """M-02 for a model tier: a closing verdict must also keep fewer replacements
        than closing everything would. The e-process and the two bounds each get
        delta/3, so the whole statement holds at 1 - delta (W0-07; U and L are
        fixed-n bounds, so they are valid at a stated check, not under every look)."""
        third = self._delta / 3
        cert = replace(eprocess_certificate(scores, wrong, alpha=self._alpha_for(verdict),
                                            delta=third, start=start), base_rate_required=True)
        if cert.threshold is None:
            return cert
        picked = [w for s, w in zip(scores, wrong) if s >= cert.threshold]
        upper = clopper_pearson_upper(sum(picked), len(picked), third)
        return replace(cert, false_keep_upper=upper,
                       base_rate_lower=self._base_rate_lower(tier, model, population, third))

    def _base_rate_lower(self, tier: str, model: str, population: str, delta: float) -> float:
        labels: dict[str, set[str]] = {}
        for h, label in self._conn.execute(_INBOX, (self._profile, self._question,
                                                    self._version, population, tier, model)):
            labels.setdefault(h, set()).add(label)
        # Only a unanimous "replaces" counts: the lower bound stays conservative.
        replaced = sum(1 for got in labels.values() if got == {RETIRING})
        return clopper_pearson_lower(replaced, len(labels), delta)

    def permit(self, tier: str, model: str, verdict: str, confidence: float | None, *,
               population: str, regime: str,
               routing_threshold: float | None = None) -> Permit | None:
        """The open certificate this decision may act on, or None (I-10, I-14, W0-13).

        The caller puts ``Permit.cert_id`` in the acted receipt's ``action_ref``. No open
        certificate row for the cell (a store without the table, or a record that was
        not written) is no permission, whatever the labels say.
        """
        from superlocalmemory.core import judge_certificates as records

        if regime not in CERTIFIABLE_REGIMES:
            return None  # W0-04: batched labels are not independent trials
        cert = self._record(tier, model, verdict, population, regime)
        if not (cert.acts and _score(confidence) >= cert.threshold):
            return None
        # W0-10: the acted set is {score >= max(t*, r)}; it is bounded only when r
        # is at or below t* or is itself a threshold the certified chain passed.
        if (routing_threshold is not None and routing_threshold > cert.threshold
                and routing_threshold not in {t for t, _n, _e, ok in cert.tested if ok}):
            return None
        row = records.open_record(self._conn, self._cell(tier, model, verdict, population, regime))
        return None if row is None else Permit(str(row[0]))

    def allows(self, tier: str, model: str, verdict: str, confidence: float | None, *,
               population: str, regime: str, routing_threshold: float | None = None) -> bool:
        return self.permit(tier, model, verdict, confidence, population=population,
                           regime=regime, routing_threshold=routing_threshold) is not None


__all__ = ["ActingGate", "CANNOT_TELL", "CERTIFIABLE_REGIMES", "Certificate", "GRID", "RETIRING",
           "Permit", "Selection", "ThresholdTest", "binom_cdf",
           "clean_decisions_needed", "clopper_pearson_lower", "clopper_pearson_upper", "cohen_kappa",
           "eprocess_certificate", "first_testable_start", "is_wrong", "select_threshold"]
