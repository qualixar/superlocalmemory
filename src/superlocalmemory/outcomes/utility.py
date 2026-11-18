# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""How often is each memory used when it is delivered? (W3-5, M-13)

A memory's *use rate* is the chance that, once delivered, the agent acts on it
within the outcome window (the ``used`` signals of M-12: marker echoed in an
edit or tool call, cited, named in an explicit success). It is descriptive, not
causal: whether a memory *helps* is M-12's question, answerable only in
aggregate (F-47). Nothing here changes memory; the harm gate (M-14) is separate.

Most memories are delivered a handful of times, so a raw rate (0 of 1, 1 of 1)
says almost nothing. The estimate is **hierarchical Beta, empirical Bayes**:

* every delivered fact's rate is drawn from Beta(kappa * mu_g, kappa * (1 - mu_g))
  for its kind g (``memory_kind``, else ``fact_type``);
* kinds' means are drawn from Beta(lambda * mu_0, lambda * (1 - mu_0)): mu_0 and
  lambda are fitted by maximum marginal likelihood (beta-binomial) over the
  kinds' pooled counts, and a kind's mean is pooled toward the store's,
  mu_g = (U_g + lambda mu_0) / (N_g + lambda), so a kind seen a few times stays
  near the store-wide rate (with one kind, mu_g and kappa are fitted jointly);
* the within-kind concentration kappa is then fitted by maximum marginal
  likelihood over the facts with each kind's mean held at mu_g, so differences
  between kinds are not mistaken for noise within one;
* a fact's posterior is Beta(kappa mu_g + u, kappa (1 - mu_g) + n - u), reported
  with its mean and a central credible interval at ``effects.ALPHA``.

**Time decay.** A delivery ``age`` days old counts 2^(-age / HALF_LIFE_DAYS),
its use with it, so the rate follows what the agent does now. Deliveries whose
outcome window is still open are not counted (their use is not yet known).
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from superlocalmemory.outcomes import effects, usage

#: Descriptive: a delivery's weight halves every this many days.
HALF_LIFE_DAYS = 30.0
#: Method: bounds of the fitted concentration (kappa); its likelihood is flat
#: beyond them, so they only keep the optimiser finite.
KAPPA_BOUNDS = (1e-2, 1e4)

_DELIVERED = """
SELECT r.fact_id, d.delivered_at,
       EXISTS (SELECT 1 FROM injection_usage u
                WHERE u.delivery_id = r.delivery_id AND u.fact_id = r.fact_id
                  AND u.signal IN ({signals})
                  AND julianday(u.observed_at) <= julianday(d.delivered_at) + ? / 24.0),
       COALESCE(f.memory_kind, f.fact_type)
  FROM injection_receipts r
  JOIN injection_deliveries d ON d.delivery_id = r.delivery_id
  LEFT JOIN atomic_facts f ON f.fact_id = r.fact_id
 WHERE d.profile_id = ? AND r.delivered = 1
   AND julianday(d.delivered_at) + ? / 24.0 <= julianday(?)
"""


@dataclass(frozen=True)
class FactUtility:
    fact_id: str
    group: str
    deliveries: int
    uses: int
    #: Time-weighted counts that enter the estimate.
    weight: float
    weighted_uses: float
    mean: float
    low: float
    high: float


@dataclass(frozen=True)
class GroupRate:
    group: str
    facts: int
    deliveries: int
    uses: int
    mean: float


@dataclass(frozen=True)
class Utility:
    #: Store-wide (time-weighted) use rate, and the within-kind concentration
    #: kappa; None before two facts are delivered.
    mean: float | None
    concentration: float | None
    groups: tuple[GroupRate, ...]
    facts: tuple[FactUtility, ...]

    def most_used(self, k: int = 10) -> list[FactUtility]:
        return sorted(self.facts, key=lambda f: (-f.mean, -f.deliveries, f.fact_id))[:k]

    def rarely_used(self, k: int = 10) -> list[FactUtility]:
        """Facts confidently below the store's rate: the whole interval under it."""
        if self.mean is None:
            return []
        low = [f for f in self.facts if f.high < self.mean]
        return sorted(low, key=lambda f: (-f.deliveries, f.high, f.fact_id))[:k]

    def never_used(self) -> int:
        return sum(1 for f in self.facts if f.uses == 0)


EMPTY = Utility(None, None, (), ())


def _decay(delivered_at: str, as_of: datetime) -> float:
    try:
        at = datetime.fromisoformat(delivered_at)
    except (TypeError, ValueError):
        return 0.0
    if at.tzinfo is None:
        at = at.replace(tzinfo=UTC)
    age = max((as_of - at).total_seconds() / 86400.0, 0.0)
    return 2.0 ** (-age / HALF_LIFE_DAYS)


def _counts(conn: sqlite3.Connection, profile_id: str,
            as_of: datetime) -> dict[str, list]:
    """fact -> [group, deliveries, uses, weight, weighted uses]."""
    sql = _DELIVERED.format(signals=", ".join("?" * len(effects.USED_SIGNALS)))
    params = [*effects.USED_SIGNALS, effects.OUTCOME_WINDOW_H, profile_id,
              effects.OUTCOME_WINDOW_H, usage.stamp(as_of)]
    out: dict[str, list] = {}
    for fact_id, at, used, group in conn.execute(sql, params):
        w = _decay(at, as_of)
        row = out.setdefault(fact_id, [group or "unknown", 0, 0, 0.0, 0.0])
        row[1] += 1
        row[2] += int(bool(used))
        row[3] += w
        row[4] += w * bool(used)
    return out


def _loglik(mus, kappa: float, ns, us) -> float:
    """Beta-binomial marginal log-likelihood of (n, u) pairs whose rates are
    Beta(kappa mu, kappa (1 - mu)); binomial coefficients (free of the
    parameters) are dropped. Counts may be fractional (time weights)."""
    import numpy as np
    from scipy.special import betaln

    m, n, u = np.asarray(mus, float), np.asarray(ns, float), np.asarray(us, float)
    a, b = kappa * m, kappa * (1.0 - m)
    return float(np.sum(betaln(a + u, b + n - u) - betaln(a, b)))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _golden(f, lo: float, hi: float, tol: float = 1e-7) -> float:
    """The minimiser of a unimodal ``f`` on [lo, hi] by golden-section search.
    Pure Python, so no native optimiser runs (no LAPACK; see ``_solve`` notes in
    ``effects``)."""
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2.0


#: Routing only: the search interval for the prior mean on the logit scale.
_LOGIT_BOUNDS = (-12.0, 12.0)


def fit_prior(ns: list[float], us: list[float]) -> tuple[float, float] | None:
    """(mu, concentration) by maximum marginal likelihood; None with fewer than two points.

    Profile likelihood: for each concentration the best mean is found by a
    golden-section search, and the concentration by another one over that
    profile (the log likelihood of a Beta-binomial is unimodal in each)."""
    pts = [(n, u) for n, u in zip(ns, us, strict=True) if n > 0]
    if len(pts) < 2:
        return None
    ns, us = [p[0] for p in pts], [p[1] for p in pts]
    lo, hi = (math.log(b) for b in KAPPA_BOUNDS)

    def best_logit(log_kappa: float) -> float:
        kappa = math.exp(log_kappa)
        return _golden(lambda t: -_loglik([_sigmoid(t)] * len(ns), kappa, ns, us),
                       *_LOGIT_BOUNDS, tol=1e-6)

    def profile(log_kappa: float) -> float:
        t = best_logit(log_kappa)
        return -_loglik([_sigmoid(t)] * len(ns), math.exp(log_kappa), ns, us)

    log_kappa = _golden(profile, lo, hi, tol=1e-5)
    return _sigmoid(best_logit(log_kappa)), math.exp(log_kappa)


def fit_concentration(mus: list[float], ns: list[float], us: list[float]) -> float:
    """kappa by maximum marginal likelihood with each point's mean held fixed."""
    lo, hi = (math.log(b) for b in KAPPA_BOUNDS)
    return math.exp(_golden(lambda x: -_loglik(mus, math.exp(x), ns, us), lo, hi, tol=1e-6))


def _interval(a, b):
    """Central credible bounds at ``effects.ALPHA`` for arrays of Beta(a, b)."""
    from scipy.stats import beta

    half = effects.ALPHA / 2.0
    return beta.ppf(half, a, b), beta.ppf(1.0 - half, a, b)


def _kind_means(by_group: dict[str, list]) -> dict[str, float] | None:
    """Each kind's pooled mean, mu_g; None when there is nothing to pool."""
    names = [n for n, g in by_group.items() if g[3] > 0]
    if len(names) >= 2:
        prior = fit_prior([by_group[n][3] for n in names], [by_group[n][4] for n in names])
        if prior is not None:
            mu0, lam = prior
            return {n: (g[4] + lam * mu0) / (g[3] + lam) for n, g in by_group.items()}
    return None


def estimate(counts: dict[str, list]) -> Utility:
    """The hierarchical estimate from per-fact counts (see the module text)."""
    live = {f: c for f, c in counts.items() if c[3] > 0}
    if len(live) < 2:
        return EMPTY
    by_group: dict[str, list] = {}
    for c in live.values():
        g = by_group.setdefault(c[0], [0, 0, 0, 0.0, 0.0])
        for i in range(1, 5):
            g[i] += c[i]
        g[0] += 1
    means = _kind_means(by_group)
    if means is None:  # one kind: its mean and kappa together
        mu, kappa = fit_prior([c[3] for c in live.values()], [c[4] for c in live.values()])
        means = {name: mu for name in by_group}
    else:
        kappa = fit_concentration([means[c[0]] for c in live.values()],
                                  [c[3] for c in live.values()], [c[4] for c in live.values()])
    groups = tuple(GroupRate(name, g[0], g[1], g[2], means[name])
                   for name, g in sorted(by_group.items()))
    ids = list(live)
    a = [kappa * means[live[f][0]] + live[f][4] for f in ids]
    b = [kappa * (1.0 - means[live[f][0]]) + live[f][3] - live[f][4] for f in ids]
    lows, highs = _interval(a, b)
    facts = [FactUtility(f, live[f][0], live[f][1], live[f][2], live[f][3], live[f][4],
                         a[i] / (a[i] + b[i]), float(lows[i]), float(highs[i]))
             for i, f in enumerate(ids)]
    rate = sum(c[4] for c in live.values()) / sum(c[3] for c in live.values())
    return Utility(rate, kappa, groups, tuple(sorted(facts, key=lambda f: f.fact_id)))


def measure(conn: sqlite3.Connection, profile_id: str, *,
            as_of: datetime | None = None) -> Utility:
    """Every delivered fact's use rate on this profile; empty on a store without the ledger."""
    try:
        counts = _counts(conn, profile_id, as_of or datetime.now(UTC))
    except sqlite3.OperationalError:
        return EMPTY
    return estimate(counts)


__all__ = ["EMPTY", "HALF_LIFE_DAYS", "KAPPA_BOUNDS", "FactUtility", "GroupRate", "Utility",
           "estimate", "fit_concentration", "fit_prior", "measure"]
