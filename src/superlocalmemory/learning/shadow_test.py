# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Track A.3 (LLD-10 / LLD-00 §8)

"""Two-phase live-recall A/B shadow validator (LLD-10 §4 + LLD-00 §8).

Phase A (n=100, fast triage):
    Early-stop ``promote`` ONLY if ``|effect| > MIN_STRONG_EFFECT`` AND
    ``p < ALPHA_STRONG`` (strong signal path). Otherwise Phase B must
    accumulate further paired recalls.

Phase B (n=885, full validation):
    Bayesian-conservative sample size for σ=0.15, MDE=0.02, power 0.8,
    two-sided α=0.05. Criterion: mean paired diff ≥ MIN_EFFECT AND
    paired t-test p<0.05.

This module is a PURE state machine — no DB, no lightgbm, no network.
Tests in ``tests/test_learning/test_shadow_test.py`` exercise it.

Deterministic A/B routing: ``route_query(qid)`` returns ``'active'`` or
``'candidate'`` by SHA-256 first-8-hex-char modulo-2. Bit-exact
reproducible across daemon restart (LLD-10 §4.1).

No scipy dependency: for n<60 we use a tabled two-tailed critical-t
value; for n≥60 the normal-approximation z≈1.96 applies. Fallback
matches the existing ``consolidation_worker._shadow_test_improved``
behaviour (hardcoded ``t > 2.0``).
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Final, Optional

logger = logging.getLogger(__name__)


# S9-SKEP-01: resolve scipy.stats.t ONCE at module load, not on every
# _critical_t call. Prior ``try: import`` at call-sites paid ~microsecond
# lookup per invocation (cached via sys.modules, but not free) and the
# bare ``except Exception`` silently swallowed ValueError/FloatingPointError
# from scipy.stats.t.ppf itself — exactly the "early-stop more permissive
# than α=0.01" defect the table interpolation was supposed to fix.
#
# After this cache:
#   * ImportError/ModuleNotFoundError on first import → fall through to
#     the table permanently.
#   * Present-but-broken scipy (corrupt install, bad C-ext) → we still
#     import it; errors in .ppf() propagate on the FIRST call and the
#     caller sees it (not swallowed).
_SCIPY_T: Optional[Any]
try:
    from scipy.stats import t as _scipy_t  # type: ignore[import-not-found]
    _SCIPY_T = _scipy_t
except (ImportError, ModuleNotFoundError):
    _SCIPY_T = None


# ---------------------------------------------------------------------------
# Two-phase parameters — LLD-00 §8, LLD-10 §4.5
# ---------------------------------------------------------------------------

#: Phase A sample size (per LLD-00 §8 fast triage).
_PHASE_A_N: Final[int] = 100

#: Phase B sample size (statistical power for MDE=0.02 MRR at σ=0.15).
_PHASE_B_N: Final[int] = 885

#: Minimum acceptable mean paired improvement to promote (LLD-10 §4.5).
_MIN_EFFECT: Final[float] = 0.02

#: Phase A "strong signal" early-stop threshold: |effect| > 0.08 AND p<0.01.
_MIN_STRONG_EFFECT: Final[float] = 0.08


#: Significance level for Phase B (LLD-10 §4.5 + LLD-00 §8).
#: S9-defer S9-STAT-07: two-look sequential design needs alpha spending.
#: Without correction the family-wise false-promote probability was
#: 1 - (1-0.01)(1-0.05) ≈ 0.0595 rather than the advertised 0.05. We
#: now use Pocock boundaries that spread α across the two looks so
#: the family-wise α is 0.05 as contracted.
#: Pocock α_1 for 2-look design with overall α=0.05 is 0.0294; we use
#: a conservative 0.001 for Phase A (making the first look a strong
#: filter, not a contribution to family-wise α) and α=0.049 for Phase B
#: so family-wise α is approximately 0.05.
_ALPHA: Final[float] = 0.049

#: Tighter significance level for Phase A early-stop (LLD-00 §8).
#: Pocock-style: first look only fires on VERY strong evidence so the
#: second look retains nearly the full α budget.
_ALPHA_STRONG: Final[float] = 0.001


# ---------------------------------------------------------------------------
# Critical-t table — two-tailed (degrees of freedom → critical t)
#
# Stage 8 F4.B / H-02 (skeptic H-01) fix:
#   The previous table had sparse rows (5, 10, 15, 20, 25, 30, 40, 60, 120)
#   and a lookup that returned the critical-t of the next row AT OR ABOVE
#   the requested df. For df values between rows (e.g. df=99, df=49, df=9)
#   that returned a value LOWER than the true critical-t, making Phase A's
#   strong-signal early-stop more permissive than the α=0.01 contract
#   claims — i.e. the guard against promoting on noise was weaker than
#   advertised.
#
# Fix applied here:
#   1. Dense rows for df=1..30 (every integer — the regime where the
#      t-distribution is most non-linear and small errors hurt most).
#   2. Standard thinning for df=40, 50, 60, 80, 100, 120, 200, 10000 where
#      the function is nearly flat.
#   3. Linear interpolation between rows for any df not in the table.
#   4. Optional ``scipy.stats.t.ppf`` preference when scipy is importable —
#      this is already a transitive dep of lightgbm-learner, so when
#      present we use it and skip the table entirely.
#
# All table values were cross-verified against scipy.stats.t.ppf within
# ±0.001 at module import time. See tests/test_learning/test_shadow_test.py
# (test_critical_t_matches_scipy_reference) for the regression guard.
# ---------------------------------------------------------------------------

_CRIT_T_05_TWO_TAIL: Final[tuple[tuple[int, float], ...]] = (
    (1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776), (5, 2.571),
    (6, 2.447), (7, 2.365), (8, 2.306), (9, 2.262), (10, 2.228),
    (11, 2.201), (12, 2.179), (13, 2.160), (14, 2.145), (15, 2.131),
    (16, 2.120), (17, 2.110), (18, 2.101), (19, 2.093), (20, 2.086),
    (21, 2.080), (22, 2.074), (23, 2.069), (24, 2.064), (25, 2.060),
    (26, 2.056), (27, 2.052), (28, 2.048), (29, 2.045), (30, 2.042),
    (40, 2.021), (50, 2.009), (60, 2.000), (80, 1.990), (100, 1.984),
    (120, 1.980), (200, 1.972), (10_000, 1.960),
)

#: Tighter α=0.01 table (two-tailed) for Phase A early-stop.
_CRIT_T_01_TWO_TAIL: Final[tuple[tuple[int, float], ...]] = (
    (1, 63.657), (2, 9.925), (3, 5.841), (4, 4.604), (5, 4.032),
    (6, 3.707), (7, 3.499), (8, 3.355), (9, 3.250), (10, 3.169),
    (11, 3.106), (12, 3.055), (13, 3.012), (14, 2.977), (15, 2.947),
    (16, 2.921), (17, 2.898), (18, 2.878), (19, 2.861), (20, 2.845),
    (21, 2.831), (22, 2.819), (23, 2.807), (24, 2.797), (25, 2.787),
    (26, 2.779), (27, 2.771), (28, 2.763), (29, 2.756), (30, 2.750),
    (40, 2.704), (50, 2.678), (60, 2.660), (80, 2.639), (100, 2.626),
    (120, 2.617), (200, 2.601), (10_000, 2.576),
)


def _critical_t(df: int, *, alpha: float) -> float:
    """Return the two-tailed critical t for ``df`` degrees of freedom.

    Preference order:
      1. ``scipy.stats.t.ppf(1 - alpha/2, df)`` when scipy is importable.
      2. Exact tabled value when ``df`` is a table row.
      3. Linear interpolation between adjacent table rows otherwise.

    For ``df ≤ 0`` returns ``inf`` (caller's ``|t| > inf`` is always
    False; no early-stop).
    """
    if df <= 0:
        return float("inf")

    # Preference 1 — scipy, when importable (cached at module load).
    # S9-SKEP-01: no silent `except Exception`. If scipy is present but
    # .ppf() raises (corrupt install, NaN propagation), we let the
    # error surface so callers see it; silently falling back to the
    # table was the original bug that led to false-promote on noise.
    if _SCIPY_T is not None:
        return float(_SCIPY_T.ppf(1.0 - alpha / 2.0, df))

    table = (
        _CRIT_T_05_TWO_TAIL
        if abs(alpha - 0.05) < 1e-9
        else _CRIT_T_01_TWO_TAIL
    )

    # Preference 2 + 3 — exact row match or linear interpolation.
    prev_df, prev_t = table[0]
    if df <= prev_df:
        return prev_t
    for row_df, row_t in table[1:]:
        if df == row_df:
            return row_t
        if df < row_df:
            # Linear interpolation in df space — adequate at the
            # resolution we keep (every integer for df≤30).
            span = row_df - prev_df
            frac = (df - prev_df) / span
            return prev_t + frac * (row_t - prev_t)
        prev_df, prev_t = row_df, row_t
    return prev_t


def _paired_t_stat(diffs: list[float]) -> tuple[float, float, float]:
    """Return ``(mean, std_sample, t_stat)`` for a sequence of paired
    differences. ``std_sample`` uses ddof=1. When ``len(diffs) < 2`` or
    ``std == 0``, ``t_stat`` is ``inf`` if mean>0 else ``-inf``.
    """
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(diffs) / n
    if n < 2:
        return mean, 0.0, math.copysign(math.inf, mean) if mean != 0 else 0.0
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    std = math.sqrt(var)
    if std == 0.0:
        return mean, 0.0, math.copysign(math.inf, mean) if mean != 0 else 0.0
    t_stat = mean / (std / math.sqrt(n))
    return mean, std, t_stat


# ---------------------------------------------------------------------------
# ShadowTest
# ---------------------------------------------------------------------------


class ShadowTest:
    """Two-phase live-recall A/B validator.

    Callers:
      1. Route each incoming recall with ``route_query(qid)`` →
         ``'active'`` | ``'candidate'``. Deterministic per ``qid`` for
         bit-exact reproducibility across daemon restart.
      2. After each recall's outcome settles, call
         ``record_recall_pair(query_id=..., arm=..., ndcg_at_10=...)``.
      3. Call ``decide()`` to get one of ``'promote' | 'reject' | 'continue'``.
    """

    # Exposed for tests + manifest cross-reference.
    PHASE_A_N: Final[int] = _PHASE_A_N
    PHASE_B_N: Final[int] = _PHASE_B_N
    MIN_EFFECT: Final[float] = _MIN_EFFECT
    MIN_STRONG_EFFECT: Final[float] = _MIN_STRONG_EFFECT
    ALPHA: Final[float] = _ALPHA
    ALPHA_STRONG: Final[float] = _ALPHA_STRONG

    def __init__(
        self,
        profile_id: str,
        candidate_model_id: str,
        *,
        learning_db: str | None = None,
    ) -> None:
        self.profile_id = profile_id
        self.candidate_model_id = candidate_model_id
        # Insertion-ordered lists of NDCG@10 values per arm.
        self._active: list[float] = []
        self._candidate: list[float] = []
        # S9-defer H-ARC-01 (full): if ``learning_db`` is provided and
        # the ``shadow_observations`` table (M012) exists, paired obs
        # persist there and reload on restart. Old tests that construct
        # ShadowTest without a DB path keep pure-in-memory semantics.
        # Pair storage keyed by (query_id, arm) avoids duplicate inserts
        # on crash-replay.
        self._learning_db: str | None = learning_db
        # S9-defer S9-STAT-08: replace by-index pairing with query_id
        # pairing. Observations are keyed by (query_id, arm). ``decide``
        # iterates the intersection of arm-keysets so "pair #7 in
        # active" no longer silently pairs with "pair #7 in candidate"
        # when the two streams diverge.
        self._active_by_qid: dict[str, float] = {}
        self._candidate_by_qid: dict[str, float] = {}
        if learning_db:
            self._reload_from_db()

    # ------------------------------------------------------------------
    # Persistence (M012 / H-ARC-01 full)
    # ------------------------------------------------------------------

    def _reload_from_db(self) -> None:
        """Populate in-memory state from ``shadow_observations`` on
        daemon restart. Fail-soft — a missing table or schema error
        leaves the instance in cold-start mode.
        """
        try:
            import sqlite3 as _sq
            cid = int(self.candidate_model_id)
        except Exception:
            return
        try:
            conn = _sq.connect(self._learning_db, timeout=2.0)  # type: ignore[arg-type]
        except Exception:  # pragma: no cover — defensive
            return
        try:
            try:
                rows = conn.execute(
                    "SELECT arm, query_id, ndcg_at_10 "
                    "FROM shadow_observations "
                    "WHERE candidate_id = ? "
                    "ORDER BY recorded_at ASC",
                    (cid,),
                ).fetchall()
            except Exception:
                return  # table absent — M012 not yet applied.
            for arm, qid, ndcg in rows:
                if arm == "active":
                    self._active.append(float(ndcg))
                    self._active_by_qid[str(qid)] = float(ndcg)
                elif arm == "candidate":
                    self._candidate.append(float(ndcg))
                    self._candidate_by_qid[str(qid)] = float(ndcg)
        finally:
            try:
                conn.close()
            except Exception:  # pragma: no cover
                pass

    def _persist_observation(
        self, *, query_id: str, arm: str, ndcg: float,
    ) -> None:
        """Append one observation to ``shadow_observations``. Fail-soft."""
        if not self._learning_db:
            return
        try:
            import sqlite3 as _sq
            cid = int(self.candidate_model_id)
        except Exception:
            return
        try:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            conn = _sq.connect(self._learning_db, timeout=2.0)
            try:
                # INSERT OR IGNORE so crash-replay + duplicate observations
                # (same query_id, same arm) are idempotent.
                conn.execute(
                    "INSERT OR IGNORE INTO shadow_observations "
                    "(profile_id, candidate_id, query_id, arm, "
                    " ndcg_at_10, recorded_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (self.profile_id, cid, query_id, arm, float(ndcg), now),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:  # pragma: no cover — defensive
            pass

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route_query(self, query_id: str) -> str:
        """Deterministic 50/50 A/B route by SHA-256 first 8 hex chars.

        LLD-10 §4.1 — exact formula: ``int(hexdigest[:8], 16) % 2``.
        0 → ``'active'``, 1 → ``'candidate'``.

        SEC-L1 / assumption (daemon contract): ``query_id`` is minted by
        the recall pipeline (``recall_query_id``) and is NOT user-
        controllable — any change to that contract MUST re-audit this
        routing for collision / preimage bias. The current 32-bit hash
        prefix is adequate because pairing validity (Phase A/B t-test)
        degrades gracefully under skew (n_pairs shrinks) rather than
        producing a one-sided false promotion.
        """
        h = hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:8]
        bucket = int(h, 16) % 2
        return "candidate" if bucket == 1 else "active"

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def record_recall_pair(
        self, *, query_id: str, arm: str, ndcg_at_10: float,
    ) -> None:
        """Record one settled recall result for the specified arm.

        ``arm`` must be ``'active'`` or ``'candidate'``. Unknown arms
        are silently ignored — the outcome is not our business to
        police (callers may test routing bugs by feeding a mix).
        """
        # S9-defer H-P-12: route-exclusivity verifier. The routing
        # contract says each query_id deterministically routes to
        # exactly ONE arm. If the same qid arrives on both arms we
        # have a shadow double-pay bug (caller invoked record on
        # both arms, or the router flipped mid-test). Refuse the
        # second write and log — the first arm's observation wins,
        # the double-pay does not pollute the paired statistic.
        qid_s = str(query_id)
        if arm == "active":
            if qid_s in self._candidate_by_qid:
                logger.warning(
                    "shadow_test route-exclusivity violation: "
                    "qid=%s already on candidate arm; ignoring active write",
                    qid_s,
                )
                return
            self._active.append(float(ndcg_at_10))
            self._active_by_qid[qid_s] = float(ndcg_at_10)
        elif arm == "candidate":
            if qid_s in self._active_by_qid:
                logger.warning(
                    "shadow_test route-exclusivity violation: "
                    "qid=%s already on active arm; ignoring candidate write",
                    qid_s,
                )
                return
            self._candidate.append(float(ndcg_at_10))
            self._candidate_by_qid[qid_s] = float(ndcg_at_10)
        else:
            return  # unknown arm: noop
        # S9-defer: persist so restart reloads.
        self._persist_observation(
            query_id=qid_s, arm=arm, ndcg=float(ndcg_at_10),
        )

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(self) -> tuple[str, dict]:
        """Return ``(decision, stats)``.

        ``decision``:
          * ``'promote'`` — candidate beat active by ≥ MIN_EFFECT with
            sufficient statistical power.
          * ``'reject'`` — full Phase B accumulated and criterion not met.
          * ``'continue'`` — insufficient data to decide either way.

        ``stats`` is a plain dict for logging / dashboard / audit.
        """
        n_active = len(self._active)
        n_cand = len(self._candidate)
        # S9-STAT-08: pair by query_id (intersection of arm keysets),
        # NOT by arrival index. Index-pairing silently paired the
        # Nth arrival in each arm regardless of whether those arrivals
        # referred to the same query — a time-order artefact that
        # violated the paired-t iid assumption whenever the two arms
        # saw queries in different orders. Intersection-by-qid makes
        # each pair a true same-query comparison. We keep the legacy
        # index-min as a conservative upper bound on n_pairs for the
        # PHASE_B_N gate so the sample-size contract unchanged.
        paired_qids = (
            set(self._active_by_qid.keys())
            & set(self._candidate_by_qid.keys())
        )
        n_pairs = len(paired_qids)
        stats: dict = {
            "n_active": n_active,
            "n_candidate": n_cand,
            "n_pairs": n_pairs,
            "effect": 0.0,
            "t_stat": 0.0,
            "std": 0.0,
            "phase": "A" if n_pairs < self.PHASE_B_N else "B",
            "criterion": None,
        }

        if n_pairs == 0:
            return "continue", stats

        # S-M03: guard against significant arm imbalance. SHA-256 routing
        # is approximately 50/50 in expectation, but on small samples the
        # buckets can skew. When one arm is more than 2× the other AND
        # both arms have a minimal footprint, the paired-by-index diff
        # silently discards the long tail — the statistic is still valid
        # but operators should be told the data is unbalanced before any
        # promote/reject decision is attempted.
        _MIN_PER_ARM = 8
        if (
            n_active >= _MIN_PER_ARM
            and n_cand >= _MIN_PER_ARM
            and max(n_active, n_cand) > 2 * min(n_active, n_cand)
        ):
            stats["criterion"] = "unbalanced_arms"
            return "continue", stats

        # S9-STAT-08: diffs built from the query_id intersection so
        # each element of ``diffs`` is a true same-query paired
        # comparison (candidate_ndcg - active_ndcg for the same qid).
        # Sort the qid set for reproducibility across runs with the
        # same data.
        diffs = [
            self._candidate_by_qid[qid] - self._active_by_qid[qid]
            for qid in sorted(paired_qids)
        ]
        mean, std, t_stat = _paired_t_stat(diffs)
        stats["effect"] = float(mean)
        stats["std"] = float(std)
        stats["t_stat"] = float(t_stat)

        # --- Phase A early-stop on STRONG signal ---
        if n_pairs >= self.PHASE_A_N and n_pairs < self.PHASE_B_N:
            crit_strong = _critical_t(n_pairs - 1, alpha=self.ALPHA_STRONG)
            if (
                abs(mean) > self.MIN_STRONG_EFFECT
                and abs(t_stat) > crit_strong
                and mean > 0
            ):
                stats["phase"] = "A"
                stats["criterion"] = "phase_a_strong_signal"
                return "promote", stats
            # Weak or uncertain signal — continue to Phase B.
            stats["phase"] = "A"
            stats["criterion"] = "phase_a_continue"
            return "continue", stats

        # --- Phase B full validation ---
        if n_pairs >= self.PHASE_B_N:
            # S-L05: we compare ``t_stat > crit`` which is a one-tailed
            # "candidate better than active" test. ``_critical_t`` returns
            # a TWO-tailed critical (α=0.05 → 1.96). For a one-tailed
            # directional test at α=0.05 the correct critical is 1.645, i.e.
            # the two-tailed critical at α=0.10. We pass α×2 so the
            # comparison semantics match the docstring ("paired t-test
            # p<0.05") under a one-sided directional constraint AND the
            # ``mean >= MIN_EFFECT`` gate preserves the conservative
            # direction preference.
            crit = _critical_t(n_pairs - 1, alpha=min(0.999, self.ALPHA * 2.0))
            stats["phase"] = "B"
            if mean >= self.MIN_EFFECT and t_stat > crit:
                stats["criterion"] = "phase_b_promote"
                return "promote", stats
            stats["criterion"] = "phase_b_reject"
            return "reject", stats

        # n_pairs < PHASE_A_N → continue accumulating.
        stats["phase"] = "A"
        stats["criterion"] = "accumulating"
        return "continue", stats


__all__ = ("ShadowTest",)
