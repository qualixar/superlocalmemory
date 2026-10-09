"""Per-query retrieval metrics, bootstrap CIs, split, abstention, one-model rule."""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

import numpy as np

SEED = 1729
N_BOOT = 10_000


def ranked_docs(scores: dict[str, float]) -> list[str]:
    """Doc ids by descending score; ties broken by doc id."""
    return [d for d, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def _parse(metric: str) -> tuple[str, int]:
    name, k = metric.split("@")
    return name, int(k)


def _one(rel: dict[str, int], ranked: list[str], name: str, k: int) -> float:
    top = ranked[:k]
    if name == "recall":
        return sum(1 for d in top if d in rel) / len(rel)
    if name == "mrr":
        return next((1 / (i + 1) for i, d in enumerate(top) if d in rel), 0.0)
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(top) if d in rel)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal


def per_query(qrels: dict, run: dict, metric: str) -> dict[str, float]:
    """Metric per query (binary relevance). Queries with no qrels are skipped."""
    name, k = _parse(metric)
    out = {}
    for qid, rel in qrels.items():
        if not rel:
            continue
        out[qid] = _one(rel, ranked_docs(run.get(qid, {})), name, k)
    return out


def bootstrap_ci(values: Sequence[float], n_boot: int = N_BOOT,
                 seed: int = SEED, alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean: (mean, low, high)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(min(lo, v.mean())), float(max(hi, v.mean()))


def paired_delta_ci(a: Sequence[float], b: Sequence[float], n_boot: int = N_BOOT,
                    seed: int = SEED) -> tuple[float, float, float]:
    """Bootstrap CI of mean(b - a) over paired per-query values."""
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if not diff.any():
        return (0.0, 0.0, 0.0)
    return bootstrap_ci(diff, n_boot=n_boot, seed=seed)


def stratified_split(queries: list[dict], dev_frac: float = 0.4) -> tuple[list, list]:
    """Hash-ordered split per stratum; stable under reordering."""
    strata: dict[str, list[dict]] = {}
    for q in queries:
        strata.setdefault(q["stratum"], []).append(q)
    dev, test = [], []
    for name in sorted(strata):
        items = sorted(strata[name], key=lambda q: hashlib.sha256(q["id"].encode()).hexdigest())
        n_dev = round(dev_frac * len(items))
        dev += items[:n_dev]
        test += items[n_dev:]
    return dev, test


def tune_threshold(scores: Sequence[float], unanswerable: Sequence[bool]) -> float:
    """Abstain when top-1 score < tau. tau maximises abstention F1 on the given
    (dev) data only; lowest tau wins ties. No unanswerables: never abstain."""
    if not any(unanswerable):
        return float("-inf")
    cands = sorted(set(scores))
    cands.append(cands[-1] + 1e-9)
    best, best_f1 = cands[0], -1.0
    for tau in cands:
        f1 = abstention_report(scores, unanswerable, tau)["f1"]
        if f1 > best_f1 + 1e-12:
            best, best_f1 = tau, f1
    return best


def _div(a: float, b: float) -> float | None:
    return a / b if b else None


def abstention_report(scores: Sequence[float], unanswerable: Sequence[bool],
                      tau: float) -> dict:
    """False-answer rate and abstention precision/recall/F1 at threshold tau."""
    abst = [s < tau for s in scores]
    tp = sum(1 for a, u in zip(abst, unanswerable) if a and u)
    fp = sum(1 for a, u in zip(abst, unanswerable) if a and not u)
    n_un = sum(unanswerable)
    prec, rec = _div(tp, tp + fp), _div(tp, n_un)
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else 0.0
    return {
        "tau": tau, "n_unanswerable": n_un, "n_answerable": len(scores) - n_un,
        "false_answer_rate": _div(n_un - tp, n_un),
        "abstention_precision": prec, "abstention_recall": rec, "f1": f1,
    }


def one_model_rule(s1: Sequence[float], s2: Sequence[float], margin: float = 0.03,
                   min_n: int = 30, n_boot: int = N_BOOT) -> dict:
    """One-model default rule on per-query text recall@5 (S1 two-space, S2 one-model).

    Pass when the 95% CI lower bound of mean(S2 - S1) is >= -margin. Also reports
    the literal form: S2's own lower CI bound >= S1's point estimate - margin.
    """
    delta, lo, hi = paired_delta_ci(s1, s2, n_boot=n_boot)
    s1_mean = float(np.mean(s1)) if len(s1) else float("nan")
    _, s2_lo, _ = bootstrap_ci(s2, n_boot=n_boot)
    passed = lo >= -margin
    verdict = "PASS" if passed else "FAIL"
    if len(s1) < min_n:
        verdict = "SAMPLE TOO SMALL TO DECIDE"
    return {
        "verdict": verdict, "n": len(s1), "margin": margin,
        "delta": delta, "delta_ci": [lo, hi], "ci_rule_pass": passed,
        "literal_form_pass": s2_lo >= s1_mean - margin,
        "s1_point": s1_mean, "s2_ci_low": s2_lo,
    }
