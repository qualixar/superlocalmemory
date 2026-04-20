# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22

"""Multi-seed Evo-Memory runner (S9-defer C10 / STAT-01 / STAT-06).

The base ``evo_memory.EvoMemoryBenchmark`` is deterministic per seed —
two runs on the same seed produce byte-identical numbers. That property
was useful for reproducibility but ALSO meant the published +18.6%
day-30 lift was an n=1 sample with no sampling-variance estimate.

This runner sweeps ``N_SEEDS`` independent fixtures (derived from the
base v1 fixture by reseeding the RNG for activity + test-query ordering;
the underlying seed facts and gold labels are unchanged) and reports:

  * per-seed day-30 lift
  * mean ± 95% confidence interval (t-based) across seeds
  * day-14 counterfactual (STAT-06): was the dip consistent across
    seeds, or an artefact of seed 12345?

Usage::

    python -m tests.test_benchmarks.evo_memory_multiseed \
        --seeds 20 \
        --out /tmp/evo_multiseed.json

The full 20-seed sweep takes ~4-6 minutes on a MacBook M1.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from tests.test_benchmarks import evo_memory as em


DEFAULT_SEEDS = 20
DEFAULT_BASE_SEED = 12345


def _t_critical_95(df: int) -> float:
    """Two-tailed t critical at α=0.05 — dense enough for the CI band."""
    if df <= 0:
        return float("inf")
    # Lookup table — covers n=2..30 which is the realistic range here.
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    if df in table:
        return table[df]
    # Large-sample → approach 1.96.
    return 1.96


def run_one_seed(
    *, seed: int, data_dir: Path,
) -> dict[str, float]:
    """Run a single Evo-Memory benchmark with ``seed`` as the activity
    RNG seed. Returns the per-day lift dict.

    We monkey-patch ``evo_memory._SEED`` for the duration of the call so
    the underlying harness picks up our seed without a new constructor
    argument. Patch is reverted in ``finally``.
    """
    orig = em._SEED
    try:
        em._SEED = seed  # type: ignore[misc]
        bench = em.EvoMemoryBenchmark(data_dir=data_dir)
        result = bench.run_full_30_day_simulation()
    finally:
        em._SEED = orig  # type: ignore[misc]
    return result


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    """Return (mean, half_width_95, stdev). Defensive on small N."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    if n == 1:
        return (float(values[0]), float("inf"), 0.0)
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    half = _t_critical_95(n - 1) * (stdev / math.sqrt(n))
    return (mean, half, stdev)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help=f"seed count (default {DEFAULT_SEEDS})")
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED,
                        help="seed for the SEED generator itself")
    parser.add_argument("--out", type=Path, required=True,
                        help="output JSON path")
    parser.add_argument("--tmp-root", type=Path,
                        default=Path("/tmp/evo_multiseed_runs"),
                        help="root dir for per-seed isolated data")
    args = parser.parse_args(argv)

    rng = random.Random(args.base_seed)
    seeds = [rng.randrange(1, 2**31 - 1) for _ in range(args.seeds)]
    per_seed_day_lifts: dict[int, list[float]] = {1: [], 7: [], 14: [], 30: []}
    per_seed_raw: list[dict[str, Any]] = []

    t0 = time.monotonic()
    for i, seed in enumerate(seeds):
        data_dir = args.tmp_root / f"seed_{seed}"
        data_dir.mkdir(parents=True, exist_ok=True)
        try:
            r = run_one_seed(seed=seed, data_dir=data_dir)
        except Exception as exc:  # pragma: no cover — defensive
            print(f"[warn] seed {seed}: {exc}", file=sys.stderr)
            continue
        per_seed_raw.append({"seed": seed, "result": r})
        day_metrics = r.get("days") or {}
        # Lift = (day_N_mrr / day_1_mrr) - 1 for each measured day.
        day_1 = day_metrics.get(1, {}).get("mrr_at_10", 0.0) or 0.0
        for d in (1, 7, 14, 30):
            dN = day_metrics.get(d, {}).get("mrr_at_10", 0.0) or 0.0
            if day_1 > 0:
                per_seed_day_lifts[d].append((dN / day_1) - 1.0)
        elapsed_so_far = time.monotonic() - t0
        print(
            f"[multiseed {i + 1}/{len(seeds)}] seed={seed} "
            f"day_1={day_1:.4f} day_30={day_metrics.get(30, {}).get('mrr_at_10', 0):.4f} "
            f"elapsed={elapsed_so_far:.1f}s"
        )

    agg: dict[int, dict[str, float]] = {}
    for d, lifts in per_seed_day_lifts.items():
        mean, half, stdev = _mean_ci(lifts)
        agg[d] = {
            "n_seeds": len(lifts),
            "mean_lift": mean,
            "stdev": stdev,
            "ci_low_95": mean - half,
            "ci_high_95": mean + half,
        }

    result = {
        "version": "3.4.22",
        "n_seeds": len(seeds),
        "seeds": seeds,
        "per_day_aggregate": agg,
        "per_seed": per_seed_raw,
        "elapsed_sec": time.monotonic() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8",
    )

    # S9-STAT-06: day-14 counterfactual — is the dip consistent across
    # seeds, or an artefact of seed 12345?
    d14 = agg.get(14, {})
    d30 = agg.get(30, {})
    print(
        f"\n[multiseed] day-14 lift {d14.get('mean_lift', 0):+.3f} "
        f"[{d14.get('ci_low_95', 0):+.3f}, {d14.get('ci_high_95', 0):+.3f}] "
        f"(n={d14.get('n_seeds', 0)})"
    )
    print(
        f"[multiseed] day-30 lift {d30.get('mean_lift', 0):+.3f} "
        f"[{d30.get('ci_low_95', 0):+.3f}, {d30.get('ci_high_95', 0):+.3f}] "
        f"(n={d30.get('n_seeds', 0)})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
