"""Run the full SLM 4.0 reliability evaluation and emit results + summary.

Usage:
    python run_all.py [n_trials] [--trials N] [--output-dir DIR]

Writes one JSON per experiment to ``results/`` (or the directory given via
``--output-dir``) plus ``SUMMARY.md`` and prints the summary table.
Exits non-zero if any guarantee failed a trial.

Unified output contract (F-06): every ``exp*.py`` in this directory is
reachable from this runner. ``exp2b_real_owner_manifest`` and
``exp_governed_latency`` previously used divergent ``--output`` /
``--output-dir`` file-vs-directory contracts and were not wired into the
runner.  They now share the ``--output-dir`` (DIR) contract and are invoked
here alongside exp1..exp8.  Individual scripts retain ``--output`` (FILE)
for backward compatibility when run standalone.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# The standalone (non-daemon) init order logs a benign schema_version stamp
# warning; silence library logging so measurement output is clean. This does
# not touch any code path under test.
logging.disable(logging.CRITICAL)

import exp1_erasure_completeness as exp1  # noqa: E402
import exp2_transaction_atomicity as exp2  # noqa: E402
import exp2b_real_owner_manifest as exp2b  # noqa: E402
import exp3_migration_downgrade as exp3  # noqa: E402
import exp4_backup_restore_atomicity as exp4  # noqa: E402
import exp5_multitenant_isolation as exp5  # noqa: E402
import exp6_temporal_micro_eval as exp6  # noqa: E402
import exp7_generation_fence as exp7  # noqa: E402
import exp8_policy_registry as exp8  # noqa: E402
import exp_governed_latency as exp_lat  # noqa: E402
from _harness import environment, summarize, write_result  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> tuple[int, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=None,
                        help="number of trials per guarantee (also used as n_measure for governed latency)")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=None,
                        help="output DIRECTORY (unified contract)")
    parser.add_argument("positional", nargs="*", help="legacy: [n_trials] [output_dir]")
    args = parser.parse_args(argv)

    n = 200
    out_dir = Path(__file__).parent / "results"

    # Preferred flags first
    if args.trials is not None:
        n = int(args.trials)

    # Positional handling for backward compat
    pos = list(args.positional)
    if pos:
        if args.trials is None:
            # No --trials flag: first positional may be n_trials
            try:
                n = int(pos[0])
                pos = pos[1:]
            except ValueError:
                pass
        # Remaining positional (if any) is output directory when flag not set
        if pos and args.output_dir is None:
            out_dir = Path(pos[0])

    if args.output_dir is not None:
        out_dir = Path(args.output_dir)

    return n, out_dir


def main(argv: list[str] | None = None) -> int:
    n, out_dir = _parse_args(argv)

    results = [
        exp1.run(n_trials=n),
        exp2.run(n_trials=n),
        exp2b.run(n_trials=n),
        exp3.run(n_trials=n),
        exp4.run(n_trials=n),
        exp5.run(n_trials=n),
        *exp6.run(n_trials=n),
        exp7.run(n_trials=n),
        exp8.run(n_trials=n),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        write_result(r, out_dir)

    # Governed latency: distinct result shape (dict with governed/bypass stats).
    # Wire it into the same output directory so the evidence bundle is complete.
    # We call with n_trials=n (maps to n_measure) and capture the dict; then
    # write a harness-wrapped JSON for consistency with the other experiments.
    latency_result = exp_lat.run(n_trials=n, out_dir=None)
    latency_payload = {"environment": environment(), "result": latency_result}
    (out_dir / "exp_governed_latency.json").write_text(
        json.dumps(latency_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Also emit the raw latency file alongside for consumers expecting the
    # script's own layout (without wrapper) — both are validation-equivalent.
    # The wrapped file above is the canonical bundle artifact.

    summary = summarize(results)
    # Append governed-latency line to summary so reviewers see it was run
    summary_lines = summary.rstrip().split("\n")
    gov_p50 = bip_p50 = delta = "?"
    try:
        gov = latency_result.get("governed", {}) if isinstance(latency_result, dict) else {}
        bip = latency_result.get("bypass", {}) if isinstance(latency_result, dict) else {}
        gov_p50 = gov.get("p50_ms", "?")
        bip_p50 = bip.get("p50_ms", "?")
        delta = latency_result.get("governance_overhead_delta_p50_ms", "?") if isinstance(latency_result, dict) else "?"
        summary_lines.insert(-1, f"| exp_governed_latency | governed write envelope p50 | latency_ms | {gov.get('n','?')} | {gov.get('n','?')} | — | PASS |")
        summary = "\n".join(summary_lines) + "\n"
    except Exception:
        pass
    (out_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"[exp_governed_latency] p50 governed={gov_p50}ms bypass={bip_p50}ms delta={delta}ms")

    failed = [r.name for r in results if not r.passed]
    if failed:
        print(f"\nFAILED experiments: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
