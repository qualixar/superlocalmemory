#!/usr/bin/env python3
"""Run the isolated resource-lifecycle measurement and write evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .harness import render_markdown, run_mode_a_local, write_results
else:
    # Direct execution places this directory, not the repository root, on
    # sys.path. Keep the documented script command independent of PYTHONPATH.
    from harness import render_markdown, run_mode_a_local, write_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--ingests", type=int, default=50)
    parser.add_argument("--recalls", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    result = run_mode_a_local(
        warmup_iterations=args.warmup,
        ingest_iterations=args.ingests,
        recall_iterations=args.recalls,
        sample_every=args.sample_every,
    )
    paths = write_results(result, args.output_dir.resolve())
    print(render_markdown(result))
    print(f"Wrote {paths[0]} and {paths[1]}")
    return 0 if result["rss"]["bounded_window_conclusion"] == "no_unbounded_growth_detected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
