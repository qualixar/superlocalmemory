#!/usr/bin/env python3
"""Build a held-out calibration report without activating runtime confidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from superlocalmemory.evaluation.calibration import (  # noqa: E402
    CalibrationArtifactError,
    build_calibration_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate non-activating held-out calibration evidence"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    try:
        artifact = json.loads(args.input.read_text(encoding="utf-8"))
        report = build_calibration_report(artifact, bin_count=args.bins)
    except (OSError, json.JSONDecodeError, CalibrationArtifactError) as exc:
        print(f"calibration-report: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
