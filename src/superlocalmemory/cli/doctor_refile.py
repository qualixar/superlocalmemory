# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``slm doctor --refile-hidden``: show, then optionally apply, the re-filing."""

from __future__ import annotations

from typing import Any


def run_refile(args: Any, use_json: bool) -> dict:
    """Dry run unless ``--yes``. Returns the JSON-ready result."""
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.memory_health import refile_hidden

    apply = bool(getattr(args, "yes", False))
    try:
        report = refile_hidden(SLMConfig.load().db_path, dry_run=not apply)
    except Exception as exc:  # noqa: BLE001 - a repair flag must not crash doctor
        if not use_json:
            print(f"\nCould not re-file hidden memories: {exc}")
        return {"error": str(exc), "applied": False, "moved": 0, "fact_ids": []}

    if not use_json:
        _print(report, apply)
    return {
        "applied": apply,
        "moved": report.moved,
        "fact_ids": list(report.fact_ids),
        "previews": list(report.previews),
    }


def _print(report: Any, apply: bool) -> None:
    if not report.fact_ids:
        print("\nNo hidden memories need re-filing.")
        return
    verb = "Re-filed" if apply else "Would re-file"
    print(f"\n{verb} {len(report.fact_ids)} memories:")
    for fid, text in zip(report.fact_ids, report.previews):
        print(f"  {fid}  {text}")
    if apply:
        print(f"Moved {report.moved}. Nothing else changed.")
    else:
        print("Dry run. Re-run with --yes to apply.")
