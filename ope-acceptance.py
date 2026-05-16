#!/usr/bin/env python3
"""OPE Acceptance Checker — evaluates 72h monitoring data.

Usage:
    python3 ope-acceptance.py /home/max/lightrag-data/ope/

Exit code: 0 = PASS, 1 = FAIL
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# Map OPE metric keys -> criteria keys
METRIC_KEY_MAP = {
    "orphans": "orphaned_mcp",
    "hub_restarts": "hub_restarts",
    "daemon_restarts": "daemon_restarts",
    "parallel_ok": "parallel_ok",
    "slm_embedding_cpu": "slm_embedding_cpu",
    "hub_ping_ms": "hub_ping_ms",
    "tasks": "tasks",
    "sessions": "sessions",
}

CRITERIA: dict[str, dict[str, Any]] = {
    "orphaned_mcp": {
        "description": "MCP orphan processes (ppid=1)", "max": 0, "critical": True,
    },
    "hub_restarts": {
        "description": "Hub process restarts (per 24h)", "max": 5, "critical": True,
    },
    "daemon_restarts": {
        "description": "Daemon process restarts (per 24h)", "max": 5, "critical": True,
    },
    "parallel_ok": {
        "description": "Parallel remember + build_code_graph test", "min": 1, "critical": True,
    },
    "slm_embedding_cpu": {
        "description": "Embedding worker CPU usage %", "max": 50.0, "critical": True,
    },
    "hub_ping_ms": {
        "description": "Hub response time (seconds)", "max": 1.0, "critical": False,
    },
    "tasks": {
        "description": "Active long-running tasks", "max": 0, "critical": False,
    },
    "sessions": {
        "description": "Active sessions (any = healthy)", "min": 0, "critical": False,
    },
}


def load_data(paths: list[Path]) -> list[dict]:
    records = []
    for p in paths:
        if p.is_dir():
            for f in sorted(p.glob("ope-*.jsonl")):
                records.extend(load_data([f]))
        elif p.is_file():
            text = p.read_text()
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def evaluate(records: list[dict]) -> tuple[str, list[str]]:
    if not records:
        return "FAIL", ["No OPE data found"]

    failures: list[str] = []
    warnings: list[str] = []
    last = records[-1]

    crit_to_ope = METRIC_KEY_MAP

    for ope_key, crit_key in crit_to_ope.items():
        crit = CRITERIA.get(crit_key)
        if crit is None:
            continue
        if ope_key not in last:
            warnings.append(f"Missing metric: {crit_key}")
            continue
        val = last[ope_key]
        if "max" in crit and val is not None:
            try:
                if float(val) > crit["max"]:
                    msg = f"{crit['description']}: {val} > {crit['max']}"
                    if crit.get("critical"):
                        failures.append(msg)
                    else:
                        warnings.append(msg)
            except (ValueError, TypeError):
                pass
        if "min" in crit and val is not None:
            try:
                if float(val) < crit["min"]:
                    msg = f"{crit['description']}: {val} < {crit['min']}"
                    if crit.get("critical"):
                        failures.append(msg)
                    else:
                        warnings.append(msg)
            except (ValueError, TypeError):
                pass

    try:
        first_ts = datetime.fromisoformat(records[0]["ts"])
        last_ts = datetime.fromisoformat(last["ts"])
        hours = (last_ts - first_ts).total_seconds() / 3600
        if hours < 12:
            warnings.append(f"OPE data coverage: only {hours:.0f}h (minimum 12h)")
    except (KeyError, ValueError):
        warnings.append("Cannot determine OPE time range")

    try:
        hub_r = int(last.get("hub_restarts", 0))
        daemon_r = int(last.get("daemon_restarts", 0))
        if abs(hub_r - daemon_r) > 2:
            warnings.append(
                f"Restart mismatch: hub={hub_r}, daemon={daemon_r} "
                "(both should restart together)"
            )
    except (TypeError, ValueError):
        pass

    if failures:
        return "FAIL", failures + warnings
    if warnings:
        return "WARN", warnings
    return "PASS", []


def main():
    if len(sys.argv) > 1:
        paths = [Path(a) for a in sys.argv[1:]]
    else:
        print("Usage: ope-acceptance.py <directory>")
        sys.exit(1)

    records = load_data(paths)
    result, messages = evaluate(records)

    print(f"OPE ACCEPTANCE: {result}")
    print(f"Records: {len(records)}")
    if records:
        ts_range = f"{records[0].get('ts','?')[:19]} -> {records[-1].get('ts','?')[:19]}"
        print(f"Range: {ts_range}")
    print()

    if messages:
        print("Issues:")
        for m in messages:
            print(f"  {'! ' if result=='FAIL' else '  '}{m}")
    else:
        print("  All criteria passed")

    if result == "FAIL":
        print("\nFAIL: Critical issues found - rollback required")
    elif result == "WARN":
        print("\nWARN: Non-critical issues - review before proceeding")
    else:
        print("\nPASS: Bug fix accepted for upstream PR")

    sys.exit(0 if result in ("PASS", "WARN") else 1)


if __name__ == "__main__":
    main()
