# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Pure pass/fail rules for the upgrade checks.

Every rule fails closed: missing evidence, any recorded error or an
unreadable value is a failure, never a pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import corpus

TARGET_SCHEMA = 54
#: Schema a store carries after ``slm db prepare-downgrade``.
DOWNGRADE_SCHEMA = 51


def parse_recall(out: str) -> tuple[list[dict], str, str]:
    """Rows, channel-status string and an error text from ``slm recall --json`` output.

    JSON without a ``results`` key is an error, not an empty answer.
    """
    try:
        data = json.loads(out[out.index("{"):])
    except (ValueError, json.JSONDecodeError):
        return [], "", "unparsable output"
    body = data.get("data", data) if isinstance(data, dict) else {}
    if not isinstance(body, dict) or not isinstance(body.get("results"), list):
        return [], "", "response has no results list"
    return body["results"], json.dumps(body.get("channel_status", {}), sort_keys=True), ""


def overlap(a: list[str], b: list[str]) -> int:
    """How many ids the two top-5 lists share."""
    return len(set(a[:5]) & set(b[:5]))


def recall_verdict(base_a: list[list[str]], base_b: list[list[str]], new: list[list[str]]) -> dict:
    """Judge ``new`` against two runs of the reference version.

    ``baseline_overlap`` is how well the reference matches itself (measured
    nondeterminism); ``new_overlap`` is the mean of new-vs-first and new-vs-second.
    ``order_identical`` is true when every top-5 list also has the same order.
    """
    baseline = sum(overlap(x, y) for x, y in zip(base_a, base_b))
    vs_a = sum(overlap(x, y) for x, y in zip(new, base_a))
    vs_b = sum(overlap(x, y) for x, y in zip(new, base_b))
    new_overlap = (vs_a + vs_b) / 2
    exact = all(set(n[:5]) == set(a[:5]) == set(b[:5]) for n, a, b in zip(new, base_a, base_b))
    order = all(n[:5] == a[:5] == b[:5] for n, a, b in zip(new, base_a, base_b))
    if exact:
        verdict = "identical"
    elif new_overlap >= baseline:
        verdict = "within_noise"
    else:
        verdict = "worse"
    return {"verdict": verdict, "baseline_overlap": baseline, "new_overlap": new_overlap,
            "possible": sum(min(5, len(a)) for a in base_a), "order_identical": order}


def expected_hits(results: list[list[str]]) -> int:
    """Queries whose expected corpus id is in the top 5 (of len(QUERIES))."""
    return sum(1 for r, q in zip(results, corpus.QUERIES) if set(q.expected) <= set(r[:5]))


def recall_check(base: dict, new_results: list[list[str]], new_errors: list[str], new_status: str) -> dict:
    """The recall check: clean on both sides, something to compare, no loss of expected hits."""
    runs = base["runs"]
    out: dict = {"errors": list(base["errors"]) + list(new_errors), "reference_status": base["status"],
                 "new_status": new_status, "passed": False}
    if len(runs) != 2 or not new_results:
        out["errors"].append("reference or new recall produced no results")
        return out
    out.update(recall_verdict(runs[0], runs[1], new_results))
    hits = [expected_hits(r) for r in runs]
    out["expected_in_top5"] = {"reference": hits, "new": expected_hits(new_results), "of": len(corpus.QUERIES)}
    if out["possible"] <= 0:
        out["errors"].append("reference returned no results for any query")
    out["passed"] = bool(not out["errors"] and out["possible"] > 0 and out["verdict"] != "worse"
                         and out["expected_in_top5"]["new"] >= min(hits))
    return out


def upgrade_ok(d: dict) -> bool:
    """Daemon up, schema at target, migrations done without failures, nothing lost, no errors."""
    ready = d.get("readiness")
    ready = ready if isinstance(ready, dict) else {}
    return bool(
        d.get("daemon_up") is True and d.get("schema_version") == TARGET_SCHEMA
        and d.get("contents_unchanged") is True and d.get("corpus_after") == len(corpus.MEMORIES)
        and ready.get("migrations") is True and not ready.get("migration_failures")
        and d.get("timed_out") is False and d.get("memory_counts_unchanged") is True
        and not d.get("errors"))


def downgrade_ok(d: dict, recall_vs_baseline: dict | None) -> bool:
    """prepare-downgrade worked, the older version opened the store at the floor schema and answered."""
    return bool(
        d.get("prepare_rc") == 0 and d.get("daemon_up") is True
        and d.get("schema_version") == DOWNGRADE_SCHEMA
        and d.get("queries_answered") == len(corpus.QUERIES)
        and d.get("corpus_before") == d.get("corpus_after") == len(corpus.MEMORIES)
        and recall_vs_baseline is not None and recall_vs_baseline.get("verdict") in {"identical", "within_noise"}
        and not d.get("errors"))


def restore_ok(want_cols: dict, got_cols: dict, want_counts: dict, got_counts: dict,
               want_schema: int | None, got_schema: int | None, contents_ok: bool) -> bool:
    """A restored database equals the fixture's: schema version, tables, columns, rows, contents."""
    return bool(want_schema == got_schema and want_cols == got_cols
                and want_counts == got_counts and contents_ok)


def snapshot_passed(restores: list[dict], restore_points_rc: int | None) -> bool:
    return bool(restores and all(r.get("ok") is True for r in restores) and restore_points_rc == 0)


def schema_unchanged(before: Path, after: Path) -> bool:
    """Both databases have exactly the schema they had (only then is 'no snapshot' expected)."""
    from _slm_env import schema_signature

    return all(schema_signature(before / n) == schema_signature(after / n) != []
               for n in ("memory.db", "learning.db"))


def check_ok(check: dict) -> bool:
    """Passed, or explicitly not applicable; anything else (including no verdict) is a failure."""
    return check.get("passed") is True or (check.get("passed") is None and check.get("status") == "n/a")


def failing(checks: dict) -> list[str]:
    return [name for name, c in checks.items() if not check_ok(c)]
