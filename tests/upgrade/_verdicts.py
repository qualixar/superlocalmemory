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


def schema_in_range(version: object, ceiling: int) -> bool:
    """After the older version ran, the store is at the floor or was re-migrated up to that version's ceiling."""
    return isinstance(version, int) and DOWNGRADE_SCHEMA <= version <= ceiling


def downgrade_ok(d: dict, recall_vs_baseline: dict | None, ceiling: int) -> bool:
    """prepare-downgrade left the floor schema, the older version opened the store and answered.

    ``ceiling`` is the newest schema the older version supports; it may re-apply its own
    additive migrations when it opens a store left at the floor, but never go past it.
    """
    return bool(
        d.get("prepare_rc") == 0 and d.get("daemon_up") is True
        and d.get("schema_after_prepare") == DOWNGRADE_SCHEMA
        and schema_in_range(d.get("schema_version"), ceiling)
        and d.get("queries_answered") == len(corpus.QUERIES)
        and d.get("corpus_before") == d.get("corpus_after") == len(corpus.MEMORIES)
        and recall_vs_baseline is not None and recall_vs_baseline.get("verdict") in {"identical", "within_noise"}
        and not d.get("errors"))


def restore_problems(want_cols: dict, got_cols: dict, want_counts: dict, got_counts: dict,
                     want_schema: int | None, got_schema: int | None, contents_ok: bool) -> list[str]:
    """Why a restored database differs from the fixture's (empty list: it equals it).

    A table the fixture lacks is tolerated only when the restored copy has it empty
    (an older version creates such tables on start); every other difference is a reason.
    """
    out: list[str] = []
    if want_schema != got_schema:
        out.append(f"schema version {got_schema} != fixture {want_schema}")
    extra = set(got_cols) - set(want_cols)
    nonempty = sorted(t for t in extra if got_counts.get(t) != 0)
    if nonempty:
        out.append(f"restored copy has extra tables that are not known to be empty: {nonempty}")
    if sorted(set(want_cols) - set(got_cols)):
        out.append(f"restored copy lacks tables: {sorted(set(want_cols) - set(got_cols))}")
    out += [f"columns of {t} differ" for t in sorted(want_cols) if t in got_cols and want_cols[t] != got_cols[t]]
    out += [f"row count of {t}: {got_counts.get(t)} != fixture {want_counts[t]}"
            for t in sorted(want_counts) if got_counts.get(t) != want_counts[t]]
    if not contents_ok:
        out.append("stored contents differ from the fixture's")
    return out


def restore_ok(want_cols: dict, got_cols: dict, want_counts: dict, got_counts: dict,
               want_schema: int | None, got_schema: int | None, contents_ok: bool) -> bool:
    """A restored database equals the fixture's: schema version, tables, columns, rows, contents."""
    return not restore_problems(want_cols, got_cols, want_counts, got_counts, want_schema, got_schema, contents_ok)


def snapshot_reasons(restores: list[dict], restore_points_rc: int | None) -> list[str]:
    """Every reason the snapshot check fails; empty only when it passes."""
    out: list[str] = []
    if not restores:
        out.append("no snapshot was restored")
    for r in restores:
        if r.get("ok") is not True:
            out += list(r.get("reasons") or ([r["error"]] if r.get("error") else [])) or [
                f"restore of {r.get('snapshot', '?')} did not verify"]
    if restore_points_rc != 0:
        out.append(f"slm db restore-points exited {restore_points_rc}")
    return out


def snapshot_passed(restores: list[dict], restore_points_rc: int | None) -> bool:
    return not snapshot_reasons(restores, restore_points_rc)


#: memory.db tables that hold what users stored; their row counts must survive an upgrade.
#: Other tables (indexes, outboxes, change logs, caches) are rebuilt or drained by migrations.
CORE_MEMORY_TABLES = ("memories", "atomic_facts", "profiles", "write_commits",
                      "ingestion_operations", "fact_temporal_validity")


def core_counts_unchanged(before: dict, after: dict) -> bool:
    """Every core table the old store had still has the same number of rows."""
    return all(after.get(t) == before[t] for t in CORE_MEMORY_TABLES if t in before) and any(
        t in before for t in CORE_MEMORY_TABLES)


def mapping_diff(before: dict, after: dict) -> dict:
    """Keys added, removed or changed between two mappings (for the verdict's evidence)."""
    return {"added": sorted(set(after) - set(before)), "removed": sorted(set(before) - set(after)),
            "changed": sorted(k for k in set(before) & set(after) if before[k] != after[k])}


def _by_name(signature: list[tuple]) -> dict[str, tuple]:
    return {r[1]: (r[0], r[2]) for r in signature}


def schema_unchanged(before: Path, after: Path) -> bool:
    """Both databases have exactly the schema they had (only then is 'no snapshot' expected)."""
    from _slm_env import schema_signature

    return all(schema_signature(before / n) == schema_signature(after / n) != []
               for n in ("memory.db", "learning.db"))


def snapshotless_verdict(before: dict[str, list[tuple]], after: dict[str, list[tuple]],
                         counts_before: dict, counts_after: dict) -> dict:
    """Judge a start that took no pre-migration copy, from the schema signatures it left.

    unchanged -> n/a; only additions (and re-created indexes/triggers) with user data
    intact -> n/a-additive; an existing table changed or removed, or user-data rows
    changed -> fail. ``diff`` is kept as evidence in every case.
    """
    diff, errors = {}, []
    for db in sorted(before):
        b, a = _by_name(before[db]), _by_name(after.get(db, []))
        diff[db] = mapping_diff(b, a)
        errors += [f"{db}: table {n} {'removed' if n not in a else 'changed'} without a snapshot"
                   for n in diff[db]["removed"] + diff[db]["changed"] if b[n][0] == "table"]
    lost = sorted(t for t in CORE_MEMORY_TABLES if t in counts_before and counts_after.get(t) != counts_before[t])
    errors += [f"user-data table {t}: {counts_before[t]} -> {counts_after.get(t)} rows without a snapshot" for t in lost]
    changed = any(any(d.values()) for d in diff.values())
    if errors:
        return {"status": "fail", "passed": False, "errors": errors, "diff": diff}
    if not changed:
        return {"status": "n/a", "passed": None, "errors": [], "diff": diff}
    return {"status": "n/a-additive", "passed": None, "errors": [], "diff": diff}


def check_ok(check: dict) -> bool:
    """Passed, or explicitly not applicable; anything else (including no verdict) is a failure."""
    return check.get("passed") is True or (
        check.get("passed") is None and check.get("status") in {"n/a", "n/a-additive"})


def failing(checks: dict) -> list[str]:
    return [name for name, c in checks.items() if not check_ok(c)]
