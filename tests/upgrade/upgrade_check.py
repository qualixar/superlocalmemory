# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Open an old-version store with this branch, then go back to 4.1.24.

    python upgrade_check.py --fixture DIR --old-python P --new-python P
                            [--down-python P] [--out F]

``--old-python`` belongs to the venv holding 4.1.24 (baseline and downgrade);
``--new-python`` to the venv holding this branch; ``--down-python`` (optional) to
a venv holding 4.1.20, for a second downgrade.  The fixture is never modified:
every step works on a copy.  Only 4.1.24, the optional 4.1.20 and this branch are
ever started, never the fixture's own version.  Writes a JSON verdict; the exit
status is non-zero unless every check passed (or is explicitly not applicable).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _slm_env as env  # noqa: E402
import corpus  # noqa: E402
from _verdicts import (  # noqa: E402,F401
    DOWNGRADE_SCHEMA, core_counts_unchanged, TARGET_SCHEMA, check_ok, downgrade_ok, expected_hits, failing, overlap,
    mapping_diff, recall_check, recall_verdict, restore_ok, schema_unchanged, snapshot_passed, upgrade_ok)

SNAP_DIR = "pre-migration-snapshots"
EXPECTED_OLD = "4.1.24"
EXPECTED_DOWN = "4.1.20"
#: Version label for instances of the 4.1 line and this branch (not before 4.0).
LINE = "4.1"


def copy_fixture(fixture: Path, label: str, work: Path) -> Path:
    dest = work / label
    (dest / "home").mkdir(parents=True)
    shutil.copytree(fixture / "data", dest / "home" / env.DATA_SUBDIR)
    return dest


def instance(bin_dir: Path, root: Path) -> env.Instance:
    return env.Instance(bin_dir=bin_dir, data_dir=root / "home" / env.DATA_SUBDIR, home=root / "home", version=LINE)


def run_queries(inst: env.Instance) -> tuple[list[list[str]], str, list[str]]:
    results, status, errors = [], "", []
    for q in corpus.QUERIES:
        ids, st, err = inst.recall_ids(q.text)
        results.append(ids)
        status = status or st
        if err:
            errors.append(f"{q.text!r}: {err}")
    return results, status, errors


def baseline_runs(old_bin: Path, root: Path) -> dict:
    """Recall the 12 queries twice with the reference version, each on a fresh start."""
    inst, runs, errors, status = instance(old_bin, root), [], [], ""
    for _ in range(2):
        with inst.session() as (up, _, note):
            if not up:
                errors.append(f"reference daemon did not start: {note}")
                break
            wait_warm(inst)
            res, st, errs = run_queries(inst)
            runs.append(res)
            errors.extend(errs)
            status = status or st
    return {"runs": runs, "status": status, "errors": errors}


def wait_migrated(inst: env.Instance, limit: int = 120) -> tuple[dict, bool]:
    """Poll /health until migrations report done; returns (readiness, timed_out)."""
    deadline, last, done = time.monotonic() + limit, {}, False
    while time.monotonic() < deadline:
        last = (inst.health() or {}).get("readiness", {})
        done = bool(last.get("migrations")) and not last.get("migration_failures")
        if done:
            break
        time.sleep(1)
    wait_warm(inst)
    return last, not done


def wait_warm(inst: env.Instance, limit: int = 45) -> None:
    """Let the recall indexes warm so the first answers are complete."""
    for _ in range(limit):
        warm = ((inst.health() or {}).get("readiness", {}).get("recall_warmup", {}) or {})
        if warm.get("warm"):
            return
        time.sleep(1)


def log_size(inst: env.Instance) -> int:
    log = inst.data_dir / "logs" / "daemon.log"
    return log.stat().st_size if log.exists() else 0


def log_text_since(inst: env.Instance, offset: int) -> list[str]:
    log = inst.data_dir / "logs" / "daemon.log"
    if not log.exists():
        return []
    raw = log.read_bytes()
    return (raw[offset:] if len(raw) >= offset else raw).decode(errors="replace").splitlines()


def migration_errors(lines: list[str]) -> list[str]:
    """Log lines that mention a failed migration."""
    return [ln.strip()[:300] for ln in lines
            if "migrat" in ln.lower() and ("failed" in ln.lower() or "error" in ln.lower())][:5]


def check_upgrade(new_bin: Path, root: Path, fixture: Path) -> tuple[dict, list[list[str]], str, list[str]]:
    """Open the copy with this branch.  Returns (detail, recall results, status, recall errors)."""
    inst = instance(new_bin, root)
    mem_db = fixture / "data" / "memory.db"
    before, counts_before = env.contents_by_ref(mem_db), env.base_table_counts(mem_db)
    started, offset = time.monotonic(), log_size(inst)
    results, status, query_errors, ready, timed_out = [], "", [], {}, False
    with inst.session() as (up, start_s, note):
        errors = [] if up else [f"daemon did not start: {note}"]
        if up:
            ready, timed_out = wait_migrated(inst)
            results, status, query_errors = run_queries(inst)
    after = env.contents_by_ref(inst.data_dir / "memory.db")
    version = env.schema_version(inst.data_dir / "learning.db")
    errors += migration_errors(log_text_since(inst, offset))
    errors += ["migrations did not finish within the time limit"] if timed_out else []
    counts_after = env.base_table_counts(inst.data_dir / "memory.db")
    detail = {"daemon_up": up, "start_seconds": round(start_s, 1), "readiness": ready, "timed_out": timed_out,
              "schema_version": version, "corpus_before": len(before), "corpus_after": len(after),
              "contents_unchanged": before == after, "errors": errors,
              "memory_counts_unchanged": core_counts_unchanged(counts_before, counts_after),
              "derived_count_changes": {k: [counts_before.get(k), counts_after.get(k)]
                                    for k in set(counts_before) | set(counts_after)
                                    if counts_before.get(k) != counts_after.get(k)},
              "seconds": round(time.monotonic() - started, 1)}
    detail["passed"] = upgrade_ok(detail)
    return detail, results, status, query_errors


def new_snapshots(data: Path, known: set[str]) -> list[Path]:
    snap = data / SNAP_DIR
    return sorted(p for p in snap.glob("*-pre-migration.db") if p.name not in known) if snap.is_dir() else []


def restore_in_copy(new_py: Path, snapshot: Path, target: Path, sha: str | None, scratch: env.Instance) -> str:
    """Restore through the product code, under the scratch HOME and data directory."""
    code = ("import sys; from pathlib import Path; "
            "from superlocalmemory.storage.backup import restore_pre_migration_snapshot as r; "
            "r(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] or None)")
    p = subprocess.run([str(new_py), "-c", code, str(snapshot), str(target), sha or ""],
                       capture_output=True, text=True, timeout=300, env=scratch.env(), cwd=scratch.home)
    return "" if p.returncode == 0 else (p.stderr or p.stdout)[-400:]


def snapshot_manifest(data: Path, snapshot: Path) -> dict:
    """The generation manifest entry (sha256 etc.) for ``snapshot``, or {}."""
    for mf in (data / SNAP_DIR).glob("manifest-*-pre-migration.json"):
        try:
            doc = json.loads(mf.read_text())
        except (OSError, ValueError):
            continue
        for entry in doc.get("files", []):
            if entry.get("snapshot") == snapshot.name:
                return entry
    return {}


def restore_one(new_py: Path, root: Path, snap: Path, fixture: Path, work: Path) -> dict:
    """Restore one snapshot over a copy of its database; compare with the fixture's."""
    kind = "memory.db" if snap.name.startswith("memory-") else "learning.db"
    data = root / "home" / env.DATA_SUBDIR
    entry = snapshot_manifest(data, snap)
    probe = work / f"restore-{kind}"
    shutil.copytree(data, probe)
    (work / "restore-home").mkdir(exist_ok=True)
    scratch = env.Instance(bin_dir=new_py.parent, data_dir=probe, home=work / "restore-home", version=LINE)
    err = restore_in_copy(new_py, probe / SNAP_DIR / snap.name, probe / kind, entry.get("sha256"), scratch)
    out: dict = {"db": kind, "snapshot": snap.name, "manifest_sha256_present": bool(entry.get("sha256"))}
    if err:
        return {**out, "ok": False, "error": f"restore failed: {err}"}
    want_db, got_db = fixture / "data" / kind, probe / kind
    want_cols, got_cols = env.table_columns(want_db), env.table_columns(got_db)
    contents_ok = kind != "memory.db" or env.contents_by_ref(got_db) == env.contents_by_ref(want_db)
    ok = restore_ok(want_cols, got_cols, env.table_counts(want_db), env.table_counts(got_db),
                    env.schema_version(want_db), env.schema_version(got_db), contents_ok)
    return {**out, "ok": ok, "contents_identical": contents_ok,
            "bytes_identical": env.sha256_file(got_db) == env.sha256_file(want_db),
            "extra_tables": sorted(set(got_cols) - set(want_cols)), "missing_tables": sorted(set(want_cols) - set(got_cols)),
            "schema_versions": {"fixture": env.schema_version(want_db), "restored": env.schema_version(got_db)}}


def _sig_diff(before: Path, after: Path) -> dict:
    return mapping_diff({r[1]: r[2] for r in env.schema_signature(before)},
                        {r[1]: r[2] for r in env.schema_signature(after)})


def check_snapshot(new_py: Path, root: Path, known: set[str], fixture: Path, work: Path) -> dict:
    """The upgrade left a snapshot, and restoring it gives back the pre-upgrade database."""
    data = root / "home" / env.DATA_SUBDIR
    snaps = new_snapshots(data, known)
    detail: dict = {"snapshots": [s.name for s in snaps], "passed": False, "errors": [], "restores": [],
                    "fixture_schema_version": env.schema_version(fixture / "data" / "learning.db")}
    rc, _, _, _ = instance(new_py.parent, root).run("db", "restore-points", "--json", timeout=60)
    detail["restore_points_rc"] = rc
    detail["schema_changes"] = {n: _sig_diff(fixture / "data" / n, data / n) for n in ("memory.db", "learning.db")}
    detail["schema_changed_without_snapshot"] = sorted(
        n for n, diff in detail["schema_changes"].items()
        if any(diff.values()) and not any(s.name.startswith(n.split(".")[0] + "-") for s in snaps))
    if not snaps and schema_unchanged(fixture / "data", data):
        detail.update(passed=None, status="n/a",
                      not_applicable="neither database changed schema, so the start takes no copy")
        return detail
    if not snaps:
        detail["errors"].append("no *-pre-migration.db was created although the schema changed")
        return detail
    detail["restores"] = [restore_one(new_py, root, s, fixture, work) for s in snaps]
    detail["errors"] += [r["error"] for r in detail["restores"] if r.get("error")]
    detail["passed"] = snapshot_passed(detail["restores"], rc)
    if rc != 0:
        detail["errors"].append(f"slm db restore-points exited {rc}")
    return detail


def check_downgrade(new_bin: Path, old_bin: Path, root: Path) -> tuple[dict, list[list[str]]]:
    """prepare-downgrade with this branch, then the older version opens and answers."""
    new_i, old_i = instance(new_bin, root), instance(old_bin, root)
    rc, out, err, secs = new_i.run("db", "prepare-downgrade", "--yes", "--json", timeout=300)
    detail: dict = {"prepare_rc": rc, "prepare_seconds": round(secs, 1), "errors": [], "passed": False}
    if rc != 0:
        detail["errors"].append(f"prepare-downgrade rc={rc}: {(err or out)[-400:]}")
    mem = root / "home" / env.DATA_SUBDIR / "memory.db"
    detail["schema_after_prepare"] = env.schema_version(mem.parent / "learning.db")
    before, offset, results = len(env.contents_by_ref(mem)), log_size(old_i), []
    with old_i.session() as (up, start_s, note):
        if up:
            wait_warm(old_i)
            results, _, errs = run_queries(old_i)
            detail["errors"] += errs
        else:
            detail["errors"].append(f"reference daemon did not start: {note}")
    refused = [ln.strip()[:300] for ln in log_text_since(old_i, offset) if "SchemaVersionError" in ln][-3:]
    detail["errors"] += refused
    detail.update(daemon_up=up, start_seconds=round(start_s, 1), corpus_before=before,
                  corpus_after=len(env.contents_by_ref(mem)),
                  schema_version=env.schema_version(mem.parent / "learning.db"),
                  queries_answered=sum(1 for r in results if r))
    return detail, results


def downgrade_to(new_py: Path, old_py: Path, root: Path, runs: list, version: str, ceiling: int) -> dict:
    """One downgrade, judged against the reference baseline."""
    detail, results = check_downgrade(new_py.parent, old_py.parent, root)
    vr = recall_verdict(runs[0], runs[1], results) if len(runs) == 2 and results else None
    detail.update(downgraded_to=version, ceiling=ceiling, recall_vs_baseline=vr,
                  passed=downgrade_ok(detail, vr, ceiling))
    return detail


def run_checks(fixture: Path, old_py: Path, new_py: Path, work: Path, down_py: Path | None = None) -> dict:
    """All checks for one fixture; each is recorded even when an earlier one fails."""
    env.require_version(old_py, EXPECTED_OLD)
    if down_py:
        env.require_version(down_py, EXPECTED_DOWN)
    manifest = json.loads((fixture / "manifest.json").read_text())
    snap_dir = fixture / "data" / SNAP_DIR
    known = {p.name for p in snap_dir.glob("*")} if snap_dir.is_dir() else set()
    base_root = copy_fixture(fixture, "baseline", work)
    new_root = copy_fixture(fixture, "upgraded", work)
    started = time.monotonic()
    base = baseline_runs(old_py.parent, base_root)
    upgrade, new_results, status, query_errors = check_upgrade(new_py.parent, new_root, fixture)
    snapshot = check_snapshot(new_py, new_root, known, fixture, work)
    checks = {"upgrade": upgrade, "snapshot": snapshot,
              "recall": recall_check(base, new_results, query_errors, status)}
    if down_py:
        alt_root = work / "upgraded-alt"
        shutil.copytree(new_root, alt_root)
        checks[f"downgrade_to_{EXPECTED_DOWN}"] = downgrade_to(new_py, down_py, alt_root, base["runs"], EXPECTED_DOWN, 53)
    checks["downgrade"] = downgrade_to(new_py, old_py, new_root, base["runs"], EXPECTED_OLD, 54)
    return {"fixture_version": manifest["version"], "fixture_embedding_mode": manifest["embedding_mode"],
            "checks": checks, "failing": failing(checks), "total_seconds": round(time.monotonic() - started, 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixture", required=True, type=Path)
    ap.add_argument("--old-python", required=True, type=Path, help="python of the 4.1.24 venv")
    ap.add_argument("--new-python", required=True, type=Path, help="python of the branch venv")
    ap.add_argument("--down-python", type=Path, help="python of a 4.1.20 venv (adds a second downgrade)")
    ap.add_argument("--out", type=Path, help="write the JSON verdict here (default: stdout)")
    ap.add_argument("--keep", action="store_true", help="keep the scratch copies")
    args = ap.parse_args(argv)
    work = env.make_work_dir()
    try:
        verdict = run_checks(args.fixture, args.old_python, args.new_python, work, args.down_python)
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    text = json.dumps(verdict, indent=2, sort_keys=True)
    args.out.write_text(text + "\n") if args.out else print(text)
    return 1 if verdict["failing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
