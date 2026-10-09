# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Open an old-version store with this branch, then go back to 4.1.24.

    python upgrade_check.py --fixture DIR --old-python P --new-python P [--out F]

``--old-python`` belongs to the venv holding 4.1.24 (baseline and downgrade);
``--new-python`` to the venv holding this branch.  The fixture is never
modified: every step works on a copy.  Writes a JSON verdict.
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

TARGET_SCHEMA = 54
SNAP_DIR = "pre-migration-snapshots"


def overlap(a: list[str], b: list[str]) -> int:
    """How many ids the two top-5 lists share."""
    return len(set(a[:5]) & set(b[:5]))


def recall_verdict(base_a: list[list[str]], base_b: list[list[str]], new: list[list[str]]) -> dict:
    """Judge ``new`` against two runs of the reference version.

    ``baseline_overlap`` is how well the reference matches itself (measured
    nondeterminism); ``new_overlap`` is the mean of new-vs-first and new-vs-second.
    """
    baseline = sum(overlap(x, y) for x, y in zip(base_a, base_b))
    vs_a = sum(overlap(x, y) for x, y in zip(new, base_a))
    vs_b = sum(overlap(x, y) for x, y in zip(new, base_b))
    new_overlap = (vs_a + vs_b) / 2
    exact = all(set(n[:5]) == set(a[:5]) == set(b[:5]) for n, a, b in zip(new, base_a, base_b))
    if exact:
        verdict = "identical"
    elif new_overlap >= baseline:
        verdict = "within_noise"
    else:
        verdict = "worse"
    return {"verdict": verdict, "baseline_overlap": baseline, "new_overlap": new_overlap,
            "possible": sum(min(5, len(a)) for a in base_a)}


def copy_fixture(fixture: Path, label: str, work: Path) -> Path:
    dest = work / label
    (dest / "home").mkdir(parents=True)
    shutil.copytree(fixture / "data", dest / "home" / env.DATA_SUBDIR)
    return dest


def instance(bin_dir: Path, root: Path) -> env.Instance:
    return env.Instance(bin_dir=bin_dir, data_dir=root / "home" / env.DATA_SUBDIR, home=root / "home")


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
        up, _, note = inst.serve_start()
        if not up:
            errors.append(f"reference daemon did not start: {note}")
            break
        wait_warm(inst)
        res, st, errs = run_queries(inst)
        runs.append(res), errors.extend(errs)
        status = status or st
        inst.serve_stop()
    return {"runs": runs, "status": status, "errors": errors}


def wait_migrated(inst: env.Instance, limit: int = 120) -> dict:
    """Poll /health until migrations report done (or the limit); returns the readiness blob."""
    deadline, last = time.monotonic() + limit, {}
    while time.monotonic() < deadline:
        h = inst.health() or {}
        last = h.get("readiness", {})
        if last.get("migrations") and not last.get("migration_failures"):
            break
        time.sleep(1)
    wait_warm(inst)
    return last


def wait_warm(inst: env.Instance, limit: int = 45) -> None:
    """Let the recall indexes warm so the first answers are complete."""
    for _ in range(limit):
        warm = ((inst.health() or {}).get("readiness", {}).get("recall_warmup", {}) or {})
        if warm.get("warm"):
            return
        time.sleep(1)


def log_lines_since(inst: env.Instance, offset: int) -> list[str]:
    """Daemon-log lines written after ``offset`` bytes that mention a failed migration."""
    log = inst.data_dir / "logs" / "daemon.log"
    if not log.exists():
        return []
    text = log.read_bytes()
    text = text[offset:] if len(text) >= offset else text
    return [ln.strip()[:300] for ln in text.decode(errors="replace").splitlines()
            if "migrat" in ln.lower() and ("failed" in ln.lower() or "error" in ln.lower())][:5]


def log_size(inst: env.Instance) -> int:
    log = inst.data_dir / "logs" / "daemon.log"
    return log.stat().st_size if log.exists() else 0


def check_upgrade(new_bin: Path, root: Path, fixture: Path) -> tuple[dict, list[list[str]], str]:
    inst = instance(new_bin, root)
    before = env.contents_by_ref(fixture / "data" / "memory.db")
    started, offset = time.monotonic(), log_size(inst)
    up, start_s, note = inst.serve_start()
    detail: dict = {"daemon_up": up, "start_seconds": round(start_s, 1)}
    errors = [] if up else [f"daemon did not start: {note}"]
    results, status = [], ""
    if up:
        detail["readiness"] = wait_migrated(inst)
        results, status, errs = run_queries(inst)
        errors += errs
    inst.serve_stop()
    after = env.contents_by_ref(inst.data_dir / "memory.db")
    version = env.schema_version(inst.data_dir / "learning.db")
    detail.update(schema_version=version, corpus_before=len(before), corpus_after=len(after),
                  contents_unchanged=before == after, seconds=round(time.monotonic() - started, 1))
    errors += log_lines_since(inst, offset)
    detail["passed"] = bool(up and version == TARGET_SCHEMA and before == after and len(after) == len(corpus.MEMORIES))
    detail["errors"] = errors
    return detail, results, status


def new_snapshots(data: Path, known: set[str]) -> list[Path]:
    snap = data / SNAP_DIR
    return sorted(p for p in snap.glob("*-pre-migration.db") if p.name not in known) if snap.is_dir() else []


def restore_in_copy(new_py: Path, snapshot: Path, target: Path, sha: str | None) -> str:
    code = ("import sys; from pathlib import Path; "
            "from superlocalmemory.storage.backup import restore_pre_migration_snapshot as r; "
            "r(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] or None)")
    p = subprocess.run([str(new_py), "-c", code, str(snapshot), str(target), sha or ""],
                       capture_output=True, text=True, timeout=300)
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


def check_snapshot(new_py: Path, root: Path, known: set[str], fixture: Path, work: Path) -> dict:
    """A snapshot made during the upgrade exists and restores the pre-upgrade store."""
    data = root / "home" / env.DATA_SUBDIR
    snaps = new_snapshots(data, known)
    mem = next((s for s in snaps if s.name.startswith("memory-")), None)
    detail: dict = {"snapshots": [s.name for s in snaps], "passed": False, "errors": [],
                    "fixture_schema_version": env.schema_version(fixture / "data" / "learning.db")}
    if mem is None:
        detail["errors"].append("no memory-*-pre-migration.db created by the upgrade")
        return detail
    entry = snapshot_manifest(data, mem)
    detail["manifest_sha256_present"] = bool(entry.get("sha256"))
    probe = work / "restore-probe"
    shutil.copytree(data, probe)
    err = restore_in_copy(new_py, probe / SNAP_DIR / mem.name, probe / "memory.db", entry.get("sha256"))
    if err:
        detail["errors"].append(f"restore failed: {err}")
        return detail
    want_rows = env.table_counts(fixture / "data" / "memory.db")
    got_rows = env.table_counts(probe / "memory.db")
    detail["bytes_identical"] = env.sha256_file(probe / "memory.db") == env.sha256_file(fixture / "data" / "memory.db")
    extra = {k: v for k, v in got_rows.items() if k not in want_rows}
    detail["extra_tables"] = extra  # tables the new version created before the copy was taken
    detail["rows_identical"] = (all(got_rows.get(k) == v for k, v in want_rows.items())
                                and not any(extra.values()))
    detail["contents_identical"] = env.contents_by_ref(probe / "memory.db") == env.contents_by_ref(fixture / "data" / "memory.db")
    detail["passed"] = bool(detail["rows_identical"] and detail["contents_identical"])
    return detail


def check_downgrade(new_bin: Path, old_bin: Path, root: Path) -> tuple[dict, list[list[str]]]:
    """prepare-downgrade with this branch, then the reference version opens and answers."""
    new_i, old_i = instance(new_bin, root), instance(old_bin, root)
    rc, out, err, secs = new_i.run("db", "prepare-downgrade", "--yes", "--json", timeout=300)
    detail: dict = {"prepare_rc": rc, "prepare_seconds": round(secs, 1), "errors": [], "passed": False}
    if rc != 0:
        detail["errors"].append(f"prepare-downgrade rc={rc}: {(err or out)[-400:]}")
    before = len(env.contents_by_ref(root / "home" / env.DATA_SUBDIR / "memory.db"))
    offset = log_size(old_i)
    up, start_s, note = old_i.serve_start()
    results: list[list[str]] = []
    if up:
        wait_warm(old_i)
        results, _, errs = run_queries(old_i)
        detail["errors"] += errs
    else:
        detail["errors"].append(f"reference daemon did not start: {note}")
    old_i.serve_stop()
    log = root / "home" / env.DATA_SUBDIR / "logs" / "daemon.log"
    raw = log.read_bytes() if log.exists() else b""
    text = (raw[offset:] if len(raw) >= offset else raw).decode(errors="replace")
    refused = [ln.strip()[:300] for ln in text.splitlines() if "SchemaVersionError" in ln][-3:]
    detail["errors"] += refused
    after = len(env.contents_by_ref(root / "home" / env.DATA_SUBDIR / "memory.db"))
    detail.update(daemon_up=up, start_seconds=round(start_s, 1), corpus_before=before, corpus_after=after,
                  schema_version=env.schema_version(root / "home" / env.DATA_SUBDIR / "learning.db"))
    answered = sum(1 for r in results if r)
    detail["queries_answered"] = answered
    detail["passed"] = bool(up and not refused and before == after == len(corpus.MEMORIES) and answered > 0)
    return detail, results


def run_checks(fixture: Path, old_py: Path, new_py: Path, work: Path) -> dict:
    """All four checks for one fixture; each is recorded even when an earlier one fails."""
    manifest = json.loads((fixture / "manifest.json").read_text())
    old_bin, new_bin = old_py.parent, new_py.parent
    known = {p.name for p in (fixture / "data" / SNAP_DIR).glob("*")} if (fixture / "data" / SNAP_DIR).is_dir() else set()
    base_root = copy_fixture(fixture, "baseline", work)
    new_root = copy_fixture(fixture, "upgraded", work)
    started = time.monotonic()
    base = baseline_runs(old_bin, base_root)
    upgrade, new_results, status = check_upgrade(new_bin, new_root, fixture)
    snapshot = check_snapshot(new_py, new_root, known, fixture, work)
    down, down_results = check_downgrade(new_bin, old_bin, new_root)
    runs = base["runs"]
    recall: dict = {"errors": base["errors"], "reference_status": base["status"], "new_status": status}
    if len(runs) == 2 and new_results:
        recall.update(recall_verdict(runs[0], runs[1], new_results))
        recall["passed"] = recall["verdict"] != "worse"
        down["recall_vs_baseline"] = recall_verdict(runs[0], runs[1], down_results) if down_results else None
    else:
        recall["passed"] = False
        recall["errors"].append("reference or new recall produced no results")
    return {"fixture_version": manifest["version"], "fixture_embedding_mode": manifest["embedding_mode"],
            "checks": {"upgrade": upgrade, "snapshot": snapshot, "recall": recall, "downgrade": down},
            "total_seconds": round(time.monotonic() - started, 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixture", required=True, type=Path)
    ap.add_argument("--old-python", required=True, type=Path, help="python of the 4.1.24 venv")
    ap.add_argument("--new-python", required=True, type=Path, help="python of the branch venv")
    ap.add_argument("--out", type=Path, help="write the JSON verdict here (default: stdout)")
    ap.add_argument("--keep", action="store_true", help="keep the scratch copies")
    args = ap.parse_args(argv)
    work = env.make_work_dir()
    try:
        verdict = run_checks(args.fixture, args.old_python, args.new_python, work)
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    text = json.dumps(verdict, indent=2, sort_keys=True)
    args.out.write_text(text + "\n") if args.out else print(text)
    return 0 if all(c.get("passed") for c in verdict["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
