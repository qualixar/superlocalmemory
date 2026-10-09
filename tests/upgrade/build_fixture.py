# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Build a store with one SLM version and the synthetic corpus.

    python build_fixture.py --version 4.1.20 --out DIR [--venv DIR] [--py 3.12]

Creates a throwaway venv, installs ``superlocalmemory==VERSION`` from PyPI,
saves the corpus through that version's own ``slm remember`` command (never by
writing SQL), waits for the store to settle, stops the daemon and writes
``DIR/manifest.json`` next to ``DIR/data``.  Synthetic memories only.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _slm_env as env  # noqa: E402
import corpus  # noqa: E402

REQUIRED = {
    "version": str, "python": str, "cli": str, "counts": dict,
    "sha256": dict, "embedding_mode": str, "built_at": str,
}
DB_NAMES = ("memory.db", "learning.db")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_manifest(m: dict) -> list[str]:
    """Return a list of problems with a manifest (empty when it is well formed)."""
    problems = []
    for key, typ in REQUIRED.items():
        if key not in m:
            problems.append(f"missing key: {key}")
        elif not isinstance(m[key], typ):
            problems.append(f"{key} must be {typ.__name__}")
    counts = m.get("counts")
    if isinstance(counts, dict) and not all(isinstance(v, int) for v in counts.values()):
        problems.append("counts values must be integers")
    shas = m.get("sha256")
    if isinstance(shas, dict) and not all(isinstance(v, str) and SHA_RE.match(v) for v in shas.values()):
        problems.append("sha256 values must be 64 hex characters")
    return problems


def make_venv(venv: Path, py: str) -> None:
    if (venv / "bin" / "slm").exists():
        return
    subprocess.run(["uv", "venv", str(venv), "-p", py, "-q"], check=True)


def install(venv: Path, version: str) -> float:
    started = time.monotonic()
    subprocess.run(["uv", "pip", "install", "-p", str(venv), f"superlocalmemory=={version}", "-q"],
                   check=True)
    return time.monotonic() - started


def classify_embedding(status_json: str) -> str:
    """Name the recall mode from a ``channel_status`` blob."""
    if not status_json:
        return "unknown (no channel_status reported)"
    try:
        status = json.loads(status_json)
    except json.JSONDecodeError:
        return "unknown (unparsable channel_status)"
    semantic = str(status.get("semantic", "n/a"))
    if semantic == "ok":
        return f"embeddings active ({status_json})"
    return f"keyword-only fallback: semantic={semantic} ({status_json})"


LOG_MARKERS = ("Query embedding returned None", "Model load failed", "no_embedding", "Embedding worker unavailable")


def log_hint(data: Path) -> str:
    """First daemon-log line showing the embedder was unavailable ('' when none)."""
    log = data / "logs" / "daemon.log"
    if not log.exists():
        return ""
    for line in log.read_text(errors="replace").splitlines():
        if any(mark in line for mark in LOG_MARKERS):
            return line.strip()[:200]
    return ""


def write_corpus(inst: env.Instance) -> list[str]:
    """Save every memory through the CLI; returns error strings (empty when all saved)."""
    errors = []
    for m in corpus.MEMORIES:
        rc, out, err, _ = inst.run("remember", m.text, "--sync", "--json", "--tags", m.tag, timeout=180)
        if rc != 0:
            errors.append(f"{m.id}: rc={rc} {(err or out)[-200:]}")
    return errors


def wait_settled(data_dir: Path, want: int = len(corpus.MEMORIES), limit: int = 240) -> int:
    """Wait until the number of corpus ids found in memory.db is stable; returns it."""
    last, stable, deadline = -1, 0, time.monotonic() + limit
    while time.monotonic() < deadline:
        now = len(env.contents_by_ref(data_dir / "memory.db"))
        stable = stable + 1 if now == last else 0
        last = now
        if now >= want and stable >= 2:
            break
        time.sleep(3)
    return last


def build(version: str, out: Path, venv: Path, py: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    make_venv(venv, py)
    install_s = install(venv, version)
    home = out / "home"
    data = home / env.DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    inst = env.Instance(bin_dir=venv / "bin", data_dir=data, home=home)
    up, start_s, note = inst.serve_start()
    errors = [] if up else [f"daemon did not start: {note}"]
    status = ""
    if up:
        errors += write_corpus(inst)
        settled = wait_settled(data)
        _, status, probe_err = inst.recall_ids(corpus.QUERIES[0].text)
        errors += [probe_err] if probe_err else []
    inst.serve_stop()
    return _finish(version, venv, data, status, settled if up else 0, errors, install_s, start_s)


def _finish(version, venv, data, status, settled, errors, install_s, start_s) -> dict:
    """Build the manifest, then move the store to ``out/data`` and drop the scratch HOME."""
    manifest = _manifest(version, venv, data, status, settled, errors, install_s, start_s)
    out = data.parent.parent
    final = out / "data"
    shutil.rmtree(final, ignore_errors=True)
    shutil.move(str(data), str(final))
    shutil.rmtree(out / "home", ignore_errors=True)
    manifest["sha256"] = {n: env.sha256_file(final / n) for n in DB_NAMES if (final / n).exists()}
    return manifest


def _embedding_mode(status: str, data: Path) -> str:
    mode = classify_embedding(status)
    hint = log_hint(data)
    if hint and "keyword-only" not in mode:
        mode = f"keyword-only fallback (from daemon log); {mode}"
    return f"{mode} | log: {hint}" if hint else mode


def _manifest(version, venv, data, status, settled, errors, install_s, start_s) -> dict:
    py = subprocess.run([str(venv / "bin" / "python"), "--version"], capture_output=True, text=True)
    counts = {}
    for name in DB_NAMES:
        for table, n in env.table_counts(data / name).items():
            counts[f"{name}:{table}"] = n
    counts["corpus_ids_found"] = settled
    return {
        "version": version, "python": (py.stdout or py.stderr).strip().replace("Python ", ""),
        "cli": "slm remember <text> --sync --json --tags <tag>; slm recall <q> --json",
        "counts": counts,
        "sha256": {n: env.sha256_file(data / n) for n in DB_NAMES if (data / n).exists()},
        "embedding_mode": _embedding_mode(status, data),
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "install_seconds": round(install_s, 1), "daemon_start_seconds": round(start_s, 1),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--venv", type=Path, help="default: /tmp-style dir next to --out")
    ap.add_argument("--py", default="3.12")
    args = ap.parse_args(argv)
    venv = args.venv or args.out.parent / f"venv-{args.version}"
    manifest = build(args.version, args.out, venv, args.py)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    problems = validate_manifest(manifest)
    print(json.dumps({"version": args.version, "problems": problems, "errors": manifest["errors"]}))
    return 1 if problems or manifest["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
