# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Isolated LightGBM work (retrain + consolidation) — OpenMP-crash guard.

**Why this exists** — the unified daemon serves the HTTP API in-process and,
by then, has already loaded PyTorch's OpenMP runtime (``torch/lib/libomp.dylib``)
with *warm* worker threads (the cross-encoder reranker + embedding warm-up at
startup). Any code path that imports ``lightgbm`` in that same process loads a
*second* OpenMP runtime — the Homebrew ``libomp.dylib`` that
``lib_lightgbm.dylib`` links against — which corrupts the shared ``__kmp``
global state and SIGSEGVs a pre-existing torch OMP worker thread. That
hard-crashes the whole daemon (observed on macOS / Apple Silicon;
``DiagnosticReports/Python-*.ips`` shows ``libomp.dylib __kmp_launch_worker``).

Two in-daemon entry points reach LightGBM training:
  * ``POST /api/learning/retrain``  → legacy ``_retrain_ranker_impl``;
  * ``POST /api/v3/learning/consolidate`` → ``ConsolidationWorker.run`` whose
    step 5 trains via the online ``_run_shadow_cycle`` (active model) or the
    legacy cold-start path.

Both are funnelled through this module so LightGBM only ever trains in a
**fresh subprocess** that never runs a torch tensor op — torch's OMP pool stays
dormant there and lightgbm's runtime is the only active one. The mechanism
mirrors the existing ``embedding_worker`` / ``reranker_worker`` isolation.

Spawned via :func:`run_retrain_isolated` / :func:`run_consolidation_isolated`,
which use a ``python -c`` bootstrap that imports lightgbm BEFORE the
``superlocalmemory`` package. That ordering is load-bearing: importing the
package transitively pulls in torch, and torch-OMP-first-then-lightgbm is
exactly the sequence that segfaults. A plain ``python -m
superlocalmemory.learning.lightgbm_subprocess`` would run the package
``__init__`` (torch) first and crash — do not invoke it that way.

The child emits a single JSON line on stdout — ``{"error": str|null, ...}`` —
plus a task payload (``trained`` for retrain, ``stats`` for consolidate).
Exit 0 = ran to completion; non-zero = handled failure. Either way the parent
(the daemon) keeps running.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Wall-clock ceilings. A retrain is 50 boosting rounds on ≤2000 rows (seconds);
# a full consolidation cycle also decays/dedups/mines/graph-analyses, so it
# gets more headroom.
RETRAIN_TIMEOUT_SEC = 300
CONSOLIDATE_TIMEOUT_SEC = 600


# ---------------------------------------------------------------------------
# Child-side task implementations (run in the isolated subprocess).
# ---------------------------------------------------------------------------

def _retrain(learning_db: str, profile_id: str, *, include_synthetic: bool) -> bool:
    """Legacy ranker retrain in this (lightgbm-first) process. Returns trained."""
    import lightgbm  # noqa: F401  (import-order side effect is intentional)

    from superlocalmemory.learning.ranker_retrain_legacy import (
        _retrain_ranker_impl,
    )

    return bool(
        _retrain_ranker_impl(
            learning_db, profile_id, include_synthetic=include_synthetic,
        )
    )


def _consolidate(
    memory_db: str, learning_db: str, profile_id: str, *, dry_run: bool,
) -> dict:
    """Full consolidation cycle (incl. step-5 LightGBM training). Returns stats."""
    import lightgbm  # noqa: F401  (import-order side effect is intentional)

    from superlocalmemory.learning.consolidation_worker import (
        ConsolidationWorker,
    )

    worker = ConsolidationWorker(memory_db, learning_db)
    return worker.run(profile_id, dry_run=dry_run)


def _emit(obj: dict) -> None:
    """Write the verdict as a single JSON line (default=str for safety)."""
    sys.stdout.write(json.dumps(obj, default=str) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="slm-lightgbm-subprocess")
    parser.add_argument("--task", choices=["retrain", "consolidate"], default="retrain")
    parser.add_argument("--learning-db", required=True)
    parser.add_argument("--memory-db")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--include-synthetic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    result: dict = {"error": None}
    try:
        if args.task == "retrain":
            result["trained"] = _retrain(
                args.learning_db, args.profile,
                include_synthetic=args.include_synthetic,
            )
        else:
            if not args.memory_db:
                raise ValueError("--memory-db is required for the consolidate task")
            result["stats"] = _consolidate(
                args.memory_db, args.learning_db, args.profile,
                dry_run=args.dry_run,
            )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        _emit(result)
        return 1

    _emit(result)
    return 0


# ---------------------------------------------------------------------------
# Caller-side helpers — spawn the child in an isolated process.
# ---------------------------------------------------------------------------

def _run_isolated(task_args: list[str], *, timeout_sec: int) -> dict:
    """Spawn the child with lightgbm imported first; return its JSON verdict.

    Never raises for the expected failure modes (non-zero exit, timeout,
    native crash with no JSON) — encodes them in an ``error`` key so the
    calling daemon stays up no matter what.
    """
    # The leading ``import lightgbm`` is load-bearing: it must run BEFORE the
    # superlocalmemory package (which transitively imports torch). ``-c`` is
    # the only way to guarantee that ordering — ``-m superlocalmemory.…`` runs
    # the package __init__ (torch) first and would reintroduce the crash.
    bootstrap = (
        "import lightgbm; "
        "from superlocalmemory.learning.lightgbm_subprocess import main; "
        "raise SystemExit(main())"
    )
    cmd = [sys.executable, "-c", bootstrap, *task_args]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout_sec}s"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"spawn failed: {exc}"}

    # The verdict is the last JSON line on stdout. Parse defensively — a native
    # crash in the child would leave no JSON, which we surface plainly.
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except ValueError:
            continue

    tail = (proc.stderr or "").strip()[-500:]
    return {
        "error": (
            f"subprocess produced no verdict (exit={proc.returncode}). "
            f"stderr tail: {tail}"
        ),
    }


def run_retrain_isolated(
    learning_db: str | Path,
    profile_id: str,
    *,
    include_synthetic: bool = False,
    timeout_sec: int = RETRAIN_TIMEOUT_SEC,
) -> dict:
    """Train the ranker in a subprocess. Returns ``{"trained": bool, "error": ...}``."""
    args = [
        "--task", "retrain",
        "--learning-db", str(learning_db),
        "--profile", profile_id,
    ]
    if include_synthetic:
        args.append("--include-synthetic")
    verdict = _run_isolated(args, timeout_sec=timeout_sec)
    verdict.setdefault("trained", False)
    verdict.setdefault("error", None)
    return verdict


def run_consolidation_isolated(
    memory_db: str | Path,
    learning_db: str | Path,
    profile_id: str,
    *,
    dry_run: bool = False,
    timeout_sec: int = CONSOLIDATE_TIMEOUT_SEC,
) -> dict:
    """Run the full consolidation cycle in a subprocess (its step-5 training
    would otherwise crash the torch-warm daemon).

    Returns ``{"stats": dict|None, "error": str|None}``.
    """
    args = [
        "--task", "consolidate",
        "--memory-db", str(memory_db),
        "--learning-db", str(learning_db),
        "--profile", profile_id,
    ]
    if dry_run:
        args.append("--dry-run")
    verdict = _run_isolated(args, timeout_sec=timeout_sec)
    verdict.setdefault("stats", None)
    verdict.setdefault("error", None)
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
