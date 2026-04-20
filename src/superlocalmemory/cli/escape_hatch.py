# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Stage 8 SB-5

"""CLI escape-hatch commands.

Implements the user-facing kill switches and reset commands promised
by MASTER-PLAN §8 that were previously advertised but missing:

- ``slm disable``      — write ``~/.superlocalmemory/.disabled`` marker
  and stop the daemon. Hooks and MCP tools no-op until re-enabled.
- ``slm enable``       — remove the marker; prompt user to start daemon.
- ``slm clear-cache``  — wipe cache-only databases (preserves memory.db
  — user memories are NEVER deleted here).
- ``slm reconfigure``  — invoke the interactive postinstall with the
  ``--reconfigure`` flag so users can change profile/knobs safely.
- ``slm benchmark``    — run the evo-memory benchmark harness against
  an isolated tmp DB (never touches ``~/.superlocalmemory``).

All handlers are stdlib-only at the command boundary; import heavier
modules lazily inside each function so CLI help stays snappy.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path


def cmd_disable(args: Namespace) -> None:
    """Write the ``.disabled`` marker and stop the daemon."""
    from superlocalmemory.core.slm_disabled import write_marker, is_disabled

    reason = getattr(args, "reason", "") or ""
    marker = write_marker(reason)
    print(f"SLM disabled. Marker: {marker}")

    # Best-effort daemon stop — non-fatal.
    try:
        from superlocalmemory.cli.daemon import stop_daemon
        if stop_daemon():
            print("Daemon stopped.")
    except Exception as exc:  # pragma: no cover — defensive
        print(f"(daemon stop skipped: {exc})", file=sys.stderr)

    # Sanity check
    if not is_disabled():  # pragma: no cover
        print("warning: disable marker wrote but is_disabled() returned False",
              file=sys.stderr)


def cmd_enable(args: Namespace) -> None:
    """Remove the ``.disabled`` marker. Daemon is NOT auto-started —
    print the one-liner so the user can start it when ready."""
    from superlocalmemory.core.slm_disabled import remove_marker, marker_path

    removed = remove_marker()
    if removed:
        print("SLM enabled. To start the daemon: slm serve start")
    else:
        print(f"SLM was already enabled (no marker at {marker_path()}).")


_CACHE_DBS = (
    "active_brain_cache.db",
    "context_cache.db",
    "entity_trigram_cache.db",
)


def cmd_clear_cache(args: Namespace) -> None:
    """Delete cache-only DBs. Preserves ``memory.db`` and ``learning.db``.

    This is an explicitly non-destructive command — user memories live
    in ``memory.db`` which is NEVER touched here. Only regenerable
    caches are wiped so SLM can rebuild them on next daemon start.
    """
    from superlocalmemory.core.slm_disabled import _slm_home

    home = _slm_home()
    removed: list[str] = []
    missing: list[str] = []
    protected = {"memory.db", "learning.db", "audit.db", "audit_chain.db",
                 "pending.db"}
    for name in _CACHE_DBS:
        if name in protected:  # pragma: no cover — defensive belt
            continue
        path = home / name
        try:
            path.unlink()
            removed.append(name)
        except FileNotFoundError:
            missing.append(name)
        except OSError as exc:
            print(f"  skip {name}: {exc}", file=sys.stderr)

    if removed:
        print("Removed cache DBs:")
        for name in removed:
            print(f"  - {name}")
    else:
        print("No cache DBs present — nothing to do.")
    print("memory.db / learning.db preserved (user memories are never "
          "touched by clear-cache).")


def cmd_reconfigure(args: Namespace) -> None:
    """Re-run the interactive postinstall with ``--reconfigure``.

    Backs up the existing ``config.toml`` to ``config.toml.bak`` before
    writing the new one.
    """
    script = _find_postinstall_script()
    if script is None:
        print("postinstall-interactive.js not found; reinstall the npm "
              "package to run this command.", file=sys.stderr)
        sys.exit(2)

    cmd = ["node", str(script), "--reconfigure"]
    # Pass through any extra CLI flags untouched.
    extras = getattr(args, "extras", None) or []
    cmd.extend(extras)
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


def _find_postinstall_script() -> Path | None:
    """Locate ``scripts/postinstall-interactive.js`` next to the package."""
    # 1. Walk up from this file.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "scripts" / "postinstall-interactive.js"
        if candidate.is_file():
            return candidate
    # 2. npm global-install layout — sibling of the bin dir.
    bin_path = shutil.which("slm")
    if bin_path:
        bin_dir = Path(bin_path).resolve().parent
        candidate = bin_dir.parent / "lib" / "node_modules" / \
            "superlocalmemory" / "scripts" / "postinstall-interactive.js"
        if candidate.is_file():
            return candidate
    return None


def cmd_benchmark(args: Namespace) -> None:
    """Run the evo-memory benchmark harness against an isolated tmp DB.

    The benchmark refuses to touch the user's ``~/.superlocalmemory`` —
    it spins up its own tmp_path DB, runs day-0 → day-30 simulation,
    prints MRR / Recall / lift tiers, and exits. CI-safe.
    """
    # Import locally — keeps the CLI top-level light.
    try:
        # The harness lives in the test tree so we add it to sys.path lazily.
        repo_root = Path(__file__).resolve().parents[3]
        tests_dir = repo_root / "tests" / "test_benchmarks"
        if tests_dir.is_dir() and str(tests_dir) not in sys.path:
            sys.path.insert(0, str(tests_dir))
        from evo_memory import EvoMemoryBenchmark  # type: ignore
    except ImportError as exc:
        print(f"evo-memory harness not available in this install "
              f"(bench fixture excluded from binary wheels): {exc}",
              file=sys.stderr)
        sys.exit(2)

    import json
    import tempfile
    as_json = bool(getattr(args, "json", False))

    with tempfile.TemporaryDirectory(prefix="slm-bench-") as d:
        bench = EvoMemoryBenchmark(profile_id="bench_v1", data_dir=Path(d))
        result = bench.run_full_30_day_simulation()

    if as_json:
        print(json.dumps(result, default=str))
        return

    print("== Evo-Memory Benchmark ==")
    print(f"fixture:       {result.get('fixture_version', 'v1')}")
    print(f"wall_seconds:  {result.get('wall_seconds'):.2f}")
    cmp = result.get("comparison", {})
    print(f"day_1 MRR@10:  {cmp.get('day_1_mrr', 0):.4f}")
    print(f"day_30 MRR@10: {cmp.get('day_n_mrr', 0):.4f}")
    print(f"lift:          {cmp.get('mrr_lift_pct', 0):.2f}%  "
          f"(gate: {'STABLE' if cmp.get('passes_10pct_gate') else 'DRAFT'})")


def cmd_rotate_token(args: Namespace) -> None:
    """S-M07: rotate the install token.

    Prints the new token's last-4 suffix for operator audit + a line
    reminding the user to restart the daemon so HMAC marker caches
    converge on the new token.
    """
    from superlocalmemory.core.security_primitives import rotate_install_token
    old, new = rotate_install_token()
    if not new:
        print("install-token rotation failed (check permissions on "
              "~/.superlocalmemory/.install_token)")
        return
    old_tail = old[-4:] if old else "(absent)"
    new_tail = new[-4:]
    print(f"install-token rotated: ...{old_tail} -> ...{new_tail}")
    print("NOTE: restart the daemon so HMAC marker caches pick up "
          "the new token (run: slm restart).")


__all__ = (
    "cmd_disable",
    "cmd_enable",
    "cmd_clear_cache",
    "cmd_reconfigure",
    "cmd_benchmark",
    "cmd_rotate_token",
)
