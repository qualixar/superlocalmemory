"""Drift gate F-16 — structurally prevents re-introduction of hand-picked migrations.

Two properties:
  (a) No exp*.py may import individual M0xx migration modules (AST check).
  (b) exp7's guarantee still holds (smoke run with non-zero hold rate).

Discover experiment files FROM DISK; do not hardcode a list.
"""
from __future__ import annotations

import ast
import os
import pathlib
import re
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BENCHMARK = _REPO_ROOT / "benchmark"

# Ensure benchmark suite resolves to declared source tree
os.environ.setdefault("SLM_SOURCE_ROOT", str(_REPO_ROOT / "src"))
if str(_BENCHMARK) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK))


_MIGRATION_MODULE_RE = re.compile(r"^M\d{3}_")


def _collect_migration_enumerations(exp_path: pathlib.Path) -> list[str]:
    """Return list of M0xx names imported from superlocalmemory.storage.migrations."""
    tree = ast.parse(exp_path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "superlocalmemory.storage.migrations" or mod.startswith(
                "superlocalmemory.storage.migrations."
            ):
                for alias in node.names:
                    # alias.name could be "M018_ingestion_operations"
                    base = alias.name.split(".")[0]
                    if _MIGRATION_MODULE_RE.match(base):
                        hits.append(f"{mod}:{alias.name}")
                    # Also flag submodule import like "superlocalmemory.storage.migrations.M018_..."
                    # handled via module path
                # Also check if module itself is a specific M0xx submodule
                # e.g. from superlocalmemory.storage.migrations.M018_x import ...
                tail = mod.split(".")[-1] if "." in mod else ""
                if _MIGRATION_MODULE_RE.match(tail):
                    hits.append(f"{mod} (submodule import)")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith("superlocalmemory.storage.migrations"):
                    # e.g. import superlocalmemory.storage.migrations.M018_...
                    parts = name.split(".")
                    for p in parts:
                        if _MIGRATION_MODULE_RE.match(p):
                            hits.append(name)
                            break
    return hits


def test_no_experiment_hand_enumerates_migrations():
    """Flag any exp*.py that imports individual M0xx migration modules.

    The correct path is superlocalmemory.storage.migration_runner.apply_all;
    hand-picking M0xx modules snapshots a schema that keeps moving (F-16).
    """
    exp_files = sorted(_BENCHMARK.glob("exp*.py"))
    assert exp_files, "no exp*.py files found"

    violations: dict[str, list[str]] = {}
    for exp in exp_files:
        hits = _collect_migration_enumerations(exp)
        if hits:
            violations[exp.name] = hits

    assert not violations, (
        "Experiments hand-enumerate migration modules — use "
        "superlocalmemory.storage.migration_runner.apply_all instead: "
        f"{violations}"
    )


def test_exp7_smoke_hold_rate_non_zero():
    """Fast smoke: exp7 must still hold its guarantee (non-zero hold rate).

    This is the liveness companion to the AST gate: a future schema change
    that breaks exp7 should turn CI red instead of silently producing 0/200.
    """
    exp7 = pytest.importorskip(
        "exp7_generation_fence", reason="benchmark suite not present"
    )
    # Small n for speed; non-zero hold rate is the invariant.
    result = exp7.run(n_trials=4)
    assert result.held > 0, (
        f"exp7 smoke failed: {result.held}/{result.trials} held — "
        f"fence guarantee silent failure (failures={result.failures[:2]})"
    )
    # Stronger: expect full pass for the small smoke.
    assert result.held == result.trials, (
        f"exp7 smoke partial: {result.held}/{result.trials}"
    )
