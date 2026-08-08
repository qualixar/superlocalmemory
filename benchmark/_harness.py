"""Shared harness for the SLM 4.0 reliability evaluation.

Every experiment imports the *installed* ``superlocalmemory`` package and drives
its real code paths — real SQLite databases, the real backup coordinator, the
real migration runner, the real scope/authorization layer.

The spine services (ErasureService, MemoryTransactionService,
OperationPolicyRegistry, CanonicalRememberRuntime) are exercised with zero
mocking.  Projection-owner implementations vary by experiment: exp1 uses the
real production Bm25Owner/TemporalOwner/VectorOwner from concrete_owners.py;
exp2 uses lightweight _TrackingOwner instances (a minimal but complete
ProjectionOwner implementation) to exercise the service and obligation paths in
isolation.  The only other synthetic input is a deterministic embedding function
used where a vector is structurally required; embedding *quality* is never the
property under measurement (plumbing correctness is).

Design rules:
  * Fail loud. A broken harness must crash the run, never silently fabricate a
    "pass". Only the *specific* exception that a guarantee promises to raise is
    caught; everything else propagates.
  * Immutable results (frozen dataclasses); deterministic per-trial seeds.
  * Each experiment exposes ``run(n_trials, seed) -> ExperimentResult``.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Source-root guard (F-18) — single implementation for all experiments.
#
# Every experiment imports from this harness, so this guard runs before
# ``superlocalmemory`` is first imported and guarantees the bundle measures
# the declared source tree.  ``SLM_SOURCE_ROOT`` may point at the repo root
# or its ``src/`` directory; when unset the guard is a no-op (installed pkg).
# ---------------------------------------------------------------------------

_HARNESS_SOURCE_ROOT = os.environ.get("SLM_SOURCE_ROOT")
_HARNESS_SRC_PATH: Path | None = None
if _HARNESS_SOURCE_ROOT:
    _harness_source_path = Path(_HARNESS_SOURCE_ROOT).expanduser().resolve()
    _HARNESS_SRC_PATH = (
        _harness_source_path
        if _harness_source_path.name == "src"
        else _harness_source_path / "src"
    )
    if not (_HARNESS_SRC_PATH / "superlocalmemory").is_dir():
        raise RuntimeError(
            "SLM_SOURCE_ROOT must point to a repository root or its src directory: "
            f"{_harness_source_path}"
        )
    if str(_HARNESS_SRC_PATH) not in sys.path:
        sys.path.insert(0, str(_HARNESS_SRC_PATH))
    # Eagerly import and verify once so every experiment that imports this
    # harness inherits the guard without needing its own copy.
    try:
        import superlocalmemory as _slm_eager  # noqa: F401

        _slm_file_eager = (
            Path(_slm_eager.__file__).resolve() if _slm_eager.__file__ else None
        )
        if _slm_file_eager is None or not _slm_file_eager.is_relative_to(
            _HARNESS_SRC_PATH
        ):
            raise RuntimeError(
                f"superlocalmemory imported from unexpected location: {_slm_file_eager}"
            )
    except ImportError:
        # superlocalmemory not yet importable (e.g. harness imported before src
        # is on sys.path in test collection); verify_slm_source_root() will
        # check on demand after the experiment imports it.
        pass


def verify_slm_source_root() -> Path | None:
    """Fail loud if ``superlocalmemory`` was imported from outside SLM_SOURCE_ROOT.

    Call after ``import superlocalmemory``.  Returns the resolved src path when
    the guard is active, else ``None``.
    """
    if _HARNESS_SOURCE_ROOT is None or _HARNESS_SRC_PATH is None:
        return None
    import superlocalmemory as _slm  # local import to avoid circular eager import

    slm_file = Path(_slm.__file__).resolve() if _slm.__file__ else None
    if slm_file is None or not slm_file.is_relative_to(_HARNESS_SRC_PATH):
        raise RuntimeError(
            f"superlocalmemory imported from unexpected location: {slm_file}"
        )
    return _HARNESS_SRC_PATH

# ---------------------------------------------------------------------------
# Result types (immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialOutcome:
    """One trial: did the guarantee hold, and the evidence for that verdict."""

    index: int
    held: bool
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentResult:
    """Aggregate of N trials of one guarantee."""

    name: str
    guarantee: str
    metric_name: str
    trials: int
    held: int
    metric_value: float
    method: str
    failures: tuple[dict, ...] = ()
    extra: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.held == self.trials

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        return d


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


def run_trials(
    name: str,
    guarantee: str,
    metric_name: str,
    n_trials: int,
    trial_fn: Callable[[int], TrialOutcome],
    *,
    method: str,
    extra: dict | None = None,
) -> ExperimentResult:
    """Run ``trial_fn`` ``n_trials`` times and aggregate.

    ``trial_fn(index)`` returns a :class:`TrialOutcome`. Any exception it raises
    is a harness/API defect and is allowed to propagate — we do not convert an
    infrastructure crash into a passing or failing trial, because that would
    corrupt the measurement.
    """
    outcomes: list[TrialOutcome] = [trial_fn(i) for i in range(n_trials)]
    held = sum(1 for o in outcomes if o.held)
    failures = tuple(o.detail for o in outcomes if not o.held)[:20]
    metric_value = held / n_trials if n_trials else 0.0
    return ExperimentResult(
        name=name,
        guarantee=guarantee,
        metric_name=metric_name,
        trials=n_trials,
        held=held,
        metric_value=metric_value,
        method=method,
        failures=failures,
        extra=extra or {},
    )


# ---------------------------------------------------------------------------
# Real-DB helpers
# ---------------------------------------------------------------------------


def fresh_db(tmp_dir: Path, name: str = "memory.db"):
    """A real :class:`DatabaseManager` built exactly as the engine builds it.

    Mirrors ``MemoryEngine._init_db_layer``: base schema, the v3.4.3/4.6/4.7
    schema extensions (mesh, learning, ingestion), then the full forward and
    deferred migration chain (RBAC, scope columns, indexes). This is the same
    on-disk schema production runs — the measurement is against the real thing.
    """
    from superlocalmemory.learning.database import LearningDatabase
    from superlocalmemory.storage import schema as real_schema
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.migration_runner import apply_all, apply_deferred
    from superlocalmemory.storage.schema_v343 import (
        apply_v343_schema,
        apply_v346_schema,
    )
    from superlocalmemory.storage.schema_v347 import apply_v347_schema

    memory_db = tmp_dir / name
    manager = DatabaseManager(memory_db)
    manager.initialize(real_schema)
    apply_v343_schema(str(memory_db))
    apply_v346_schema(str(memory_db))
    apply_v347_schema(str(memory_db))
    learning_db = tmp_dir / "learning.db"
    LearningDatabase(learning_db)
    apply_all(learning_db, memory_db)
    apply_deferred(learning_db, memory_db)
    return manager


def add_profile(db, profile_id: str) -> None:
    db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name, description) "
        "VALUES (?, ?, '')",
        (profile_id, profile_id),
    )


class TempWorkspace:
    """Context manager yielding a fresh temp directory, cleaned up after use."""

    def __init__(self) -> None:
        self._path: Path | None = None

    def __enter__(self) -> Path:
        self._path = Path(tempfile.mkdtemp(prefix="slm-reliab-"))
        return self._path

    def __exit__(self, *exc) -> None:
        if self._path and self._path.exists():
            shutil.rmtree(self._path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def environment() -> dict:
    import superlocalmemory

    return {
        "package": "superlocalmemory",
        "version": getattr(superlocalmemory, "__version__", "unknown"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp": datetime.now(UTC).isoformat(),
    }


def write_result(result: ExperimentResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"environment": environment(), "result": result.to_dict()}
    path = out_dir / f"{result.name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def summarize(results: list[ExperimentResult]) -> str:
    """Render a markdown summary table across all experiments."""
    env = environment()
    lines = [
        "# SLM 4.0 — Reliability Evaluation Results",
        "",
        f"- Package version: **{env['version']}**",
        f"- Python: {env['python']}",
        f"- Platform: {env['platform']}",
        f"- Generated: {env['timestamp']}",
        "",
        "| Experiment | Guarantee | Metric | Trials | Held | Rate | Verdict |",
        "|---|---|---|---:|---:|---:|:--:|",
    ]
    for r in results:
        verdict = "PASS" if r.passed else "FAIL"
        lines.append(
            f"| {r.name} | {r.guarantee} | {r.metric_name} | "
            f"{r.trials} | {r.held} | {r.metric_value:.4f} | {verdict} |"
        )
    lines.append("")
    total = sum(r.trials for r in results)
    held = sum(r.held for r in results)
    lines.append(
        f"**Aggregate: {held}/{total} trials upheld their guarantee "
        f"({(held / total if total else 0):.4%}).**"
    )
    return "\n".join(lines) + "\n"
