"""``slm loop`` — inspect and demonstrate bounded loops backed by SLM memory.

Subcommands:

* ``slm loop demo``            run a built-in, keyless convergence demo (stub
                              proposer + deterministic gate) and persist every
                              lap to SLM memory. Proves the engine + ledger
                              end to end without a credentialed agent.
* ``slm loop history [--name]`` list recorded loop runs from SLM memory.
* ``slm loop show <run_id>``   show every lap of one run.

The durable value here is that a loop's history lives in the same SLM data
root as everything else the agent remembers — queryable via ``slm recall`` and
visible in the dashboard. Reads and the demo run through an in-process engine
store rooted at the active data root (``SLM_DATA_DIR`` or
``~/.superlocalmemory``); point ``SLM_DATA_DIR`` elsewhere to sandbox a run.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from superlocalmemory.infra.data_root import canonical_data_root
from superlocalmemory.loops import (
    Bounds,
    LapResult,
    SLMMemoryLedger,
    Verdict,
    open_engine_store,
    run_bounded_loop,
)


def _data_root() -> Path:
    return canonical_data_root()


class _FailOpenLedger:
    """Wrap a ledger so a memory write hiccup never aborts a running loop.

    A ledger is observability; the loop's correctness comes from the gate.
    Write errors are counted and surfaced after the run rather than raised
    mid-flight, so we never silently pretend they did not happen.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.write_errors: list[str] = []

    def record(self, entry: Any) -> None:
        try:
            self._inner.record(entry)
        except Exception as exc:  # pragma: no cover - defensive path
            self.write_errors.append(f"lap {getattr(entry, 'lap', '?')}: {exc}")

    def laps(self, run_id: str) -> list:
        return self._inner.laps(run_id)

    def runs(self, name: str) -> list:
        return self._inner.runs(name)


def _open_ledger() -> tuple[SLMMemoryLedger, Any]:
    store = open_engine_store(_data_root() / "memory.db")
    return SLMMemoryLedger(store), store


def cmd_loop(args: Namespace) -> None:
    action = getattr(args, "loop_command", None)
    if action == "demo":
        _cmd_demo(args)
    elif action == "history":
        _cmd_history(args)
    elif action == "show":
        _cmd_show(args)
    else:
        print("Usage: slm loop {demo|history|show} [options]")


def _cmd_demo(args: Namespace) -> None:
    """Run the convergence demo: the gate fails twice, then passes on lap 3."""
    iterations = int(getattr(args, "iterations", 10) or 10)
    as_json = bool(getattr(args, "json", False))
    pass_on = 3

    ledger, store = _open_ledger()
    guarded = _FailOpenLedger(ledger)
    try:
        outcome = run_bounded_loop(
            "convergence-demo",
            bounds=Bounds(max_iterations=iterations),
            runner=lambda lap: LapResult(changed=True, tokens=8),
            gate=lambda lap: Verdict(lap >= pass_on, f"demo gate: lap {lap}"),
            ledger=guarded,
        )
        laps = guarded.laps(outcome.run_id)
    finally:
        store.close()

    if as_json:
        print(json.dumps({
            "status": outcome.status.value,
            "reason": outcome.reason,
            "laps": outcome.laps,
            "run_id": outcome.run_id,
            "ledger": [
                {"lap": e.lap, "decision": e.decision, "passed": e.passed}
                for e in laps
            ],
            "write_errors": guarded.write_errors,
        }, indent=2))
        return

    mark = "✓" if outcome.ok else "✗"
    print(f"{mark} [{outcome.status.value}] {outcome.reason} (laps: {outcome.laps})")
    for e in laps:
        gate = "gate-pass" if e.passed else "gate-fail"
        print(f"   lap {e.lap}: {e.decision:<8} {gate}  {e.detail}")
    print(f"run_id: {outcome.run_id}  (recall with tag loop:convergence-demo)")
    if guarded.write_errors:
        print(f"WARNING: {len(guarded.write_errors)} ledger write error(s): "
              f"{guarded.write_errors[0]}")


def _cmd_history(args: Namespace) -> None:
    name = getattr(args, "name", None) or "convergence-demo"
    as_json = bool(getattr(args, "json", False))
    ledger, store = _open_ledger()
    try:
        run_ids = ledger.runs(name)
        rows = []
        for rid in run_ids:
            laps = ledger.laps(rid)
            last = laps[-1] if laps else None
            rows.append({
                "run_id": rid,
                "laps": len(laps),
                "final": last.decision if last else "unknown",
                "ts": last.ts if last else "",
            })
    finally:
        store.close()

    if as_json:
        print(json.dumps({"name": name, "runs": rows}, indent=2))
        return
    if not rows:
        print(f"No recorded runs for loop '{name}'.")
        return
    print(f"Runs for loop '{name}':")
    for r in rows:
        print(f"  {r['run_id']:<28} laps={r['laps']:<3} final={r['final']:<8} {r['ts']}")


def _cmd_show(args: Namespace) -> None:
    run_id = getattr(args, "run_id", None)
    as_json = bool(getattr(args, "json", False))
    if not run_id:
        print("Usage: slm loop show <run_id>")
        return
    ledger, store = _open_ledger()
    try:
        laps = ledger.laps(run_id)
    finally:
        store.close()

    if as_json:
        print(json.dumps({
            "run_id": run_id,
            "laps": [
                {"lap": e.lap, "ts": e.ts, "decision": e.decision,
                 "passed": e.passed, "detail": e.detail, "budget": e.budget}
                for e in laps
            ],
        }, indent=2))
        return
    if not laps:
        print(f"No ledger entries for run '{run_id}'.")
        return
    print(f"Loop run {run_id} ({laps[0].name}):")
    for e in laps:
        gate = "gate-pass" if e.passed else "gate-fail"
        print(f"  lap {e.lap}: {e.decision:<8} {gate}  {e.detail}  "
              f"[tokens={e.budget.get('tokens', 0)}]")
