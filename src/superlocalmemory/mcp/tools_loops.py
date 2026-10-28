# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLM v3.8.0 — MCP bounded-loop tools.

Exposes the ``superlocalmemory.loops`` engine over MCP so an agent connected
via MCP — not only the ``slm loop`` CLI or the ``/slm-loop`` command — can run
a gated, bounded loop and inspect its durable ledger. Bounded loops therefore
ship on three surfaces: CLI, plugin command, and MCP.

The loop's one invariant holds identically here: an INDEPENDENT gate decides
when the loop is done, never the agent's own claim. Over MCP the gate is an SLM
*recall*: the loop converges the first lap a memory matching ``gate_query``
becomes retrievable with confidence (the "verification lives in memory" model
the engine is designed around). That makes ``slm_loop_run`` a safe, shell-free
multi-agent coordination primitive — one agent waits, under strict bounds, for
a memory another agent will write into shared SLM.

Three tools:
  * ``slm_loop_run``     — run one bounded, gate-verified loop to a terminal
                           outcome. Blocks (polling the gate) until the gate
                           passes or a bound trips. Every lap is persisted to
                           SLM memory (tag ``loop:<name>``) and shows on the
                           dashboard.
  * ``slm_loop_history`` — list recorded runs for a loop name (read-only).
  * ``slm_loop_show``    — show every lap of one run (read-only).

Fail-open: every tool body returns a dict; internal errors surface as
``ok: False`` with a message, never a raised exception.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable


from superlocalmemory.mcp.tool_hints import READ_ONLY, hints
from superlocalmemory.core.admission import admits
from superlocalmemory.core.operation_request import OperationKind
from superlocalmemory.loops import (
    engine_backed_ledger,
)
from superlocalmemory.loops.stop_reasons import last_gate_reason
from superlocalmemory.retrieval.answer_check_status import REQUEST_NO_REORDER

logger = logging.getLogger("slm.mcp.tools_loops")

# ─── Exported tool name list (used by server.py + tests) ─────────────────────

_LOOP_TOOL_NAMES = (
    "slm_loop_run",
    "slm_loop_history",
    "slm_loop_show",
)

# ─── Hard caps (a loop tool must never hang the daemon or spin unbounded) ────

_MAX_ITERATIONS = 200
_MAX_WALLCLOCK_S = 120.0
_MIN_POLL_S = 0.25
_MAX_NAME_CHARS = 128
_MAX_QUERY_CHARS = 2000
#: Results each gate lap asks for — and so the most memories its answer check
#: ever judges.
_GATE_LIMIT = 3


def _gate_recall_kwargs(recall: Callable) -> dict:
    """The gate's per-recall request, for a recall that can carry it.

    The gate needs the answer check's verdict, never the opt-in reorder. A pool
    whose transport does not forward the request yet is called as before.
    """
    try:
        params = inspect.signature(recall).parameters
    except (TypeError, ValueError):
        return {}
    takes_it = "answer_check" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    return {"answer_check": REQUEST_NO_REORDER} if takes_it else {}


def _loop_profile(engine: Any, profile_id: str) -> tuple[str | None, dict | None]:
    """``(profile, refusal)`` for a loop read; ``None`` = the engine's own."""
    from superlocalmemory.mcp.request_profile import requested_profile, tool_profile

    if not requested_profile(profile_id):
        return None, None
    return tool_profile(engine, profile_id)


def register_loop_tools(
    server,
    get_engine: Callable,
    get_pool: Callable | None = None,
) -> None:
    """Register the 3 bounded-loop tools on *server*.

    *server* is duck-typed: must support the ``@server.tool()`` decorator.
    Compatible with FastMCP, _FilteredServer, and the test mock server.
    """

    @server.tool(title="Run Bounded Loop", annotations=hints(open_world=True))
    @admits(OperationKind.REMEMBER)
    async def slm_loop_run(
        name: str,
        gate_query: str,
        gate_min_score: float = 0.0,
        max_iterations: int = 20,
        max_wallclock_s: float = 15.0,
        poll_interval_s: float = 1.0,
        max_tokens: int = 0,
        no_progress_window: int = 0,
        require_support: bool = False,
        gate_contract: dict | None = None,
    ) -> dict:
        """Wait under hard bounds for independently executed verification.

        DONE requires gate_contract={"job_id": <operator registration nonce>}.
        The operator pins check files, declares mutable artifact paths/globs,
        watcher name and freshness via loops.receipt_jobs.register_job, then
        executes run_registered_check in a private copy. DONE copies and hashes
        current inputs again. Older caller-selected contracts fail closed.
        Hosts without secure no-follow directory handles refuse registration;
        the watcher exposes the platform or receipt reason in its result.
        Pre-start evidence requires explicit permission in that registration.
        Recall locates context; related text or support alone cannot pass.
        require_support additionally requires a supported answer check.
        Each lap recalls at most three memories and asks the answer check once;
        no reorder is requested. An online check may be billed for each lap.
        Missing, failed, stale or mismatched receipts halt at the configured bound.
        """
        from superlocalmemory.mcp.loop_runner import run_loop

        return await run_loop(
            get_engine,
            get_pool,
            name,
            gate_query,
            gate_min_score,
            max_iterations,
            max_wallclock_s,
            poll_interval_s,
            max_tokens,
            no_progress_window,
            require_support,
            gate_contract,
        )

    @server.tool(title="List Bounded Loop Runs", annotations=READ_ONLY)
    async def slm_loop_history(name: str, limit: int = 20, profile_id: str = "") -> dict:
        """List recorded bounded-loop runs for a loop name, newest laps summarised.

        Reads the durable SLM-backed ledger (the same rows ``slm loop history``
        and the dashboard show). Read-only.

        Args:
            name: Loop name to list runs for.
            limit: Maximum runs to return (1–200).
            profile_id: The profile whose runs to list (empty = the active one).
        """
        try:
            name = (name or "").strip()
            if not name:
                return {"ok": False, "error": "name is required"}
            lim = max(1, min(int(limit), 200))
            engine = get_engine()
            pid, refused = _loop_profile(engine, profile_id)
            if refused:
                return {"ok": False, **refused}
            ledger = engine_backed_ledger(engine, pid)

            def _collect() -> list[dict]:
                run_ids = ledger.runs(name)[:lim]
                rows: list[dict] = []
                for rid in run_ids:
                    laps = ledger.laps(rid)
                    last = laps[-1] if laps else None
                    rows.append(
                        {
                            "run_id": rid,
                            "laps": len(laps),
                            "final": last.decision if last else "unknown",
                            "reason": last_gate_reason(laps),
                            "ts": last.ts if last else "",
                        }
                    )
                return rows

            rows = await asyncio.to_thread(_collect)
            return {"ok": True, "name": name, "count": len(rows), "runs": rows}
        except Exception as exc:
            logger.exception("slm_loop_history failed (fail-open)")
            return {"ok": False, "error": str(exc)}

    @server.tool(title="Show Bounded Loop Run", annotations=READ_ONLY)
    async def slm_loop_show(run_id: str, limit: int = 200, profile_id: str = "") -> dict:
        """Show every lap of one bounded-loop run, in order, from SLM memory.

        Read-only. Each lap carries the gate verdict plus the agent's recorded
        (advisory, never loop-terminating) done-claim.

        Args:
            run_id: Run identifier returned by ``slm_loop_run``.
            limit: Maximum laps to return (1–1000).
            profile_id: The profile the run was recorded in (empty = the active one).
        """
        try:
            run_id = (run_id or "").strip()
            if not run_id:
                return {"ok": False, "error": "run_id is required"}
            lim = max(1, min(int(limit), 1000))
            engine = get_engine()
            pid, refused = _loop_profile(engine, profile_id)
            if refused:
                return {"ok": False, **refused}
            ledger = engine_backed_ledger(engine, pid)

            def _collect() -> list[dict]:
                return [
                    {
                        "lap": e.lap,
                        "ts": e.ts,
                        "decision": e.decision,
                        "passed": e.passed,
                        "detail": e.detail,
                        "agent_claimed_done": e.agent_claimed_done,
                        "tokens": e.budget.get("tokens", 0),
                    }
                    for e in ledger.laps(run_id)[:lim]
                ]

            laps = await asyncio.to_thread(_collect)
            return {"ok": True, "run_id": run_id, "count": len(laps), "laps": laps}
        except Exception as exc:
            logger.exception("slm_loop_show failed (fail-open)")
            return {"ok": False, "error": str(exc)}
