"""SuperLocalMemory bounded loops.

Agent loops that stop when an independent gate passes — not when the agent
says it is done — with the run's history persisted in SLM's durable memory.

Public API::

    from superlocalmemory.loops import (
        Bounds, Rung, Status, Verdict, LapResult, Outcome,
        run_bounded_loop, InMemoryLedger, SLMMemoryLedger, open_engine_store,
    )
"""

from __future__ import annotations

from superlocalmemory.loops.engine import run_bounded_loop
from superlocalmemory.loops.ledger import (
    InMemoryLedger,
    LedgerEntry,
    LedgerStore,
    SLMMemoryLedger,
    engine_backed_ledger,
    open_engine_store,
)
from superlocalmemory.loops.models import (
    Bounds,
    LapResult,
    Outcome,
    Rung,
    Status,
    Verdict,
)
from superlocalmemory.loops.rules import (
    no_progress,
    rung_requires_approval,
    stop_condition_met,
)

__all__ = [
    "Bounds",
    "Rung",
    "Status",
    "Verdict",
    "LapResult",
    "Outcome",
    "run_bounded_loop",
    "LedgerEntry",
    "LedgerStore",
    "InMemoryLedger",
    "SLMMemoryLedger",
    "engine_backed_ledger",
    "open_engine_store",
    "no_progress",
    "rung_requires_approval",
    "stop_condition_met",
]
