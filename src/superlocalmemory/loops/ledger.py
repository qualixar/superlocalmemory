"""Durable, queryable ledger for bounded-loop runs.

Every lap produces one append-only :class:`LedgerEntry`. A ledger persists
those entries so a run can be inspected, resumed, and audited after the fact.

Two implementations ship:

* :class:`InMemoryLedger` — a dict-backed store used by tests and as a safe
  fallback when no SLM data root is available.
* :class:`SLMMemoryLedger` — the real backend. It writes each lap through a
  SuperLocalMemory engine so the ledger *is* memory: queryable via ``slm
  recall``, visible in the dashboard, and resumable across sessions. This is
  what makes SLM's take on bounded loops distinct — the loop's history lives
  in the same durable store as everything else the agent remembers.

The engine-backed store mirrors the exact profile-scoped SQL contract the
shipped framework adapters already rely on, so it stays valid as the engine
evolves.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

LEDGER_TAG = "slm-loop"
_LEDGER_IMPORTANCE = 2  # below ordinary user memories so laps never crowd recall
_SESSION_PREFIX = "loop:"


@dataclass(frozen=True)
class LedgerEntry:
    """One immutable row recording what happened on a single lap."""

    run_id: str
    name: str
    lap: int
    ts: str
    decision: str  # continue | done | halt | pause | killed | error
    passed: bool
    detail: str
    # The runner's own claim (audit-only; never terminates the loop) + its log.
    agent_claimed_done: bool = False
    runner_log: str = ""
    budget: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "LedgerEntry | None":
        try:
            data = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or "run_id" not in data or "lap" not in data:
            return None
        return cls(
            run_id=str(data.get("run_id", "")),
            name=str(data.get("name", "")),
            lap=int(data.get("lap", 0)),
            ts=str(data.get("ts", "")),
            decision=str(data.get("decision", "")),
            passed=bool(data.get("passed", False)),
            detail=str(data.get("detail", "")),
            agent_claimed_done=bool(data.get("agent_claimed_done", False)),
            runner_log=str(data.get("runner_log", "")),
            budget=data.get("budget") if isinstance(data.get("budget"), dict) else {},
        )


@runtime_checkable
class LedgerStore(Protocol):
    """Append-only audit trail keyed by run."""

    def record(self, entry: LedgerEntry) -> None: ...
    def laps(self, run_id: str) -> list[LedgerEntry]: ...
    def runs(self, name: str) -> list[str]: ...


class InMemoryLedger:
    """Process-local ledger. Never raises; ideal for tests and offline demos."""

    def __init__(self) -> None:
        self._by_run: dict[str, list[LedgerEntry]] = {}

    def record(self, entry: LedgerEntry) -> None:
        self._by_run.setdefault(entry.run_id, []).append(entry)

    def laps(self, run_id: str) -> list[LedgerEntry]:
        return list(self._by_run.get(run_id, ()))

    def runs(self, name: str) -> list[str]:
        seen: list[str] = []
        for run_id, entries in self._by_run.items():
            if entries and entries[0].name == name and run_id not in seen:
                seen.append(run_id)
        return seen


class SLMMemoryLedger:
    """Ledger backed by a SuperLocalMemory engine store.

    ``store`` is any object exposing ``add(content, *, session_id, metadata)``,
    ``list_session(session_id)`` and ``list_prefix(prefix)`` — the small
    contract :func:`open_engine_store` provides. Injecting it keeps this class
    free of engine-construction concerns and trivially testable with a fake.
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    @staticmethod
    def _session_id(run_id: str) -> str:
        return f"{_SESSION_PREFIX}{run_id}"

    def record(self, entry: LedgerEntry) -> None:
        self._store.add(
            entry.to_json(),
            session_id=self._session_id(entry.run_id),
            metadata={
                "integration": "slm-loop",
                "loop_name": entry.name,
                "loop_run_id": entry.run_id,
                "loop_lap": entry.lap,
                "loop_decision": entry.decision,
                "tags": [LEDGER_TAG, f"loop:{entry.name}"],
                "importance": _LEDGER_IMPORTANCE,
                "project_name": "slm-loop",
            },
        )

    def laps(self, run_id: str) -> list[LedgerEntry]:
        rows = self._store.list_session(self._session_id(run_id))
        entries = [LedgerEntry.from_json(r.get("content", "")) for r in rows]
        return [e for e in entries if e is not None]

    def runs(self, name: str) -> list[str]:
        """Run ids for ``name``, newest run first.

        ``list_prefix`` returns rows created_at DESC, so the first time a run's
        id is seen it is its most-recent lap; ``slm loop history`` therefore
        lists the most recent runs first.
        """
        rows = self._store.list_prefix(_SESSION_PREFIX)
        ordered: list[str] = []
        for row in rows:
            entry = LedgerEntry.from_json(row.get("content", ""))
            if entry is not None and entry.name == name and entry.run_id not in ordered:
                ordered.append(entry.run_id)
        return ordered


class _EngineLedgerStore:
    """Minimal profile-scoped store over a SuperLocalMemory engine.

    Uses the engine's non-blocking write-through path when available and
    direct, escaped, profile-scoped reads.  The ``store`` fallback preserves
    compatibility with lightweight adapter/test engines that predate
    ``store_fast``.
    """

    def __init__(self, engine: Any, *, owns_engine: bool = True) -> None:
        self._engine = engine
        # When False, this store does NOT own the engine's lifecycle (the
        # caller — e.g. the MCP daemon — keeps it), so close() must not tear
        # down a shared engine. open_engine_store() passes True (it built the
        # engine); engine_backed_ledger() passes False (daemon-owned engine).
        self._owns_engine = owns_engine

    def add(self, content: str, *, session_id: str, metadata: dict) -> None:
        # A loop ledger needs the durable parent row and immediate lexical
        # recall, not synchronous embeddings/entity/graph enrichment. Loading
        # the heavyweight embedding worker for every bounded-loop lap can stall
        # the loop for the full worker timeout and consume ~1 GB for metadata.
        # The write-through path persists the same session-scoped content in
        # milliseconds; ordinary background enrichment can still promote it.
        fast_metadata = {**metadata, "session_id": session_id}
        store_fast = getattr(self._engine, "store_fast", None)
        if callable(store_fast):
            store_fast(
                content,
                metadata=fast_metadata,
                index_external=False,
            )
            return

        self._engine.store(
            content,
            session_id=session_id,
            metadata=metadata,
        )

    def list_session(self, session_id: str) -> list[dict]:
        # Cap the read: a bounded-loop run is capped at max_iterations laps, so
        # a legitimate run is small; the LIMIT stops a pathologically long
        # session_id from forcing an unbounded materialization on every
        # `slm loop show` / history lookup.
        rows = self._engine.db.execute(
            "SELECT content, created_at FROM memories "
            "WHERE profile_id=? AND session_id=? "
            "ORDER BY created_at ASC, rowid ASC LIMIT 5000",
            (self._engine.profile_id, session_id),
        )
        return [dict(row) for row in rows]

    def list_prefix(self, prefix: str) -> list[dict]:
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        # Cap the scan so a long-lived, high-volume ledger can't force an
        # unbounded read on `slm loop history`.
        rows = self._engine.db.execute(
            "SELECT content, created_at FROM memories "
            "WHERE profile_id=? AND session_id LIKE ? ESCAPE '\\' "
            "ORDER BY created_at DESC, rowid DESC LIMIT 5000",
            (self._engine.profile_id, escaped + "%"),
        )
        return [dict(row) for row in rows]

    def close(self) -> None:
        if self._owns_engine:
            self._engine.close()


def engine_backed_ledger(engine: Any) -> SLMMemoryLedger:
    """Build an SLM-backed ledger over an ALREADY-OPEN engine.

    Unlike :func:`open_engine_store`, this neither creates nor owns the engine —
    the caller (e.g. the MCP daemon, which keeps one long-lived engine per
    profile) retains full ownership and lifecycle. The returned ledger never
    closes the engine, so it is safe to build one per tool call. The engine's
    per-call, WAL-mode connection model makes the ledger's reads/writes safe
    from a worker thread.
    """
    return SLMMemoryLedger(_EngineLedgerStore(engine, owns_engine=False))


def open_engine_store(db_path: str | Path) -> _EngineLedgerStore:
    """Build an engine-backed ledger store rooted at ``db_path``.

    Raises ``ImportError`` with an install hint if the SLM runtime is missing.
    """
    from dataclasses import replace

    try:
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.engine import MemoryEngine
        from superlocalmemory.storage.models import Mode
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(
            "SuperLocalMemory runtime is required for the SLM-backed loop "
            "ledger. Install it with: python -m pip install superlocalmemory."
        ) from exc

    path = Path(db_path).expanduser().resolve()
    config = SLMConfig.for_mode(Mode.A, base_dir=path.parent)
    config.db_path = path
    config.forgetting = replace(config.forgetting, enabled=False)
    config.retrieval.use_cross_encoder = False
    engine = MemoryEngine(config)
    engine.initialize()
    return _EngineLedgerStore(engine)
