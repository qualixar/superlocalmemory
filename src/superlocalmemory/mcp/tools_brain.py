"""Portable Agent Experience receipt tools for the SLM Brain.

Receipts are scoped to the MCP engine's active profile.  A host cannot write
to another profile by supplying a different ``profile_id`` in its payload.
The receipts are evidence/observability records only: recall and ranking do
not consume them synchronously, so an unavailable receipt store never slows
or changes a memory answer.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any, Callable

from mcp.types import ToolAnnotations

from superlocalmemory.brain.truth import BrainTruthService
from superlocalmemory.core.admission import admits
from superlocalmemory.core.operation_request import OperationKind
from superlocalmemory.infra.data_root import state_path
from superlocalmemory.integrations.bounded_loops_mcp import (
    BridgeUnavailable,
    observe_installed,
    observe_installed_v2,
)
from superlocalmemory.storage.execution_learning import (
    ExecutionLearningStore,
    ExecutionLearningValidationError,
)
from superlocalmemory.storage.agent_experience import (
    AgentExperienceConflictError,
    AgentExperienceStore,
    CognitiveTurnTransitionError,
    LearningWriteBusyError,
    ProfileAdmissionError,
)
from superlocalmemory.storage.external_evidence import (
    ExternalEvidenceConflictError,
    ExternalEvidenceStore,
    ExternalEvidenceValidationError,
)


def _store_for(engine: Any) -> AgentExperienceStore:
    active_profile = engine.profile_id
    return AgentExperienceStore(
        Path(state_path("learning.db")),
        is_profile_active=lambda profile_id: profile_id == active_profile,
    )


def _external_store_for(engine: Any) -> ExternalEvidenceStore:
    active_profile = engine.profile_id
    return ExternalEvidenceStore(
        Path(state_path("learning.db")),
        is_profile_active=lambda profile_id: profile_id == active_profile,
    )


def _execution_store_for(engine: Any) -> ExecutionLearningStore:
    active_profile = engine.profile_id
    return ExecutionLearningStore(
        Path(state_path("learning.db")),
        is_profile_active=lambda profile_id: profile_id == active_profile,
    )


def _brain_truth_for(engine: Any) -> dict[str, Any]:
    """Read the portable truth snapshot without opening an engine or a writer."""
    return BrainTruthService(
        memory_db_path=state_path("memory.db"),
        learning_db_path=state_path("learning.db"),
    ).snapshot(engine.profile_id)


def _legacy_agent_experience(truth: dict[str, Any]) -> dict[str, Any]:
    """Keep the v4.0.4 MCP alias during the one-release transition window."""
    evidence = truth["agent_experience"]
    available = evidence["availability"] == "available"
    claimed = evidence["claimed_experiences_total"]
    turns = evidence["cognitive_turns_total"]
    states = evidence["cognitive_turns_by_state"]
    return {
        "is_real": available,
        "availability": evidence["availability"],
        "experiences_total": claimed if available else 0,
        "turns_total": turns if available else 0,
        "turns_by_state": states if available else {},
        # The old name remains an alias only.  BrainTruth deliberately calls
        # these declared claims, never independently verified learning.
        "claimed_evidence_experiences": claimed if available else 0,
        "source": evidence["source"],
    }


def _legacy_external_evidence(truth: dict[str, Any]) -> dict[str, Any]:
    """Keep the v4.0.4 external-graph alias without a second database read."""
    evidence = truth["external_evidence"]
    available = evidence["availability"] == "available"
    return {
        "is_real": available,
        "availability": evidence["availability"],
        "total": evidence["receipts_total"] if available else 0,
        "by_run_state": evidence["receipts_by_run_state"] if available else {},
        "demonstrations": evidence["demonstrations_total"] if available else 0,
        "control_plane": "observation_only",
    }


def _require_active_profile(engine: Any, payload: dict[str, Any]) -> str | None:
    supplied = payload.get("profile_id")
    if supplied != engine.profile_id:
        return "profile_id must equal the active MCP profile"
    return None


class _ExternalEvidenceWriteError(Exception):
    """Preserve committed receipt count when a later snapshot item fails."""

    def __init__(self, created: int, cause: Exception) -> None:
        super().__init__(str(cause))
        self.created = created
        self.cause = cause


def _record_external_evidence(store: ExternalEvidenceStore, observed: list[dict[str, Any]]) -> int:
    """Write bounded evidence off the async MCP loop; return durable inserts."""
    created = 0
    for payload in observed:
        try:
            if store.record(payload):
                created += 1
        except Exception as exc:
            raise _ExternalEvidenceWriteError(created, exc) from exc
    return created


def register_brain_tools(server: Any, get_engine: Callable[[], Any]) -> None:
    """Register transport-neutral receipt reads and writes.

    The same tools work through Codex, Claude, Cursor, VS Code, and direct
    HTTP MCP.  Callers supply contract-shaped JSON; the active-profile check
    keeps multi-profile data isolated at the public boundary.
    """

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def get_brain_evidence_status() -> dict[str, Any]:
        """Get profile-scoped, observation-only Living Brain evidence totals.

        ``brain_truth`` is the canonical v1 payload.  The legacy aliases are
        retained for one release so existing hosts can move independently.
        """
        engine = get_engine()
        truth = _brain_truth_for(engine)
        return {
            "success": True,
            "profile_id": engine.profile_id,
            "brain_truth": truth,
            "agent_experience": _legacy_agent_experience(truth),
            "external_evidence": truth["external_evidence"],
            "external_graph_evidence": _legacy_external_evidence(truth),
            "control_plane": truth["control_plane"],
        }

    @server.tool()
    @admits(OperationKind.REMEMBER)
    async def record_agent_experience(payload: dict[str, Any]) -> dict[str, Any]:
        """Record a contract-validated terminal experience receipt.

        Supply only evidence you can substantiate.  SLM records the declared
        verification authority and does not use this receipt to alter recall,
        ranking, or model routing automatically.
        """
        engine = get_engine()
        error = _require_active_profile(engine, payload)
        if error:
            return {"success": False, "durable": False, "error": error}
        try:
            created = _store_for(engine).record_experience(payload)
        except (AgentExperienceConflictError, ProfileAdmissionError) as exc:
            return {"success": False, "durable": False, "error": str(exc)}
        except LearningWriteBusyError as exc:
            return {"success": False, "durable": False, "retryable": True, "error": str(exc)}
        except (TypeError, ValueError) as exc:
            return {"success": False, "durable": False, "error": str(exc)}
        return {"success": True, "durable": True, "created": created}

    @server.tool()
    @admits(OperationKind.REMEMBER)
    async def record_cognitive_turn(payload: dict[str, Any]) -> dict[str, Any]:
        """Open one contract-validated cognitive-turn provenance receipt."""
        engine = get_engine()
        error = _require_active_profile(engine, payload)
        if error:
            return {"success": False, "durable": False, "error": error}
        try:
            created = _store_for(engine).create_cognitive_turn(payload)
        except (
            AgentExperienceConflictError,
            CognitiveTurnTransitionError,
            ProfileAdmissionError,
        ) as exc:
            return {"success": False, "durable": False, "error": str(exc)}
        except LearningWriteBusyError as exc:
            return {"success": False, "durable": False, "retryable": True, "error": str(exc)}
        except (TypeError, ValueError) as exc:
            return {"success": False, "durable": False, "error": str(exc)}
        return {"success": True, "durable": True, "created": created}

    @server.tool()
    @admits(OperationKind.REMEMBER)
    async def finalize_cognitive_turn(receipt_id: str, outcome: dict[str, Any]) -> dict[str, Any]:
        """Finalize an active-profile cognitive turn with outcome evidence."""
        engine = get_engine()
        try:
            finalized = _store_for(engine).finalize_cognitive_turn(
                engine.profile_id, receipt_id, outcome
            )
        except (
            AgentExperienceConflictError,
            CognitiveTurnTransitionError,
            ProfileAdmissionError,
        ) as exc:
            return {"success": False, "durable": False, "error": str(exc)}
        except LearningWriteBusyError as exc:
            return {"success": False, "durable": False, "retryable": True, "error": str(exc)}
        except (TypeError, ValueError) as exc:
            return {"success": False, "durable": False, "error": str(exc)}
        return {"success": True, "durable": True, "finalized": finalized}

    @server.tool()
    @admits(OperationKind.REMEMBER)
    async def observe_bounded_loop_evidence(workspace: str) -> dict[str, Any]:
        """Import one explicit, read-only snapshot from installed Bounded Loops.

        Bounded Loops remains optional.  This tool negotiates its public MCP
        contract at runtime, records only compatible terminal evidence, and
        never changes recall, ranking, routing, or learned behaviour.
        """
        engine = get_engine()
        created = 0
        try:
            observed = await observe_installed(workspace=workspace, profile_id=engine.profile_id)
            store = _external_store_for(engine)
            created = await asyncio.to_thread(_record_external_evidence, store, observed)
        except _ExternalEvidenceWriteError as exc:
            return {
                "success": False,
                "durable": exc.created > 0,
                "created": exc.created,
                "retryable": isinstance(exc.cause, LearningWriteBusyError),
                "error": str(exc.cause),
            }
        except (
            BridgeUnavailable,
            ExternalEvidenceConflictError,
            ExternalEvidenceValidationError,
            ProfileAdmissionError,
            sqlite3.Error,
        ) as exc:
            return {
                "success": False,
                "durable": created > 0,
                "created": created,
                "error": str(exc),
            }
        except LearningWriteBusyError as exc:
            return {"success": False, "durable": False, "retryable": True, "error": str(exc)}
        return {
            "success": True,
            "durable": True,
            "observed": len(observed),
            "created": created,
            "control_plane": "observation_only",
        }

    @server.tool()
    @admits(OperationKind.REMEMBER)
    async def observe_bounded_loop_execution_learning(workspace: str) -> dict[str, Any]:
        """Ingest negotiated bridge-v2 terminal evidence into the execution plane.

        The producer is reached only through the installed bounded-loops MCP
        capability handshake.  This never writes semantic facts or preferences.
        """
        engine = get_engine()
        try:
            observed = await observe_installed_v2(workspace=workspace, profile_id=engine.profile_id)
            store = _execution_store_for(engine)
            created = 0
            for payload in observed:
                created += int(await asyncio.to_thread(store.ingest, payload))
            return {"success": True, "durable": True, "observed": len(observed),
                    "created": created, "control_plane": "execution_reliability_only"}
        except (BridgeUnavailable, ExecutionLearningValidationError, ProfileAdmissionError,
                sqlite3.Error) as exc:
            return {"success": False, "durable": False, "error": str(exc)}
