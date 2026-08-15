"""Portable Agent Experience receipt tools for the SLM Brain.

Receipts are scoped to the MCP engine's active profile.  A host cannot write
to another profile by supplying a different ``profile_id`` in its payload.
The receipts are evidence/observability records only: recall and ranking do
not consume them synchronously, so an unavailable receipt store never slows
or changes a memory answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mcp.types import ToolAnnotations

from superlocalmemory.core.admission import admits
from superlocalmemory.core.operation_request import OperationKind
from superlocalmemory.infra.data_root import state_path
from superlocalmemory.integrations.bounded_loops_mcp import BridgeUnavailable, observe_installed
from superlocalmemory.storage.agent_experience import (
    AgentExperienceConflictError,
    AgentExperienceStore,
    CognitiveTurnTransitionError,
    LearningWriteBusyError,
    ProfileAdmissionError,
    get_profile_receipt_summary,
)
from superlocalmemory.storage.external_evidence import (
    ExternalEvidenceConflictError,
    ExternalEvidenceStore,
    ExternalEvidenceValidationError,
    get_profile_external_evidence_summary,
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


def _require_active_profile(engine: Any, payload: dict[str, Any]) -> str | None:
    supplied = payload.get("profile_id")
    if supplied != engine.profile_id:
        return "profile_id must equal the active MCP profile"
    return None


def register_brain_tools(server: Any, get_engine: Callable[[], Any]) -> None:
    """Register transport-neutral receipt reads and writes.

    The same tools work through Codex, Claude, Cursor, VS Code, and direct
    HTTP MCP.  Callers supply contract-shaped JSON; the active-profile check
    keeps multi-profile data isolated at the public boundary.
    """

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def get_brain_evidence_status() -> dict[str, Any]:
        """Get profile-scoped, observation-only Brain evidence totals."""
        engine = get_engine()
        return {
            "success": True,
            "profile_id": engine.profile_id,
            "agent_experience": get_profile_receipt_summary(
                state_path("learning.db"), engine.profile_id
            ),
            "external_graph_evidence": get_profile_external_evidence_summary(
                state_path("learning.db"), engine.profile_id
            ),
            "control_plane": "observation_only",
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
        try:
            observed = await observe_installed(workspace=workspace, profile_id=engine.profile_id)
            store = _external_store_for(engine)
            created = sum(1 for payload in observed if store.record(payload))
        except (
            BridgeUnavailable,
            ExternalEvidenceConflictError,
            ExternalEvidenceValidationError,
            ProfileAdmissionError,
        ) as exc:
            return {"success": False, "durable": False, "error": str(exc)}
        except LearningWriteBusyError as exc:
            return {"success": False, "durable": False, "retryable": True, "error": str(exc)}
        return {
            "success": True,
            "durable": True,
            "observed": len(observed),
            "created": created,
            "control_plane": "observation_only",
        }
