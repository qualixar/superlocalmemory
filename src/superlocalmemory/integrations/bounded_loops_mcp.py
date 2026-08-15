"""Versioned, read-only contract boundary for Bounded Loops MCP evidence."""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Awaitable, Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

CONTRACT_ID = "bounded-loops.dev/slm-bridge/v1"
_ADVERTISEMENT = {
    "id": CONTRACT_ID,
    "tool": "bl_graph_evidence",
    "operation": "observe_terminal_run",
}


class BridgeUnavailable(ValueError):
    """The installed producer does not advertise a compatible bridge contract."""


def supports_bridge(capabilities: dict[str, Any]) -> bool:
    """Negotiate on the declared public contract, never producer semver."""
    advertised = capabilities.get("evidence_contracts")
    return isinstance(advertised, list) and any(
        isinstance(item, dict)
        and all(item.get(key) == value for key, value in _ADVERTISEMENT.items())
        for item in advertised
    )


def bridge_payload(evidence: dict[str, Any], *, profile_id: str) -> dict[str, Any]:
    """Attach active-profile identity after refusing incompatible evidence."""
    if evidence.get("contract") != CONTRACT_ID:
        raise BridgeUnavailable("unsupported bounded-loops evidence contract")
    if evidence.get("eligible_for_learning") is not False:
        raise BridgeUnavailable("bounded-loops evidence is not observation-only")
    # The producer has organisation/project metadata for its own control plane.
    # SLM stores only the v1 observation receipt needed by its profile-scoped
    # learning database; retaining arbitrary producer extensions would turn a
    # versioned contract into an unbounded schema sink.
    fields = (
        "contract",
        "workspace_id",
        "run_ref",
        "run_id",
        "outcome",
        "run_state",
        "demonstration",
        "eligible_for_learning",
        "terminal_at",
        "graph_digest",
        "plan_digest",
        "policy_digest",
        "receipt",
        "nodes",
    )
    if any(field not in evidence for field in fields):
        raise BridgeUnavailable("bounded-loops evidence is missing required v1 fields")
    return {field: deepcopy(evidence[field]) for field in fields} | {"profile_id": profile_id}


async def observe_terminal_runs(
    call_tool: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]], *, profile_id: str
) -> list[dict[str, Any]]:
    """Collect only producer-advertised terminal evidence over an injected MCP transport."""
    discovery = await call_tool("bl_capabilities", {})
    if discovery.get("status") != "ok" or not supports_bridge(discovery.get("capabilities", {})):
        raise BridgeUnavailable("bounded-loops does not advertise slm-bridge/v1")
    listing = await call_tool("bl_graph_terminal_runs", {"limit": 100})
    if listing.get("status") != "ok" or listing.get("contract") != CONTRACT_ID:
        raise BridgeUnavailable("bounded-loops terminal listing is unavailable")
    runs = listing.get("runs")
    if not isinstance(runs, list):
        raise BridgeUnavailable("bounded-loops terminal listing is malformed")
    observed: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get("run_ref"), str):
            raise BridgeUnavailable("bounded-loops terminal listing is malformed")
        response = await call_tool("bl_graph_evidence", {"run_ref": run["run_ref"]})
        if response.get("status") == "unavailable":
            continue
        if response.get("status") != "ok" or not isinstance(response.get("evidence"), dict):
            raise BridgeUnavailable("bounded-loops evidence response is malformed")
        observed.append(bridge_payload(response["evidence"], profile_id=profile_id))
    return observed


async def observe_from_stdio(*, command: str, cwd: str, profile_id: str) -> list[dict[str, Any]]:
    """Run one bounded, explicit MCP 2 observation; never call from recall or remember."""
    executable, workspace = Path(command), Path(cwd)
    if (
        not executable.is_absolute()
        or not executable.is_file()
        or not workspace.is_absolute()
        or not workspace.is_dir()
    ):
        raise BridgeUnavailable(
            "bounded-loops bridge requires an approved executable and workspace"
        )
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async with stdio_client(
        StdioServerParameters(command=str(executable), args=[], cwd=str(workspace))
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            async def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
                result = await session.call_tool(name, arguments)
                texts = [item.text for item in result.content if hasattr(item, "text")]
                if len(texts) != 1:
                    raise BridgeUnavailable("bounded-loops returned an invalid MCP payload")
                try:
                    payload = json.loads(texts[0])
                except json.JSONDecodeError as exc:
                    raise BridgeUnavailable("bounded-loops returned invalid JSON") from exc
                if not isinstance(payload, dict):
                    raise BridgeUnavailable("bounded-loops returned an invalid MCP payload")
                return payload

            try:
                return await asyncio.wait_for(
                    observe_terminal_runs(call, profile_id=profile_id), timeout=5.0
                )
            except TimeoutError as exc:
                raise BridgeUnavailable("bounded-loops observation timed out") from exc


async def observe_installed(*, workspace: str, profile_id: str) -> list[dict[str, Any]]:
    """Observe a user-installed producer without accepting an agent command.

    Discovery deliberately resolves exactly the public ``bounded-loops-mcp``
    executable.  It does not accept a command, shell fragment, or arguments
    from an MCP caller; the only caller-supplied value is the existing project
    workspace whose Bounded Loops state is to be read.
    """
    command = shutil.which("bounded-loops-mcp")
    if command is None:
        raise BridgeUnavailable("bounded-loops-mcp is not installed")
    return await observe_from_stdio(
        command=str(Path(command).resolve()), cwd=workspace, profile_id=profile_id
    )
