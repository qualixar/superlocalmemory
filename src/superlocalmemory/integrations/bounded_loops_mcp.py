"""Versioned, read-only contract boundary for Bounded Loops MCP evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import stat
from collections.abc import Awaitable, Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from superlocalmemory.storage.execution_learning import (
    VerifiedExecutionEvidence,
    _seal_verified_execution_evidence,
)

CONTRACT_ID = "bounded-loops.dev/slm-bridge/v1"
CONTRACT_V2_ID = "bounded-loops.dev/slm-bridge/v2"
_OBSERVATION_TIMEOUT_SECONDS = 5.0
_MAX_MCP_TEXT_BYTES = 2 * 1024 * 1024
_ADVERTISEMENT = {
    "id": CONTRACT_ID,
    "tool": "bl_graph_evidence",
    "operation": "observe_terminal_run",
}
_ADVERTISEMENT_V2 = {
    "id": CONTRACT_V2_ID,
    "tool": "bl_graph_execution_evidence",
    "operation": "observe_verified_terminal_run",
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


def supports_bridge_v2(capabilities: dict[str, Any]) -> bool:
    """Negotiate the additive v2 capability without altering v1 semantics."""
    advertised = capabilities.get("evidence_contracts")
    return isinstance(advertised, list) and any(
        isinstance(item, dict)
        and all(item.get(key) == value for key, value in _ADVERTISEMENT_V2.items())
        for item in advertised
    )


def _bridge_v2_payload(
    evidence: dict[str, Any], *, profile_id: str, terminal_run: dict[str, Any]
) -> dict[str, Any]:
    """Bind v2 evidence to the terminal run independently enumerated this session.

    A v2 producer cannot promote an arbitrary schema-shaped JSON object: the
    evidence must agree exactly with the separately fetched terminal listing.
    The resulting payload is sealed below before it can reach storage.
    """
    if evidence.get("contract") != CONTRACT_V2_ID:
        raise BridgeUnavailable("unsupported bounded-loops execution evidence contract")
    required_listing = ("run_ref", "run_id", "run_state", "terminal_at")
    if any(not isinstance(terminal_run.get(field), str) for field in required_listing):
        raise BridgeUnavailable("bounded-loops terminal listing is malformed")
    if any(evidence.get(field) != terminal_run[field] for field in required_listing):
        raise BridgeUnavailable("bounded-loops execution evidence does not match terminal listing")
    payload = deepcopy(evidence)
    payload["profile_id"] = profile_id
    return payload


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
    # The producer's limit is advisory.  Keep this explicit operation bounded
    # even against a compatible but faulty/malicious producer.
    runs = runs[:100]
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


async def observe_terminal_runs_v2(
    call_tool: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]], *, profile_id: str,
    producer_identity: str,
) -> list[VerifiedExecutionEvidence]:
    """Collect v2 receipts only after verified local producer provenance.

    The producer identity is measured by the stdio launcher, not supplied by
    producer JSON.  The capability and terminal-listing digests record the
    exact independently observed control-plane statements that authorized
    each sealed payload.
    """
    if not isinstance(producer_identity, str) or not producer_identity:
        raise BridgeUnavailable("bounded-loops execution producer identity is unavailable")
    discovery = await call_tool("bl_capabilities", {})
    if discovery.get("status") != "ok" or not supports_bridge_v2(discovery.get("capabilities", {})):
        raise BridgeUnavailable("bounded-loops does not advertise slm-bridge/v2")
    listing = await call_tool("bl_graph_terminal_runs", {"limit": 100})
    if (
        listing.get("status") != "ok"
        or listing.get("contract") != CONTRACT_ID
        or not isinstance(listing.get("runs"), list)
    ):
        raise BridgeUnavailable("bounded-loops terminal listing is unavailable")
    capability_digest = _canonical_digest(discovery["capabilities"])
    listing_digest = _canonical_digest(listing["runs"])
    observed: list[VerifiedExecutionEvidence] = []
    for run in listing["runs"][:100]:
        if not isinstance(run, dict) or any(
            not isinstance(run.get(field), str)
            for field in ("run_ref", "run_id", "run_state", "terminal_at")
        ):
            raise BridgeUnavailable("bounded-loops terminal listing is malformed")
        response = await call_tool("bl_graph_execution_evidence", {"run_ref": run["run_ref"]})
        if response.get("status") == "unavailable":
            continue
        if response.get("status") != "ok" or not isinstance(response.get("evidence"), dict):
            raise BridgeUnavailable("bounded-loops execution evidence response is malformed")
        payload = _bridge_v2_payload(
            response["evidence"], profile_id=profile_id, terminal_run=run
        )
        try:
            observed.append(_seal_verified_execution_evidence(
                payload,
                producer_identity=producer_identity,
                capability_digest=capability_digest,
                terminal_listing_digest=listing_digest,
            ))
        except ValueError as exc:
            raise BridgeUnavailable("bounded-loops execution evidence is invalid") from exc
    return observed


def _canonical_digest(value: Any) -> str:
    """Stable audit identity for a negotiated MCP control-plane document."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_digest(path: Path, digest: hashlib._Hash) -> None:
    """Feed one regular file into an already-bound identity hash."""
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(64 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BridgeUnavailable("bounded-loops producer source could not be identified") from exc


def _package_source_identity(executable: Path) -> str:
    """Digest the bounded_loops source loaded by the MCP launcher's venv.

    The console script is only a shim. Its shebang identifies the interpreter
    whose site-packages directory owns the running MCP server, so bind the
    source files from that environment rather than whichever package happens
    to be importable by SLM itself.
    """
    try:
        first_line = executable.open("rb").readline(4096).decode("utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise BridgeUnavailable("bounded-loops executable has no readable interpreter") from exc
    if not first_line.startswith("#!"):
        raise BridgeUnavailable("bounded-loops executable has no absolute interpreter")
    interpreter = Path(first_line[2:])
    if not interpreter.is_absolute() or not interpreter.is_file():
        raise BridgeUnavailable("bounded-loops executable interpreter is unavailable")
    if interpreter.parent.name not in {"bin", "Scripts"}:
        raise BridgeUnavailable("bounded-loops executable is not an isolated environment launcher")
    environment = interpreter.parent.parent
    candidates = sorted(
        candidate.resolve(strict=True)
        for candidate in (environment / "lib").glob("python*/site-packages/bounded_loops")
        if candidate.is_dir() and not candidate.is_symlink()
    )
    if len(candidates) != 1:
        raise BridgeUnavailable("bounded-loops package source is unavailable or ambiguous")
    package = candidates[0]
    digest = hashlib.sha256()
    source_files = sorted(
        path for path in package.rglob("*.py") if path.is_file() and not path.is_symlink()
    )
    if not source_files:
        raise BridgeUnavailable("bounded-loops package source is empty")
    for path in source_files:
        digest.update(path.relative_to(package).as_posix().encode("utf-8"))
        digest.update(b"\0")
        _file_digest(path, digest)
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _producer_identity(executable: Path) -> str:
    """Bind the trusted launcher and actual bounded_loops package source."""
    digest = hashlib.sha256()
    digest.update(b"bounded-loops-launcher\0")
    _file_digest(executable, digest)
    digest.update(b"\0bounded-loops-package\0")
    digest.update(_package_source_identity(executable).encode("ascii"))
    return "sha256:" + digest.hexdigest()


def _assert_trusted_executable(executable: Path) -> None:
    """Raise BridgeUnavailable if executable does not pass the bridge trust checks."""
    try:
        st = executable.stat()
    except OSError as exc:
        raise BridgeUnavailable("bounded-loops bridge path is unavailable") from exc
    mode = st.st_mode
    if not stat.S_ISREG(mode) or (
        os.name != "nt" and mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise BridgeUnavailable("bounded-loops executable is not a trusted regular file")
    # Windows trust-check limitation: only S_ISREG applies on Windows.
    # The mode-bit check above (group/other writable) and the uid/owner check
    # below are both skipped because:
    #   - Python's os.stat() on Windows returns emulated Unix-style mode bits
    #     that do not reflect real ACL permissions; the check would be meaningless.
    #   - os.geteuid() does not exist on Windows (AttributeError); st_uid is
    #     always 0 there, so the ownership check cannot identify non-root owners.
    # Proper Windows ownership verification requires Win32 ACL APIs (advapi32),
    # which would introduce a heavy optional dependency.  Scoped for a future release.
    if os.name != "nt" and st.st_uid not in {0, os.geteuid()}:
        raise BridgeUnavailable("bounded-loops executable owner is not trusted")

async def observe_from_stdio(*, command: str, cwd: str, profile_id: str) -> list[dict[str, Any]]:
    """Run one bounded, explicit MCP 2 observation; never call from recall or remember."""
    executable, workspace = Path(command), Path(cwd)
    if (
        not executable.is_absolute()
        or not executable.is_file()
        or not workspace.is_absolute()
        or not workspace.is_dir()
        or workspace.is_symlink()
    ):
        raise BridgeUnavailable(
            "bounded-loops bridge requires an approved executable and workspace"
        )
    try:
        executable = executable.resolve(strict=True)
        workspace = workspace.resolve(strict=True)
    except OSError as exc:
        raise BridgeUnavailable("bounded-loops bridge path is unavailable") from exc
    _assert_trusted_executable(executable)

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    try:
        parameters = StdioServerParameters(
            command=str(executable), args=[], cwd=str(workspace)
        )
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(
                read,
                write,
                # MCP 2.x passes this directly to AnyIO's timeout machinery,
                # which accepts a numeric duration rather than timedelta.
                read_timeout_seconds=_OBSERVATION_TIMEOUT_SECONDS,
            ) as session:
                async def observe() -> list[dict[str, Any]]:
                    await session.initialize()

                    async def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
                        result = await session.call_tool(name, arguments)
                        if result.is_error:
                            raise BridgeUnavailable(
                                "bounded-loops rejected the observation request"
                            )
                        texts = [item.text for item in result.content if hasattr(item, "text")]
                        if len(texts) != 1 or len(texts[0].encode("utf-8")) > _MAX_MCP_TEXT_BYTES:
                            raise BridgeUnavailable("bounded-loops returned an invalid MCP payload")
                        try:
                            payload = json.loads(texts[0])
                        except json.JSONDecodeError as exc:
                            raise BridgeUnavailable("bounded-loops returned invalid JSON") from exc
                        if not isinstance(payload, dict):
                            raise BridgeUnavailable("bounded-loops returned an invalid MCP payload")
                        return payload

                    return await observe_terminal_runs(call, profile_id=profile_id)
                return await asyncio.wait_for(observe(), timeout=_OBSERVATION_TIMEOUT_SECONDS)
    except BridgeUnavailable:
        raise
    except Exception as exc:
        raise BridgeUnavailable("bounded-loops observation timed out or could not start") from exc


async def observe_v2_from_stdio(
    *, command: str, cwd: str, profile_id: str
) -> list[VerifiedExecutionEvidence]:
    """Use the same trusted stdio boundary for additive v2 evidence."""
    # Reuse the hardened v1 launcher while selecting only v2 after capability
    # negotiation.  The injected helper is intentionally local to this call.
    executable, workspace = Path(command), Path(cwd)
    if (
        not executable.is_absolute()
        or not executable.is_file()
        or not workspace.is_absolute()
        or not workspace.is_dir()
        or workspace.is_symlink()
    ):
        raise BridgeUnavailable("bounded-loops bridge requires an approved executable and workspace")
    try:
        executable = executable.resolve(strict=True)
        workspace = workspace.resolve(strict=True)
    except OSError as exc:
        raise BridgeUnavailable("bounded-loops bridge path is unavailable") from exc
    _assert_trusted_executable(executable)
    producer_identity = _producer_identity(executable)
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    try:
        parameters = StdioServerParameters(command=str(executable), args=[], cwd=str(workspace))
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=_OBSERVATION_TIMEOUT_SECONDS) as session:
                await session.initialize()
                async def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
                    result = await session.call_tool(name, arguments)
                    texts = [item.text for item in result.content if hasattr(item, "text")]
                    if result.is_error or len(texts) != 1 or len(texts[0].encode("utf-8")) > _MAX_MCP_TEXT_BYTES:
                        raise BridgeUnavailable("bounded-loops returned an invalid MCP payload")
                    payload = json.loads(texts[0])
                    if not isinstance(payload, dict):
                        raise BridgeUnavailable("bounded-loops returned an invalid MCP payload")
                    return payload
                return await asyncio.wait_for(
                    observe_terminal_runs_v2(
                        call,
                        profile_id=profile_id,
                        producer_identity=producer_identity,
                    ),
                    _OBSERVATION_TIMEOUT_SECONDS,
                )
    except BridgeUnavailable:
        raise
    except Exception as exc:
        raise BridgeUnavailable("bounded-loops execution observation timed out or could not start") from exc


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
    if not Path(command).is_absolute():
        raise BridgeUnavailable("bounded-loops-mcp discovery returned an unsafe path")
    if Path(command).resolve().name not in {"bounded-loops-mcp", "bounded-loops-mcp.exe"}:
        raise BridgeUnavailable("bounded-loops-mcp discovery returned an unsafe executable")
    return await observe_from_stdio(
        command=str(Path(command).resolve()), cwd=workspace, profile_id=profile_id
    )


async def observe_installed_v2(
    *, workspace: str, profile_id: str
) -> list[VerifiedExecutionEvidence]:
    """Observe only from the installed bounded-loops MCP executable."""
    command = shutil.which("bounded-loops-mcp")
    if command is None or not Path(command).is_absolute():
        raise BridgeUnavailable("bounded-loops-mcp is not installed")
    executable = Path(command).resolve()
    if executable.name not in {"bounded-loops-mcp", "bounded-loops-mcp.exe"}:
        raise BridgeUnavailable("bounded-loops-mcp discovery returned an unsafe executable")
    return await observe_v2_from_stdio(command=str(executable), cwd=workspace,
                                       profile_id=profile_id)
