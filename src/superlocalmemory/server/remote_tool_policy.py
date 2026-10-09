# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Which MCP tools an AI tool on another computer may call.

Default deny. A tool that is in neither :data:`READ_TOOLS` nor
:data:`WRITE_TOOLS` is host-only: no remote caller can run it, whatever its key.
Host-only tools manage the SLM computer itself - switching its active profile,
indexing local folders, maintenance, retention, mesh, loops, deleting by
pattern. ``tests/test_security/test_remote_tool_policy.py`` fails when a tool is
registered without being classified here.

:class:`RemoteToolScopeASGI` enforces the policy on every MCP request that
carries a remote principal (set by :mod:`server.remote_access`):

* only ``initialize``, ``server/discover``, ``ping``, ``tools/list``, ``tools/call`` and client
  notifications are accepted; every other MCP method is refused;
* tool names are matched exactly - there are no aliases, so ``Remember`` or
  ``remember `` is simply an unknown, refused name;
* a body with a repeated JSON key, a batch (JSON array), or a body over 1 MiB
  is refused before the MCP server parses it;
* ``tools/list`` answers list only the tools the key may call.

A refused ``tools/call`` is answered as an MCP tool error (``isError``), so the
caller sees the reason in the tool result.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger("superlocalmemory.remote")
audit_logger = logging.getLogger("superlocalmemory.remote.audit")

MAX_BODY_BYTES = 1_048_576
MAX_RESPONSE_BYTES = 4 * 1_048_576

READ_TOOLS: frozenset[str] = frozenset({
    "fetch", "get_assertions", "get_attribution", "get_behavioral_patterns",
    "get_brain_evidence_status", "get_learned_patterns", "get_lifecycle_status",
    "get_memory_summary", "get_mode", "get_retention_stats", "get_soft_prompts",
    "get_status", "get_version", "health", "list_corrections", "list_recent",
    "memory_kinds_status", "memory_used", "prestage_context", "recall", "recall_trace",
    "review_memory_kinds", "run_view", "search", "skill_health", "skill_lineage",
    "slm_cache_get",
    "slm_loop_history", "slm_loop_show", "slm_optimize_stats", "slm_retrieve",
})

WRITE_ONLY_TOOLS: frozenset[str] = frozenset({
    "close_session", "confirm_memory_kinds", "contradict_assertion", "core_memory",
    "correct_pattern", "delete_memory", "finalize_cognitive_turn", "log_tool_event",
    "manage_view",
    "observe", "record_agent_experience", "record_cognitive_turn",
    "reinforce_assertion", "remember", "report_feedback", "report_outcome",
    "review_correction", "session_init", "set_memory_kind", "settle_session_outcomes",
    "slm_cache_set", "slm_compress", "update_memory",
})

WRITE_TOOLS: frozenset[str] = READ_TOOLS | WRITE_ONLY_TOOLS

# Mesh tools stay denied to remote callers until a verified per-app identity
# exists; flipping this alone does not expose them.
REMOTE_MESH_TOOLS_ENABLED = False

HOST_ONLY_TOOLS: frozenset[str] = frozenset({
    "apply_refactor", "audit_trail", "backup_status", "build_code_graph", "build_graph",
    "code_entity_history", "code_memory_search", "code_stale_check", "compact_memories",
    "consistency_check", "consolidate_cognitive", "detect_changes", "enrich_blast_radius",
    "evolve_skill", "find_large_functions", "forget", "get_affected_flows", "get_media",
    "get_architecture_overview", "get_blast_radius", "get_community", "get_flow",
    "get_review_context", "link_memory_to_code", "list_communities",
    "list_failed_operations", "list_flows", "list_graph_stats", "mesh_events",
    "mesh_inbox", "mesh_lock", "mesh_peers", "mesh_send", "mesh_state", "mesh_status",
    "mesh_summary", "mesh_wait", "observe_bounded_loop_evidence",
    "observe_bounded_loop_execution_learning", "quantize", "query_graph",
    "reap_processes", "refactor_preview", "remember_media", "resolve_operation", "run_maintenance",
    "semantic_search_code", "set_mode", "set_retention_policy", "slm_loop_run",
    "switch_profile", "update_code_graph",
})

#: Image tools. They are in :data:`HOST_ONLY_TOOLS` and stay there until the owner decides how a
#: remote key gets image rights; flipping this switch alone changes nothing.
MEDIA_TOOLS: frozenset[str] = frozenset({"remember_media", "get_media"})
REMOTE_MEDIA_TOOLS_ENABLED = False

#: MCP methods a remote caller may send. Everything else is refused.
ALLOWED_METHODS: frozenset[str] = frozenset({
    "initialize", "server/discover", "ping", "tools/list", "tools/call",
    "notifications/initialized", "notifications/cancelled",
})

DENIAL_CODE = "remote_tool_not_allowed"


def tool_allowed(scope: str, name: object) -> bool:
    """Exact-name membership; anything unexpected is a no."""
    if not isinstance(name, str):
        return False
    if scope == "read":
        return name in READ_TOOLS
    if scope == "write":
        return name in WRITE_TOOLS
    return False


class PolicyViolation(ValueError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise PolicyViolation(400, "duplicate_json_key",
                                  f"The request repeats the JSON key '{key}'.")
        seen[key] = value
    return seen


def parse_message(body: bytes) -> dict[str, Any]:
    """The single JSON-RPC message in ``body``. Raises :class:`PolicyViolation`."""
    try:
        message = json.loads(body.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except PolicyViolation:
        raise
    except (UnicodeDecodeError, ValueError) as exc:
        raise PolicyViolation(400, "invalid_jsonrpc", "The request is not valid JSON.") from exc
    if isinstance(message, list):
        raise PolicyViolation(400, "batch_not_supported",
                              "Batched MCP requests are not accepted over remote access.")
    if not isinstance(message, dict):
        raise PolicyViolation(400, "invalid_jsonrpc", "The request is not a JSON-RPC message.")
    method = message.get("method")
    if not isinstance(method, str):
        raise PolicyViolation(400, "invalid_jsonrpc",
                              "Only JSON-RPC requests and notifications are accepted.")
    return message


def called_tool(message: dict[str, Any]) -> str | None:
    """The tool name of a ``tools/call`` message, ``None`` for other methods."""
    if message.get("method") != "tools/call":
        return None
    params = message.get("params")
    name = params.get("name") if isinstance(params, dict) else None
    if not isinstance(name, str):
        raise PolicyViolation(400, "invalid_jsonrpc", "tools/call needs a string tool name.")
    return name


#: Appended to a refusal for a read-only key, so a client can stop sending
#: writes it will never be allowed (the Hermes plugin does).
READ_ONLY_TAG = "[remote_key_read_only]"


def denial_message(tool: str, key_name: str, scope: str) -> str:
    if tool in HOST_ONLY_TOOLS:
        return (f"'{tool}' manages the SLM computer and is not available over remote "
                f"access. Run it on the SLM computer. [{DENIAL_CODE}]")
    if tool in WRITE_ONLY_TOOLS and scope == "read":
        return (f"'{tool}' changes memory, and remote key '{key_name}' is read-only. "
                f"Use a write key (slm remote keys add <name>) to save from this tool. "
                f"{READ_ONLY_TAG}")
    return f"'{tool}' is not available to remote key '{key_name}'. [{DENIAL_CODE}]"


# -- ASGI helpers --------------------------------------------------------------------


async def _send_json(send: Callable[..., Awaitable[None]], status: int,
                     payload: dict[str, Any],
                     extra_headers: tuple[tuple[bytes, bytes], ...] = ()) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                            *extra_headers]})
    await send({"type": "http.response.body", "body": body})


#: Answer to any non-POST request from another computer. The MCP transport is
#: stateless here, so a GET event stream could never carry a message; it would
#: only hold a connection (and a worker) open for as long as the caller liked.
METHOD_NOT_ALLOWED = {
    "error": "remote_method_not_allowed",
    "message": "Remote MCP accepts POST only.",
}


async def _read_body(receive: Callable[[], Awaitable[dict[str, Any]]]) -> bytes | None:
    """The whole request body, or ``None`` when it is over the cap."""
    chunks: list[bytes] = []
    size = 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            break
        chunk = message.get("body", b"") or b""
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            return None
        chunks.append(chunk)
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


def _replay(body: bytes, receive: Callable[[], Awaitable[dict[str, Any]]]):
    delivered = False

    async def _receive() -> dict[str, Any]:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return _receive


def _filter_tools_list(body: bytes, scope: str) -> bytes:
    try:
        payload = json.loads(body.decode("utf-8"))
        tools = payload["result"]["tools"]
    except (UnicodeDecodeError, ValueError, KeyError, TypeError):
        return body
    if not isinstance(tools, list):
        return body
    kept = [t for t in tools if isinstance(t, dict) and tool_allowed(scope, t.get("name"))]
    result = dict(payload["result"], tools=kept)
    return json.dumps(dict(payload, result=result)).encode("utf-8")


def _redact_call_answer(body: bytes) -> bytes:
    """The tool answer with host details removed. Fails closed: an answer that
    cannot be read as a JSON-RPC response is not forwarded as it is."""
    from superlocalmemory.server.remote_redaction import redact_rpc_error, redact_tool_result

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        payload = None
    if not isinstance(payload, dict):
        return json.dumps({"jsonrpc": "2.0", "id": None, "error": {
            "code": -32603, "message": "The tool answer could not be checked."}}).encode()
    if "error" in payload:
        payload = dict(payload, error=redact_rpc_error(payload["error"]))
    if isinstance(payload.get("result"), dict):
        payload = dict(payload, result=redact_tool_result(payload["result"]))
    return json.dumps(payload).encode()


def _filter_discovery(body: bytes) -> bytes:
    """Expose only the remote tools capability; never share scoped discovery."""
    try:
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("invalid_discovery")
        if "error" in payload:
            return _redact_call_answer(body)
        source = payload["result"]
        versions = source["supportedVersions"]
        if not isinstance(versions, list) or len(versions) > 32 or not all(isinstance(v, str) and len(v) <= 32 for v in versions):
            raise ValueError("invalid_discovery")
        result = {"cacheScope": "private", "ttlMs": 0, "resultType": "complete",
                  "supportedVersions": versions, "capabilities": {"tools": {}},
                  "instructions": "Scoped SuperLocalMemory access. Use tools/list for available tools."}
        return json.dumps({"jsonrpc": "2.0", "id": payload.get("id"), "result": result}).encode()
    except (UnicodeDecodeError, ValueError, TypeError, KeyError):
        return json.dumps({"jsonrpc": "2.0", "id": None, "error": {
            "code": -32603, "message": "Discovery could not be checked."}}).encode()


class _JsonAnswerFilter:
    """Buffers the single JSON answer and rewrites it with ``transform``. A
    streamed (non-JSON) answer is not forwarded at all (fail closed)."""

    def __init__(self, send: Callable[..., Awaitable[None]],
                 transform: Callable[[bytes], bytes]) -> None:
        self._send = send
        self._transform = transform
        self._start: dict[str, Any] | None = None
        self._chunks: list[bytes] = []
        self._size = 0
        self._refused = False

    async def __call__(self, message: dict[str, Any]) -> None:
        if self._refused:
            return
        if message["type"] == "http.response.start":
            self._start = message
            return
        if message["type"] != "http.response.body":
            await self._send(message)
            return
        chunk = message.get("body", b"") or b""
        self._size += len(chunk)
        if self._size > MAX_RESPONSE_BYTES:
            self._refused = True
            self._chunks.clear()
            await _send_json(self._send, 502, {"error": "remote_answer_too_large"})
            return
        self._chunks.append(chunk)
        if message.get("more_body", False):
            return
        start = self._start or {"type": "http.response.start", "status": 500, "headers": []}
        headers = dict((k.lower(), v) for k, v in start.get("headers", []))
        if b"application/json" not in headers.get(b"content-type", b""):
            await _send_json(self._send, 502, {"error": "remote_answer_unfilterable"})
            return
        body = self._transform(b"".join(self._chunks))
        new_headers = [(k, v) for k, v in start.get("headers", [])
                       if k.lower() not in {b"content-length", b"cache-control", b"etag", b"last-modified", b"age"}]
        new_headers.append((b"cache-control", b"no-store"))
        new_headers.append((b"content-length", str(len(body)).encode()))
        await self._send(dict(start, headers=new_headers))
        await self._send({"type": "http.response.body", "body": body})


def _audit(principal: Any, scope: dict[str, Any], tool: str, decision: str) -> None:
    from superlocalmemory.mcp.agent_context import sanitize_agent_id

    root = scope.get("root_path", "") or ""
    path = scope.get("path", "") or ""
    agent = path[len(root):].lstrip("/").split("/")[0] if path.startswith(root) else ""
    audit_logger.info(
        "remote tools/call key_id=%s key_name=%s scope=%s profile=%s agent=%s tool=%s "
        "decision=%s",
        principal.key_id, principal.name, principal.scope, principal.profile or "(active)",
        sanitize_agent_id(agent) or "-", sanitize_agent_id(tool), decision,
    )


def _tool_error(message: dict[str, Any], text: str,
                code: str = DENIAL_CODE) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message.get("id"),
            "result": {"content": [{"type": "text", "text": text}], "isError": True,
                       "structuredContent": {"error": code, "message": text}}}


def _with_body(scope: dict[str, Any], body: bytes) -> dict[str, Any]:
    """``scope`` with its Content-Length matching a rewritten ``body``."""
    headers = [(k, v) for k, v in scope.get("headers", []) if k.lower() != b"content-length"]
    headers.append((b"content-length", str(len(body)).encode()))
    return dict(scope, headers=headers)


def _method_error(message: dict[str, Any], method: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message.get("id"),
            "error": {"code": -32601,
                      "message": f"'{method}' is not available over remote access."}}


class RemoteToolScopeASGI:
    """Wraps the MCP app. Local callers pass straight through.

    ``runtime_for`` finds the daemon's profile runtime for a request (default:
    the one on ``scope["app"].state``). Without one a remote tool call is
    refused - there would be no way to hold it to its key's profile.
    """

    def __init__(self, app: Any, runtime_for: Callable[[dict[str, Any]], Any] | None = None
                 ) -> None:
        self.app = app
        if runtime_for is None:
            from superlocalmemory.server.remote_profile_binding import runtime_from_scope

            runtime_for = runtime_from_scope
        self._runtime_for = runtime_for

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        from superlocalmemory.server.remote_access import (
            is_trusted_local_peer,
            principal_from_scope,
        )

        principal = principal_from_scope(scope)
        if principal is None:
            if is_trusted_local_peer(scope):
                await self.app(scope, receive, send)
                return
            # Fail closed: a network request reached the MCP app without a
            # principal (the auth middleware did not run or was bypassed).
            await _send_json(send, 401, {"error": "remote_auth_required"})
            return
        if scope.get("method") != "POST":
            await _send_json(send, 405, METHOD_NOT_ALLOWED, ((b"allow", b"POST"),))
            return
        body = await _read_body(receive)
        if body is None:
            await _send_json(send, 413, {"error": "body_too_large",
                                         "message": "Remote MCP requests are limited to 1 MiB."})
            return
        try:
            message = parse_message(body)
            method = message["method"]
            if method not in ALLOWED_METHODS:
                await _send_json(send, 200, _method_error(message, method))
                return
            tool = called_tool(message)
        except PolicyViolation as violation:
            await _send_json(send, violation.status,
                             {"error": violation.code, "message": str(violation)})
            return
        if tool is not None:
            allowed = tool_allowed(principal.scope, tool)
            _audit(principal, scope, tool, "allow" if allowed else "deny")
            if not allowed:
                await _send_json(send, 200, _tool_error(
                    message, denial_message(tool, principal.name, principal.scope)))
                return
        if message["method"] == "tools/list":
            downstream_send = _JsonAnswerFilter(
                send, lambda raw: _filter_tools_list(raw, principal.scope))
        elif message["method"] == "server/discover":
            downstream_send = _JsonAnswerFilter(send, _filter_discovery)
        elif tool is not None:
            # Host details (paths, home, account, environment) never leave
            # this computer in a tool answer (server/remote_redaction).
            downstream_send = _JsonAnswerFilter(send, _redact_call_answer)
        else:
            downstream_send = send
        from superlocalmemory.mcp.remote_caller import remote_caller

        # Per-agent stores (cache, reversible compression) are keyed by this
        # key as well as by the caller-chosen /mcp/<agent> segment.
        with remote_caller(principal.key_id):
            if tool is None:
                await self.app(scope, _replay(body, receive), downstream_send)
            else:
                await self._call_in_bound_profile(principal, scope, receive, send,
                                                  downstream_send, message, tool)

    async def _call_in_bound_profile(self, principal: Any, scope: dict[str, Any],
                                     receive: Any, send: Any, downstream_send: Any,
                                     message: dict[str, Any], tool: str) -> None:
        """Run one tool call for the key's profile (server/remote_profile_binding)."""
        from superlocalmemory.server.remote_profile_binding import (
            PROFILE_FREE_TOOLS,
            ROUTED_TOOLS,
            inactive_refusal,
            profile_lease,
        )

        runtime = self._runtime_for(scope)
        if runtime is None:
            await _send_json(send, 503, {"error": "remote_profile_unavailable",
                                         "message": "The SLM profile state is not ready."})
            return
        if tool in ROUTED_TOOLS or tool in PROFILE_FREE_TOOLS:
            # Routed per request (or touches no profile): the host's active
            # profile is not involved, so no lease and no active check.
            bound = principal.profile or runtime.snapshot.profile_id
            await self._run_bound(principal, scope, receive, send, downstream_send,
                                  message, tool, bound)
            return
        async with profile_lease(runtime) as active_profile:
            if principal.profile is not None and active_profile != principal.profile:
                refusal = inactive_refusal(tool, principal.name, principal.profile)
                _audit(principal, scope, tool, f"deny:{refusal.code}")
                await _send_json(send, 200, _tool_error(message, str(refusal), refusal.code))
                return
            await self._run_bound(principal, scope, receive, send, downstream_send,
                                  message, tool, principal.profile or active_profile)

    async def _run_bound(self, principal: Any, scope: dict[str, Any], receive: Any,
                         send: Any, downstream_send: Any, message: dict[str, Any],
                         tool: str, bound: str) -> None:
        from superlocalmemory.server.remote_profile_binding import (
            BindingRefusal,
            bind_arguments,
        )

        params = message.get("params") or {}
        try:
            arguments = bind_arguments(tool, params.get("arguments"),
                                       key_name=principal.name, bound=bound)
        except BindingRefusal as refusal:
            _audit(principal, scope, tool, f"deny:{refusal.code}")
            await _send_json(send, 200, _tool_error(message, str(refusal), refusal.code))
            return
        body = json.dumps(dict(message, params=dict(params, arguments=arguments))).encode()
        await self.app(_with_body(scope, body), _replay(body, receive), downstream_send)

__all__ = [
    "ALLOWED_METHODS",
    "DENIAL_CODE",
    "HOST_ONLY_TOOLS",
    "MEDIA_TOOLS",
    "REMOTE_MEDIA_TOOLS_ENABLED",
    "MAX_BODY_BYTES",
    "METHOD_NOT_ALLOWED",
    "PolicyViolation",
    "READ_ONLY_TAG",
    "READ_TOOLS",
    "RemoteToolScopeASGI",
    "WRITE_ONLY_TOOLS",
    "WRITE_TOOLS",
    "called_tool",
    "denial_message",
    "parse_message",
    "tool_allowed",
]
