# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""The MCP tools, hosted in a command-line process (``slm tool``).

The host is the same thing ``slm mcp`` builds for a stdio app: an
:class:`SLMFastMCP` with every tool registered by ``mcp/tool_registry``, so
names, descriptions, input schemas, admission gates and the way each tool
reaches the daemon are the MCP server's own. Two things differ, both decided
here and never by an argument:

* the channel is ``cli``: decisions made through this door are recorded as the
  command line's (``core/agent_attribution.client_channel``);
* nothing long-lived starts: no stdio watchdog, no stdin monitor, no session
  warm-up. The process answers one call and exits.

The agent is ``SLM_AGENT_ID``, read the way the stdio server reads it.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

#: The tool set ``slm tool`` uses when none is named: every registered tool.
#: A tool set exists to keep an app's tool list short; a command line has no
#: such list, and access is decided by each tool's own admission gate.
DEFAULT_TOOL_SET = "whole"
#: ``--tool-set mcp``: exactly what an MCP connection started now would see.
CONNECTION_TOOL_SET = "mcp"
CLI_CHANNEL = "cli"
_SESSION_VARS = ("SLM_SESSION_ID", "CLAUDE_SESSION_ID", "CLAUDE_CODE_SESSION_ID")

_engine = None
_engine_lock = threading.Lock()


class UnknownToolSet(ValueError):
    """The named tool set does not exist."""


def claim_cli_channel() -> None:
    """This process is the command line's door, whatever its environment says.

    ``SLM_CLIENT_CHANNEL=mcp`` in a shell must not let a script record its
    decisions as an MCP app's. Set before any daemon request is made.
    """
    os.environ["SLM_CLIENT_CHANNEL"] = CLI_CHANNEL


def cli_engine() -> Any:
    """The LIGHT engine a tool uses in process, as the stdio server builds it."""
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            from superlocalmemory.core.config import SLMConfig
            from superlocalmemory.core.engine import MemoryEngine
            from superlocalmemory.core.engine_capabilities import Capabilities

            engine = MemoryEngine(SLMConfig.load(), capabilities=Capabilities.LIGHT)
            engine.initialize()
            _engine = engine
    return _engine


def _reset_cli_engine() -> None:
    """``set_mode`` asks its host to drop the engine it built (the stdio server's reset)."""
    global _engine
    with _engine_lock:
        _engine = None


cli_engine.reset = _reset_cli_engine  # type: ignore[attr-defined]


def _prestage(query: str, limit: int, profile_id: str, as_of: str | None = None) -> list[dict]:
    from superlocalmemory.mcp.prestage_bridge import prestage_recall

    return prestage_recall(cli_engine, query, limit, profile_id, as_of)


def allowed_tools(tool_set: str | None) -> frozenset[str] | None:
    """The tool names ``tool_set`` allows (the MCP server's own filter), ``None`` for all."""
    from superlocalmemory.mcp import tool_surface
    from superlocalmemory.mcp.profile_names import UnknownProfileError

    name = (tool_set or DEFAULT_TOOL_SET).strip().lower()
    if name == CONNECTION_TOOL_SET:
        return tool_surface.connection_surface()
    try:
        return tool_surface.surface_for(
            all_tools=False, allowlist="", profile=name,
            essential=tool_surface.essential_tools())
    except UnknownProfileError as exc:
        raise UnknownToolSet(str(exc)) from exc


def build_host(tool_set: str | None = None) -> Any:
    """A tool host with every tool registered, narrowed to ``tool_set``."""
    from superlocalmemory.mcp._daemon_proxy import choose_pool
    from superlocalmemory.mcp.http_transport import SLMFastMCP
    from superlocalmemory.mcp.tool_registry import register_tools

    claim_cli_channel()
    allowed = allowed_tools(tool_set)
    host = SLMFastMCP("SuperLocalMemory CLI")
    register_tools(host, cli_engine, choose_pool, _prestage)
    if allowed is not None:
        host.configure_tool_surface(allowed)
    return host


def registered_tools(host: Any) -> list[Any]:
    """Every registered tool (``Tool`` records), whatever the tool set or feature state."""
    return list(host._tool_manager.list_tools())


def listed_tools(host: Any) -> list[Any]:
    """The tools the host lists, exactly as an MCP ``tools/list`` would."""
    return asyncio.run(host.list_tools())


def one_line(text: str | None) -> str:
    """The first sentence-ish line of a tool description."""
    for line in (text or "").strip().splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def tool_record(tool: Any) -> dict[str, Any]:
    """One listed tool as plain JSON (``tools/list`` shape, plus the CLI form)."""
    annotations = getattr(tool, "annotations", None)
    return {
        "name": tool.name,
        "title": getattr(tool, "title", None) or "",
        "description": tool.description or "",
        "summary": one_line(tool.description),
        "input_schema": tool.input_schema,
        "annotations": annotations.model_dump(exclude_none=True) if annotations else {},
        "cli": cli_form(tool.name, tool.input_schema),
    }


def cli_form(name: str, schema: dict[str, Any]) -> str:
    """How to call ``name`` from a shell, with its required arguments."""
    required = list((schema or {}).get("required") or [])
    if not required:
        return f"slm tool {name}"
    parts = [f"--{arg.replace('_', '-')} <{arg}>" for arg in required]
    return f"slm tool {name} " + " ".join(parts)


def default_mesh_session(agent_id: str, project: str) -> str:
    """A stable mesh session for a command line that named none.

    The broker keeps one peer (and one inbox) per session id, so a fresh id per
    process would lose every message between calls. Same agent and folder:
    same peer. ``--session`` or ``SLM_SESSION_ID`` gives a private one.
    """
    digest = hashlib.sha256(f"{agent_id}\0{project}".encode("utf-8")).hexdigest()[:12]
    return f"cli-{digest}"


def bind_mesh_session(session: str | None) -> str:
    """Set the mesh session for this process; returns the one in force."""
    if session:
        os.environ["SLM_SESSION_ID"] = session
        return session
    for var in _SESSION_VARS:
        if os.environ.get(var):
            return os.environ[var]
    chosen = default_mesh_session(os.environ.get("SLM_AGENT_ID", ""), os.getcwd())
    os.environ["SLM_SESSION_ID"] = chosen
    return chosen


def result_payload(result: Any, image_dir: Path | None = None) -> tuple[Any, bool]:
    """``(json payload, is_error)`` for one ``CallToolResult``.

    The payload is the tool's structured result when it has one, else its single
    JSON text block, else every block as data. Images are written to
    ``image_dir`` when given (the block then names the file), else kept as base64.
    """
    blocks = [_block(b, image_dir, i) for i, b in enumerate(result.content or [])]
    images = [b for b in blocks if b.get("type") == "image"]
    texts = [b for b in blocks if b.get("type") == "text"]
    payload: Any
    if result.structured_content is not None:
        payload = result.structured_content
    elif len(texts) == 1:
        payload = _json_or_text(texts[0]["text"])
    else:
        payload = {"content": blocks}
        images = []
    is_error = bool(result.is_error)
    if is_error and isinstance(payload, dict) and set(payload) == {"text"}:
        payload = {"success": False, "error": payload["text"]}  # a plain-text refusal
    if images:
        payload = {"result": payload, "images": images} if not isinstance(payload, dict) \
            else {**payload, "images": images}
    return payload, is_error


def _json_or_text(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return {"text": text}


def _block(block: Any, image_dir: Path | None, index: int) -> dict[str, Any]:
    kind = getattr(block, "type", "")
    if kind == "text":
        return {"type": "text", "text": block.text}
    if kind == "image":
        mime = getattr(block, "mime_type", "") or "application/octet-stream"
        if image_dir is None:
            return {"type": "image", "mime_type": mime, "data": block.data}
        image_dir.mkdir(parents=True, exist_ok=True)
        ext = mime.split("/")[-1].split("+")[0] or "bin"
        path = image_dir / f"image-{index}.{ext}"
        path.write_bytes(base64.b64decode(block.data))
        return {"type": "image", "mime_type": mime, "path": str(path)}
    return block.model_dump(mode="json", exclude_none=True)


__all__ = [
    "CLI_CHANNEL", "CONNECTION_TOOL_SET", "DEFAULT_TOOL_SET", "UnknownToolSet",
    "allowed_tools", "bind_mesh_session", "build_host", "claim_cli_channel", "cli_engine",
    "cli_form", "default_mesh_session", "listed_tools", "one_line", "registered_tools",
    "result_payload", "tool_record",
]
