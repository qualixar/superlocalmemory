"""Native Hermes adapter for SuperLocalMemory.

This module intentionally imports no SuperLocalMemory Python package.  It uses
Hermes's allowlisted MCP client for lifecycle work and the installed ``slm``
console entry point only for user-invoked CLI compatibility commands.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
INVENTORY = json.loads((ROOT / "command-inventory.json").read_text(encoding="utf-8"))
COMMANDS = tuple(INVENTORY["primary_commands"])
HIGH_IMPACT = frozenset(INVENTORY["high_impact"])
ROLES = {
    "memory": "slm-memory-advisor.md",
    "governance": "slm-governance-advisor.md",
    "optimize": "slm-optimize-advisor.md",
    "loop": "slm-loop-runner.md",
}
_MAX_TEXT = 8_000
_RELEASE_SLM_VERSION = (4, 1, 12)
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"(?i)\b(api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,}\]]+"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
)


def _bounded(value: Any, limit: int = _MAX_TEXT) -> str:
    text = "" if value is None else str(value)
    return text[:limit] + ("… [truncated]" if len(text) > limit else "")


def _redact(value: Any, limit: int = _MAX_TEXT) -> str:
    """Bound hook telemetry and keep secrets/PII out of the memory boundary."""
    try:
        text = json.dumps(value, sort_keys=True, default=str) if not isinstance(value, str) else value
    except (TypeError, ValueError):
        text = str(value)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]", text)
    return _bounded(text, limit)


def _deserialize_handle(value: dict[str, Any]) -> Any:
    from agent.subagent_lifecycle import SubagentHandle
    return SubagentHandle.from_dict(value)


def _json_result(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(value)
    return value if isinstance(value, dict) else {"result": str(value)}


def _slm_version(binary: str) -> str | None:
    try:
        completed = subprocess.run([binary, "--version"], text=True, capture_output=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", completed.stdout + completed.stderr)
    return match.group(0) if completed.returncode == 0 and match else None


def _supported_slm(binary: str) -> bool:
    raw = _slm_version(binary)
    if not raw:
        return False
    # This Hermes artifact is an exact release companion, not a floating
    # compatibility promise. Refuse a different runtime until that contract is
    # explicitly versioned and tested in a later release.
    return tuple(int(part) for part in raw.split(".")) == _RELEASE_SLM_VERSION


def _safe_mcp(ctx: Any, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Use Hermes's capability-gated MCP path; observers must always fail open."""
    try:
        return ctx.call_mcp("superlocalmemory", tool, arguments, timeout=3)
    except Exception:
        return {"ok": False}


def _session_args(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": str(kwargs.get("session_id") or ""),
        "agent_id": "hermes",
        "project_path": str(kwargs.get("project_path") or kwargs.get("cwd") or ""),
        "query": _redact(kwargs.get("query") or "", 4_000),
    }


class SlmHermesPlugin:
    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self._sessions: set[str] = set()
        self._lock = threading.Lock()

    def on_session_start(self, **kwargs: Any) -> None:
        args = _session_args(kwargs)
        if not args["session_id"]:
            return
        _safe_mcp(self.ctx, "session_init", args)
        with self._lock:
            self._sessions.add(args["session_id"])

    def pre_llm_call(self, **kwargs: Any) -> dict[str, str] | None:
        session_id = str(kwargs.get("session_id") or "")
        query = _redact(kwargs.get("user_message"), 4_000)
        if not session_id or not query:
            return None
        result = _safe_mcp(self.ctx, "recall", {"query": query, "session_id": session_id, "agent_id": "hermes", "limit": 5})
        if not result.get("ok"):
            return None
        rendered = _redact(result.get("result"), 6_000)
        if not rendered:
            return None
        return {"context": "[UNTRUSTED SLM EVIDENCE — verify before acting]\n" + rendered + "\n[END UNTRUSTED SLM EVIDENCE]"}

    def post_tool_call(self, **kwargs: Any) -> None:
        args = {
            "session_id": str(kwargs.get("session_id") or ""), "agent_id": "hermes",
            "tool_name": str(kwargs.get("tool_name") or ""),
            "event_type": "error" if str(kwargs.get("status") or "") in {"error", "failed", "blocked"} else "complete",
            "duration_ms": int(kwargs.get("duration_ms") or 0),
            "input_summary": _redact(kwargs.get("args"), 2_000), "output_summary": _redact(kwargs.get("result"), 2_000),
            "metadata": json.dumps({"tool_call_id": str(kwargs.get("tool_call_id") or ""), "status": str(kwargs.get("status") or "")}, sort_keys=True),
        }
        _safe_mcp(self.ctx, "log_tool_event", args)

    def post_llm_call(self, **kwargs: Any) -> None:
        if self.ctx.get_config("capture_turns", False) is not True:
            return
        _safe_mcp(self.ctx, "observe", {
            "session_id": str(kwargs.get("session_id") or ""), "agent_id": "hermes",
            "content": "User: " + _redact(kwargs.get("user_message"), 4_000) + "\nAssistant: " + _redact(kwargs.get("assistant_response"), 6_000),
        })

    def on_session_end(self, **kwargs: Any) -> None:
        # Hermes emits this after every turn. Per-turn settlement selects only
        # outcomes that already carry real engagement evidence; evidence-free
        # recalls remain pending for later turns or durable finalization.
        _safe_mcp(self.ctx, "settle_session_outcomes", {
            "session_id": str(kwargs.get("session_id") or ""), "agent_id": "hermes",
        })

    def on_session_finalize(self, **kwargs: Any) -> None:
        session_id = str(kwargs.get("session_id") or "")
        _safe_mcp(self.ctx, "settle_session_outcomes", {
            "session_id": session_id, "agent_id": "hermes", "finalize": True,
        })
        _safe_mcp(self.ctx, "close_session", {"session_id": session_id, "agent_id": "hermes"})
        with self._lock:
            self._sessions.discard(session_id)

    def on_session_reset(self, **kwargs: Any) -> None:
        old_id = str(kwargs.get("old_session_id") or kwargs.get("session_id") or "")
        if old_id:
            self.on_session_finalize(session_id=old_id)
        new_id = str(kwargs.get("new_session_id") or "")
        if new_id:
            self.on_session_start(session_id=new_id, model=kwargs.get("model"), platform=kwargs.get("platform"))

    def on_skill_lifecycle(self, **kwargs: Any) -> None:
        _safe_mcp(self.ctx, "log_tool_event", {
            "session_id": str(kwargs.get("session_id") or ""), "agent_id": "hermes",
            "tool_name": "skill:" + str(kwargs.get("skill_name") or ""), "event_type": "complete",
            "metadata": json.dumps({"status": str(kwargs.get("status") or "")}, sort_keys=True),
        })

    def subagent_start(self, **kwargs: Any) -> None:
        _safe_mcp(self.ctx, "log_tool_event", {
            "session_id": str(kwargs.get("parent_session_id") or ""), "agent_id": "hermes",
            "tool_name": "slm-advisor:" + str(kwargs.get("child_role") or ""), "event_type": "invoke",
            "metadata": json.dumps({"status": "started"}, sort_keys=True),
        })

    def subagent_stop(self, **kwargs: Any) -> None:
        result = _redact(kwargs.get("child_summary"), 4_000)
        _safe_mcp(self.ctx, "log_tool_event", {
            "session_id": str(kwargs.get("parent_session_id") or ""), "agent_id": "hermes",
            "tool_name": "slm-advisor:" + str(kwargs.get("child_role") or ""), "event_type": "complete",
            "output_summary": result,
            "metadata": json.dumps({
                "status": str(kwargs.get("child_status") or ""),
                "result_sha256": hashlib.sha256(result.encode()).hexdigest() if result else "",
            }, sort_keys=True),
        })

    def _launch(self, role: str, goal: str) -> dict[str, Any]:
        if role not in ROLES:
            return {"ok": False, "error": "role must be memory, governance, optimize, or loop"}
        if not goal.strip():
            return {"ok": False, "error": "goal is required"}
        from agent.subagent_lifecycle import SubagentLaunchRequest
        prompt = (ROOT / "agents" / ROLES[role]).read_text(encoding="utf-8")
        try:
            # Hermes validates allowed_toolsets against its static registry;
            # dynamic MCP toolsets such as mcp-superlocalmemory cannot be named
            # there.  Omission deliberately inherits the parent session's
            # already-authorized MCP surface rather than launching an advisor
            # that cannot use SLM at all.  This is inherits_parent_mcp.
            handle = self.ctx.subagent_lifecycle.launch(SubagentLaunchRequest(goal=_redact(goal, 8_000), context=_bounded(prompt, 16_000), role="leaf", correlation_id=f"slm-{role}-{uuid.uuid4().hex}"))
            return {"ok": True, "handle": handle.to_dict()}
        except Exception as exc:
            return {"ok": False, "error": _bounded(exc, 500)}

    def agent_tool(self, role: str = "", goal: str = "", **_: Any) -> dict[str, Any]:
        return self._launch(role, goal)

    def agent_status_tool(self, handle: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        try:
            return _json_result(self.ctx.subagent_lifecycle.status(_deserialize_handle(handle or {})))
        except Exception as exc:
            return {"ok": False, "error": _bounded(exc, 500)}

    def agent_cancel_tool(self, handle: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        try:
            return _json_result(self.ctx.subagent_lifecycle.cancel(_deserialize_handle(handle or {}), reason="User requested cancellation"))
        except Exception as exc:
            return {"ok": False, "error": _bounded(exc, 500)}

    def agent_result_tool(self, handle: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        try:
            return _json_result(self.ctx.subagent_lifecycle.result(_deserialize_handle(handle or {})))
        except Exception as exc:
            return {"ok": False, "error": _bounded(exc, 500)}

    def slash_agent(self, raw_args: str) -> str:
        role, _, goal = raw_args.strip().partition(" ")
        return json.dumps(self._launch(role, goal), sort_keys=True)

    def slash_agent_status(self, raw_args: str) -> str:
        try:
            return json.dumps(self.agent_status_tool(json.loads(raw_args)), sort_keys=True)
        except ValueError:
            return "Usage: /slm-agent-status <JSON handle from /slm-agent>"

    def slash_agent_cancel(self, raw_args: str) -> str:
        try:
            return json.dumps(self.agent_cancel_tool(json.loads(raw_args)), sort_keys=True)
        except ValueError:
            return "Usage: /slm-agent-cancel <JSON handle from /slm-agent>"

    def slash_agent_result(self, raw_args: str) -> str:
        try:
            return json.dumps(self.agent_result_tool(json.loads(raw_args)), sort_keys=True)
        except ValueError:
            return "Usage: /slm-agent-result <JSON handle from /slm-agent>"

    def slash_router(self, raw_args: str, forced: str | None = None) -> str:
        try:
            argv = shlex.split(raw_args)
        except ValueError as exc:
            return f"Invalid arguments: {exc}"
        if forced:
            argv.insert(0, forced)
        if not argv:
            return "Usage: /slm <command> [args]. Use /slm help for the SLM command tree."
        command = INVENTORY["aliases"].get(argv[0], argv[0])
        if command not in COMMANDS:
            return f"Unsupported SLM command: {argv[0]}"
        if command in HIGH_IMPACT and "CONFIRM" not in argv:
            return f"Preview required. Re-run /slm {' '.join(argv)} CONFIRM to execute this high-impact command."
        argv = [arg for arg in argv if arg != "CONFIRM"]
        binary = shutil.which("slm")
        if not binary:
            return "SLM CLI is unavailable. Install the owning runtime; the Hermes plugin never installs Python packages."
        if not _supported_slm(binary):
            return "Hermes SLM plugin requires exactly SLM CLI 4.1.12 from the owning runtime; refusing an incompatible or unverifiable executable."
        argv[0] = command
        if "--json" not in argv and command not in {"mcp", "dashboard", "serve", "proxy", "warmup", "setup"}:
            argv.append("--json")
        try:
            completed = subprocess.run([binary, *argv], text=True, capture_output=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"SLM command unavailable: {_bounded(exc, 500)}"
        return _bounded(completed.stdout or completed.stderr or f"slm exited {completed.returncode}")


def register(ctx: Any) -> None:
    plugin = SlmHermesPlugin(ctx)
    # Explicit registrations keep generated output auditable and make a missing skill visible in doctor tests.
    ctx.register_skill("slm-cache", ROOT / "skills" / "slm-cache" / "SKILL.md")
    ctx.register_skill("slm-compress", ROOT / "skills" / "slm-compress" / "SKILL.md")
    ctx.register_skill("slm-governance", ROOT / "skills" / "slm-governance" / "SKILL.md")
    ctx.register_skill("slm-graph", ROOT / "skills" / "slm-graph" / "SKILL.md")
    ctx.register_skill("slm-loop", ROOT / "skills" / "slm-loop" / "SKILL.md")
    ctx.register_skill("slm-mesh", ROOT / "skills" / "slm-mesh" / "SKILL.md")
    ctx.register_skill("slm-profile", ROOT / "skills" / "slm-profile" / "SKILL.md")
    ctx.register_skill("slm-recall", ROOT / "skills" / "slm-recall" / "SKILL.md")
    ctx.register_skill("slm-remember", ROOT / "skills" / "slm-remember" / "SKILL.md")
    ctx.register_skill("slm-scope", ROOT / "skills" / "slm-scope" / "SKILL.md")
    ctx.register_skill("slm-session", ROOT / "skills" / "slm-session" / "SKILL.md")
    ctx.register_skill("slm-status", ROOT / "skills" / "slm-status" / "SKILL.md")
    for hook in ("on_session_start", "pre_llm_call", "post_tool_call", "post_llm_call", "on_session_end", "on_session_finalize", "on_session_reset", "on_skill_lifecycle", "subagent_start", "subagent_stop"):
        ctx.register_hook(hook, getattr(plugin, hook))
    ctx.register_command("slm", plugin.slash_router, "Run a SuperLocalMemory command", "<command> [args]")
    for command in COMMANDS:
        ctx.register_command(f"slm-{command}", lambda raw_args, command=command: plugin.slash_router(raw_args, command), f"SLM {command}", "[args]")
    ctx.register_command("slm-agent", plugin.slash_agent, "Launch a bounded SLM advisor child", "<role> <goal>")
    ctx.register_command("slm-agent-status", plugin.slash_agent_status, "Inspect an SLM advisor child", "<handle-json>")
    ctx.register_command("slm-agent-cancel", plugin.slash_agent_cancel, "Cancel an SLM advisor child", "<handle-json>")
    ctx.register_command("slm-agent-result", plugin.slash_agent_result, "Get an SLM advisor child result", "<handle-json>")
    ctx.register_tool("slm_agent", "slm", {"type": "object", "properties": {"role": {"type": "string", "enum": sorted(ROLES)}, "goal": {"type": "string"}}, "required": ["role", "goal"]}, plugin.agent_tool, "Launch an explicit SLM advisor child agent")
    ctx.register_tool("slm_agent_status", "slm", {"type": "object", "properties": {"handle": {"type": "object"}}, "required": ["handle"]}, plugin.agent_status_tool, "Inspect an SLM advisor child")
    ctx.register_tool("slm_agent_cancel", "slm", {"type": "object", "properties": {"handle": {"type": "object"}}, "required": ["handle"]}, plugin.agent_cancel_tool, "Cancel an SLM advisor child")
    ctx.register_tool("slm_agent_result", "slm", {"type": "object", "properties": {"handle": {"type": "object"}}, "required": ["handle"]}, plugin.agent_result_tool, "Get an SLM advisor child result")
