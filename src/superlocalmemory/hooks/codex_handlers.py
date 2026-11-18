# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Codex lifecycle hooks: ``slm hook codex-start|codex-prompt|codex-stop``.

Moved out of ``hook_handlers`` (4.1.22, which re-exports these names) and
rebuilt around one enforceable deadline per event (``hook_deadline``):

=================  ===================  =====================  =================
event              host timeout         SLM's own deadline     output
=================  ===================  =====================  =================
SessionStart       15 s (codex_hooks)   10 s, watchdog 11 s    one JSON object
UserPromptSubmit   5 s                  3.5 s, watchdog 4 s    one JSON object
Stop               12 s                 9 s, watchdog 10 s     ``{}`` (JSON)
=================  ===================  =====================  =================

Every path, including a hung ``slm mcp`` child, a malformed reply, a cold
daemon and an exception, ends with exactly one well-formed output before the
host's limit. Codex documents plain text as invalid output for ``Stop``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

from superlocalmemory import __version__
from superlocalmemory.hooks.hook_deadline import (
    ChildRegistry,
    HookDeadline,
    McpStdio,
    Watchdog,
    run_bounded,
)

START_DEADLINE_S = 10.0  # + interpreter start, well under the 15 s host limit
PROMPT_DEADLINE_S = 5.5
STOP_DEADLINE_S = 9.0
WATCHDOG_GRACE_S = 1.0
_SESSION_INIT_UNAVAILABLE = "SLM session_init unavailable"


def _codex_payload() -> dict:
    """Read Codex's documented JSON hook payload without failing closed."""
    try:
        value = json.load(sys.stdin)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _apply_codex_session(payload: dict) -> str:
    """Map Codex lifecycle fields onto the shared SLM session primitives."""
    project_dir = payload.get("cwd")
    if not isinstance(project_dir, str) or not project_dir:
        project_dir = os.getcwd()
    os.environ["CLAUDE_PROJECT_DIR"] = project_dir
    session_id = payload.get("session_id")
    if isinstance(session_id, str) and session_id:
        # Shared handlers use this neutral lifecycle identity despite its
        # historical environment-variable name.  It is never sent to a host.
        os.environ["CLAUDE_SESSION_ID"] = session_id
        # Presence is fail-open: it shows Codex is genuinely active.
        try:
            from superlocalmemory.hooks.session_registry import (
                mark_active,
                resolve_active_profile,
            )
            mark_active(
                session_id,
                agent_type="codex",
                profile_id=resolve_active_profile(),
            )
        except Exception:
            pass
    return project_dir


def _child_env(deadline: HookDeadline) -> dict[str, str]:
    """The ``slm mcp`` child answers "starting" before this hook must stop."""
    env = dict(os.environ)
    env["SLM_DAEMON_START_WAIT_S"] = f"{max(0.0, deadline.remaining(reserve=3.0)):.1f}"
    return env


def _codex_mcp_session_init(
    project_dir: str,
    payload: dict,
    deadline: HookDeadline | None = None,
    registry: ChildRegistry | None = None,
) -> dict:
    """Call ``session_init`` through a packaged ``slm mcp`` child, by the deadline.

    Fail-open: any failure returns ``{}`` and the hook still emits valid output.
    """
    deadline = deadline or HookDeadline(START_DEADLINE_S)
    query = payload.get("prompt") or payload.get("user_prompt") or Path(project_dir).name
    if not isinstance(query, str):
        query = Path(project_dir).name
    client = None
    try:
        client = McpStdio(["slm", "mcp"], env=_child_env(deadline), registry=registry)
        client.send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {
                           "name": "superlocalmemory-codex-hook",
                           "version": __version__,
                       }},
        })
        if client.reply(1, deadline) is None:
            return {}
        client.send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        client.send({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "session_init", "arguments": {
                "project_path": project_dir, "query": query[:500],
                "max_results": 10, "max_age_days": 30,
            }},
        })
        response = client.reply(2, deadline)
        if not isinstance(response, dict):
            return {}
        result = response.get("result")
        if not isinstance(result, dict):
            return {}
        content = result.get("content", [])
        if content and isinstance(content[0], dict):
            parsed = json.loads(content[0].get("text") or "{}")
            return parsed if isinstance(parsed, dict) else {}
        return result
    except Exception:
        return {}
    finally:
        if client is not None:
            client.close()


def _session_context(project_name: str, deadline: HookDeadline,
                     registry: ChildRegistry | None = None) -> str:
    os.environ["SLM_CLIENT"] = "codex"  # names the host on the injection receipt (W0-14)
    result = run_bounded(
        ["slm", "session-context", project_name or "general"],
        deadline=deadline, registry=registry,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _session_message(session: dict) -> str:
    if not session.get("session_id"):
        reason = _unavailable_reason(session)
        message = _SESSION_INIT_UNAVAILABLE + (f" ({reason})" if reason else "")
        if reason == "recovery_needed":
            message += (": a restore stopped part-way. Run `slm backup restore-safety-copy`, "
                        "then `slm serve start`.")
        elif reason == "restore_running":
            message += ": a restore is running; wait for it to finish."
        return message
    message = (
        f"SLM session_init OK: {session['session_id']} "
        f"({session.get('memory_count', 0)} memories, "
        f"{session.get('retrieval_mode', 'unknown')})"
    )
    # session_init carries the answer-check verdict; never drop it.
    if session.get("abstention_reason") == "judged_insufficient":
        message += " — answer check: none of these memories answers the query."
    return message


def _unavailable_reason(session: dict) -> str:
    """A short, path-free reason token, e.g. ``daemon_starting``."""
    error = str(session.get("error") or "")
    if error.startswith("DAEMON_UNAVAILABLE (") and ")" in error:
        token = error[len("DAEMON_UNAVAILABLE ("):error.index(")")]
        if token.replace("_", "").isalnum():
            return token
    return ""


def _start_output(session_msg: str, context: str) -> str:
    return json.dumps({
        "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": (
            f"{session_msg}\n\n" + (context or "(SLM context unavailable.)")
        )},
        "systemMessage": session_msg,
    })


def hook_codex_start() -> None:
    """Codex SessionStart: session_init and fast context, in parallel, bounded."""
    deadline = HookDeadline(START_DEADLINE_S)
    registry = ChildRegistry()
    state = {"session": {}, "context": ""}
    timed_out_msg = _SESSION_INIT_UNAVAILABLE + " (timed out)"
    watchdog = Watchdog(
        START_DEADLINE_S + WATCHDOG_GRACE_S,
        lambda: _start_output(timed_out_msg, state["context"]),
        registry=registry,
    ).start()
    try:
        payload = _codex_payload()
        project_dir = _apply_codex_session(payload)

        def _init() -> None:
            state["session"] = _codex_mcp_session_init(project_dir, payload, deadline, registry)

        def _context() -> None:
            state["context"] = _session_context(Path(project_dir).name, deadline, registry)

        workers = [threading.Thread(target=fn, daemon=True) for fn in (_init, _context)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=deadline.remaining() + 0.5)
        registry.stop_all()
    except Exception:
        pass
    finally:
        session = state["session"] if isinstance(state["session"], dict) else {}
        watchdog.emit(_start_output(_session_message(session), state["context"]))
        watchdog.cancel()


def _context_from(stdout: str) -> str:
    """Context text from a child hook's stdout: JSON envelope, ``{}`` or text."""
    text = (stdout or "").strip()
    if not text or text == "{}":
        return ""
    try:
        value = json.loads(text)
    except ValueError:
        return text
    if isinstance(value, dict):
        specific = value.get("hookSpecificOutput")
        if isinstance(specific, dict) and isinstance(specific.get("additionalContext"), str):
            return specific["additionalContext"].strip()
        return ""
    return text


def _prompt_output(parts: list[str]) -> str:
    context = "\n\n".join(part for part in parts if part)
    if not context:
        return "{}"
    return json.dumps({"hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit", "additionalContext": context,
    }})


def hook_codex_prompt() -> None:
    """Codex prompt event: per-prompt recall (behind the need gate, W3-7d) and the
    topic-shift and re-query signals, in parallel, bounded."""
    deadline = HookDeadline(PROMPT_DEADLINE_S)
    registry = ChildRegistry()
    watchdog = Watchdog(PROMPT_DEADLINE_S + 0.5, lambda: "{}", registry=registry).start()
    parts: dict[str, str] = {}
    try:
        payload = _codex_payload()
        _apply_codex_session(payload)
        text = json.dumps(payload)

        def _run(action: str) -> None:
            result = run_bounded(["slm", "hook", action], deadline=deadline,
                                 input_text=text, registry=registry)
            parts[action] = _context_from(result.stdout) if result.returncode == 0 else ""

        workers = [threading.Thread(target=_run, args=(a,), daemon=True)
                   for a in ("auto_recall", "user_prompt_rehash", "topic_shift")]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=deadline.remaining() + 0.3)
        registry.stop_all()
    except Exception:
        pass
    finally:
        watchdog.emit(_prompt_output([parts.get("auto_recall", ""),
                                      parts.get("user_prompt_rehash", ""),
                                      parts.get("topic_shift", "")]))
        watchdog.cancel()


def hook_codex_stop(stop_body) -> None:
    """Codex Stop: the shared session checkpoint, then the turn's overlap and
    finalize (``stop_outcome_hook.settle``), bounded, then ``{}``."""
    deadline = HookDeadline(STOP_DEADLINE_S)
    watchdog = Watchdog(STOP_DEADLINE_S + WATCHDOG_GRACE_S, lambda: "{}").start()
    payload: dict = {}
    try:
        payload = _codex_payload()
        _apply_codex_session(payload)
        stop_body(deadline)
    except SystemExit:
        pass
    except Exception:
        pass
    try:
        # W3 open item: the overlap and the turn's finalize, as the Claude Code
        # Stop does (Codex's Stop input carries ``last_assistant_message`` too).
        from superlocalmemory.hooks.stop_outcome_hook import settle

        if deadline.remaining() > 0:
            settle(payload)
    except Exception:
        pass
    finally:
        watchdog.emit("{}")
        watchdog.cancel()
