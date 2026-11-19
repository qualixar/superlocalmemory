# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.43 — Pre-web recall on WebSearch/WebFetch

"""Pre-web recall hook — fires SLM recall before any WebSearch/WebFetch call.

Dispatch: `slm hook before_web` (PreToolUse, matcher "WebSearch|WebFetch").

WHY THIS HOOK EXISTS
====================
End users typically have hundreds-to-thousands of relevant memories in their
local SLM. When Claude is about to issue a WebSearch or WebFetch, there's a
high chance the answer (or strong constraints on the answer) is already in
SLM. This hook forces a recall pass on the search query/URL and injects the
top hits as a system-reminder BEFORE the web call fires. Claude must consider
the local memories before committing to the external call.

PERFORMANCE
===========
Cost: ~500-800ms warm (full recall via SLM daemon). Fires only on
WebSearch and WebFetch (5-20× per typical session), so per-session overhead
is ~5-15s in exchange for grounded answers. NOT suitable for UserPromptSubmit
(too frequent — would be a perf disaster).

CONTRACT
========
- Reads Claude Code stdin: {"tool_input": {"query"|"url"|"prompt": "..."}}
- On non-trivial query: asks the daemon's hook queue (only memories the answer
  check supports, never one this conversation already got) and injects them as a
  system-reminder block. A queue that found nothing relevant is the answer.
- Only when the queue cannot answer (daemon down, no answer in time, an error)
  does it run `slm recall <query> --limit 5` instead, within what is left of
  _HOOK_BUDGET_SEC, so the whole hook ends before the host's 5 s timeout.
- On empty/short query / recall failure / SLM down: silent exit 0.
- Always exit 0 — never blocks the web call.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any

_MIN_QUERY_LEN = 5
_QUERY_TRUNCATE = 200
_RECALL_LIMIT = 5
_RECALL_TIMEOUT_SEC = 3
#: The whole hook, both paths together, stays under the plugin's 5 s PreToolUse timeout
#: (Claude Code kills a hook at its timeout; on 20 Nov 2026 this hook was 42 of 50
#: timeouts in a day on the author's Mac, when the queue wait and a fresh `slm recall`
#: added up to 6 s). A fallback recall needs at least _MIN_FALLBACK_SEC to be worth it.
_HOOK_BUDGET_SEC = 4.0
_MIN_FALLBACK_SEC = 1.0
_RECALLED_MAX_CHARS = 3000
_RECALLED_MIN_USEFUL = 50
_PREVIEW_CHARS = 80

_SHIM_PREFIX = "[SLM PRE-WEB RECALL"


def _extract_query(payload: dict[str, Any]) -> str:
    """Pull the search query / URL / prompt from Claude Code stdin payload."""
    ti = payload.get("tool_input") or {}
    if not isinstance(ti, dict):
        return ""
    raw = ti.get("query") or ti.get("prompt") or ti.get("url") or ""
    if not isinstance(raw, str):
        return ""
    return raw[:_QUERY_TRUNCATE].strip()


def _read_input() -> dict[str, Any]:
    """Parse stdin JSON. Returns empty dict on any failure."""
    try:
        raw = sys.stdin.read()
        if not raw:
            return {}
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
        return {}
    except (json.JSONDecodeError, ValueError, OSError):
        return {}


def _run_recall(query: str, timeout: float = _RECALL_TIMEOUT_SEC) -> str:
    """Run `slm recall <query> --limit N`. Returns trimmed output or empty."""
    try:
        # Bounded query length (already truncated to 200 chars). Subprocess
        # timeout caps daemon-down risk at 3s.
        proc = subprocess.run(
            ["slm", "recall", query, "--limit", str(_RECALL_LIMIT)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            return ""
        out = (proc.stdout or "")[:_RECALLED_MAX_CHARS]
        if len(out) < _RECALLED_MIN_USEFUL:
            return ""
        return out
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return ""


def _queue_context(query: str, session_id: str) -> str | None:
    """W3 open item: the recall through the daemon's hook queue, so the answer
    gets a ``hook_recall`` receipt with this conversation's id (usage joins it)
    and a memory the conversation already got is not shown again. Only when the
    daemon's hook socket is up (else the queue would wait for nobody).

    Returns the rendered, untrusted-framed evidence; ``""`` when the queue answered
    and nothing was relevant enough to show (the answer check's verdict, as in the
    per-prompt hook: the caller must not replace it with weaker matches); ``None``
    when the queue could not answer (no socket, no answer in time, an error), the
    one case where the command path may try instead."""
    try:
        from superlocalmemory.hooks.hook_daemon import owned_socket_path

        if owned_socket_path() is None:
            return None
        from superlocalmemory.core.recall_queue import RecallQueue
        from superlocalmemory.hooks import auto_recall_hook as ar

        queue = RecallQueue(ar._get_queue_db_path())
        try:
            request_id = queue.enqueue(
                query=query, limit_n=_RECALL_LIMIT, mode=ar._detect_mode(),
                agent_id="before_web_hook", session_id=session_id, priority="high",
                stall_timeout_s=_RECALL_TIMEOUT_SEC)
            result = queue.poll_result(request_id, timeout_s=_RECALL_TIMEOUT_SEC)
        finally:
            queue.close()
        if not isinstance(result, dict) or result.get("ok") is False:
            return None
        envelope = ar._format_envelope(result.get("results") or [], response=result)
        context = (envelope.get("hookSpecificOutput") or {}).get("additionalContext", "")
        return context if isinstance(context, str) else ""
    except Exception:  # noqa: BLE001 - fail open to the command path
        return None


def main() -> int:
    """Entry point. Always returns 0 — fail-open contract."""
    try:
        payload = _read_input()
        query = _extract_query(payload)
        if len(query) < _MIN_QUERY_LEN:
            return 0

        preview = query[:_PREVIEW_CHARS].replace('"', "'")
        session_id = payload.get("session_id")
        started = time.monotonic()
        memory_evidence = _queue_context(query, session_id if isinstance(session_id, str)
                                         else "")
        if memory_evidence is None:
            left = _HOOK_BUDGET_SEC - (time.monotonic() - started)
            if left < _MIN_FALLBACK_SEC:
                return 0
            recalled = _run_recall(query, timeout=min(left, _RECALL_TIMEOUT_SEC))
            if not recalled:
                return 0
            from superlocalmemory.core.injection import render_untrusted_text
            memory_evidence = render_untrusted_text(
                recalled,
                source_type="before-web-recall",
                source_id=preview,
            )
        if not memory_evidence:
            return 0
        sys.stdout.write(
            "<system-reminder>\n"
            f'{_SHIM_PREFIX} — fired before WebSearch/WebFetch on query: "{preview}"]\n'
            "You're about to search the web. SLM already has these relevant memories.\n"
            "READ THEM FIRST. If they answer the question, skip the web call. If they\n"
            "contradict what you'd find on the web, surface the contradiction. Do not\n"
            "ignore them.\n\n"
            f"{memory_evidence}\n"
            "</system-reminder>\n"
        )
    except Exception:  # noqa: BLE001 — fail-open contract
        pass
    return 0
