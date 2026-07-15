# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""UserPromptSubmit hook — production auto-recall via recall_queue.

Entry point for Claude Code's UserPromptSubmit event. Invoked as:
  python3 -m superlocalmemory.hooks.auto_recall_hook
  OR: slm hook auto_recall

Flow:
  1. Read stdin JSON (Claude Code payload)
  2. Detect ack prompts → silent (exit 0, empty JSON)
  3. Substantive → enqueue to recall_queue.db → poll for result
  4. Format top-K memories as additionalContext
  5. Write Claude Code envelope to stdout

HARD RULES:
  - stdlib-only imports at module load. SLM modules delayed-imported.
  - NEVER imports MemoryEngine (memory blast risk).
  - NEVER raises to Claude Code — always exits 0.
  - Fail-open: any failure → {} to stdout.

MEMORY SAFETY: recall goes through recall_queue.db → QueueConsumer
(daemon background thread) → pool.recall() → recall_worker subprocess.
Engine is ONLY in the recall_worker. This process stays at ~20MB.
"""

from __future__ import annotations

import json
import re
import sys

_DEFAULT_LIMIT = 15           # raised from 3 for formatter candidate pool

_MODE_TIMEOUTS = {
    "A": 10.0,
    "B": 25.0,
    "C": 40.0,
}

_ACK_RE = re.compile(
    r"^\s*"
    r"(?:yes|no|ok|okay|approved|thanks|thank you|go|sure|yep|nope|"
    r"done|y|n|cool|got it|right|correct)"
    r"(?:\s+(?:yes|no|ok|okay|approved|thanks|done|\d+))*"
    r"\s*[.!?]?\s*$",
    re.IGNORECASE,
)


def _is_ack(prompt: str) -> bool:
    return len(prompt) <= 30 and bool(_ACK_RE.match(prompt))


def _get_mode_timeout(mode: str) -> float:
    return _MODE_TIMEOUTS.get(mode.upper(), _MODE_TIMEOUTS["B"])


def _detect_mode() -> str:
    try:
        from superlocalmemory.core.config import SLMConfig
        config = SLMConfig.load()
        return getattr(config, "mode", "B").upper()
    except Exception:
        return "B"


def _get_queue_db_path():
    from superlocalmemory.infra.data_root import state_path

    return state_path("recall_queue.db")


def _try_socket_first(prompt: str, session_id: str) -> dict | None:
    """Try the persistent hook daemon socket. Returns full envelope or None.

    The socket path returns an already-formatted Claude Code envelope.
    If this returns a non-None dict, the caller writes it to stdout directly
    (skip _do_recall + _format_envelope). Returns None on any failure,
    triggering the subprocess fallback.
    """
    try:
        from superlocalmemory.hooks.hook_daemon import try_socket_recall
        response = try_socket_recall(
            prompt=prompt,
            session_id=session_id,
            timeout=_get_mode_timeout(_detect_mode()),
        )
        if response is None or not isinstance(response, dict):
            return None
        if not response:
            return {}
        return response
    except Exception:
        return None


def _do_recall(query: str, limit: int = _DEFAULT_LIMIT, session_id: str = "") -> list[dict] | None:
    """Enqueue recall to queue, poll for result. Returns list of dicts or None."""
    try:
        from superlocalmemory.core.recall_queue import RecallQueue

        mode = _detect_mode()
        timeout = _get_mode_timeout(mode)
        stall_timeout = max(timeout - 5.0, 5.0)
        db_path = _get_queue_db_path()
        queue = RecallQueue(db_path)

        try:
            request_id = queue.enqueue(
                query=query,
                limit_n=limit,
                mode=mode,
                agent_id="auto_recall_hook",
                session_id=session_id,
                priority="high",
                stall_timeout_s=stall_timeout,
            )

            result = queue.poll_result(request_id, timeout_s=timeout)

            if isinstance(result, dict):
                if result.get("ok") is False:
                    return None
                results = result.get("results", [])
                if isinstance(results, list):
                    return results
            return None
        finally:
            queue.close()

    except Exception:
        return _fallback_recall(query, limit, session_id)


def _fallback_recall(query: str, limit: int, session_id: str) -> list[dict] | None:
    """Fallback: call daemon HTTP /recall if queue path fails."""
    try:
        import urllib.parse
        import urllib.request

        params = urllib.parse.urlencode({"q": query, "limit": limit})
        url = f"http://127.0.0.1:47152/recall?{params}"

        req = urllib.request.Request(url, method="GET")
        req.add_header("X-SLM-Session-Id", session_id)

        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("results", [])
    except Exception:
        return None


def _format_envelope(results: list[dict]) -> dict:
    """Format recall results as Claude Code envelope. Uses shared formatter
    (v3.4.65). Fail closed if the mandatory renderer is unavailable."""
    try:
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.injection import InjectableMemory, render_context
        cfg = SLMConfig.load().injection
        mode = _detect_mode()
        inj = [
            InjectableMemory(
                content=str(r.get("content", "")),
                score=float(r.get("score", 0) or 0),
                fact_id=str(r.get("fact_id", "")),
                importance=float(r.get("importance", 0) or 0),
                access_count=int(r.get("access_count", 0) or 0),
                source_type=str(r.get("source_type", "recall")),
                source_id=str(r.get("source_id", "auto-recall-hook")),
            )
            for r in results
        ]
        wrapped = render_context(inj, mode=mode, cfg=cfg, wrap=True)
        return {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": wrapped,
            }
        }
    except Exception:
        # A weaker ad-hoc formatter would create a second security contract and
        # could leak unredacted stored text. No memory is safer than unbounded,
        # unredacted memory when the mandatory renderer fails.
        return {}


def main() -> int:
    try:
        raw = sys.stdin.read()
    except Exception:
        sys.stdout.write("{}")
        return 0

    if not raw or not raw.strip():
        sys.stdout.write("{}")
        return 0

    try:
        payload = json.loads(raw)
    except Exception:
        sys.stdout.write("{}")
        return 0

    if not isinstance(payload, dict):
        sys.stdout.write("{}")
        return 0

    prompt = payload.get("prompt", "")
    session_id = payload.get("session_id", "")

    if not isinstance(prompt, str) or not prompt.strip():
        sys.stdout.write("{}")
        return 0

    if _is_ack(prompt):
        sys.stdout.write("{}")
        return 0

    try:
        socket_result = _try_socket_first(prompt, session_id)
        if socket_result is not None:
            sys.stdout.write(json.dumps(socket_result) if socket_result else "{}")
            return 0
    except Exception:
        pass

    try:
        results = _do_recall(prompt, limit=_DEFAULT_LIMIT, session_id=session_id)
    except Exception:
        sys.stdout.write("{}")
        return 0

    if not results:
        sys.stdout.write("{}")
        return 0

    try:
        envelope = _format_envelope(results)
        sys.stdout.write(json.dumps(envelope))
    except Exception:
        sys.stdout.write("{}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
