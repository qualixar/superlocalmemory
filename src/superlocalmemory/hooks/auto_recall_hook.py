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


class _Results(list):
    """The result list, carrying the whole response it came from.

    A plain ``list`` for every caller and test that treats it as one; the
    envelope reads ``.response`` for the answer-check verdict, if any.
    """

    response: dict | None = None


def _do_recall(query: str, limit: int = _DEFAULT_LIMIT, session_id: str = "") -> list[dict] | None:
    """Enqueue recall to queue, poll for result. Returns list of dicts or None."""
    response = _do_recall_response(query, limit=limit, session_id=session_id)
    if response is None:
        return None
    results = response.get("results", [])
    if not isinstance(results, list):
        return None
    carried = _Results(results)
    carried.response = response
    return carried


def _do_recall_response(query: str, limit: int = _DEFAULT_LIMIT,
                        session_id: str = "") -> dict | None:
    """Like ``_do_recall`` but the whole response, so a verdict can travel."""
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
                if isinstance(result.get("results", []), list):
                    return result
            return None
        finally:
            queue.close()

    except Exception:
        return _fallback_response(query, limit, session_id)


def _fallback_recall(query: str, limit: int, session_id: str) -> list[dict] | None:
    data = _fallback_response(query, limit, session_id)
    return data["results"] if data is not None else None


def _fallback_response(query: str, limit: int, session_id: str) -> dict | None:
    """Fallback: ask the daemon's /recall directly if the queue path fails.

    Only through the identity-checked daemon client: the prompt the user just
    typed must never go to whatever happens to listen on the default port
    (on a shared Mac, another account), and nothing such a stranger answers
    may be injected into the agent's context as "memories".

    S-M2: this recall injects context for a prompt typed to the agent, not a
    question to memory, so the daemon is asked to skip the answer check
    (nothing judged, nothing sent off the machine).
    """
    try:
        import urllib.parse

        from superlocalmemory.cli.daemon import daemon_request
        from superlocalmemory.core.answer_check_scope import (
            ANSWER_CHECK_PARAM,
            ANSWER_CHECK_SKIP,
        )

        params = urllib.parse.urlencode({
            "q": query, "limit": limit, "session_id": session_id or "",
            ANSWER_CHECK_PARAM: ANSWER_CHECK_SKIP,
            # 4.1.25: a hook injects a glance (one-line previews); the agent
            # asks recall or expand for more.
            "depth": "0",
        })
        data = daemon_request("GET", f"/recall?{params}", timeout_seconds=5.0)
        if not isinstance(data, dict):
            return None
        results = data.get("results", [])
        # The whole response: its tier and warming-up flag decide the injection.
        return data if isinstance(results, list) else None
    except Exception:
        return None


def _format_envelope(results: list[dict], response: dict | None = None) -> dict:
    """Format recall results as Claude Code envelope. Uses shared formatter
    (v3.4.65). Fail closed if the mandatory renderer is unavailable.

    ``response`` is the whole recall response when the caller has it: if it
    carries an answer-check verdict, the same one line every surface uses is
    put above the memories — never in their place (H-3).
    """
    try:
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.injection import (
            InjectableMemory,
            render_context,
            response_injectable,
        )
        # Closest memories, an abstained recall, or a warming-up one inject
        # nothing, not even a verdict line: only supported results go in.
        if not response_injectable(response):
            return {}
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
                support_tier=str(r.get("tier", "") or ""),
            )
            for r in results
        ]
        wrapped = render_context(inj, mode=mode, cfg=cfg, wrap=True)
        if isinstance(response, dict):
            from superlocalmemory.core.answer_check_notice import answer_check_line

            verdict_line = answer_check_line(response)
            if verdict_line:
                wrapped = f"{verdict_line}\n\n{wrapped}" if wrapped else verdict_line
        if not wrapped:
            return {}
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


def _switched_off() -> bool:
    """``auto_recall.on_every_prompt`` (or auto-recall as a whole) is off."""
    try:
        from superlocalmemory.hooks import rules_engine

        return not rules_engine.load().per_prompt_recall_on()
    except Exception:  # fail open
        return False


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

    if _is_ack(prompt) or _switched_off():
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
        envelope = _format_envelope(results, response=getattr(results, "response", None))
        sys.stdout.write(json.dumps(envelope))
    except Exception:
        sys.stdout.write("{}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
