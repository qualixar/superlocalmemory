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
import time

_DEFAULT_LIMIT = 15           # raised from 3 for formatter candidate pool

#: Routing only (W3-7d): how long the per-prompt hook waits for the daemon's
#: socket before it injects nothing. The hook's own limit in ``hooks.json`` is
#: 7 s; the daemon stops waiting at 4 s (``hook_daemon.PROMPT_BUDGET_S``).
CLIENT_BUDGET_S = 5.0
#: One total for the whole hook, whichever path answers (A-15): the socket, then (only if the
#: socket is absent) the queue, then HTTP, each cut to what is left. Under the 7 s host limit.
TOTAL_BUDGET_S = CLIENT_BUDGET_S
#: Below this a wait is not worth starting.
MIN_WAIT_S = 0.2

#: One prompt event can reach this hook twice: a Codex user with the plugin AND the installed host
#: hooks has per-prompt recall registered by both (the plugin's ``hooks.json`` and the installed
#: ``codex-prompt``, which runs this action as a child). The first to start owns the turn; the
#: other answers ``{}`` (A-17). Routing-only: both fire together, well inside the hook's 7 s limit.
TURN_WINDOW_S = 20.0

#: Swapped by tests.
_clock = time.monotonic
_wall = time.time


class _Budget:
    """What is left of the hook's one budget."""

    def __init__(self, total: float = TOTAL_BUDGET_S) -> None:
        self._end = _clock() + total

    def remaining(self) -> float:
        return max(0.0, self._end - _clock())

    def spent(self) -> bool:
        return self.remaining() < MIN_WAIT_S


class _DaemonBusy(Exception):
    """The hook socket exists and did not answer in time: end the hook with ``{}``."""

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


def _turn_folder() -> str:
    from superlocalmemory.infra.data_root import canonical_data_root

    return str(canonical_data_root() / "session_state" / "auto-recall-turns")


def _first_for_this_turn(session_id: str, prompt: str) -> bool:
    """True when no other ``auto_recall`` hook has taken this turn in the last
    :data:`TURN_WINDOW_S` seconds (so this one should recall); False when one has.

    The turn is the session and the prompt, hashed (no text is stored). The marker is created
    with ``O_EXCL``, so two hooks started together cannot both win. Fail-open: a folder that
    cannot be written, or any error, answers True: recall is never suppressed by a bookkeeping
    failure. Markers older than the window are removed here.
    """
    import hashlib
    import os

    try:
        folder = _turn_folder()
        os.makedirs(folder, mode=0o700, exist_ok=True)
        name = hashlib.sha256(f"{session_id}\0{prompt}".encode()).hexdigest()[:24]
        path = os.path.join(folder, name)
        now = _wall()
        for old in os.listdir(folder):
            try:
                if now - os.stat(os.path.join(folder, old)).st_mtime > TURN_WINDOW_S:
                    os.unlink(os.path.join(folder, old))
            except OSError:
                continue
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return False
        os.close(fd)
        return True
    except Exception:
        return True


def _try_socket_first(prompt: str, session_id: str, budget: _Budget | None = None) -> dict | None:
    """Try the persistent hook daemon socket. Returns full envelope or None.

    The socket path returns an already-formatted Claude Code envelope.
    If this returns a non-None dict, the caller writes it to stdout directly
    (skip _do_recall + _format_envelope). Returns None when there is no socket, or it
    failed at once (the caller may fall back). Raises ``_DaemonBusy`` when the socket exists and
    did not answer in time: the daemon is busy, so nothing else is tried (A-15).
    """
    from superlocalmemory.hooks import hook_daemon

    wait = CLIENT_BUDGET_S if budget is None else min(CLIENT_BUDGET_S, budget.remaining())
    response, outcome = hook_daemon.socket_recall(prompt=prompt, session_id=session_id,
                                                  timeout=wait)
    if outcome == hook_daemon.SOCKET_TIMEOUT:
        raise _DaemonBusy()
    if response is None or not isinstance(response, dict):
        return None
    if not response:
        return {}
    return response


class _Results(list):
    """The result list, carrying the whole response it came from.

    A plain ``list`` for every caller and test that treats it as one; the
    envelope reads ``.response`` for the answer-check verdict, if any.
    """

    response: dict | None = None


def _do_recall(query: str, limit: int = _DEFAULT_LIMIT, session_id: str = "",
               budget: _Budget | None = None) -> list[dict] | None:
    """Enqueue recall to queue, poll for result. Returns list of dicts or None."""
    response = _do_recall_response(query, limit=limit, session_id=session_id, budget=budget)
    if response is None:
        return None
    results = response.get("results", [])
    if not isinstance(results, list):
        return None
    carried = _Results(results)
    carried.response = response
    return carried


def _do_recall_response(query: str, limit: int = _DEFAULT_LIMIT,
                        session_id: str = "", budget: _Budget | None = None) -> dict | None:
    """Like ``_do_recall`` but the whole response, so a verdict can travel."""
    try:
        from superlocalmemory.core.recall_queue import RecallQueue

        mode = _detect_mode()
        timeout = min(_get_mode_timeout(mode), CLIENT_BUDGET_S)
        if budget is not None:
            if budget.spent():
                return None
            timeout = min(timeout, budget.remaining())
        stall_timeout = timeout
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
                deadline_s=timeout,  # A-14: what we stop waiting for is never "delivered"
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
        return _fallback_response(query, limit, session_id, budget)


def _fallback_recall(query: str, limit: int, session_id: str) -> list[dict] | None:
    data = _fallback_response(query, limit, session_id)
    return data["results"] if data is not None else None


def _fallback_response(query: str, limit: int, session_id: str,
                       budget: _Budget | None = None) -> dict | None:
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
            "via": "hook",  # W3-1: names the injection receipt's path
        })
        wait = CLIENT_BUDGET_S / 2
        if budget is not None:
            if budget.spent():
                return None
            wait = min(wait, budget.remaining())
        data = daemon_request("GET", f"/recall?{params}", timeout_seconds=wait)
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
                # WS-5: the recall result's own origin (``why.origin``), so an untrusted
                # memory is never crowned core by the renderer.
                taint=str(((r.get("why") or {}).get("origin") or {}).get("taint", "") or ""),
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


def _per_prompt_off() -> bool:
    """W3-7d (D-47): ``delivery.need_gate`` is off (the default) or the kill
    switch is on, so this hook answers ``{}`` before reaching the daemon.
    Fails closed: an unreadable flag means off."""
    try:
        from superlocalmemory.core.need_gate import flag_state
        from superlocalmemory.infra.data_root import canonical_data_root

        return flag_state(canonical_data_root()) is None
    except Exception:
        return True


def main() -> int:
    budget = _Budget()  # the one budget starts here, before anything is read or checked
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

    if _switched_off() or _per_prompt_off():
        sys.stdout.write("{}")
        return 0

    # W3-7d: the daemon's need gate decides and logs every prompt, the obvious
    # ones too (their miss rate is measured), so they go to the socket as well.
    if budget.spent() or not _first_for_this_turn(session_id, prompt):
        sys.stdout.write("{}")
        return 0
    try:
        socket_result = _try_socket_first(prompt, session_id, budget)
        if socket_result is not None:
            sys.stdout.write(json.dumps(socket_result) if socket_result else "{}")
            return 0
    except _DaemonBusy:
        # The socket is there and did not answer: the daemon is busy. A second recall or an
        # HTTP call would only add to it, and the budget is spent (A-15).
        sys.stdout.write("{}")
        return 0
    except Exception:
        pass

    # No socket: the gate's rules, without its log or its score.
    try:
        from superlocalmemory.core.need_gate import rule

        skip = rule(prompt) is not None
    except Exception:
        skip = _is_ack(prompt)
    if skip:
        sys.stdout.write("{}")
        return 0

    try:
        results = _do_recall(prompt, limit=_DEFAULT_LIMIT, session_id=session_id, budget=budget)
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
