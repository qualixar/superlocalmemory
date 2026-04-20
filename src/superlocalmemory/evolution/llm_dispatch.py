# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-11 §Dispatch

"""Central LLM dispatch for the skill-evolution subsystem.

Enforces MASTER-PLAN D2 (no top-tier "O-family" Claude models, no
``gpt-4-turbo``) and LLD-00 §5 (every LLM-bound prompt passes through
``redact_secrets(aggression='high')`` FIRST).

Every evolution LLM call funnels through :func:`_dispatch_llm`. Writes an
audit row to ``evolution_llm_cost_log`` after the dispatch succeeds — the
row stores only the *redacted* prompt length and the model, never the
raw prompt, so no canary can leak via the cost log.

SB-2/SB-3/SB-4 fix cluster (v3.4.22 Stage 8):
  * All backend entry points (claude CLI, ollama, Anthropic/OpenAI API)
    live HERE, not in ``skill_evolver``. ``SkillEvolver._llm_call``
    delegates to ``_dispatch_llm`` so the validate → redact → log
    invariants can never be bypassed.
  * The claude CLI backend routes through
    ``core.security_primitives.run_subprocess_safe`` — no bare
    ``subprocess.run`` in evolution code (SB-4).

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import atexit
import logging
import os
import sqlite3
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from superlocalmemory.core.security_primitives import (
    redact_secrets,
    run_subprocess_safe,
)

logger = logging.getLogger(__name__)


# M-P-09: one cached writer connection per learning_db path — LLM cost
# logging used to pay a fresh ``sqlite3.connect`` + fsync per call. At
# current volume (<10 calls/cycle) the cost is small, but caching keeps
# the dispatch code consistent with the rest of the "one-cached-writer"
# pattern the codebase standardised on (reward.py, trigram_index.py).
#
# S9-W2 C4 (fork safety): a cached SQLite handle inherited across
# ``os.fork()`` corrupts the DB because both processes think they hold an
# exclusive lock. We clear the cache in any forked child via
# ``os.register_at_fork`` AND keyed-by-pid within ``_get_cost_conn`` so
# a child that somehow missed the registrar still behaves correctly.
# On Windows / platforms without ``register_at_fork`` the fork path
# cannot happen so the pid-check is free insurance.
#
# S9-W2 C9 (serialization): ``_COST_CONN_LOCK`` is now ONLY held during
# the get/create-cache flip, NOT during the ``execute+commit`` inside
# ``_log_cost``. SQLite's own writer serialisation (BEGIN IMMEDIATE +
# busy_timeout) is the correct tool for write ordering; the Python lock
# was converting 10 parallel candidates × 3-8ms fsync into a single
# 30-80ms tail. Cache structure remains intact; the lock's scope shrinks.
_COST_CONN_CACHE: dict[str, sqlite3.Connection] = {}
_COST_CONN_LOCK = threading.Lock()
_COST_CONN_OWNER_PID: int | None = None


def _resolve_cost_key(learning_db: Path) -> str:
    """Resolve a DB path to a stable cache key.

    M-P-03 fix: ``~/.slm/learning.db`` and ``/home/u/.slm/learning.db``
    previously cached to separate conns on the same inode, producing two
    writers contending over WAL. ``os.path.realpath`` collapses them.
    """
    try:
        return os.path.realpath(str(learning_db))
    except OSError:  # pragma: no cover — defensive
        return str(learning_db)


def _reset_cost_cache_for_child() -> None:
    """Close any inherited handles in the fork child.

    C4: ``os.register_at_fork(after_in_child=...)`` fires before any user
    code runs in the child, so closing here is safe even if the parent
    was mid-write (the child never participated in that transaction).
    """
    global _COST_CONN_OWNER_PID
    # Do NOT close parent-owned handles — let the parent keep using them.
    # We only clear our cache reference so the child opens fresh ones.
    _COST_CONN_CACHE.clear()
    _COST_CONN_OWNER_PID = os.getpid()


def _get_cost_conn(learning_db: Path) -> sqlite3.Connection:
    """Return a cached writer connection for ``learning_db``. Never raises."""
    global _COST_CONN_OWNER_PID
    key = _resolve_cost_key(learning_db)
    with _COST_CONN_LOCK:
        # Belt-and-suspenders: if we somehow missed the fork registrar
        # (embedded interpreter, non-POSIX fork path), detect pid drift
        # and reset before handing out a potentially-corrupt handle.
        current_pid = os.getpid()
        if _COST_CONN_OWNER_PID is not None and (
            _COST_CONN_OWNER_PID != current_pid
        ):
            _COST_CONN_CACHE.clear()
        _COST_CONN_OWNER_PID = current_pid
        conn = _COST_CONN_CACHE.get(key)
        if conn is not None:
            return conn
        conn = sqlite3.connect(key, check_same_thread=False, timeout=2.0)
        _COST_CONN_CACHE[key] = conn
        return conn


def _close_cost_conns() -> None:
    """Close every cached cost-log connection (atexit)."""
    with _COST_CONN_LOCK:
        conns = list(_COST_CONN_CACHE.items())
        _COST_CONN_CACHE.clear()
    for _key, conn in conns:
        try:
            conn.close()
        except Exception:  # pragma: no cover
            pass


atexit.register(_close_cost_conns)
# C4: wipe inherited caches in any forked child. ``register_at_fork`` is
# POSIX-only; Windows simply doesn't fork so there is nothing to register.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_cost_cache_for_child)


# ---------------------------------------------------------------------------
# Allow-list and deny-list
# ---------------------------------------------------------------------------
#
# Allow-list explicitly names every model evolution may invoke. Deny-list
# catches substrings that must NEVER appear in an evolution-issued model
# id — notably the O-tier Claude family (MASTER-PLAN D2) and OpenAI's
# ``gpt-4-turbo`` (cost + behaviour regressions observed in prod).
#
# NOTE on the deny-list strings: the Stage-5b CI gate scans ``src/`` for
# the full banned model-family literal. That literal must NEVER appear in
# this file or any other source file. We check for the shorter substring
# ``opus`` instead; that catches every Claude O-family id variant without
# putting the banned literal anywhere in source.

ALLOWED_LLM_MODELS: frozenset[str] = frozenset({
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "ollama:llama3",
    "ollama:qwen2.5",
})

FORBIDDEN_MODEL_SUBSTRINGS: tuple[str, ...] = ("opus", "gpt-4-turbo")

MAX_TOKENS_CAP: int = 500


# ---------------------------------------------------------------------------
# Backends (SB-2, SB-4) — moved out of skill_evolver.py
# ---------------------------------------------------------------------------
#
# Every backend has a uniform signature::
#
#     backend(prompt: str, *, model: str, max_tokens: int) -> str
#
# They receive the ALREADY-REDACTED prompt from ``_dispatch_llm``. They
# must never log the prompt. They return an empty string on any
# transport failure (fail-closed: caller treats "" as "no evolution").


def _call_claude_cli_backend(
    prompt: str, *, model: str, max_tokens: int,
) -> str:
    """Spawn ``claude --model <model>`` via ``run_subprocess_safe``.

    SB-4: bare ``subprocess.run`` is banned in evolution code — every
    shell-out goes through ``run_subprocess_safe`` which strips the
    inherited env down to a vetted allow-list.
    """
    # Translate the allow-listed model id to the CLI short name.
    cli_model = "haiku"
    if "sonnet" in model:
        cli_model = "sonnet"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False,
    ) as f:
        f.write(prompt)
        prompt_file = f.name

    try:
        result = run_subprocess_safe(
            ["claude", "--model", cli_model, "--print", "--no-input",
             "--max-tokens", str(max_tokens),
             "--prompt-file", prompt_file],
            timeout=120.0,
            env={
                "CLAUDE_CODE_ENTRYPOINT": "cli",
                "ECC_SKIP_OBSERVE": "1",
            },
        )
        stdout = getattr(result, "stdout", "") or ""
        rc = getattr(result, "returncode", 1)
        return stdout.strip() if rc == 0 else ""
    except Exception as exc:  # noqa: BLE001 — fail-closed, never crash caller
        logger.debug("claude CLI backend failed: %s", exc)
        return ""
    finally:
        try:
            os.unlink(prompt_file)
        except OSError:
            pass


def _call_ollama_backend(
    prompt: str, *, model: str, max_tokens: int,
) -> str:
    """Call local Ollama HTTP API for LLM completion.

    ``model`` is expected to be an allow-listed id of the form
    ``"ollama:<model-name>"`` — the prefix is stripped before dispatch.
    """
    import json as _json
    import urllib.request

    ollama_model = model.split(":", 1)[1] if model.startswith("ollama:") else model
    payload = _json.dumps({
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }).encode()

    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            data = _json.loads(resp.read())
            return data.get("response", "") or ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("Ollama backend failed: %s", exc)
        return ""


def _call_claude_api_backend(
    prompt: str, *, model: str, max_tokens: int,
) -> str:
    """Call the Anthropic Messages API directly.

    The API model id is the allow-listed name itself — no client-side
    mapping table, so adding a new allow-listed model is a one-line
    edit to :data:`ALLOWED_LLM_MODELS`.
    """
    try:
        import anthropic  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        logger.debug("anthropic sdk unavailable: %s", exc)
        return ""

    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        content = getattr(msg, "content", None)
        if content and len(content) > 0:
            first = content[0]
            text = getattr(first, "text", None)
            if isinstance(text, str):
                return text
        return ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("Anthropic API backend failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Backend registry — dispatches by (allow-listed) model id
# ---------------------------------------------------------------------------


def _fail_closed_backend(
    prompt: str, *, model: str, max_tokens: int,
) -> str:
    """S9-SKEP-14: explicit fail-closed backend for unroutable models.

    Returns "" (the fail-closed sentinel every dispatch treats as
    "no evolution happened") and logs a warning. Previously the
    fallthrough silently routed any unknown id to the paid Anthropic
    API — a misconfigured entry in ``ALLOWED_LLM_MODELS`` would burn
    user money without anyone noticing.
    """
    logger.warning(
        "llm_dispatch: no backend registered for model=%r — "
        "fail-closed (returning empty string)", model,
    )
    return ""


def _pick_backend(model: str) -> Callable[..., str]:
    """Resolve an allow-listed model id to its backend callable.

    Contract: ``model`` is already validated against ``ALLOWED_LLM_MODELS``
    by the caller (``_dispatch_llm`` runs ``_validate_model`` first).

    S9-SKEP-14: routing is prefix-exact — we no longer default unknown
    models to the Claude API path. An allow-listed model without a
    backend entry hits ``_fail_closed_backend`` and returns ""
    instead of silently spending money on the wrong vendor.
    """
    if model.startswith("ollama:"):
        return _call_ollama_backend
    if model.startswith("claude-"):
        # Claude CLI path is an alternative — selected when an explicit
        # env flag is set. Default path is the Anthropic API backend.
        if os.environ.get("SLM_EVOLUTION_BACKEND") == "claude-cli":
            return _call_claude_cli_backend
        return _call_claude_api_backend
    return _fail_closed_backend


def _actual_llm_call(prompt: str, *, model: str, max_tokens: int) -> str:
    """Dispatch the redacted prompt to the backend registered for ``model``.

    Kept as a stable module-level function so tests can ``monkeypatch``
    it with a deterministic stub (see ``record_backend`` fixture in
    ``test_llm_dispatch.py``). Production callers never invoke this
    directly — they go through :func:`_dispatch_llm`.
    """
    backend = _pick_backend(model)
    return backend(prompt, model=model, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def _validate_model(model: str) -> None:
    """Raise ``ValueError`` if the model is forbidden or not allow-listed."""
    if not isinstance(model, str) or not model:
        raise ValueError(f"model must be a non-empty str, got {model!r}")
    lowered = model.lower()
    for forbidden in FORBIDDEN_MODEL_SUBSTRINGS:
        if forbidden in lowered:
            raise ValueError(
                f"forbidden model: {model!r} (contains {forbidden!r})"
            )
    if model not in ALLOWED_LLM_MODELS:
        raise ValueError(
            f"model not in ALLOWED_LLM_MODELS: {model!r} "
            f"(allowed: {sorted(ALLOWED_LLM_MODELS)})"
        )


def _log_cost(
    *,
    learning_db: Path,
    profile_id: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float = 0.0,
    cycle_id: str | None = None,
) -> None:
    """Append a redacted cost-log row. Never stores prompt/response text.

    H-16 (Stage 8): ``profile_id`` must be a non-empty string. The schema
    has ``NOT NULL`` on the column but SQLite accepts empty strings — that
    would break the dashboard's per-profile cost widget by silently
    aggregating unattributed spend. We raise here instead so the caller
    fixes the upstream bug rather than learning about it weeks later from
    a mis-reported invoice.
    """
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError(
            "evolution_llm_cost_log.profile_id must be a non-empty string "
            f"(got {profile_id!r})"
        )
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        # S9-W2 C9: the cached conn is ``check_same_thread=False`` and
        # SQLite's own writer serialisation (BEGIN IMMEDIATE + 2 s
        # ``busy_timeout`` in the connect() call) is the right tool for
        # write ordering. Previously we held _COST_CONN_LOCK across the
        # execute+commit fsync, converting 10 parallel candidates' worth
        # of 3-8 ms commits into a single 30-80 ms tail. Release the
        # Python lock BEFORE the SQL round-trip.
        conn = _get_cost_conn(Path(learning_db))
        conn.execute(
            "INSERT INTO evolution_llm_cost_log "
            "(profile_id, ts, model, tokens_in, tokens_out, cost_usd, cycle_id) "
            "VALUES (?,?,?,?,?,?,?)",
            (profile_id, now, model, tokens_in, tokens_out, cost_usd, cycle_id),
        )
        conn.commit()
    except sqlite3.Error as e:
        logger.warning("cost log write failed: %s", e)


def _dispatch_llm(
    prompt: str,
    *,
    model: str,
    learning_db: Path | str,
    profile_id: str,
    max_tokens: int = MAX_TOKENS_CAP,
    cycle_id: str | None = None,
) -> str:
    """Central choke-point for every evolution LLM call.

    Validates model against allow/deny lists, caps ``max_tokens``, runs the
    prompt through ``redact_secrets(aggression='high')``, dispatches, and
    logs a redacted cost row. Raises ``ValueError`` on any contract breach.
    """
    _validate_model(model)

    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(
            f"max_tokens must be a positive int, got {max_tokens!r}"
        )
    if max_tokens > MAX_TOKENS_CAP:
        raise ValueError(
            f"max_tokens {max_tokens} > {MAX_TOKENS_CAP} cap (LLD-11)"
        )
    # S9-W2 H-SKEP-05: validate profile_id BEFORE paying for the LLM
    # call. Previously the check lived in _log_cost, AFTER the paid
    # Anthropic round-trip — a misconfigured profile with empty id would
    # spend the money, raise ValueError in _log_cost, and return "" from
    # _llm_call's except. Net: cost incurred, no cost-log row, caller
    # may retry and burn more. Validate up-front, fail-closed, zero cost.
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError(
            "profile_id must be a non-empty string "
            f"(got {profile_id!r})"
        )

    # LLD-00 §5 — redact BEFORE dispatch. Never log the raw prompt.
    safe_prompt = redact_secrets(prompt, aggression="high")

    # S9-defer H-P-10: per-cycle retry-cost DoS guard. If the caller
    # (or an orchestrator layer) keeps retrying a failing dispatch on
    # the same ``cycle_id``, cost escalates without bound — a crafted
    # adversarial scenario could make evolution burn through the
    # daily USD cap in minutes. Count prior calls for this cycle_id
    # in ``evolution_llm_cost_log`` and refuse once the retry cap is
    # hit. The EvolutionBudget object already caps overall LLM calls
    # per cycle to 10; this is the per-cycle-ID guard for retries on
    # the SAME logical step (distinct from 10 different LLM calls for
    # 10 different steps).
    _RETRY_CAP_PER_CYCLE = int(
        os.environ.get("SLM_EVOLUTION_RETRY_CAP", "5")
    )
    if cycle_id:
        try:
            _conn = _get_cost_conn(Path(learning_db))
            row = _conn.execute(
                "SELECT COUNT(*) FROM evolution_llm_cost_log "
                "WHERE profile_id = ? AND cycle_id = ?",
                (profile_id, cycle_id),
            ).fetchone()
            prior = int(row[0]) if row and row[0] is not None else 0
            if prior >= _RETRY_CAP_PER_CYCLE:
                logger.warning(
                    "evolution retry cap hit: profile=%s cycle_id=%s "
                    "prior=%d cap=%d — refusing dispatch",
                    profile_id, cycle_id, prior, _RETRY_CAP_PER_CYCLE,
                )
                raise RuntimeError(
                    f"evolution retry cap exceeded for cycle {cycle_id}"
                )
        except sqlite3.Error:
            # Cost log unavailable — fail-open on this guard (the
            # outer EvolutionBudget still enforces the 10-call cap).
            pass

    response = _actual_llm_call(
        safe_prompt, model=model, max_tokens=max_tokens,
    )

    # Cost-log row: lengths only, no text content. This guarantees the
    # redaction canary (e.g. a ``ghp_...`` GitHub PAT) cannot end up in
    # the audit log — we never persist the redacted prompt either.
    _log_cost(
        learning_db=Path(learning_db),
        profile_id=profile_id,
        model=model,
        tokens_in=len(safe_prompt),
        tokens_out=len(response) if isinstance(response, str) else 0,
        cycle_id=cycle_id,
    )
    return response
