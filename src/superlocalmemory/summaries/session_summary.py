# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Session Summary — issue #113 bounded summary.

CRITICAL DATA HONESTY NOTE
---------------------------
On a real 3,294-fact store, only 127 facts (3.9%) carry a session_id.
A Session Summary that presents itself as "everything you did this session"
while silently covering 4% of the facts is overclaiming — the same defect
removed from brain/truth.py.

Coverage is ALWAYS disclosed.  If the session has too few facts to summarise
meaningfully we say so; we never silently return partial data as complete.

DETERMINISTIC FALLBACK IS MANDATORY
-------------------------------------
Mode A users have no LLM at all.  Mode B/C users lose theirs whenever Ollama
or the network is down.  Every call to generate_session_summary() MUST return
a SummaryResult.  It may say the data is insufficient; it may never return
None or an empty result.  The extractive fallback path is always active.

HOT PATH EXCLUSION
-------------------
This module must NEVER be imported from core/recall_pipeline.py or
core/store_pipeline.py.  Summary generation performs multi-table reads and
optional LLM calls.  It belongs in background maintenance or on explicit
user request only.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, timezone
from pathlib import Path

from superlocalmemory.storage.database import visible_fact_clause_for_connection

from .base import (
    clean_llm_summary,
    format_highlight,
    SUMMARY_SYSTEM_PROMPT,
    COVERAGE_FULL,
    COVERAGE_INSUFFICIENT,
    COVERAGE_NO_SESSION,
    COVERAGE_PARTIAL,
    COVERAGE_UNAVAILABLE,
    GENERATED_BY_EXTRACTIVE,
    GENERATED_BY_LLM_B,
    GENERATED_BY_LLM_C,
    HiddenOf,
    SummaryResult,
    without_hidden,
    get_mode_str,
)

logger = logging.getLogger("superlocalmemory.summaries.session")

# Minimum number of session facts to attempt a meaningful summary.
# Below this threshold we return coverage=insufficient and a stub.
_MIN_FACTS = 3

# Maximum facts to include in the content body (extractive mode).
_BODY_FACTS = 7

# Maximum character length for a single fact in the body.
_MAX_FACT_CHARS = 250


def generate_session_summary(
    db_path: str | Path,
    session_id: str,
    profile_id: str = "default",
    config: object | None = None,
    *,
    hidden_of: HiddenOf | None = None,
) -> SummaryResult:
    """Generate a Session Summary for a specific session.

    COVERAGE DISCLOSURE: session data is sparse on real stores (~3.9% of
    facts carry a session_id).  The returned SummaryResult.coverage is always
    set to an accurate value — never "full" unless the session is truly
    complete.

    Args:
        db_path:    Path to memory.db.
        session_id: The session identifier (atomic_facts.session_id).
        profile_id: Profile scope — never mix profiles.
        config:     Optional SLMConfig for LLM mode (B/C).  None = Mode A
                    (extractive only, always deterministic).

    Returns:
        SummaryResult with source_fact_ids for every contributing fact.
        Never returns None.

    Note:
        This function is NOT on the hot path.  It belongs in background
        maintenance or on explicit user request, never inside recall or store.
    """
    db_path = Path(db_path)

    # ── query ────────────────────────────────────────────────────────────────
    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        try:
            # Withheld and deleted rows are not memories a person may be shown.
            visible = visible_fact_clause_for_connection(conn)
            rows = conn.execute(
                f"""
                SELECT fact_id, content, created_at, importance, lifecycle
                FROM   atomic_facts
                WHERE  profile_id  = ?
                  AND  session_id  = ?
                  AND  lifecycle  != 'archived'{visible}
                ORDER  BY importance DESC, created_at ASC, fact_id ASC
                """,  # noqa: S608 - the clause is built from constants only
                (profile_id, session_id),
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("session summary query failed for %s: %s", session_id, exc)
        return SummaryResult(
            kind="session",
            profile_id=profile_id,
            content=(
                f"Session summary for '{session_id}' is unavailable: "
                f"data access error."
            ),
            source_fact_ids=[],
            coverage=COVERAGE_UNAVAILABLE,
            generated_by=GENERATED_BY_EXTRACTIVE,
            metadata={"session_id": session_id, "error": str(exc)},
        )

    rows = without_hidden([dict(r) for r in rows], hidden_of)

    # ── coverage decision ─────────────────────────────────────────────────────
    if not rows:
        return SummaryResult(
            kind="session",
            profile_id=profile_id,
            content=(
                f"No facts found for session '{session_id}'.  "
                f"Note: only a small fraction of facts carry a session_id "
                f"on the current store — coverage is inherently partial."
            ),
            source_fact_ids=[],
            coverage=COVERAGE_NO_SESSION,
            generated_by=GENERATED_BY_EXTRACTIVE,
            metadata={"session_id": session_id, "fact_count": 0},
        )

    facts = [dict(r) for r in rows]
    source_fact_ids = [f["fact_id"] for f in facts]
    fact_count = len(facts)

    if fact_count < _MIN_FACTS:
        coverage = COVERAGE_INSUFFICIENT
    else:
        # Session summaries are structurally partial: they cover only the facts
        # that happened to carry a session_id.  On a real store that is ~3.9%
        # of the corpus.  Reporting "full" would be dishonest.
        coverage = COVERAGE_PARTIAL

    # ── extractive summary (deterministic, always available) ──────────────────
    extractive_content = _build_extractive_content(session_id, facts, fact_count)

    # ── LLM enrichment (optional) ─────────────────────────────────────────────
    mode = get_mode_str(config)
    if fact_count >= _MIN_FACTS and mode in ("b", "c"):
        llm_content, llm_mode = _try_llm(
            f"Summarise the key activities in session {session_id}",
            facts,
            config,
            mode,
        )
        if llm_content:
            return SummaryResult(
                kind="session",
                profile_id=profile_id,
                content=llm_content,
                source_fact_ids=source_fact_ids,
                coverage=coverage,
                generated_by=llm_mode,
                metadata={"session_id": session_id, "fact_count": fact_count},
            )

    return SummaryResult(
        kind="session",
        profile_id=profile_id,
        content=extractive_content,
        source_fact_ids=source_fact_ids,
        coverage=coverage,
        generated_by=GENERATED_BY_EXTRACTIVE,
        metadata={"session_id": session_id, "fact_count": fact_count},
    )


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_extractive_content(
    session_id: str,
    facts: list[dict],
    fact_count: int,
) -> str:
    """Build a deterministic extractive summary from session facts.

    Coverage is disclosed in the header.  Top facts by importance are listed.
    This is the fallback path — it must never fail or return an empty string.
    """
    lines = [
        f"Session: {session_id}",
        f"Facts recorded: {fact_count}",
        f"Coverage: partial (session facts are a subset of the full store)",
        "",
        "Top recorded facts:",
    ]
    for f in facts[:_BODY_FACTS]:
        content = f.get("content", "")
        content = format_highlight(content)
        lines.append(f"  - {content}")
    if fact_count > _BODY_FACTS:
        lines.append(f"  ... and {fact_count - _BODY_FACTS} more facts.")
    return "\n".join(lines)


def _try_llm(
    prompt: str,
    facts: list[dict],
    config: object | None,
    mode: str,
) -> tuple[str | None, str]:
    """Attempt LLM summarisation.  Returns (content, generated_by) or (None, extractive)."""
    if mode == "c":
        result = _call_cloud_llm(prompt, facts, config)
        if result:
            return result, GENERATED_BY_LLM_C
    # Mode C uses the configured provider only: the Ollama path below
    # would post to the cloud host in config.llm.api_base.
    if mode == "b":
        result = _call_ollama(prompt, facts, config)
        if result:
            return result, GENERATED_BY_LLM_B
    return None, GENERATED_BY_EXTRACTIVE


def _call_ollama(
    prompt: str,
    facts: list[dict],
    config: object | None,
) -> str | None:
    """Mode B: call Ollama.  Returns None on any failure."""
    try:
        import json
        import urllib.request

        from superlocalmemory.core import outbound_http

        api_base = "http://localhost:11434"
        model = "llama3.2"
        timeout = 30
        if config and hasattr(config, "llm"):
            api_base = getattr(config.llm, "api_base", api_base) or api_base
            model = getattr(config.llm, "model", model) or model
            timeout = (
                getattr(config.llm, "timeout_seconds", None)
                or getattr(config.llm, "timeout", None)
                or timeout
            )

        fact_texts = "\n".join(f"- {f['content']}" for f in facts[:10])
        full_prompt = f"{prompt}\n\nFacts:\n{fact_texts}\n\nRespond in 2-4 sentences."
        payload = json.dumps({
            "model": model,
            "prompt": full_prompt,
            "system": SUMMARY_SYSTEM_PROMPT,
            "stream": False,
            "options": {"num_predict": 200},
        }).encode()
        req = urllib.request.Request(
            f"{api_base}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = outbound_http.urlopen(req, timeout=timeout)
        data = json.loads(resp.read().decode())
        text = clean_llm_summary(data.get("response", ""))
        return text if text and len(text) > 20 else None
    except Exception as exc:
        logger.debug("Ollama session summary failed: %s", exc)
        return None


def _call_cloud_llm(
    prompt: str,
    facts: list[dict],
    config: object | None,
) -> str | None:
    """Mode C: call the configured cloud LLM.  Returns None on any failure."""
    if not config or not hasattr(config, "llm"):
        return None
    try:
        from superlocalmemory.llm.backbone import LLMBackbone
        llm = LLMBackbone(config.llm)
        if not llm.is_available():
            return None
        fact_texts = "\n".join(f"- {f['content']}" for f in facts[:10])
        full_prompt = f"{prompt}\n\nFacts:\n{fact_texts}\n\nRespond in 2-4 sentences."
        text = llm.generate(
            prompt=full_prompt,
            system=SUMMARY_SYSTEM_PROMPT,
            max_tokens=200,
            temperature=0.1,
        )
        cleaned = clean_llm_summary(text or "")
        return cleaned if len(cleaned) > 20 else None
    except Exception as exc:
        logger.debug("Cloud LLM session summary failed: %s", exc)
        return None
