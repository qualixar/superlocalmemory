# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Daily Reflection — issue #113 bounded summary.

Produces a human-readable summary of all facts recorded on a specific date,
grouped by time of day and sorted by importance.

MEASURED DATA REALITY
---------------------
On a real 3,294-fact store there are 48 distinct active days.
Per-day volumes range from 4 to 454 facts.  Daily reflections are
structurally viable for this store.

DETERMINISTIC FALLBACK
-----------------------
The extractive path groups facts by topic entities and summarises the top-N
by importance.  It never calls an LLM and never fails to return a result.
Mode B/C enrichment is attempted when configured, but falls back to extractive
on any failure — silence is not acceptable.

HOT PATH EXCLUSION
-------------------
This module must NEVER be imported from core/recall_pipeline.py or
core/store_pipeline.py.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date
from pathlib import Path

from superlocalmemory.storage.database import visible_fact_clause_for_connection

from .base import (
    clean_llm_summary,
    local_day_modifier,
    format_highlight,
    SUMMARY_SYSTEM_PROMPT,
    COVERAGE_FULL,
    COVERAGE_INSUFFICIENT,
    COVERAGE_UNAVAILABLE,
    GENERATED_BY_EXTRACTIVE,
    GENERATED_BY_LLM_B,
    GENERATED_BY_LLM_C,
    HiddenOf,
    SummaryResult,
    without_hidden,
    get_mode_str,
)

logger = logging.getLogger("superlocalmemory.summaries.daily")

_MIN_FACTS = 2          # Below this threshold, coverage=insufficient
_BODY_FACTS = 10        # Top-N facts shown in extractive summary
_MAX_FACT_CHARS = 300   # Per-fact character cap in the body


def generate_daily_reflection(
    db_path: str | Path,
    target_date: str | date,
    profile_id: str = "default",
    config: object | None = None,
    *,
    tz_offset_minutes: int = 0,
    hidden_of: HiddenOf | None = None,
) -> SummaryResult:
    """Generate a Daily Reflection for a specific date.

    Args:
        db_path:     Path to memory.db.
        target_date: The date to reflect on.  Accepts ``date`` objects or
                     ISO-format strings ("2026-08-17").
        profile_id:  Profile scope — never mix profiles.
        config:      Optional SLMConfig for LLM enrichment.  None = Mode A
                     (extractive, always deterministic).
        tz_offset_minutes: The caller's offset east of UTC (330 for India).
                     ``created_at`` is stored in UTC, so without it the day is
                     the UTC day and, east of UTC, memories saved after local
                     midnight land in the previous day's reflection.
        hidden_of:   Names the facts the caller may not see; they are left out.

    Returns:
        SummaryResult with source_fact_ids for every contributing fact.
        Never returns None.
    """
    db_path = Path(db_path)
    date_str = target_date.isoformat() if isinstance(target_date, date) else str(target_date)
    day_shift = local_day_modifier(tz_offset_minutes)

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
                SELECT fact_id, content, created_at, importance,
                       canonical_entities_json, lifecycle
                FROM   atomic_facts
                WHERE  profile_id = ?
                  AND  DATE(created_at, ?) = ?
                  AND  lifecycle != 'archived'{visible}
                ORDER  BY importance DESC, created_at ASC, fact_id ASC
                """,  # noqa: S608 - the clause is built from constants only
                (profile_id, day_shift, date_str),
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("daily reflection query failed for %s: %s", date_str, exc)
        return SummaryResult(
            kind="daily",
            profile_id=profile_id,
            content=f"Daily reflection for {date_str} is unavailable: data access error.",
            source_fact_ids=[],
            coverage=COVERAGE_UNAVAILABLE,
            generated_by=GENERATED_BY_EXTRACTIVE,
            metadata={"date": date_str, "error": str(exc)},
        )

    rows = without_hidden([dict(r) for r in rows], hidden_of)

    # ── coverage decision ─────────────────────────────────────────────────────
    if not rows:
        return SummaryResult(
            kind="daily",
            profile_id=profile_id,
            content=f"No facts recorded on {date_str}.",
            source_fact_ids=[],
            coverage=COVERAGE_INSUFFICIENT,
            generated_by=GENERATED_BY_EXTRACTIVE,
            metadata={"date": date_str, "fact_count": 0,
                      "tz_offset_minutes": tz_offset_minutes},
        )

    facts = [dict(r) for r in rows]
    source_fact_ids = [f["fact_id"] for f in facts]
    fact_count = len(facts)

    if fact_count < _MIN_FACTS:
        coverage = COVERAGE_INSUFFICIENT
    else:
        coverage = COVERAGE_FULL

    # ── extractive summary (deterministic, always available) ──────────────────
    extractive_content = _build_extractive_content(date_str, facts, fact_count, db_path, profile_id)

    # ── LLM enrichment (optional) ─────────────────────────────────────────────
    mode = get_mode_str(config)
    if fact_count >= _MIN_FACTS and mode in ("b", "c"):
        llm_content, llm_mode = _try_llm(date_str, facts, config, mode)
        if llm_content:
            return SummaryResult(
                kind="daily",
                profile_id=profile_id,
                content=llm_content,
                source_fact_ids=source_fact_ids,
                coverage=coverage,
                generated_by=llm_mode,
                metadata={"date": date_str, "fact_count": fact_count,
                          "tz_offset_minutes": tz_offset_minutes},
            )

    return SummaryResult(
        kind="daily",
        profile_id=profile_id,
        content=extractive_content,
        source_fact_ids=source_fact_ids,
        coverage=coverage,
        generated_by=GENERATED_BY_EXTRACTIVE,
        metadata={"date": date_str, "fact_count": fact_count,
                          "tz_offset_minutes": tz_offset_minutes},
    )


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_extractive_content(
    date_str: str,
    facts: list[dict],
    fact_count: int,
    db_path: str | Path,
    profile_id: str,
) -> str:
    """Build a deterministic extractive daily reflection.

    ``db_path`` / ``profile_id`` are needed only to turn entity IDs into names
    for the "Active entities" line — the facts themselves are already loaded.
    """
    import json

    lines = [
        f"Daily reflection: {date_str}",
        f"Total facts recorded: {fact_count}",
        "",
        "Highlights:",
    ]
    for f in facts[:_BODY_FACTS]:
        content = f.get("content", "")
        content = format_highlight(content)
        lines.append(f"  - {content}")
    if fact_count > _BODY_FACTS:
        lines.append(f"  ... and {fact_count - _BODY_FACTS} additional facts.")

    # Entity summary: which entities appeared most
    entity_counts: dict[str, int] = {}
    for f in facts:
        raw = f.get("canonical_entities_json") or "[]"
        try:
            entities = json.loads(raw)
        except (ValueError, TypeError):
            entities = []
        for e in entities:
            entity_counts[e] = entity_counts.get(e, 0) + 1

    if entity_counts:
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        # Resolve to names. canonical_entities_json stores entity IDs, so this
        # line previously read "Active entities: 1666abc512904473 (127),
        # 84288f2dde994afe (124)" — five 16-hex identifiers, which tell a reader
        # nothing. They resolve to 'Fixed', 'Gateway', 'REVISED' and so on. This
        # is the same defect 4.0.6 fixed in the Living Brain, where source
        # quality listed internal identifiers; it survived here because this
        # generator was never reachable to be looked at.
        names = _entity_names(db_path, [e for e, _ in top_entities], profile_id)
        labelled = [
            (names.get(eid) or eid, count) for eid, count in top_entities
        ]
        entity_str = ", ".join(f"{name} ({c})" for name, c in labelled)
        lines.append("")
        lines.append(f"Active entities: {entity_str}")

    return "\n".join(lines)


def _entity_names(
    db_path: str | Path, entity_ids: list[str], profile_id: str
) -> dict[str, str]:
    """Map entity_id → canonical_name. Missing or unreadable rows are omitted.

    Fail-open: a summary is still useful with a raw id in it, so a query problem
    must not lose the whole line. The caller falls back to the id per entity.
    """
    if not entity_ids:
        return {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            placeholders = ",".join("?" for _ in entity_ids)
            rows = conn.execute(
                f"SELECT entity_id, canonical_name FROM canonical_entities "
                f"WHERE entity_id IN ({placeholders}) AND profile_id = ?",
                (*entity_ids, profile_id),
            ).fetchall()
        finally:
            conn.close()
        return {r[0]: r[1] for r in rows if r[1]}
    except sqlite3.Error:
        return {}


def _try_llm(
    date_str: str,
    facts: list[dict],
    config: object | None,
    mode: str,
) -> tuple[str | None, str]:
    """Attempt LLM-based daily reflection.  Returns (content, generated_by)."""
    prompt = (
        f"Write a concise daily reflection for {date_str} based on these facts. "
        f"Focus on main themes and accomplishments in 3-5 sentences."
    )
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
    """Mode B: call Ollama.  Returns None on any failure (extractive fallback)."""
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

        fact_texts = "\n".join(f"- {f['content']}" for f in facts[:15])
        full_prompt = f"{prompt}\n\nFacts from {len(facts)} recorded:\n{fact_texts}"
        payload = json.dumps({
            "model": model,
            "prompt": full_prompt,
            "system": SUMMARY_SYSTEM_PROMPT,
            "stream": False,
            "options": {"num_predict": 300},
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
        logger.debug("Ollama daily reflection failed: %s", exc)
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
        fact_texts = "\n".join(f"- {f['content']}" for f in facts[:15])
        full_prompt = f"{prompt}\n\nFacts:\n{fact_texts}"
        text = llm.generate(
            prompt=full_prompt,
            system=SUMMARY_SYSTEM_PROMPT,
            max_tokens=300,
            temperature=0.1,
        )
        cleaned = clean_llm_summary(text or "")
        return cleaned if len(cleaned) > 20 else None
    except Exception as exc:
        logger.debug("Cloud LLM daily reflection failed: %s", exc)
        return None
