# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Project Work Log — issue #113 bounded summary.

Produces a human-readable log of tool activity and recorded facts for a
specific project, identified by tool_events.project_path.

CRITICAL IMPLEMENTATION NOTE — DO NOT USE entity_profiles.project_name
-----------------------------------------------------------------------
entity_profiles.project_name has 1,148 rows and EXACTLY ONE distinct value
on a real store.  Grouping by it yields a single meaningless bucket for all
projects.  This is measured fact, not assumption.

Project scope MUST come from tool_events.project_path.
  - 1,899 rows across 13 real projects on the same store.
  - Top project: ".../testing - automation" (336 events).
  - This is the only reliable project discriminator in the schema.

DETERMINISTIC FALLBACK
-----------------------
The extractive path aggregates tool events and the top facts for the project.
It never calls an LLM and never fails to return a result.  Mode B/C
enrichment is attempted when configured, but falls back to extractive on any
failure.

HOT PATH EXCLUSION
-------------------
This module must NEVER be imported from core/recall_pipeline.py or
core/store_pipeline.py.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from pathlib import Path

from superlocalmemory.storage.database import visible_fact_clause_for_connection

from .base import (
    clean_llm_summary,
    format_highlight,
    SUMMARY_SYSTEM_PROMPT,
    COVERAGE_FULL,
    COVERAGE_INSUFFICIENT,
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

logger = logging.getLogger("superlocalmemory.summaries.project")

_MIN_EVENTS = 1         # Minimum tool_events rows to attempt a summary
_MIN_FACTS = 1          # Minimum atomic_facts rows to include facts section
_TOP_TOOLS = 8          # How many tools to list in the summary
_TOP_FACTS = 8          # How many facts to include in the extractive body
_MAX_FACT_CHARS = 300   # Per-fact character cap in the body


def generate_project_work_log(
    db_path: str | Path,
    project_path: str,
    profile_id: str = "default",
    config: object | None = None,
    *,
    hidden_of: HiddenOf | None = None,
) -> SummaryResult:
    """Generate a Project Work Log for a specific project.

    Scope is determined by tool_events.project_path — NOT by
    entity_profiles.project_name (which has 1 distinct value on a real store
    and is therefore useless for scoping).

    Args:
        db_path:      Path to memory.db.
        project_path: The project path as stored in tool_events.project_path.
                      Exact match — callers may pass a prefix and use
                      generate_project_work_log_by_prefix() for fuzzy matching.
        profile_id:   Profile scope — never mix profiles.
        config:       Optional SLMConfig for LLM enrichment.  None = Mode A.

    Returns:
        SummaryResult with source_fact_ids.  Never returns None.
    """
    db_path = Path(db_path)

    # ── query tool events ────────────────────────────────────────────────────
    tool_rows, facts_rows, query_error = _query_project_data(
        db_path, project_path, profile_id
    )

    facts_rows = without_hidden(facts_rows, hidden_of)

    if query_error:
        return SummaryResult(
            kind="project",
            profile_id=profile_id,
            content=(
                f"Project work log for '{project_path}' is unavailable: "
                f"data access error."
            ),
            source_fact_ids=[],
            coverage=COVERAGE_UNAVAILABLE,
            generated_by=GENERATED_BY_EXTRACTIVE,
            metadata={"project_path": project_path, "error": query_error},
        )

    if not tool_rows and not facts_rows:
        return SummaryResult(
            kind="project",
            profile_id=profile_id,
            content=(
                f"No tool events or facts found for project '{project_path}'.\n"
                f"Note: a project is matched by its name (the last part of a "
                f"path, ignoring case) in tool events and in the project each "
                f"memory was saved under."
            ),
            source_fact_ids=[],
            coverage=COVERAGE_INSUFFICIENT,
            generated_by=GENERATED_BY_EXTRACTIVE,
            metadata={"project_path": project_path, "event_count": 0},
        )

    source_fact_ids = [f["fact_id"] for f in facts_rows]
    event_count = len(tool_rows)
    fact_count = len(facts_rows)

    # A project work log has two inputs: what was DONE (tool events) and what was
    # LEARNED (facts). "full" means both were there.
    #
    # The previous rule was `events >= 1 OR facts >= 1`, which reported "full" for
    # a project with 86 tool events and zero facts — rendered in the dashboard as
    # "Built from 0 memories · coverage: full". Claiming full coverage of nothing
    # is precisely the dishonest summary issue #113 exists to prevent.
    if event_count >= _MIN_EVENTS and fact_count >= _MIN_FACTS:
        coverage = COVERAGE_FULL
    elif event_count >= _MIN_EVENTS or fact_count >= _MIN_FACTS:
        coverage = COVERAGE_PARTIAL
    else:
        coverage = COVERAGE_INSUFFICIENT

    extractive_content = _build_extractive_content(
        project_path, tool_rows, facts_rows, event_count, fact_count
    )

    # ── LLM enrichment (optional) ─────────────────────────────────────────────
    mode = get_mode_str(config)
    if mode in ("b", "c"):
        llm_content, llm_mode = _try_llm(
            project_path, tool_rows, facts_rows, config, mode
        )
        if llm_content:
            return SummaryResult(
                kind="project",
                profile_id=profile_id,
                content=llm_content,
                source_fact_ids=source_fact_ids,
                coverage=coverage,
                generated_by=llm_mode,
                metadata={
                    "project_path": project_path,
                    "event_count": event_count,
                    "fact_count": fact_count,
                },
            )

    return SummaryResult(
        kind="project",
        profile_id=profile_id,
        content=extractive_content,
        source_fact_ids=source_fact_ids,
        coverage=coverage,
        generated_by=GENERATED_BY_EXTRACTIVE,
        metadata={
            "project_path": project_path,
            "event_count": event_count,
            "fact_count": fact_count,
        },
    )


def generate_project_work_log_by_prefix(
    db_path: str | Path,
    project_prefix: str,
    profile_id: str = "default",
    config: object | None = None,
) -> list[SummaryResult]:
    """Generate work logs for all projects whose path starts with project_prefix.

    Useful when the user knows only the parent directory, not the exact path.
    Returns one SummaryResult per distinct project_path found.

    Each result uses the EXACT project_path as the key, so source_fact_ids
    and tool events are correctly scoped per sub-project.
    """
    db_path = Path(db_path)
    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        try:
            rows = conn.execute(
                """
                SELECT DISTINCT project_path
                FROM   tool_events
                WHERE  profile_id    = ?
                  AND  project_path  LIKE ?
                  AND  project_path != ''
                ORDER  BY project_path
                """,
                (profile_id, f"{project_prefix}%"),
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return []

    results = []
    for row in rows:
        path = dict(row)["project_path"]
        result = generate_project_work_log(db_path, path, profile_id, config)
        results.append(result)
    return results


# ── internal helpers ──────────────────────────────────────────────────────────

def _like_pattern(key: str) -> str:
    """A LIKE pattern that narrows the rows to read; project_key decides.

    "Contains", not "ends with": a stored path may end in a slash. SQLite's
    LIKE folds case for ASCII only, so for any other name every row is read.
    """
    if not key.isascii():
        return "%"
    escaped = key.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _query_project_data(
    db_path: Path,
    project_path: str,
    profile_id: str,
) -> tuple[list[dict], list[dict], str | None]:
    """Query tool events and associated facts for the project.

    Returns (tool_rows, facts_rows, error_message_or_None).

    4.1.21 (#150): "the project" is ``core.project_identity.project_key`` -
    the same rule recall uses - so a target of "superlocalmemory" and one of
    "/Users/x/superlocalmemory" name the same project. Facts come from two
    sources: memories SAVED under the project (``remember(project=...)``, the
    session-end hook, the backfill), and facts recorded in sessions whose tool
    events ran in it. Before this only the second existed, so a project whose
    memories were tagged but never reached by a hook-observed session
    reported "No tool events or facts found".
    """
    from superlocalmemory.core.project_identity import project_key

    key = project_key(project_path)
    if key is None:
        return [], [], None
    pattern = _like_pattern(key)
    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        try:
            # Tool events whose directory is this project. The LIKE narrows to
            # paths ending in the name; project_key decides.
            event_rows = [dict(r) for r in conn.execute(
                """
                SELECT tool_name, event_type, session_id, project_path,
                       input_summary, output_summary, created_at, duration_ms
                FROM   tool_events
                WHERE  profile_id = ?
                  AND  project_path LIKE ? ESCAPE '\\'
                ORDER  BY created_at ASC, id ASC
                """,
                (profile_id, pattern),
            ).fetchall()]
            tool_rows = [r for r in event_rows if project_key(r["project_path"]) == key]
            paths = sorted({r["project_path"] for r in tool_rows})

            # Withheld and deleted rows are not memories a person may be shown.
            visible = visible_fact_clause_for_connection(conn, "af")
            session_facts: list[dict] = []
            if paths:
                marks = ",".join("?" * len(paths))
                session_facts = [dict(r) for r in conn.execute(
                    f"""
                    SELECT DISTINCT af.fact_id, af.content, af.created_at,
                                    af.importance, af.canonical_entities_json
                    FROM   atomic_facts  af
                    JOIN   tool_events   te
                           ON  te.session_id  = af.session_id
                           AND te.profile_id  = af.profile_id
                    WHERE  af.profile_id   = ?
                      AND  te.project_path IN ({marks})
                      AND  af.lifecycle   != 'archived'{visible}
                    """,  # noqa: S608 - the clauses are built from constants only
                    (profile_id, *paths),
                ).fetchall()]

            saved_rows = [dict(r) for r in conn.execute(
                f"""
                SELECT af.fact_id, af.content, af.created_at, af.importance,
                       af.canonical_entities_json,
                       json_extract(m.metadata_json, '$.project') AS project
                FROM   atomic_facts af
                JOIN   memories     m ON m.memory_id = af.memory_id
                WHERE  af.profile_id = ?
                  AND  af.lifecycle != 'archived'
                  AND  json_valid(m.metadata_json)
                  AND  json_extract(m.metadata_json, '$.project') LIKE ? ESCAPE '\\'{visible}
                """,  # noqa: S608 - the clause is built from constants only
                (profile_id, pattern),
            ).fetchall()]
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("project work log query failed for %s: %s", project_path, exc)
        return [], [], str(exc)

    saved_facts = [
        {k: v for k, v in r.items() if k != "project"}
        for r in saved_rows if project_key(r.get("project")) == key
    ]
    merged = {r["fact_id"]: r for r in [*session_facts, *saved_facts]}
    facts_rows = sorted(
        merged.values(),
        key=lambda r: (-(r.get("importance") or 0.0), str(r.get("created_at") or ""),
                       str(r["fact_id"])),
    )
    return tool_rows, facts_rows, None


def _build_extractive_content(
    project_path: str,
    tool_rows: list[dict],
    facts_rows: list[dict],
    event_count: int,
    fact_count: int,
) -> str:
    """Build a deterministic extractive project work log."""
    # Readable project name: last 2 path components.
    parts = project_path.rstrip("/").split("/")
    display_name = "/".join(parts[-2:]) if len(parts) >= 2 else project_path

    lines = [
        f"Project Work Log: {display_name}",
        f"Full path: {project_path}",
        f"Tool events: {event_count}",
        f"Associated facts: {fact_count}",
        "",
    ]

    # Tool usage breakdown.
    if tool_rows:
        tool_counter: Counter = Counter()
        for r in tool_rows:
            tool_counter[r.get("tool_name") or "unknown"] += 1
        top_tools = tool_counter.most_common(_TOP_TOOLS)
        lines.append("Tool usage:")
        for tool, count in top_tools:
            lines.append(f"  {tool}: {count} event(s)")
        if len(tool_counter) > _TOP_TOOLS:
            lines.append(f"  ... and {len(tool_counter) - _TOP_TOOLS} other tools.")
        lines.append("")

    # Top facts by importance.
    if facts_rows:
        lines.append("Key facts from project sessions:")
        for f in facts_rows[:_TOP_FACTS]:
            content = f.get("content", "")
            content = format_highlight(content)
            lines.append(f"  - {content}")
        if fact_count > _TOP_FACTS:
            lines.append(f"  ... and {fact_count - _TOP_FACTS} more facts.")

    return "\n".join(lines)


def _try_llm(
    project_path: str,
    tool_rows: list[dict],
    facts_rows: list[dict],
    config: object | None,
    mode: str,
) -> tuple[str | None, str]:
    """Attempt LLM-based project summary.  Returns (content, generated_by)."""
    parts = project_path.rstrip("/").split("/")
    display_name = "/".join(parts[-2:]) if len(parts) >= 2 else project_path
    prompt = (
        f"Write a concise project work log for '{display_name}' based on "
        f"{len(tool_rows)} tool events and {len(facts_rows)} recorded facts."
    )
    if mode == "c":
        result = _call_cloud_llm(prompt, tool_rows, facts_rows, config)
        if result:
            return result, GENERATED_BY_LLM_C
    # Mode C uses the configured provider only: the Ollama path below
    # would post to the cloud host in config.llm.api_base.
    if mode == "b":
        result = _call_ollama(prompt, tool_rows, facts_rows, config)
        if result:
            return result, GENERATED_BY_LLM_B
    return None, GENERATED_BY_EXTRACTIVE


def _call_ollama(
    prompt: str,
    tool_rows: list[dict],
    facts_rows: list[dict],
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

        tool_summary = ", ".join(
            f"{r['tool_name']}({r.get('event_type', '')})"
            for r in tool_rows[:10]
        )
        fact_texts = "\n".join(f"- {f['content']}" for f in facts_rows[:8])
        full_prompt = (
            f"{prompt}\n\n"
            f"Tools used: {tool_summary}\n\n"
            f"Key facts:\n{fact_texts}\n\n"
            f"Respond in 3-5 sentences."
        )
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
        logger.debug("Ollama project work log failed: %s", exc)
        return None


def _call_cloud_llm(
    prompt: str,
    tool_rows: list[dict],
    facts_rows: list[dict],
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
        tool_summary = ", ".join(
            f"{r['tool_name']}({r.get('event_type', '')})"
            for r in tool_rows[:10]
        )
        fact_texts = "\n".join(f"- {f['content']}" for f in facts_rows[:8])
        full_prompt = (
            f"{prompt}\n\nTools: {tool_summary}\n\nFacts:\n{fact_texts}"
        )
        text = llm.generate(
            prompt=full_prompt,
            system=SUMMARY_SYSTEM_PROMPT,
            max_tokens=300,
            temperature=0.1,
        )
        cleaned = clean_llm_summary(text or "")
        return cleaned if len(cleaned) > 20 else None
    except Exception as exc:
        logger.debug("Cloud LLM project work log failed: %s", exc)
        return None
