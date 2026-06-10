# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — Core Block Hygiene (v3.6.6)

"""Core Memory Block hygiene helpers (v3.6.6 F-5).

Three pure functions used by the block compiler:
  - dedupe_block_content: normalized-line dedup within a block
  - filter_low_quality_block_facts: drop is_low_quality facts
  - compile_block_content: full compile pipeline (filter + dedup + cap)

Also exports: _recompile_core_blocks — the hook called by MaintenanceScheduler.

These are pure functions (no I/O). All I/O lives in the scheduler / consolidation
engine that calls them.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_BLOCK_SEPARATOR = "\n---\n"
_PLACEHOLDER = "No data available."


# ---------------------------------------------------------------------------
# dedupe_block_content
# ---------------------------------------------------------------------------

def dedupe_block_content(lines: list[str]) -> list[str]:
    """Remove duplicate and empty lines from a block content line-list.

    Normalization: lowercased, whitespace-collapsed.
    Order is preserved; only the FIRST occurrence is kept.

    Returns a new list — never mutates input.
    """
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        key = " ".join(stripped.lower().split())
        if key in seen:
            continue
        seen.add(key)
        result.append(line)
    return result


# ---------------------------------------------------------------------------
# filter_low_quality_block_facts
# ---------------------------------------------------------------------------

def filter_low_quality_block_facts(facts: list[dict]) -> list[dict]:
    """Filter fact dicts whose content is low-quality or prompt-template.

    Delegates to injection.is_low_quality and injection.is_prompt_template.
    Returns a new list — never mutates input.
    """
    try:
        from superlocalmemory.core.injection import is_low_quality, is_prompt_template
    except Exception:
        return list(facts)

    result: list[dict] = []
    for f in facts:
        content = f.get("content", "") or ""
        if is_low_quality(content):
            continue
        if is_prompt_template(content):
            continue
        result.append(f)
    return result


# ---------------------------------------------------------------------------
# compile_block_content
# ---------------------------------------------------------------------------

def compile_block_content(
    facts: list[dict],
    max_chars: int = 2000,
) -> str:
    """Compile facts into block content with hygiene and char cap.

    Pipeline:
      1. filter_low_quality_block_facts
      2. Extract content lines from each fact (split by newline / separator)
      3. dedupe_block_content across all lines
      4. Join with separator, truncate to max_chars

    Returns a string ≤ max_chars. Returns empty string if all facts filtered.
    """
    clean_facts = filter_low_quality_block_facts(facts)
    if not clean_facts:
        return ""

    all_lines: list[str] = []
    for f in clean_facts:
        content = (f.get("content") or "").strip()
        if not content:
            continue
        # Split by separator or newline to treat each line independently
        for line in content.replace(_BLOCK_SEPARATOR, "\n").split("\n"):
            all_lines.append(line)

    deduped = dedupe_block_content(all_lines)
    if not deduped:
        return ""

    joined = _BLOCK_SEPARATOR.join(deduped)
    return joined[:max_chars]


# ---------------------------------------------------------------------------
# _recompile_core_blocks — scheduler hook (F-5 daily recompile)
# ---------------------------------------------------------------------------

def _recompile_core_blocks(
    db,
    config,
    profile_id: str,
) -> dict:
    """Recompile core memory blocks with hygiene applied.

    Called by MaintenanceScheduler._run() on the daily cycle.
    Delegates to ConsolidationEngine.compile_core_blocks_mode_a() with
    the hygiene improvements applied at the _facts_to_content step.

    Returns a dict with stats: {blocks_compiled, profile_id}.
    """
    try:
        from superlocalmemory.core.consolidation_engine import ConsolidationEngine
        engine = ConsolidationEngine(db=db, config=config.consolidation)
        result = engine.compile_core_blocks_mode_a(profile_id)
        logger.info(
            "Daily core-block recompile: profile=%s blocks=%s",
            profile_id, result.get("blocks_compiled", 0),
        )
        return {**result, "profile_id": profile_id}
    except Exception as exc:
        logger.warning("Core-block recompile failed: %s", exc)
        return {"blocks_compiled": 0, "profile_id": profile_id, "error": str(exc)}
