# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — Ingest Gate (v3.6.6)

"""Ingest gate for the remember/store write path (v3.6.6 F-4).

Responsibilities:
  1. Hard reject content > 1MB (nobody's memory is a megabyte).
  2. Clamp content > 24000 chars to head 70% + tail 30% + truncation marker.
     Head+tail (not head-only) preserves OPEN ITEMS that session-close facts put
     at the END. Only pathological pastes (167KB JSON, 50KB logs) are touched;
     normal dense memories (6-15K session handoffs) pass through intact.
     Full original preserved in result.full_content for storage in memories table.
  3. Prompt-template firewall: reject content matching _PROMPT_TEMPLATE_PATTERNS
     (extends v3.6.4 remember-write-02 quality gate).

Kill-switch: SLM_INGEST_NO_GATE=1 bypasses rules 2 and 3 (but NOT the 1MB hard cap).

Usage::
    from superlocalmemory.core.ingest_gate import apply_ingest_gate
    result = apply_ingest_gate(content)
    if result.rejected:
        return {"success": False, "error": result.rejection_reason}
    store(result.fact_content, full_content=result.full_content)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# v3.6.6: clamp only pathological monsters; preserve normal dense memories.
# 24K chars ≈ 6K tokens — well beyond any legitimate single memory.
_MAX_VERBATIM_CHARS = 24000
_HEAD_FRACTION = 0.70  # head 70% + tail 30% so OPEN ITEMS at the end survive
_MAX_INGEST_BYTES = 1_048_576  # 1MB — hard cap, NOT bypassed by kill-switch
_TRUNCATION_MARKER = "\n…[content truncated at ingest; full text in source memory]…\n"


@dataclass
class IngestGateResult:
    """Result of applying the ingest gate to content.

    fact_content:     Content to store in atomic_facts.content (may be truncated).
    full_content:     Original unmodified content (for memories table storage).
    rejected:         True if content should be rejected outright.
    rejection_reason: Human-readable reason for rejection (when rejected=True).
    truncated:        True if fact_content was head-sliced.
    """
    fact_content: str
    full_content: str
    rejected: bool = False
    rejection_reason: str = ""
    truncated: bool = False


def apply_ingest_gate(
    content: str,
    max_verbatim_chars: int = _MAX_VERBATIM_CHARS,
    max_ingest_bytes: int = _MAX_INGEST_BYTES,
) -> IngestGateResult:
    """Apply the ingest quality gate to content before storing.

    Returns IngestGateResult. Callers MUST check result.rejected before
    proceeding with storage.

    Kill-switch SLM_INGEST_NO_GATE=1 bypasses the verbatim size clamp and
    template firewall but NOT the 1MB hard cap (safety boundary).
    """
    gate_active = os.environ.get("SLM_INGEST_NO_GATE", "0") != "1"

    # --- Hard cap: 1MB regardless of kill-switch ---
    try:
        byte_len = len(content.encode("utf-8", errors="replace"))
    except Exception:
        byte_len = len(content)
    if byte_len > max_ingest_bytes:
        return IngestGateResult(
            fact_content=content,
            full_content=content,
            rejected=True,
            rejection_reason=(
                f"Content size {byte_len} bytes exceeds maximum "
                f"{max_ingest_bytes} bytes (1MB). "
                "Nobody's memory is a megabyte."
            ),
        )

    # --- Gate bypassed ---
    if not gate_active:
        return IngestGateResult(
            fact_content=content,
            full_content=content,
            rejected=False,
            truncated=False,
        )

    # --- Prompt-template firewall ---
    # Extends the v3.6.4 remember-write-02 quality gate.
    try:
        from superlocalmemory.core.injection import is_prompt_template
        if is_prompt_template(content):
            return IngestGateResult(
                fact_content=content,
                full_content=content,
                rejected=True,
                rejection_reason=(
                    "Content matches internal prompt-template patterns "
                    "(low-quality gate). Prompt machinery must not be stored as memory."
                ),
            )
    except Exception:
        pass  # Defensive: gate failure must never block a store

    # --- Verbatim size clamp (head 70% + tail 30%) ---
    if len(content) > max_verbatim_chars:
        budget = max_verbatim_chars - len(_TRUNCATION_MARKER)
        head_len = int(budget * _HEAD_FRACTION)
        tail_len = budget - head_len
        fact_content = content[:head_len] + _TRUNCATION_MARKER + content[-tail_len:]
        return IngestGateResult(
            fact_content=fact_content,
            full_content=content,
            rejected=False,
            truncated=True,
        )

    return IngestGateResult(
        fact_content=content,
        full_content=content,
        rejected=False,
        truncated=False,
    )
