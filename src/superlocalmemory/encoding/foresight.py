# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Foresight Signal Extraction.

Extracts time-bounded intentions and planned events from conversations.
V1 extracted foresight signals but NEVER STORED them. Now persisted as TemporalEvents.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

from superlocalmemory.storage.models import AtomicFact, TemporalEvent

logger = logging.getLogger(__name__)

# Patterns that indicate future intent or planned events
_FORESIGHT_PATTERNS = [
    re.compile(r"\b(plan(?:ning|s|ned)?)\s+to\b", re.I),
    re.compile(r"\b(going)\s+to\b", re.I),
    re.compile(r"\b(will|shall)\s+\w+", re.I),
    re.compile(r"\b(schedul(?:e|ed|ing))\b", re.I),
    re.compile(r"\b(appointment|reservation|booking)\b", re.I),
    re.compile(r"\b(remind(?:er)?|don't forget)\b", re.I),
    re.compile(r"\b(taking|starting|beginning)\s+\w+\s+(next|tomorrow|soon)\b", re.I),
    re.compile(r"\b(deadline|due date|due by)\b", re.I),
    re.compile(r"\bnext\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I),
]


def extract_foresight_signals(fact: AtomicFact) -> list[dict]:
    """Extract foresight signals from a fact's content.

    Returns list of dicts with: content, pattern_matched.
    These get converted to TemporalEvents by the encoding pipeline.
    """
    signals = []
    for pattern in _FORESIGHT_PATTERNS:
        match = pattern.search(fact.content)
        if match:
            signals.append({
                "content": fact.content,
                "pattern": match.group(0),
                "fact_id": fact.fact_id,
            })
            break  # One signal per fact is sufficient

    return signals


def foresight_to_temporal_events(
    fact: AtomicFact,
    entity_ids: list[str],
    profile_id: str,
) -> list[TemporalEvent]:
    """Convert foresight signals into TemporalEvents for persistence.

    Each signal becomes a temporal event linked to the fact's entities.
    The temporal_parser handles actual date extraction — we just mark
    the fact as having foresight intent.
    """
    signals = extract_foresight_signals(fact)
    if not signals:
        return []

    events = []
    for eid in entity_ids:
        event = TemporalEvent(
            profile_id=profile_id,
            scope=fact.scope,
            shared_with=fact.shared_with,
            entity_id=eid,
            fact_id=fact.fact_id,
            observation_date=fact.observation_date,
            referenced_date=fact.referenced_date,
            interval_start=fact.interval_start,
            interval_end=fact.interval_end,
            description=f"Foresight: {fact.content[:200]}",
        )
        events.append(event)

    return events


def has_foresight(text: str) -> bool:
    """Quick check if text contains foresight signals."""
    return any(p.search(text) for p in _FORESIGHT_PATTERNS)
