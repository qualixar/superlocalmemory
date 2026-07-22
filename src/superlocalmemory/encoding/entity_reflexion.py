# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Reflexion self-review over LLM-extracted entities (Wave Q1).

After primary LLM extraction (Mode B/C only), one bounded self-review pass
audits the extracted entities against the source text and:

  - drops hallucinated entities (not supported by the source), and
  - adds clearly-missed entities (whose exact text appears in the source).

This closes the two distinct extraction failure modes the market leaders
separate: coverage (missed) and grounding (hallucinated). Coverage is the
"gleaning" idea (re-present what was extracted, ask for misses); grounding is
the "reflexion" idea (verify each extracted entity is real).

Design guarantees:
  - Mode A never reaches here — the caller gates on Mode B/C + LLM available.
  - Fail-open: any error, unavailable LLM, or unparseable output returns the
    input facts unchanged. This module NEVER raises.
  - Immutable: corrected facts are new objects; inputs are not mutated.
  - Bounded: one LLM call, capped facts reviewed, capped entities per fact.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from typing import Any

from superlocalmemory.storage.models import AtomicFact

logger = logging.getLogger(__name__)

_MAX_ENTITIES_PER_FACT = 12
_MIN_ENTITY_LEN = 2

_SYSTEM = (
    "You are a precise information-extraction auditor. You verify that "
    "entities extracted from text are actually supported by that text, and "
    "you catch clearly-named entities that were missed. You never invent "
    "entities. You reply with JSON only."
)


class EntityReflexion:
    """Bounded, fail-open reflexion pass over extracted entities."""

    def __init__(
        self,
        llm: Any,
        max_facts: int = 8,
        max_tokens: int = 512,
    ) -> None:
        self._llm = llm
        self._max_facts = max(1, int(max_facts))
        self._max_tokens = max(64, int(max_tokens))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refine(
        self, source_text: str, facts: list[AtomicFact],
    ) -> list[AtomicFact]:
        """Return facts with entity lists corrected, or unchanged on failure."""
        if not self._llm or not facts or not source_text or not source_text.strip():
            return facts
        try:
            is_avail = getattr(self._llm, "is_available", None)
            if callable(is_avail) and not is_avail():
                return facts
        except Exception:
            return facts

        subject = facts[: self._max_facts]
        try:
            raw = self._invoke(source_text, subject)
        except Exception as exc:
            logger.debug("EntityReflexion LLM call failed: %s", exc)
            return facts

        corrections = self._parse(raw)
        if not corrections:
            return facts
        try:
            return self._apply(facts, corrections, source_text)
        except Exception as exc:
            logger.debug("EntityReflexion apply failed: %s", exc)
            return facts

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _invoke(self, source_text: str, subject: list[AtomicFact]) -> str:
        lines = []
        for i, f in enumerate(subject):
            lines.append(f'{i}: "{f.content}" entities={list(f.entities)}')
        facts_block = "\n".join(lines)
        prompt = (
            "Audit the extracted entities for each fact against the SOURCE "
            "text below.\n"
            "For each fact, report:\n"
            "- drop: entities in the fact's list that are NOT supported by the "
            "source (hallucinated).\n"
            "- add: named entities clearly present in the source but missing "
            "from the fact's list. Only add entities whose exact text appears "
            "in the source.\n\n"
            f"--- SOURCE ---\n{source_text}\n--- END ---\n\n"
            f"--- FACTS ---\n{facts_block}\n--- END ---\n\n"
            'Respond with ONLY a JSON array: '
            '[{"index": <int>, "drop": [..], "add": [..]}]. '
            "Use [] for an empty list. Omit facts that need no change."
        )
        out = self._llm.generate(
            prompt=prompt,
            system=_SYSTEM,
            temperature=0.0,
            max_tokens=self._max_tokens,
        )
        return out if isinstance(out, str) else str(out or "")

    @staticmethod
    def _parse(raw: str) -> list[dict]:
        if not raw or not raw.strip():
            return []
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            return []
        if not isinstance(data, list):
            return []
        return [d for d in data if isinstance(d, dict)]

    def _apply(
        self,
        facts: list[AtomicFact],
        corrections: list[dict],
        source_text: str,
    ) -> list[AtomicFact]:
        by_index: dict[int, dict] = {}
        for c in corrections:
            idx = c.get("index")
            if isinstance(idx, int) and 0 <= idx < self._max_facts:
                by_index[idx] = c

        src_low = source_text.lower()
        out: list[AtomicFact] = []
        for i, fact in enumerate(facts):
            c = by_index.get(i)
            if c is None:
                out.append(fact)
                continue

            ents = list(fact.entities)
            drop = {
                str(d).strip().lower()
                for d in _as_list(c.get("drop"))
                if str(d).strip()
            }
            if drop:
                ents = [e for e in ents if e.strip().lower() not in drop]

            seen = {e.strip().lower() for e in ents}
            for a in _as_list(c.get("add")):
                cand = str(a).strip()
                low = cand.lower()
                if len(cand) < _MIN_ENTITY_LEN or low in seen:
                    continue
                # Grounding guard: only add entities literally present in source.
                if low not in src_low:
                    continue
                ents.append(cand)
                seen.add(low)
                if len(ents) >= _MAX_ENTITIES_PER_FACT:
                    break

            if ents != list(fact.entities):
                out.append(replace(fact, entities=ents))
            else:
                out.append(fact)
        return out


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]
