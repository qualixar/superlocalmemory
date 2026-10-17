# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3.2 | https://qualixar.com

"""Generate contextual descriptions for facts (A-MEM pattern).

WHY a memory matters, not just WHAT it says. Enriches retrieval by
providing a secondary searchable signal and helps consolidation
understand memory relationships.

NEVER imports core/engine.py (Rule 06).
Receives LLM backbone via __init__, not engine.

References:
  - A-MEM (agiresearch/A-mem-sys): contextual_description field
  - Zep/Graphiti: relationship summaries on temporal edges

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from superlocalmemory.storage.models import AtomicFact, FactType, SignalType

logger = logging.getLogger(__name__)

#: One sentence plus 3-5 keywords; the backbone default (4,096) let a model
#: run on. Measured answers are ~75 tokens.
CONTEXT_MAX_TOKENS = 160


def _json_object(text: str) -> dict:
    """The JSON object in a model answer, with or without a ```json fence.

    4.2.9: gemma4:e2b fences its answer; a bare json.loads rejected it and
    every context fell back to rules. Raises ValueError when there is none.
    """
    body = (text or "").strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the model answer")
    parsed = json.loads(body[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("the model answer is not a JSON object")
    return parsed


@dataclass(frozen=True)
class ContextResult:
    """Immutable result of context generation."""

    description: str
    keywords: list[str]
    generated_by: str  # "rules" | "ollama" | "cloud"


class ContextGenerator:
    """Generates WHY a memory matters, not just WHAT it says.

    Mode A: deterministic template-based (zero LLM).
    Mode B: Ollama LLM one-sentence generation.
    Mode C: Cloud LLM high-quality generation.

    NEVER imports core/engine.py (Rule 06).
    Receives LLM backbone via __init__, not engine.
    """

    def __init__(self, llm=None) -> None:
        """
        Args:
            llm: LLMBackbone instance or None (Mode A).
                 Must implement .generate(prompt, system) -> str
                 and .is_available() -> bool.
        """
        self._llm = llm

    def generate(self, fact: AtomicFact, mode: str = "a") -> ContextResult:
        """Generate contextual description for a fact.

        Args:
            fact: The AtomicFact to generate context for.
            mode: "a" (rules), "b" (Ollama), "c" (cloud).

        Returns:
            ContextResult with description, keywords, and generator type.
        """
        try:
            if mode == "a" or self._llm is None or not self._llm.is_available():
                return self._rules_based(fact)
            return self._llm_based(fact, mode)
        except Exception as exc:
            logger.debug("Context generation failed, falling back to rules: %s", exc)
            return self._rules_based(fact)

    def _rules_based(self, fact: AtomicFact) -> ContextResult:
        """Deterministic template for Mode A (zero LLM).

        Template:
        'This {fact_type} about {entities} records {signal_type}
        observed on {observation_date} regarding {topic}.'

        Keywords extracted from entities + content tokens.
        """
        entities_str = (
            ", ".join(fact.canonical_entities[:5])
            if fact.canonical_entities
            else "general knowledge"
        )
        date_str = fact.observation_date or fact.created_at[:10]
        topic = fact.content[:60].rstrip(".")

        # Map fact_type to human-readable category
        type_labels = {
            FactType.EPISODIC: "episodic event",
            FactType.SEMANTIC: "semantic knowledge",
            FactType.OPINION: "opinion or preference",
            FactType.PROSPECTIVE: "time-bounded event",
        }
        type_label = type_labels.get(fact.fact_type, "memory")

        # Map signal_type to human-readable signal description
        signal_labels = {
            SignalType.FACTUAL: "factual information",
            SignalType.EMOTIONAL: "emotional context",
            SignalType.TEMPORAL: "temporal relationship",
            SignalType.OPINION: "subjective judgment",
            SignalType.REQUEST: "a request or action item",
            SignalType.SOCIAL: "social interaction",
        }
        signal_label = signal_labels.get(fact.signal_type, "an observation")

        description = (
            f"This {type_label} about {entities_str} records {signal_label} "
            f"observed on {date_str} regarding {topic}."
        )

        # Extract keywords from entities + content significant words
        keywords = list(fact.canonical_entities[:5])
        stop = {
            "this", "that", "with", "from", "about",
            "which", "their", "there", "would", "could", "should",
        }
        content_words = [
            w.lower().strip(".,!?;:\"'()[]")
            for w in fact.content.split()
            if len(w) > 4
        ]
        for w in content_words[:10]:
            if w not in stop and w not in keywords:
                keywords.append(w)

        return ContextResult(
            description=description,
            keywords=keywords[:10],
            generated_by="rules",
        )

    def _llm_based(self, fact: AtomicFact, mode: str) -> ContextResult:
        """LLM-generated contextual description for Mode B/C.

        Prompt follows A-MEM pattern: explain WHY this memory
        would be useful in the future.
        """
        prompt = (
            "You are a memory analyst. Given a memory fact, write ONE sentence "
            "explaining WHY this memory would be useful to recall in the future. "
            "Focus on what it reveals about the person, project, or decision -- "
            "not what it literally says.\n\n"
            f"Memory: {fact.content}\n"
            f"Type: {fact.fact_type.value}\n"
            f"Entities: {', '.join(fact.canonical_entities[:5])}\n\n"
            "Also extract 3-5 keywords that would help find this memory later.\n\n"
            "Respond in this exact JSON format:\n"
            '{"description": "...", "keywords": ["...", "..."]}'
        )

        response = self._llm.generate(
            prompt,
            system="You are a precise memory analyst. Respond ONLY with valid JSON.",
            max_tokens=CONTEXT_MAX_TOKENS,
            think=False,
        )

        try:
            parsed = _json_object(response)
            return ContextResult(
                description=parsed.get("description", "")[:500],
                keywords=parsed.get("keywords", [])[:10],
                generated_by="ollama" if mode == "b" else "cloud",
            )
        except (ValueError, KeyError, TypeError):
            logger.debug("LLM context response unparseable, falling back to rules")
            return self._rules_based(fact)
