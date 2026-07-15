# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Intent Classifier (Stage 0: Ingestion).

Classifies an incoming observation as an assertion (a fact to remember) vs.
a query or directive (a question, or an instruction/command). This is the
diagram's "unsolved component" — errors here propagate to every downstream
guarantee, and adversarial content is built to blur exactly this line — so
the classifier is deliberately heuristic and conservative: it flags rather
than silently drops non-assertions, letting the trust-gated merge pipeline
(encoding/consolidator.py) route flagged content through quarantine instead
of trusting it outright.

Heuristic only (no new ML dependency): question markers, interrogative
openers, and imperative/second-person command patterns commonly used to
smuggle instructions into stored content (prompt-injection defense).

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Intent = Literal["assertion", "query", "directive"]

_QUESTION_WORDS = (
    "who", "what", "when", "where", "why", "how",
    "is", "are", "do", "does", "did", "can", "could",
    "would", "should", "will", "may", "might",
)

_DIRECTIVE_VERBS = (
    "ignore", "disregard", "forget", "delete", "override", "bypass",
    "run", "execute", "call", "invoke", "send", "post", "fetch",
    "act as", "pretend", "you must", "you are now", "system prompt",
    "from now on", "new instructions",
)

_WORD_RE = re.compile(r"[a-zA-Z']+")


@dataclass(frozen=True, slots=True)
class IntentResult:
    intent: Intent
    confidence: float


def classify_intent(content: str) -> IntentResult:
    """Classify *content* as assertion, query, or directive.

    Returns confidence in [0, 1]. Low-confidence results should be treated
    as "assertion" by callers (fail open) — this is a coarse heuristic
    filter, not a guarantee.
    """
    text = (content or "").strip()
    if not text:
        return IntentResult("assertion", 0.0)

    lowered = text.lower()
    first_word_match = _WORD_RE.match(lowered)
    first_word = first_word_match.group(0) if first_word_match else ""

    # --- Query signals ---
    ends_with_question_mark = text.rstrip().endswith("?")
    starts_with_question_word = first_word in _QUESTION_WORDS

    if ends_with_question_mark and starts_with_question_word:
        return IntentResult("query", 0.9)
    if ends_with_question_mark:
        return IntentResult("query", 0.7)
    if starts_with_question_word and len(lowered.split()) <= 12:
        # Short interrogative-opener sentence without a "?" — still likely
        # a question (e.g. transcribed speech missing punctuation).
        return IntentResult("query", 0.55)

    # --- Directive signals ---
    for marker in _DIRECTIVE_VERBS:
        if marker in lowered:
            return IntentResult("directive", 0.75)

    # Leading imperative second-person command: "You should X", "Please do Y".
    if lowered.startswith(("please ", "you should ", "you need to ")):
        return IntentResult("directive", 0.6)

    return IntentResult("assertion", 1.0)
