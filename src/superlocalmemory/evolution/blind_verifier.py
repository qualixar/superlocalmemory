# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Blind Verifier — information-isolated skill verification.

The key insight from EvoSkills (arXiv:2604.01687): when a generator
creates a skill and the same model verifies it, confirmation bias is
nearly guaranteed. The verifier must be BLIND to the generator's reasoning.

This verifier:
- Uses a DIFFERENT model from the generator (Haiku vs Sonnet)
- CANNOT see: original skill, mutation rationale, generator's reasoning
- CAN see: task description (what the skill should do), evolved SKILL.md
- Evaluates independently: "Does this skill correctly address the task?"

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerificationResult:
    """Result of blind verification."""
    passed: bool
    confidence: float  # 0.0-1.0
    issues: tuple[str, ...] = ()
    reasoning: str = ""


def build_verification_prompt(
    skill_name: str,
    skill_description: str,
    evolved_content: str,
) -> str:
    """Build blind verification prompt.

    The verifier sees ONLY:
    - What the skill is supposed to do (name + description)
    - The evolved skill content

    The verifier does NOT see:
    - The original skill
    - Why it was evolved
    - What evidence triggered evolution
    - The generator's reasoning
    """
    return f"""You are an independent skill quality reviewer. You have NOT seen the original
version of this skill or why it was modified. Evaluate it purely on its merits.

SKILL PURPOSE: {skill_name}
EXPECTED BEHAVIOR: {skill_description}

SKILL CONTENT TO REVIEW:
{evolved_content[:8000]}

EVALUATE:
1. Does the skill clearly explain what to do? (clarity)
2. Are the instructions specific and actionable? (specificity)
3. Are there any obvious errors, contradictions, or missing steps? (correctness)
4. Would an AI agent be able to follow these instructions? (executability)

RESPOND IN JSON FORMAT:
{{
  "passed": true/false,
  "confidence": 0.0-1.0,
  "issues": ["issue1", "issue2"],
  "reasoning": "brief explanation"
}}

Be strict. Only pass skills that are genuinely clear, correct, and actionable.
A mediocre skill that might work sometimes should FAIL — evolution should produce
clear improvements, not marginal changes."""


def parse_verification_response(response: str) -> VerificationResult:
    """Parse the verifier's JSON response."""
    # Try parsing JSON from response
    json_match = re.search(r"\{[^{}]*\"passed\"[^{}]*\}", response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return VerificationResult(
                passed=bool(data.get("passed", False)),
                confidence=float(data.get("confidence", 0.5)),
                issues=tuple(data.get("issues", [])),
                reasoning=str(data.get("reasoning", "")),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Fallback: keyword detection.
    # Negative keywords are checked first so that negating phrases such as
    # "cannot approve" or "not approved" are never mistakenly matched by the
    # positive keyword "approve".
    lower = response.lower()
    if any(kw in lower for kw in (
        "\"passed\": false", "passed: false",
        "reject", "fail",
        "cannot approve", "not approve",
    )):
        return VerificationResult(passed=False, confidence=0.6, reasoning="keyword match")

    if any(kw in lower for kw in ("\"passed\": true", "passed: true", "approve", "looks good")):
        return VerificationResult(passed=True, confidence=0.6, reasoning="keyword match")

    # Default: reject if can't parse (conservative)
    return VerificationResult(
        passed=False,
        confidence=0.3,
        reasoning="Could not parse verification response",
        issues=("Unparseable response",),
    )
