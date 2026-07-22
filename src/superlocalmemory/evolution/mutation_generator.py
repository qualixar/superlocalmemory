# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Mutation Generator — LLM-driven skill improvement.

Reads the original SKILL.md + failure evidence + performance data,
generates an improved version. Apply-retry cycle (3 attempts) for
malformed output.

Token-driven termination: <EVOLUTION_COMPLETE> or <EVOLUTION_FAILED>.
Adopted from OpenSpace evolver.py patterns.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from superlocalmemory.evolution.types import (
    EvolutionCandidate,
    EvolutionType,
)

logger = logging.getLogger(__name__)

MAX_APPLY_RETRIES = 3
MAX_CONTENT_CHARS = 12_000  # Truncate skill content in prompt


def build_mutation_prompt(
    candidate: EvolutionCandidate,
    original_content: str,
) -> str:
    """Build the LLM prompt for skill mutation."""
    truncated = original_content[:MAX_CONTENT_CHARS]
    evidence_text = "\n".join(f"- {e}" for e in candidate.evidence)

    if candidate.evolution_type == EvolutionType.FIX:
        return _fix_prompt(candidate.skill_name, truncated, evidence_text, candidate.effective_score)
    elif candidate.evolution_type == EvolutionType.DERIVED:
        return _derived_prompt(candidate.skill_name, truncated, evidence_text, candidate.effective_score)
    else:
        return _captured_prompt(candidate.skill_name, evidence_text)


def parse_mutation_output(output: str) -> Optional[str]:
    """Extract evolved SKILL.md content from LLM output.

    Looks for content between markdown code fences or after
    <EVOLUTION_COMPLETE> token. Returns None if <EVOLUTION_FAILED>
    or no valid content found.
    """
    if "<EVOLUTION_FAILED>" in output:
        return None

    # Try extracting from code fence
    fence_match = re.search(
        r"```(?:markdown|md)?\s*\n(---\s*\n.*?)```",
        output,
        re.DOTALL,
    )
    if fence_match:
        return fence_match.group(1).strip()

    # Try extracting after EVOLUTION_COMPLETE token
    complete_match = re.search(
        r"<EVOLUTION_COMPLETE>\s*(---\s*\n.*)",
        output,
        re.DOTALL,
    )
    if complete_match:
        return complete_match.group(1).strip()

    # Try finding YAML frontmatter directly
    frontmatter_match = re.search(
        r"(---\s*\nname:.*?)(?:\n---|\Z)",
        output,
        re.DOTALL,
    )
    if frontmatter_match:
        # Return everything from the frontmatter start
        idx = output.index(frontmatter_match.group(0))
        return output[idx:].strip()

    return None


# An evolved SKILL.md is auto-loaded into future sessions, so a mutation that
# hallucinated (or was prompt-injected into producing) code-execution or
# secret-exfiltration instructions must never be persisted. Structure checks
# alone are insufficient — reject dangerous content outright.
_SKILL_DENY_PATTERNS: tuple[str, ...] = (
    "os.environ", "subprocess", "exec(", "eval(", "__import__", "import os",
    "import subprocess", "pickle.loads", "curl ", "wget ", "rm -rf",
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AWS_SECRET", "/.ssh/", ".install_token",
)


def validate_skill_content(content: str) -> Optional[str]:
    """Validate evolved skill content. Returns error message or None if valid."""
    if not content or len(content) < 50:
        return "Content too short (< 50 chars)"
    if "---" not in content:
        return "Missing YAML frontmatter (no --- found)"
    if content.count("---") >= 2 and "name:" not in content.split("---")[1]:
        return "Missing 'name:' in frontmatter"
    lowered = content.lower()
    for pat in _SKILL_DENY_PATTERNS:
        if pat.lower() in lowered:
            return (f"Rejected: evolved skill contains a disallowed pattern "
                    f"({pat!r}) — skills must not execute code or access secrets")
    return None


def build_retry_prompt(original_prompt: str, error: str, attempt: int) -> str:
    """Build retry prompt after failed mutation attempt."""
    return (
        f"{original_prompt}\n\n"
        f"--- RETRY (attempt {attempt}/{MAX_APPLY_RETRIES}) ---\n"
        f"Previous output was invalid: {error}\n"
        f"Please generate a valid SKILL.md with proper YAML frontmatter "
        f"(--- / name: / description: / ---) followed by markdown instructions.\n"
        f"End with <EVOLUTION_COMPLETE> or <EVOLUTION_FAILED>."
    )


# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

def _fix_prompt(skill_name: str, content: str, evidence: str, score: float) -> str:
    return f"""You are a skill evolution engine. A skill is underperforming and needs repair.

SKILL NAME: {skill_name}
EFFECTIVE SCORE: {score:.0%} (approximate)

CURRENT SKILL CONTENT:
{content}

EVIDENCE OF PROBLEMS:
{evidence}

YOUR TASK:
Generate an improved version of this SKILL.md that addresses the identified problems.
Keep the same overall structure and purpose. Fix what's broken, don't rewrite from scratch.

OUTPUT FORMAT:
Return the complete improved SKILL.md content inside a markdown code fence.
The file must start with YAML frontmatter (--- / name: / description: / ---).
End your response with <EVOLUTION_COMPLETE> if you generated a valid improvement,
or <EVOLUTION_FAILED> if you cannot improve this skill."""


def _derived_prompt(skill_name: str, content: str, evidence: str, score: float) -> str:
    return f"""You are a skill evolution engine. A skill works for some tasks but not others.
Create a specialized variant for the failing task type.

PARENT SKILL: {skill_name}
EFFECTIVE SCORE: {score:.0%} (moderate — works sometimes, fails sometimes)

PARENT SKILL CONTENT:
{content}

EVIDENCE:
{evidence}

YOUR TASK:
Create a specialized DERIVED variant that handles the failing cases better.
Give it a new name (e.g., "{skill_name}-specialized" or a descriptive name).
Keep the parent's strengths. Add specific handling for the failure patterns.

OUTPUT FORMAT:
Return the complete new SKILL.md inside a markdown code fence.
Must start with YAML frontmatter (--- / name: / description: / ---).
End with <EVOLUTION_COMPLETE> or <EVOLUTION_FAILED>."""


def _captured_prompt(skill_name: str, evidence: str) -> str:
    return f"""You are a skill evolution engine. A repeated workflow pattern was detected
that no existing skill covers. Create a new skill to codify this pattern.

PATTERN NAME: {skill_name}
EVIDENCE:
{evidence}

YOUR TASK:
Create a new SKILL.md that codifies this workflow pattern into a reusable skill.
Make it specific and actionable — not generic advice.

OUTPUT FORMAT:
Return the complete SKILL.md inside a markdown code fence.
Must start with YAML frontmatter (--- / name: / description: / ---).
End with <EVOLUTION_COMPLETE> or <EVOLUTION_FAILED>."""
