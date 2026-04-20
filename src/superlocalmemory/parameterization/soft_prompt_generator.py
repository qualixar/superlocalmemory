# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3.3

"""SoftPromptGenerator — Convert extracted patterns to text soft prompts.

Pure text personality encoding. No LoRA, no model weights.
Token budget management with priority ordering by category.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superlocalmemory.core.config import ParameterizationConfig

from superlocalmemory.parameterization.pattern_extractor import (
    PatternAssertion,
    PatternCategory,
)
from superlocalmemory.parameterization.pii_filter import PIIFilter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category templates — natural language patterns for each category
# ---------------------------------------------------------------------------

CATEGORY_TEMPLATES: dict[str, str] = {
    "identity": (
        "The user is {role} with expertise in {domains}. "
        "They work at {organization}."
    ),
    "tech_preference": (
        "The user's preferred technology stack includes: {technologies}. "
        "Default to these when generating code or making recommendations."
    ),
    "communication_style": (
        "The user prefers {style} responses. {specific_preferences}"
    ),
    "workflow_pattern": (
        "The user typically {workflow_description}. "
        "Anticipate this workflow when assisting."
    ),
    "project_context": (
        "Current active project: {project_name}. "
        "Key context: {context_summary}."
    ),
    "decision_history": (
        "Recent key decisions: {decisions}. "
        "These reflect the user's current direction."
    ),
    "avoidance": (
        "The user has explicitly asked to avoid: {avoid_list}. "
        "Do not suggest or use these."
    ),
    # v3.4.7: Behavioral assertions — learned patterns from tool usage
    "behavioral": (
        "Learned behavior: When {trigger}, {action}. "
        "(Confidence: {confidence}%, based on {evidence} observations)"
    ),
}

CATEGORY_PRIORITY_ORDER: list[str] = [
    "identity",
    "tech_preference",
    "communication_style",
    "behavioral",  # v3.4.7: behavioral assertions after communication style
    "workflow_pattern",
    "project_context",
    "decision_history",
    "avoidance",
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SoftPromptTemplate:
    """A generated soft prompt for one category."""

    prompt_id: str
    profile_id: str
    category: str
    content: str
    source_pattern_ids: list[str]
    confidence: float
    effectiveness: float
    token_count: int
    retention_score: float
    active: bool
    version: int


# ---------------------------------------------------------------------------
# SoftPromptGenerator class
# ---------------------------------------------------------------------------

class SoftPromptGenerator:
    """Convert extracted pattern assertions into natural language soft prompts.

    Respects token budget and priority ordering. Filters PII.
    """

    def __init__(self, config: ParameterizationConfig) -> None:
        self._config = config
        self._pii_filter = PIIFilter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        patterns: list[PatternAssertion],
        profile_id: str,
        *,
        high_reward_source_ids: set[str] | None = None,
    ) -> list[SoftPromptTemplate]:
        """Master generation pipeline: filter, group, render, budget-trim.

        Args:
            patterns: Extracted pattern assertions.
            profile_id: Target profile.
            high_reward_source_ids: Optional v3.4.22 (LLD-12 §6) filter —
                when provided, only patterns whose source_ids intersect
                this set are considered. When None (default), behaviour
                matches pre-v3.4.22 and every pattern flows through.

        Returns:
            List of SoftPromptTemplate, ordered by category priority,
            within token budget.
        """
        # Filter to enabled categories
        enabled = set(self._config.categories_enabled)
        filtered = [
            p for p in patterns if p.category.value in enabled
        ]

        # v3.4.22 (LLD-12 §6): reward-aware filter — opt-in only.
        if high_reward_source_ids is not None:
            filtered = [
                p for p in filtered
                if set(p.source_ids) & high_reward_source_ids
            ]

        # Group by category
        grouped: dict[str, list[PatternAssertion]] = defaultdict(list)
        for p in filtered:
            grouped[p.category.value].append(p)

        prompts: list[SoftPromptTemplate] = []
        # Reserve tokens for assembly header + separators
        header_overhead = self._estimate_tokens(
            "# User Profile (auto-learned)"
        ) + 2  # newlines
        running_token_count = header_overhead

        for category in CATEGORY_PRIORITY_ORDER:
            if category not in grouped:
                continue

            template = self._render_category(
                category, grouped[category], profile_id,
            )
            if template is None:
                continue

            token_count = self._estimate_tokens(template.content)

            if running_token_count + token_count > self._config.max_prompt_tokens:
                remaining = self._config.max_prompt_tokens - running_token_count
                template = self._trim_content(template, remaining)
                token_count = self._estimate_tokens(template.content)
                if not template.content or len(template.content) < 20:
                    break

            template.token_count = token_count
            prompts.append(template)
            running_token_count += token_count

        return self._validate(prompts)

    def assemble(self, prompts: list[SoftPromptTemplate]) -> str:
        """Assemble prompts into a single text block with header.

        Args:
            prompts: List of templates to assemble.

        Returns:
            Assembled text or empty string if no prompts.
        """
        if not prompts:
            return ""

        lines = ["# User Profile (auto-learned)", ""]

        # Sort by priority order
        order_map = {
            cat: idx for idx, cat in enumerate(CATEGORY_PRIORITY_ORDER)
        }
        sorted_prompts = sorted(
            prompts,
            key=lambda p: order_map.get(p.category, len(CATEGORY_PRIORITY_ORDER)),
        )

        for prompt in sorted_prompts:
            lines.append(prompt.content)
            lines.append("")

        result = "\n".join(lines).rstrip()
        return result

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_category(
        self,
        category: str,
        patterns: list[PatternAssertion],
        profile_id: str,
    ) -> SoftPromptTemplate | None:
        """Render patterns for a single category into a SoftPromptTemplate."""
        template_str = CATEGORY_TEMPLATES.get(category, "{key}: {value}")
        values = self._extract_template_values(category, patterns)

        # Fill template with defaultdict for missing keys
        safe_values = defaultdict(str, values)
        try:
            content = template_str.format_map(safe_values)
        except (KeyError, IndexError, ValueError):  # pragma: no cover
            content = ", ".join(p.value for p in patterns)  # pragma: no cover

        # Clean up
        content = self._clean_content(content)

        # Filter PII
        content = self._pii_filter.filter_text(content)
        if not content.strip():
            return None

        # Trim to 100 tokens per category
        content = self._trim_to_tokens(content, 100)

        # Aggregate confidence
        confidence = (
            sum(p.confidence for p in patterns) / len(patterns)
            if patterns else 0.0
        )

        # Collect source IDs
        source_ids = [sid for p in patterns for sid in p.source_ids]

        return SoftPromptTemplate(
            prompt_id=str(uuid.uuid4()),
            profile_id=profile_id,
            category=category,
            content=content,
            source_pattern_ids=source_ids,
            confidence=confidence,
            effectiveness=0.5,
            token_count=0,
            retention_score=1.0,
            active=True,
            version=1,
        )

    @staticmethod
    def _extract_template_values(
        category: str,
        patterns: list[PatternAssertion],
    ) -> dict[str, str]:
        """Extract placeholder values from patterns for a category template."""
        values: dict[str, str] = {}
        pat_values = [p.value for p in patterns]

        if category == "identity":
            values["role"] = pat_values[0] if pat_values else ""
            values["domains"] = ", ".join(pat_values)
            org_patterns = [
                p for p in patterns
                if "organization" in p.key or "company" in p.key
            ]
            values["organization"] = (
                org_patterns[0].value if org_patterns else ""
            )

        elif category == "tech_preference":
            values["technologies"] = ", ".join(pat_values)

        elif category == "communication_style":
            values["style"] = pat_values[0] if pat_values else ""
            values["specific_preferences"] = ", ".join(pat_values[1:])

        elif category == "workflow_pattern":
            values["workflow_description"] = "; ".join(pat_values)

        elif category == "project_context":
            values["project_name"] = pat_values[0] if pat_values else ""
            values["context_summary"] = ", ".join(pat_values[1:])

        elif category == "decision_history":
            values["decisions"] = ", ".join(pat_values)

        elif category == "avoidance":
            values["avoid_list"] = ", ".join(pat_values)

        else:
            # Generic fallback
            values["key"] = patterns[0].key if patterns else ""
            values["value"] = ", ".join(pat_values)

        return values

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(
        self, prompts: list[SoftPromptTemplate],
    ) -> list[SoftPromptTemplate]:
        """Validate prompts: no empty content, no PII, within token limits."""
        valid: list[SoftPromptTemplate] = []
        seen_categories: set[str] = set()

        for prompt in prompts:
            if not prompt.content.strip():
                logger.warning(
                    "Skipping empty prompt for category %s", prompt.category,
                )
                continue
            if self._pii_filter.has_pii(prompt.content):
                logger.warning(
                    "PII detected in prompt for category %s, skipping",
                    prompt.category,
                )
                continue
            if prompt.token_count > 100:
                logger.warning(
                    "Prompt for %s exceeds 100 tokens (%d), trimming",
                    prompt.category, prompt.token_count,
                )
                prompt.content = self._trim_to_tokens(prompt.content, 100)
                prompt.token_count = self._estimate_tokens(prompt.content)
            if prompt.category in seen_categories:
                logger.warning(
                    "Duplicate active category %s, skipping", prompt.category,
                )
                continue
            seen_categories.add(prompt.category)
            valid.append(prompt)

        return valid

    # ------------------------------------------------------------------
    # Token utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count from text (~1.3 tokens per word)."""
        word_count = len(text.split())
        return max(1, int(word_count * 1.3))

    @staticmethod
    def _trim_to_tokens(text: str, max_tokens: int) -> str:
        """Trim text to fit within a token budget by removing sentences."""
        current = SoftPromptGenerator._estimate_tokens(text)
        if current <= max_tokens:
            return text

        # Split into sentences
        import re
        sentences = re.split(r"(?<=\.)\s+", text)
        result_sentences: list[str] = []
        accumulated = 0

        for sentence in sentences:
            sent_tokens = SoftPromptGenerator._estimate_tokens(sentence)
            if accumulated + sent_tokens > max_tokens:
                break
            result_sentences.append(sentence)
            accumulated += sent_tokens

        if not result_sentences:
            # Take first N words if no sentence fits
            words = text.split()
            target_words = max(1, int(max_tokens / 1.3))
            return " ".join(words[:target_words]).rstrip(".,;: ") + "."

        result = " ".join(result_sentences)
        if not result.endswith("."):  # pragma: no cover — regex split keeps "." with chunks
            result += "."  # pragma: no cover
        return result

    @staticmethod
    def _trim_content(
        template: SoftPromptTemplate,
        remaining_budget: int,
    ) -> SoftPromptTemplate:
        """Create a new template with content trimmed to budget (immutable)."""
        trimmed = SoftPromptGenerator._trim_to_tokens(
            template.content, remaining_budget,
        )
        return SoftPromptTemplate(
            prompt_id=template.prompt_id,
            profile_id=template.profile_id,
            category=template.category,
            content=trimmed,
            source_pattern_ids=template.source_pattern_ids,
            confidence=template.confidence,
            effectiveness=template.effectiveness,
            token_count=SoftPromptGenerator._estimate_tokens(trimmed),
            retention_score=template.retention_score,
            active=template.active,
            version=template.version,
        )

    # ------------------------------------------------------------------
    # Content cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_content(content: str) -> str:
        """Strip extra whitespace, remove empty sentences."""
        import re
        # Collapse multiple spaces
        content = re.sub(r"\s+", " ", content).strip()
        # Remove empty sentences (". .")
        content = re.sub(r"\.\s*\.", ".", content)
        # Ensure ends with period
        if content and not content.endswith("."):
            content += "."
        return content
