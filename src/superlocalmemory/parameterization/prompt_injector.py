# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3.3

"""PromptInjector — Inject soft prompts into context with token budget.

Priority: soft prompts > regular memories (soft prompts come first).
Token budget split: prompt_budget (500 default) + memory_budget (1500 default).
Dedup with existing context.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.core.config import ParameterizationConfig

from superlocalmemory.parameterization.soft_prompt_generator import (
    SoftPromptGenerator,
    SoftPromptTemplate,
)

logger = logging.getLogger(__name__)


class PromptInjector:
    """Inject soft prompts into conversation context.

    Handles retrieval from DB, token budget enforcement, and
    combination with regular memory context.
    """

    def __init__(
        self,
        db: DatabaseManager,
        generator: SoftPromptGenerator,
        config: ParameterizationConfig,
    ) -> None:
        self._db = db
        self._generator = generator
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_injection_context(self, profile_id: str) -> str:
        """Retrieve stored soft prompts, assemble for injection.

        Args:
            profile_id: Profile to load prompts for.

        Returns:
            Assembled soft prompt text, or "" if disabled/empty.
        """
        if not self._config.enabled:
            return ""

        rows = self._db.execute(
            "SELECT prompt_id, category, content, confidence, "
            "token_count, retention_score, profile_id, "
            "source_pattern_ids, effectiveness, active, version "
            "FROM soft_prompt_templates "
            "WHERE profile_id = ? AND active = 1 AND retention_score >= 0.1 "
            "ORDER BY confidence DESC",
            (profile_id,),
        )

        if not rows:
            return ""

        templates: list[SoftPromptTemplate] = []
        for row in rows:
            # DatabaseManager returns sqlite3.Row which lacks .get() — convert to dict.
            r = dict(row)
            raw_ids = r.get("source_pattern_ids", "[]")
            try:
                source_ids = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
            except (json.JSONDecodeError, TypeError):
                source_ids = []

            templates.append(SoftPromptTemplate(
                prompt_id=r["prompt_id"],
                profile_id=r.get("profile_id", profile_id),
                category=r["category"],
                content=r["content"],
                source_pattern_ids=source_ids,
                confidence=r["confidence"],
                effectiveness=r.get("effectiveness", 0.5),
                token_count=r.get("token_count", 0),
                retention_score=r.get("retention_score", 1.0),
                active=bool(r.get("active", 1)),
                version=r.get("version", 1),
            ))

        # Budget enforcement
        total = sum(t.token_count for t in templates)
        if total > self._config.max_prompt_tokens:
            selected: list[SoftPromptTemplate] = []
            accumulated = 0
            for t in templates:  # Already sorted by confidence DESC
                if accumulated + t.token_count > self._config.max_prompt_tokens:
                    break
                selected.append(t)
                accumulated += t.token_count
            templates = selected

        return self._generator.assemble(templates)

    def inject_into_context(
        self,
        soft_prompt_text: str,
        memory_context: str,
        total_budget: int | None = None,
    ) -> str:
        """Combine soft prompts with regular memory context.

        Args:
            soft_prompt_text: Assembled soft prompt text.
            memory_context: Regular memory recall context.
            total_budget: Optional total token budget override.

        Returns:
            Combined context string with soft prompts first.
        """
        budget = total_budget or (
            self._config.max_prompt_tokens + self._config.max_memory_tokens
        )
        prompt_budget = self._config.max_prompt_tokens
        memory_budget = self._config.max_memory_tokens

        # Trim soft prompt to its budget
        if soft_prompt_text:
            soft_tokens = SoftPromptGenerator._estimate_tokens(soft_prompt_text)
            if soft_tokens > prompt_budget:
                soft_prompt_text = SoftPromptGenerator._trim_to_tokens(
                    soft_prompt_text, prompt_budget,
                )

        # Trim memory context to its budget
        if memory_context:
            mem_tokens = SoftPromptGenerator._estimate_tokens(memory_context)
            if mem_tokens > memory_budget:
                memory_context = SoftPromptGenerator._trim_to_tokens(
                    memory_context, memory_budget,
                )

        if not soft_prompt_text:
            return memory_context
        if not memory_context:
            return soft_prompt_text

        return f"{soft_prompt_text}\n\n{memory_context}"

    def store_prompts(
        self, templates: list[SoftPromptTemplate],
    ) -> int:
        """Persist generated soft prompts, deactivating old versions.

        Args:
            templates: Generated templates to store.

        Returns:
            Number of templates stored.
        """
        count = 0
        for template in templates:
            # Deactivate existing for same (profile_id, category)
            self._db.execute(
                "UPDATE soft_prompt_templates SET active = 0, "
                "updated_at = datetime('now') "
                "WHERE profile_id = ? AND category = ? AND active = 1",
                (template.profile_id, template.category),
            )

            # Get max version for this (profile_id, category)
            version_rows = self._db.execute(
                "SELECT COALESCE(MAX(version), 0) AS max_version "
                "FROM soft_prompt_templates "
                "WHERE profile_id = ? AND category = ?",
                (template.profile_id, template.category),
            )
            max_version = 0
            if version_rows:
                row = version_rows[0]
                max_version = (dict(row) if hasattr(row, "keys") else {"max_version": row[0]}).get("max_version", 0) or 0
            new_version = max_version + 1

            # Insert new prompt
            self._db.execute(
                "INSERT INTO soft_prompt_templates "
                "(prompt_id, profile_id, category, content, "
                "source_pattern_ids, confidence, effectiveness, "
                "token_count, retention_score, active, version, "
                "created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, "
                "datetime('now'), datetime('now'))",
                (
                    template.prompt_id,
                    template.profile_id,
                    template.category,
                    template.content,
                    json.dumps(template.source_pattern_ids),
                    template.confidence,
                    template.effectiveness,
                    template.token_count,
                    template.retention_score,
                    new_version,
                ),
            )
            count += 1

        return count
