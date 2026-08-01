# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3.3

"""PromptLifecycleManager — Ebbinghaus decay and effectiveness for soft prompts.

Applies Ebbinghaus forgetting curve to soft prompts. Higher effectiveness
and more versions (evidence proxy) slow decay. 48h floor prevents cold-start death.

[AUDIT FIX F-2] 48h floor on prompt strength.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.math.ebbinghaus import EbbinghausCurve
    from superlocalmemory.core.config import ParameterizationConfig

from superlocalmemory.parameterization.pattern_extractor import (
    PatternAssertion,
)

logger = logging.getLogger(__name__)

# Minimum prompt strength floor — prevents cold-start death
_PROMPT_STRENGTH_FLOOR = 48.0


class PromptLifecycleManager:
    """Manage soft prompt lifecycle: effectiveness tracking + Ebbinghaus decay.

    Effectiveness is updated via session feedback signals.
    Retention decays via Ebbinghaus curve with strength derived from
    effectiveness * version (evidence proxy).
    """

    def __init__(
        self,
        db: DatabaseManager,
        config: ParameterizationConfig,
        *,
        ebbinghaus: EbbinghausCurve | None = None,
    ) -> None:
        self._db = db
        self._config = config
        self._ebbinghaus = ebbinghaus

    # ------------------------------------------------------------------
    # Effectiveness tracking
    # ------------------------------------------------------------------

    def update_effectiveness(
        self,
        profile_id: str,
        category: str,
        signals: dict[str, float],
    ) -> float:
        """Update prompt effectiveness from feedback signals.

        Signals: followed, session_success (positive),
                 corrected, session_failure (negative).
        Uses exponential moving average: 0.7 * raw + 0.3 * current.

        Args:
            profile_id: Target profile.
            category: Prompt category to update.
            signals: Feedback signal dict.

        Returns:
            New effectiveness score in [0.0, 1.0].
        """
        rows = self._db.execute(
            "SELECT prompt_id, effectiveness, retention_score "
            "FROM soft_prompt_templates "
            "WHERE profile_id = ? AND category = ? AND active = 1",
            (profile_id, category),
        )
        if not rows:
            return 0.5

        row = rows[0]
        current_effectiveness = row.get("effectiveness", 0.5)
        prompt_id = row["prompt_id"]

        # Compute signal summary
        positive = (
            signals.get("followed", 0.0)
            + signals.get("session_success", 0.0)
        )
        negative = (
            signals.get("corrected", 0.0)
            + signals.get("session_failure", 0.0)
        )
        total = positive + negative + 1.0  # Laplace smoothing

        raw_score = positive / total
        new_effectiveness = 0.7 * raw_score + 0.3 * current_effectiveness
        new_effectiveness = max(0.0, min(1.0, new_effectiveness))

        self._db.execute(
            "UPDATE soft_prompt_templates "
            "SET effectiveness = ?, updated_at = datetime('now') "
            "WHERE prompt_id = ?",
            (new_effectiveness, prompt_id),
        )

        return new_effectiveness

    # ------------------------------------------------------------------
    # Ebbinghaus decay
    # ------------------------------------------------------------------

    def compute_prompt_retention(self, prompt_id: str) -> float:
        """Compute Ebbinghaus retention for a soft prompt.

        Strength = max(48h floor, 2 * effectiveness * version).
        R(t) = e^(-t/S) via EbbinghausCurve.

        Args:
            prompt_id: Prompt to compute retention for.

        Returns:
            Retention score in [0.0, 1.0], or 0.0 if not found.
        """
        rows = self._db.execute(
            "SELECT prompt_id, effectiveness, retention_score, "
            "version, created_at, updated_at "
            "FROM soft_prompt_templates WHERE prompt_id = ?",
            (prompt_id,),
        )
        if not rows:
            return 0.0

        row = rows[0]
        effectiveness = row.get("effectiveness", 0.5)
        version = max(1, row.get("version", 1))
        updated_at = row.get("updated_at", "")

        # Compute hours since last update
        now = datetime.now(timezone.utc)
        try:
            last_update = datetime.fromisoformat(updated_at)
            if last_update.tzinfo is None:
                last_update = last_update.replace(tzinfo=timezone.utc)
            hours = (now - last_update).total_seconds() / 3600.0
        except (ValueError, TypeError):
            hours = 0.0

        # Prompt strength: effectiveness * version proxy
        s_raw = 2.0 * effectiveness * version
        s_prompt = max(_PROMPT_STRENGTH_FLOOR, s_raw)

        if self._ebbinghaus is None:
            from superlocalmemory.math.ebbinghaus import EbbinghausCurve
            from superlocalmemory.core.config import ForgettingConfig
            self._ebbinghaus = EbbinghausCurve(ForgettingConfig())

        retention = self._ebbinghaus.retention(hours, s_prompt)
        return retention

    # ------------------------------------------------------------------
    # Lifecycle review
    # ------------------------------------------------------------------

    def run_lifecycle_review(self, profile_id: str) -> dict:
        """Periodic review: decay prompts, deactivate dead ones.

        Args:
            profile_id: Profile to review.

        Returns:
            Stats dict: {reviewed, decayed, removed, refreshed}.
        """
        rows = self._db.execute(
            "SELECT prompt_id, category, effectiveness, retention_score, "
            "version, created_at, updated_at "
            "FROM soft_prompt_templates "
            "WHERE profile_id = ? AND active = 1",
            (profile_id,),
        )

        stats = {"reviewed": 0, "decayed": 0, "removed": 0, "refreshed": 0}

        for row in rows:
            prompt_id = row["prompt_id"]
            old_retention = row.get("retention_score", 1.0)
            stats["reviewed"] += 1

            new_retention = self.compute_prompt_retention(prompt_id)

            if new_retention < 0.1:
                # Deactivate
                self._db.execute(
                    "UPDATE soft_prompt_templates "
                    "SET active = 0, retention_score = ?, "
                    "updated_at = datetime('now') WHERE prompt_id = ?",
                    (new_retention, prompt_id),
                )
                stats["removed"] += 1
                logger.info(
                    "Prompt %s (category=%s) deactivated: retention %.3f < 0.1",
                    prompt_id, row.get("category", "?"), new_retention,
                )
            else:
                # Update retention score
                self._db.execute(
                    "UPDATE soft_prompt_templates "
                    "SET retention_score = ?, updated_at = datetime('now') "
                    "WHERE prompt_id = ?",
                    (new_retention, prompt_id),
                )
                delta = abs(new_retention - old_retention)
                if delta > 0.1:
                    stats["decayed"] += 1

        return stats

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def evolve_prompt(
        self,
        profile_id: str,
        category: str,
        new_pattern: PatternAssertion,
    ) -> str:
        """Handle preference evolution: new pattern vs existing prompt.

        Args:
            profile_id: Target profile.
            category: Category being evolved.
            new_pattern: New pattern assertion to compare.

        Returns:
            "new" | "replaced" | "kept_existing" | "user_review_needed"
        """
        rows = self._db.execute(
            "SELECT prompt_id, category, confidence, effectiveness "
            "FROM soft_prompt_templates "
            "WHERE profile_id = ? AND category = ? AND active = 1",
            (profile_id, category),
        )

        if not rows:
            return "new"

        current = rows[0]
        current_confidence = current.get("confidence", 0.0)

        if new_pattern.confidence > current_confidence + 0.1:
            # New pattern clearly better — replace
            self._db.execute(
                "UPDATE soft_prompt_templates "
                "SET active = 0, updated_at = datetime('now') "
                "WHERE prompt_id = ?",
                (current["prompt_id"],),
            )
            return "replaced"

        if new_pattern.confidence < current_confidence - 0.1:
            logger.info(
                "Keeping existing prompt for %s (conf=%.2f > new=%.2f)",
                category, current_confidence, new_pattern.confidence,
            )
            return "kept_existing"

        # Close confidence — needs user review
        logger.warning(
            "Equal confidence conflict in %s for %s. User review needed.",
            category, new_pattern.key,
        )
        return "user_review_needed"
