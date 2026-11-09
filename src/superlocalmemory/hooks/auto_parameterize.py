# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3.3

"""AutoParameterizeHook — Trigger parameterization on consolidation events.

Runs the full pipeline: extract -> generate -> store -> lifecycle review.
Rate-limited to config.refresh_interval_hours. Tracks session effectiveness.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superlocalmemory.core.config import ParameterizationConfig
    from superlocalmemory.parameterization.pattern_extractor import PatternExtractor
    from superlocalmemory.parameterization.prompt_injector import PromptInjector
    from superlocalmemory.parameterization.prompt_lifecycle import PromptLifecycleManager
    from superlocalmemory.parameterization.soft_prompt_generator import SoftPromptGenerator

logger = logging.getLogger(__name__)


class AutoParameterizeHook:
    """Hook that triggers soft prompt parameterization on consolidation.

    Called by the consolidation engine after a consolidation cycle completes.
    Rate-limited to prevent excessive recomputation.
    """

    def __init__(
        self,
        extractor: PatternExtractor,
        generator: SoftPromptGenerator,
        injector: PromptInjector,
        lifecycle: PromptLifecycleManager,
        config: ParameterizationConfig,
    ) -> None:
        self._extractor = extractor
        self._generator = generator
        self._injector = injector
        self._lifecycle = lifecycle
        self._config = config
        self._last_run: str | None = None

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_consolidation_complete(self, profile_id: str) -> dict:
        """Triggered after consolidation engine finishes.

        Runs: extract -> generate -> store -> lifecycle review.

        Args:
            profile_id: Profile that was consolidated.

        Returns:
            Status dict with pipeline results.
        """
        if not self._config.enabled:
            return {"status": "disabled"}

        # Rate limit check
        if self._last_run is not None:
            try:
                last = datetime.fromisoformat(self._last_run)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                interval = timedelta(hours=self._config.refresh_interval_hours)
                if now - last < interval:
                    return {"status": "rate_limited"}
            except (ValueError, TypeError):
                pass

        # Step 1: Extract patterns
        patterns = self._extractor.extract(profile_id)
        if not patterns:
            return {"status": "no_patterns", "count": 0}

        # Step 2: Generate prompts
        prompts = self._generator.generate(patterns, profile_id)
        if not prompts:
            return {"status": "no_prompts", "patterns": len(patterns)}

        # Step 3: Store prompts. A failure here must not skip step 4: it once
        # ended every run (an unstorable category), so no stale prompt was ever
        # retired, and it was only logged at debug level by the caller.
        try:
            stored = self._injector.store_prompts(prompts)
        except Exception as exc:  # noqa: BLE001 - the review below still runs
            logger.warning("Soft prompts not stored for %s: %s", profile_id, exc)
            stored = 0

        # Step 4: Run lifecycle review
        lifecycle_stats = self._lifecycle.run_lifecycle_review(profile_id)

        # Update last run timestamp
        self._last_run = datetime.now(timezone.utc).isoformat()

        return {
            "status": "success",
            "patterns": len(patterns),
            "prompts": stored,
            "lifecycle": lifecycle_stats,
        }

    def on_session_end(
        self, profile_id: str, session_outcome: str,
    ) -> None:
        """Triggered at session end for effectiveness tracking.

        Maps session outcome to feedback signals and updates
        effectiveness for all active prompt categories.

        Args:
            profile_id: Profile for the ending session.
            session_outcome: "success" | "failure" | "partial"
        """
        if not self._config.effectiveness_tracking:
            return

        # Map outcome to signals
        signal_map: dict[str, dict[str, float]] = {
            "success": {"session_success": 1.0},
            "failure": {"session_failure": 1.0},
            "partial": {"session_success": 0.5},
        }
        signals = signal_map.get(session_outcome, {})
        if not signals:
            return

        # Update effectiveness for all active categories
        for category in [
            "identity", "tech_preference", "communication_style",
            "workflow_pattern", "project_context", "decision_history",
            "avoidance",
        ]:
            try:
                self._lifecycle.update_effectiveness(
                    profile_id, category, signals,
                )
            except Exception as exc:
                logger.debug(
                    "Failed to update effectiveness for %s: %s",
                    category, exc,
                )
