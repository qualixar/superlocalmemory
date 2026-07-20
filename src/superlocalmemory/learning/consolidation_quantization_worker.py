# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""CCQ Worker — background scheduling for Cognitive Consolidation Quantization.

Wraps CognitiveConsolidator with scheduling logic:
  - Runs every N stores (store_count_trigger, default 100)
  - Runs on session end
  - Respects enabled flag

Integration point: ConsolidationEngine._step7_ccq() calls this worker.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superlocalmemory.core.config import CCQConfig
    from superlocalmemory.encoding.cognitive_consolidator import (
        CCQPipelineResult,
        CognitiveConsolidator,
    )

logger = logging.getLogger(__name__)


class CCQWorker:
    """Background CCQ scheduling and execution.

    Wraps CognitiveConsolidator.run_pipeline() with trigger logic.
    """

    __slots__ = ("_consolidator", "_config", "_run_count")

    def __init__(
        self,
        consolidator: CognitiveConsolidator,
        config: CCQConfig,
    ) -> None:
        self._consolidator = consolidator
        self._config = config
        self._run_count: int = 0

    def run(self, profile_id: str) -> CCQPipelineResult:
        """Execute one CCQ pipeline run.

        Returns empty result if disabled. Otherwise delegates to
        CognitiveConsolidator.run_pipeline().
        """
        from superlocalmemory.encoding.cognitive_consolidator import (
            CCQPipelineResult,
        )

        if not self._config.enabled:
            return CCQPipelineResult(
                clusters_processed=0,
                blocks_created=0,
                facts_archived=0,
                total_bytes_before=0,
                total_bytes_after=0,
                compression_ratio=0.0,
                audit_entries=(),
                errors=(),
            )

        self._run_count += 1
        result = self._consolidator.run_pipeline(profile_id)

        logger.info(
            "CCQ run #%d: clusters=%d, blocks=%d, archived=%d, ratio=%.2f",
            self._run_count,
            result.clusters_processed,
            result.blocks_created,
            result.facts_archived,
            result.compression_ratio,
        )

        return result

    def should_run(self, store_count: int, is_session_end: bool) -> bool:
        """Determine if CCQ should run now.

        Args:
            store_count: Current store count since last trigger.
            is_session_end: Whether the current session is ending.

        Returns:
            True if CCQ should execute.
        """
        if not self._config.enabled:
            return False

        if is_session_end and self._config.run_on_session_end:
            return True

        if (
            store_count > 0
            and store_count % self._config.store_count_trigger == 0
        ):
            return True

        return False

    def get_stats(self) -> dict[str, Any]:
        """Return worker statistics."""
        return {
            "total_runs": self._run_count,
            "enabled": self._config.enabled,
        }
