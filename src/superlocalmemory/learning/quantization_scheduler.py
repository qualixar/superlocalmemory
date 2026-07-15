# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Quantization Scheduler -- combined SAGQ + EAP precision management.

Background quantization worker that periodically reviews all memories and
applies both EAP (forgetting) and SAGQ (network centrality) signals together.

Conflict resolution: max(EAP_precision, SAGQ_precision) -- safety first.
Core Memory blocks are immune to quantization (HR-01).
Every precision change is logged to fact_access_log for audit trail (HR-09).

HR-01: Core Memory immune.
HR-07: No-op when config.enabled=False.
HR-08: Synchronous execution (no threading).
HR-09: Audit trail for every change.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, TYPE_CHECKING

from superlocalmemory.storage.models import _new_id

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.dynamics.activation_guided_quantization import (
        ActivationGuidedQuantizer,
        SAGQPrecision,
    )
    from superlocalmemory.core.config import SAGQConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes (frozen -- all immutable, Rule 10)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecisionChange:
    """Record of a single precision change for audit trail."""

    fact_id: str
    old_bit_width: int
    new_bit_width: int
    action: str       # "upgrade" | "downgrade"
    centrality: float
    sagq_signal: int
    eap_signal: int
    timestamp: str    # ISO 8601 datetime string


@dataclass(frozen=True)
class SchedulerRunResult:
    """Result of a single scheduler run."""

    total_facts: int
    upgrades: int
    downgrades: int
    skipped: int
    errors: int
    changes: tuple[PrecisionChange, ...]
    duration_ms: float


# ---------------------------------------------------------------------------
# Bit-width -> quantization level mapping
# ---------------------------------------------------------------------------

_BW_TO_LEVEL: dict[int, str] = {
    32: "float32",
    8: "int8",
    4: "polar4",
    2: "polar2",
    0: "deleted",
}


# ---------------------------------------------------------------------------
# QuantizationScheduler
# ---------------------------------------------------------------------------


class QuantizationScheduler:
    """Combined SAGQ + EAP quantization scheduler.

    Runs synchronously (HR-08). Called by the consolidation engine
    or CLI command on schedule.
    """

    def __init__(
        self,
        db: Any,
        sagq: Any,
        eap_mapper: Callable[[str], int],
        quantized_store: Any,
        vector_store: Any,
        config: Any,
    ) -> None:
        """Initialize scheduler. No side effects."""
        self._db = db
        self._sagq = sagq
        self._eap_mapper = eap_mapper
        self._quantized_store = quantized_store
        self._vector_store = vector_store
        self._config = config

    def run(self, profile_id: str) -> SchedulerRunResult:
        """Execute one combined SAGQ + EAP quantization pass.

        Algorithm:
          1. Compute SAGQ precision recommendations (centrality + EAP + max())
          2. Exclude core memory facts (HR-01)
          3. Execute upgrades/downgrades with error isolation
          4. Log audit trail for each change (HR-09)
          5. Return summary

        Returns SchedulerRunResult with totals and change records.
        """
        # HR-07: No-op when disabled
        if not self._config.enabled:
            return SchedulerRunResult(
                total_facts=0, upgrades=0, downgrades=0,
                skipped=0, errors=0, changes=(), duration_ms=0.0,
            )

        start_time = time.monotonic()

        # Step 2: Get SAGQ precision recommendations
        recommendations = self._sagq.compute_sagq_precision_batch(
            profile_id, self._eap_mapper,
        )

        if not recommendations:
            duration_ms = (time.monotonic() - start_time) * 1000
            return SchedulerRunResult(
                total_facts=0, upgrades=0, downgrades=0,
                skipped=0, errors=0, changes=(), duration_ms=duration_ms,
            )

        # Step 5a: Get core memory fact IDs (HR-01)
        core_fact_ids = self._get_core_fact_ids(profile_id)

        # Step 6: Process each recommendation
        upgrades = 0
        downgrades = 0
        skipped = 0
        errors = 0
        changes: list[PrecisionChange] = []

        for prec in recommendations:
            # HR-01: Core Memory immune
            if prec.fact_id in core_fact_ids:
                skipped += 1
                continue

            if prec.action == "skip":
                skipped += 1
                continue

            change = self._process_precision_change(prec, profile_id)

            if change is None:
                # Error or unable to process
                if prec.action in ("downgrade", "upgrade"):
                    errors += 1
                else:
                    skipped += 1
                continue

            if change.action == "downgrade":
                downgrades += 1
            elif change.action == "upgrade":
                upgrades += 1

            changes.append(change)

        duration_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "SAGQ scheduler: %d upgrades, %d downgrades, %d skipped, "
            "%d errors in %.1fms",
            upgrades, downgrades, skipped, errors, duration_ms,
        )

        return SchedulerRunResult(
            total_facts=len(recommendations),
            upgrades=upgrades,
            downgrades=downgrades,
            skipped=skipped,
            errors=errors,
            changes=tuple(changes),
            duration_ms=duration_ms,
        )

    def _process_precision_change(
        self, prec: Any, profile_id: str,
    ) -> PrecisionChange | None:
        """Process a single precision change with error isolation.

        Each fact is independent -- one failure does not block others.
        Returns PrecisionChange on success, None on failure.
        """
        try:
            if prec.action == "downgrade":
                # Fetch float32 embedding
                emb = self._vector_store.get_embedding(prec.fact_id, profile_id)
                if emb is None:
                    logger.warning(
                        "SAGQ: No float32 for %s, skip downgrade", prec.fact_id,
                    )
                    return None
                # Compress to target bit-width
                self._quantized_store.compress_fact(
                    prec.fact_id, profile_id, emb, prec.final_bit_width,
                )

            elif prec.action == "upgrade":
                # Upgrade = re-compress at higher bit_width from float32 backup
                emb = self._vector_store.get_embedding(prec.fact_id, profile_id)
                if emb is None:
                    logger.warning(
                        "SAGQ: No float32 for %s, skip upgrade", prec.fact_id,
                    )
                    return None
                self._quantized_store.compress_fact(
                    prec.fact_id, profile_id, emb, prec.final_bit_width,
                )

            else:
                return None  # skip

            # Update embedding_metadata (Q7)
            level = self._bit_width_to_quantization_level(prec.final_bit_width)
            self._db.execute(
                "UPDATE embedding_metadata "
                "SET bit_width = ?, quantization_level = ? "
                "WHERE fact_id = ? AND profile_id = ?",
                (prec.final_bit_width, level, prec.fact_id, profile_id),
            )

            # Audit trail (Q10 -- HR-09)
            self._db.execute(
                "INSERT INTO fact_access_log "
                "(log_id, fact_id, profile_id, accessed_at, access_type, session_id) "
                "VALUES (?, ?, ?, datetime('now'), 'consolidation', 'sagq_scheduler')",
                (_new_id(), prec.fact_id, profile_id),
            )

            now_iso = datetime.now(UTC).isoformat()
            return PrecisionChange(
                fact_id=prec.fact_id,
                old_bit_width=prec.current_bit_width,
                new_bit_width=prec.final_bit_width,
                action=prec.action,
                centrality=prec.centrality,
                sagq_signal=prec.sagq_bit_width,
                eap_signal=prec.eap_bit_width,
                timestamp=now_iso,
            )

        except Exception as exc:
            logger.error(
                "SAGQ: precision change failed for %s: %s", prec.fact_id, exc,
            )
            return None

    def _get_core_fact_ids(self, profile_id: str) -> set[str]:
        """Get fact IDs referenced by core_memory_blocks (immune to quantization).

        Uses json_each() to extract from source_fact_ids JSON array (Q8).
        """
        try:
            rows = self._db.execute(
                "SELECT json_each.value as fact_id "
                "FROM core_memory_blocks, json_each(core_memory_blocks.source_fact_ids) "
                "WHERE core_memory_blocks.profile_id = ?",
                (profile_id,),
            )
            return {dict(r)["fact_id"] for r in rows}
        except Exception as exc:
            logger.debug("SAGQ: core_memory_blocks query failed: %s", exc)
            return set()

    def _bit_width_to_quantization_level(self, bit_width: int) -> str:
        """Map bit-width integer to quantization level string."""
        return _BW_TO_LEVEL.get(bit_width, "float32")

    def should_run(self, last_run_at: str | None) -> bool:
        """Check if enough time has passed since the last run.

        Args:
            last_run_at: ISO 8601 datetime of last run, or None if never run.

        Returns True if the scheduler should run now.
        """
        if last_run_at is None:
            return True

        try:
            last_run = datetime.fromisoformat(last_run_at)
            now = datetime.now(UTC)
            # Ensure both are timezone-aware for subtraction
            if last_run.tzinfo is None:
                last_run = last_run.replace(tzinfo=UTC)
            hours_since = (now - last_run).total_seconds() / 3600
            return hours_since >= self._config.scheduler_interval_hours
        except (ValueError, TypeError) as exc:
            logger.warning("SAGQ: Could not parse last_run_at '%s': %s", last_run_at, exc)
            return True
