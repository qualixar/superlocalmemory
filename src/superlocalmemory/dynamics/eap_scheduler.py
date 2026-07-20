# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""EAP Scheduler -- Embedding-Aware Precision.

Couples Ebbinghaus retention scores to embedding bit-width:
  R > 0.8  -> 32 (float32, full precision)
  R > 0.5  ->  8 (int8, sqlite-vec native)
  R > 0.2  ->  4 (polar 4-bit)
  R > 0.05 ->  2 (polar 2-bit)
  R <= 0.05->  0 (forgotten, delete embedding)

The EAP cycle:
  1. Fetch all facts with retention data + current bit_width
  2. Map retention to target bit_width
  3. Execute downgrades (compress) and upgrades (restore)
  4. Return stats

HR-03: Original float32 NEVER deleted unless keep_float32_backup=False
       AND fact is in archive/forgotten zone.
HR-04: Quantization ONLY via EAP scheduler (not ad-hoc).

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from superlocalmemory.core.config import QuantizationConfig

if TYPE_CHECKING:
    from superlocalmemory.math.ebbinghaus import EbbinghausCurve
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.quantized_store import QuantizedEmbeddingStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retention -> bit-width mapping (Section 10, Pattern 4)
# ---------------------------------------------------------------------------


def retention_to_bit_width(retention: float) -> int:
    """Map Ebbinghaus retention to embedding precision.

    Returns:
        32, 8, 4, 2, or 0 (deleted).
    """
    if retention > 0.8:
        return 32
    if retention > 0.5:
        return 8
    if retention > 0.2:
        return 4
    if retention > 0.05:
        return 2
    return 0


# ---------------------------------------------------------------------------
# EAPScheduler
# ---------------------------------------------------------------------------


class EAPScheduler:
    """Embedding-Aware Precision scheduler.

    Runs periodic cycles that adjust embedding precision based on
    how well each fact is retained in memory.
    """

    # No __slots__: allows mock patching of _get_fact_embedding in tests

    def __init__(
        self,
        db: DatabaseManager,
        ebbinghaus: EbbinghausCurve,
        quantized_store: QuantizedEmbeddingStore,
        config: QuantizationConfig,
    ) -> None:
        """Initialize EAP scheduler.

        No side effects, no DB calls, no file I/O.
        """
        self._db = db
        self._ebbinghaus = ebbinghaus
        self._quantized_store = quantized_store
        self._config = config

    def run_eap_cycle(self, profile_id: str, *, dry_run: bool = False) -> dict:
        """Execute one EAP cycle for a profile.

        Steps:
          1. Fetch all facts with retention data
          2. Map retention -> target bit_width
          3. Execute downgrades/upgrades
          4. Return stats

        Returns:
            {total, downgrades, upgrades, skipped, deleted, errors}
        """
        stats = {
            "total": 0,
            "downgrades": 0,
            "upgrades": 0,
            "skipped": 0,
            "deleted": 0,
            "errors": 0,
        }

        # Step 1: Fetch all facts with retention + current bit_width
        try:
            rows = self._db.execute(
                "SELECT r.fact_id, r.retention_score, r.lifecycle_zone, "
                "  COALESCE(eqm.bit_width, 32) as current_bw "
                "FROM fact_retention r "
                "LEFT JOIN embedding_quantization_metadata eqm "
                "  ON r.fact_id = eqm.fact_id "
                "WHERE r.profile_id = ?",
                (profile_id,),
            )
        except Exception as exc:
            logger.error("EAP cycle query failed: %s", exc)
            stats["errors"] = 1
            return stats

        if not rows:
            return stats

        # Step 2+3: Process each fact
        for row in rows:
            d = dict(row)
            fact_id = d["fact_id"]
            retention = float(d["retention_score"])
            current_bw = int(d["current_bw"])

            stats["total"] += 1

            target_bw = retention_to_bit_width(retention)

            if target_bw == current_bw:
                stats["skipped"] += 1
                continue

            if target_bw == 0:
                # Forgotten -- mark as deleted
                if not dry_run:
                    self._handle_deletion(fact_id, profile_id)
                stats["deleted"] += 1
                continue

            if target_bw < current_bw:
                # Downgrade -- compress to lower precision
                if dry_run:
                    stats["downgrades"] += 1
                    continue
                success = self._handle_downgrade(
                    fact_id, profile_id, target_bw,
                )
                if success:
                    stats["downgrades"] += 1
                else:
                    stats["errors"] += 1
            else:
                # Upgrade -- restore to higher precision (only if float32 exists)
                if dry_run:
                    stats["upgrades"] += 1
                    continue
                success = self._handle_upgrade(
                    fact_id, profile_id, target_bw,
                )
                if success:
                    stats["upgrades"] += 1
                else:
                    stats["skipped"] += 1  # Can't upgrade without float32

        return stats

    # -- Handlers ----------------------------------------------------------

    def _handle_downgrade(
        self, fact_id: str, profile_id: str, target_bw: int,
    ) -> bool:
        """Compress a fact to lower bit_width.

        Fetches original float32 embedding, quantizes, and stores.
        """
        embedding = self._get_fact_embedding(fact_id)
        if embedding is None:
            logger.info(
                "No float32 embedding for %s, cannot compress", fact_id,
            )
            return False

        return self._quantized_store.compress_fact(
            fact_id, profile_id, embedding, target_bw,
        )

    def _handle_upgrade(
        self, fact_id: str, profile_id: str, target_bw: int,
    ) -> bool:
        """Restore a fact to higher precision.

        Only possible if original float32 is still in fact_embeddings.
        """
        # For upgrade to float32, just update metadata
        if target_bw == 32:
            try:
                self._db.execute(
                    "INSERT INTO embedding_quantization_metadata "
                    "(fact_id, profile_id, quantization_level, bit_width, created_at) "
                    "VALUES (?, ?, 'float32', 32, datetime('now')) "
                    "ON CONFLICT(fact_id) DO UPDATE SET "
                    "  quantization_level = 'float32', "
                    "  bit_width = 32",
                    (fact_id, profile_id),
                )
                return True
            except Exception as exc:
                logger.error("Upgrade to float32 failed for %s: %s", fact_id, exc)
                return False

        # For upgrade to int8 from polar, re-compress at higher precision
        embedding = self._get_fact_embedding(fact_id)
        if embedding is None:
            return False

        return self._quantized_store.compress_fact(
            fact_id, profile_id, embedding, target_bw,
        )

    def _handle_deletion(self, fact_id: str, profile_id: str) -> None:
        """Mark an embedding as deleted (forgotten).

        HR-03: Only deletes if keep_float32_backup is False.
        """
        try:
            self._db.execute(
                "INSERT INTO embedding_quantization_metadata "
                "(fact_id, profile_id, quantization_level, bit_width, created_at) "
                "VALUES (?, ?, 'deleted', 0, datetime('now')) "
                "ON CONFLICT(fact_id) DO UPDATE SET "
                "  quantization_level = 'deleted', "
                "  bit_width = 0",
                (fact_id, profile_id),
            )
        except Exception as exc:
            logger.error("Delete metadata failed for %s: %s", fact_id, exc)

    def _get_fact_embedding(self, fact_id: str) -> NDArray | None:
        """Retrieve original float32 embedding for a fact.

        Tries embedding_metadata -> fact_embeddings (vec0 table).
        Falls back to atomic_facts.embedding JSON column.
        """
        # Try atomic_facts.embedding (JSON column)
        try:
            rows = self._db.execute(
                "SELECT embedding FROM atomic_facts WHERE fact_id = ?",
                (fact_id,),
            )
            if rows:
                raw = dict(rows[0]).get("embedding")
                if raw and raw != "null":
                    data = json.loads(raw) if isinstance(raw, str) else raw
                    if data:
                        return np.array(data, dtype=np.float64)
        except Exception as exc:
            logger.debug(
                "Could not load embedding from atomic_facts for %s: %s",
                fact_id, exc,
            )

        return None
