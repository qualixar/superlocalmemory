# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Quantized embedding storage and retrieval.

Manages polar_embeddings and embedding_quantization_metadata tables.
Bridges PolarQuantEncoder/QJLEncoder to SQLite persistence.

HR-05: All SQL uses parameterized queries.
HR-06: BLOB columns use Python bytes, not base64.
HR-07: QJL is optional -- system works without it.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from superlocalmemory.core.config import QuantizationConfig
from superlocalmemory.math.polar_quant import PolarQuantEncoder, QuantizedEmbedding
from superlocalmemory.math.qjl import QJLEncoder

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quantization level names
# ---------------------------------------------------------------------------

_BW_TO_LEVEL: dict[int, str] = {
    32: "float32",
    8: "int8",
    4: "polar4",
    2: "polar2",
    0: "deleted",
}


class QuantizedEmbeddingStore:
    """CRUD for quantized polar embeddings with optional QJL correction.

    Uses two tables:
      - polar_embeddings: stores angle indices, radius, QJL bits
      - embedding_quantization_metadata: tracks quantization state per fact
    """

    __slots__ = ("_db", "_polar", "_qjl", "_config")

    def __init__(
        self,
        db: DatabaseManager,
        polar: PolarQuantEncoder,
        qjl: QJLEncoder | None,
        config: QuantizationConfig,
    ) -> None:
        self._db = db
        self._polar = polar
        self._qjl = qjl
        self._config = config

    # -- CRUD: polar_embeddings --------------------------------------------

    def store(
        self, fact_id: str, profile_id: str, qe: QuantizedEmbedding,
    ) -> bool:
        """UPSERT a quantized embedding into polar_embeddings.

        Also updates embedding_quantization_metadata.
        Returns True on success, False on error.
        """
        try:
            # UPSERT polar embedding (HR-05: parameterized)
            self._db.execute(
                "INSERT INTO polar_embeddings "
                "(fact_id, profile_id, radius, angle_indices, qjl_bits, bit_width, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, datetime('now')) "
                "ON CONFLICT(fact_id) DO UPDATE SET "
                "  radius = excluded.radius, "
                "  angle_indices = excluded.angle_indices, "
                "  qjl_bits = excluded.qjl_bits, "
                "  bit_width = excluded.bit_width",
                (fact_id, profile_id, qe.radius, qe.angle_indices,
                 qe.qjl_bits, qe.bit_width),
            )

            # UPSERT quantization metadata
            level = _BW_TO_LEVEL.get(qe.bit_width, "float32")
            compressed_size = len(qe.angle_indices)
            if qe.qjl_bits:
                compressed_size += len(qe.qjl_bits)

            self._db.execute(
                "INSERT INTO embedding_quantization_metadata "
                "(fact_id, profile_id, quantization_level, bit_width, "
                " compressed_size_bytes, created_at) "
                "VALUES (?, ?, ?, ?, ?, datetime('now')) "
                "ON CONFLICT(fact_id) DO UPDATE SET "
                "  quantization_level = excluded.quantization_level, "
                "  bit_width = excluded.bit_width, "
                "  compressed_size_bytes = excluded.compressed_size_bytes",
                (fact_id, profile_id, level, qe.bit_width, compressed_size),
            )

            return True
        except Exception as exc:
            logger.error("store failed for fact_id=%s: %s", fact_id, exc)
            return False

    def load(
        self, fact_id: str, profile_id: str,
    ) -> QuantizedEmbedding | None:
        """Load a quantized embedding by fact_id and profile_id.

        Returns None if not found.
        """
        try:
            rows = self._db.execute(
                "SELECT radius, angle_indices, qjl_bits, bit_width "
                "FROM polar_embeddings "
                "WHERE fact_id = ? AND profile_id = ?",
                (fact_id, profile_id),
            )
            if not rows:
                return None

            row = dict(rows[0])
            return QuantizedEmbedding(
                fact_id=fact_id,
                radius=float(row["radius"]),
                angle_indices=bytes(row["angle_indices"]),
                bit_width=int(row["bit_width"]),
                qjl_bits=bytes(row["qjl_bits"]) if row["qjl_bits"] else None,
            )
        except Exception as exc:
            logger.error("load failed for fact_id=%s: %s", fact_id, exc)
            return None

    def search(
        self,
        query_embedding: NDArray,
        profile_id: str,
        top_k: int = 50,
        *,
        bit_widths: tuple[int, ...] | None = None,
    ) -> list[tuple[str, float]]:
        """Search polar embeddings for a profile.

        Pre-filters by lifecycle_zone (excludes 'forgotten').
        ``bit_widths`` selects a persisted precision tier; ``None`` preserves
        the legacy all-quantized-row behavior.
        Returns [(fact_id, similarity)] sorted descending.
        """
        if bit_widths is not None and not bit_widths:
            return []
        try:
            width_clause = ""
            params: list[object] = [profile_id]
            if bit_widths is not None:
                widths = tuple(int(width) for width in bit_widths)
                width_clause = (
                    " AND pe.bit_width IN ("
                    + ",".join("?" for _ in widths)
                    + ")"
                )
                params.extend(widths)
            rows = self._db.execute(
                "SELECT pe.fact_id, pe.radius, pe.angle_indices, "
                "  pe.qjl_bits, pe.bit_width "
                "FROM polar_embeddings pe "
                "JOIN fact_retention fr "
                "  ON pe.fact_id = fr.fact_id AND fr.profile_id = pe.profile_id "
                "WHERE pe.profile_id = ? "
                "  AND fr.lifecycle_zone NOT IN ('forgotten')"
                + width_clause,
                tuple(params),
            )
        except Exception as exc:
            logger.error("search query failed: %s", exc)
            return []

        if not rows:
            return []

        # Build QuantizedEmbedding objects and compute similarities
        results: list[tuple[str, float]] = []
        for row in rows:
            d = dict(row)
            qe = QuantizedEmbedding(
                fact_id=d["fact_id"],
                radius=float(d["radius"]),
                angle_indices=bytes(d["angle_indices"]),
                bit_width=int(d["bit_width"]),
                qjl_bits=bytes(d["qjl_bits"]) if d["qjl_bits"] else None,
            )

            sim = self._polar.approximate_similarity(query_embedding, qe)

            # QJL correction (HR-07: optional)
            if qe.qjl_bits and self._qjl:
                correction = self._qjl.estimate_correction(
                    query_embedding, qe.qjl_bits,
                )
                sim += correction

            results.append((qe.fact_id, sim))

        # Sort descending by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # -- Compression helpers -----------------------------------------------

    def compress_fact(
        self,
        fact_id: str,
        profile_id: str,
        original_embedding: NDArray,
        target_bit_width: int,
    ) -> bool:
        """Quantize an embedding and store it.

        For bit_width <= 4, also computes QJL residual correction
        (if QJL encoder is available).

        Returns True on success.
        """
        try:
            qe = self._polar.encode(original_embedding, target_bit_width)

            # QJL residual for low bit-widths (HR-07: optional)
            qjl_bits: bytes | None = None
            if target_bit_width <= 4 and self._qjl:
                decoded = self._polar.decode(qe)
                residual = original_embedding - decoded
                qjl_bits = self._qjl.encode_residual(residual)

            # Build final QuantizedEmbedding with fact_id and QJL bits
            qe_final = QuantizedEmbedding(
                fact_id=fact_id,
                radius=qe.radius,
                angle_indices=qe.angle_indices,
                bit_width=qe.bit_width,
                qjl_bits=qjl_bits,
            )

            return self.store(fact_id, profile_id, qe_final)
        except Exception as exc:
            logger.error(
                "compress_fact failed for fact_id=%s: %s", fact_id, exc,
            )
            return False

    def batch_compress(
        self,
        fact_ids: list[str],
        profile_id: str,
        embeddings: dict[str, NDArray],
        target_bit_width: int,
    ) -> int:
        """Compress a batch of facts. Returns count of successful compressions."""
        count = 0
        for fact_id in fact_ids:
            if fact_id in embeddings:
                if self.compress_fact(
                    fact_id, profile_id, embeddings[fact_id], target_bit_width,
                ):
                    count += 1
        return count
