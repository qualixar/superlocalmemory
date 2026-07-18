# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory v3.4.5 — LanceDB Vector Backend.

Embedded vector database backend powered by LanceDB (Apache-2.0).
Replaces sqlite-vec for embedding storage and similarity search.

Verified API: lancedb v0.30.2, connect(path), create_table, search().metric('cosine')

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import sqlite3
import struct
from pathlib import Path
from typing import Any

from superlocalmemory.storage.sqlite_vectors import iter_canonical_vectors

logger = logging.getLogger(__name__)

# Optional import
try:
    import lancedb
    _LANCEDB_AVAILABLE = True
except ImportError:
    lancedb = None  # type: ignore[assignment]
    _LANCEDB_AVAILABLE = False


class LanceDBError(Exception):
    """Base exception for LanceDB backend failures."""


class LanceDBNotAvailable(LanceDBError):
    """LanceDB not installed. Install with: pip install superlocalmemory[lancedb]"""


# ---------------------------------------------------------------------------
# LanceDBVectorBackend
# ---------------------------------------------------------------------------

class LanceDBVectorBackend:
    """Embedded vector backend powered by LanceDB.

    Columnar storage (Lance format). Cosine similarity search.
    Tier-aware: hot+warm vectors searched by default.
    """

    # Valid tier values (F-27: validated before interpolation)
    VALID_TIERS: frozenset[str] = frozenset({"active", "warm", "cold", "archived"})

    def __init__(self, db_path: str) -> None:
        if not _LANCEDB_AVAILABLE:
            raise LanceDBNotAvailable(
                "LanceDB not installed. Run: pip install superlocalmemory[lancedb]"
            )
        path = Path(db_path)
        path.mkdir(parents=True, exist_ok=True)
        self._db_path = str(path)
        self._db = lancedb.connect(self._db_path)  # type: ignore[union-attr]
        self._table = self._open_or_create_table()

    def _open_or_create_table(self):
        """Open existing table or create empty one."""
        try:
            return self._db.open_table("embeddings")
        except Exception:
            import pyarrow as pa
            schema = pa.schema([
                pa.field("fact_id", pa.string(), nullable=False),
                pa.field("vector", pa.list_(pa.float32(), list_size=768), nullable=False),
                pa.field("tier", pa.string(), nullable=False),
                pa.field("profile_id", pa.string(), nullable=False),
            ])
            return self._db.create_table("embeddings", schema=schema)

    def close(self) -> None:
        """Release this backend's native table and connection references."""
        for resource in (self._table, self._db):
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("LanceDB resource close failed", exc_info=True)
        self._table = None
        self._db = None

    # ------------------------------------------------------------------
    # Write Path
    # ------------------------------------------------------------------

    def add_vectors(
        self,
        fact_ids: list[str],
        embeddings: list[list[float]],
        tiers: list[str],
        profile_id: str = "default",
    ) -> int:
        """Idempotently insert or replace vectors by canonical fact ID."""
        if not fact_ids:
            return 0
        data = [
            {"fact_id": fid, "vector": emb, "tier": tier, "profile_id": profile_id}
            for fid, emb, tier in zip(fact_ids, embeddings, tiers)
        ]
        # `add()` permits duplicate fact IDs on retries.  Store/retry paths are
        # at-least-once by design, so use LanceDB's merge operation to keep the
        # projection one-row-per-fact and safe to replay after a restart.
        (
            self._table.merge_insert("fact_id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(data)
        )
        return len(data)

    @staticmethod
    def _fact_predicate(fact_id: str) -> str:
        """Build a Lance SQL literal without allowing predicate injection."""
        return "fact_id = '" + fact_id.replace("'", "''") + "'"

    def remove_vector(self, fact_id: str) -> None:
        """Delete one derived vector after its canonical fact is deleted."""
        self._table.delete(self._fact_predicate(fact_id))

    # ------------------------------------------------------------------
    # Read Path
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query_vector: list[float],
        top_k: int = 50,
        tier_filter: list[str] | None = None,
        profile_id: str = "default",
    ) -> list[tuple[str, float]]:
        """ANN search with optional tier filter.

        Returns [(fact_id, similarity_score), ...] where 1.0 = identical.
        Uses cosine metric — _distance is (1 - cosine_similarity)
        so we return (1.0 - _distance).
        """
        if tier_filter is None:
            tier_filter = ["active", "warm"]

        # F-27: Validate tiers
        assert all(t in self.VALID_TIERS for t in tier_filter), (
            f"Invalid tier filter: {set(tier_filter) - self.VALID_TIERS}"
        )

        try:
            search = self._table.search(query_vector).metric("cosine").limit(top_k)

            # Build tier filter string for LanceDB SQL-like where clause
            tier_str = ", ".join(f"'{t}'" for t in tier_filter)
            profile_literal = profile_id.replace("'", "''")
            results = search.where(
                f"tier IN ({tier_str}) AND profile_id = '{profile_literal}'"
            ).to_list()

            # Convert distance → similarity (F-08)
            return [(r["fact_id"], 1.0 - r["_distance"]) for r in results]
        except Exception as exc:
            logger.warning("LanceDB similarity search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Bulk Import (sqlite-vec → LanceDB)
    # ------------------------------------------------------------------

    def bulk_import_from_sqlite(
        self, conn: sqlite3.Connection, profile_id: str = "default",
    ) -> int:
        """Export embeddings from sqlite-vec → LanceDB.

        Reads the supported vec0 virtual table and joins row IDs through
        ``embedding_metadata``. Shadow-table layouts are sqlite-vec internals
        and must not be treated as a stable migration API.

        Returns number of vectors imported.
        """
        imported = 0
        batch: list[tuple[str, list[float], str, str]] = []
        for rowid, fact_id, tier, row_profile_id, blob in iter_canonical_vectors(
            conn, profile_id
        ):
            try:
                vector = self._decode_vector_blob(blob)
            except Exception as exc:
                raise LanceDBError(
                    f"Invalid canonical vector for rowid {rowid}: {exc}"
                ) from exc
            batch.append((fact_id, vector, tier, row_profile_id))
            if len(batch) >= 256:
                imported += self._flush_import_batch(batch)
                batch.clear()
        if batch:
            imported += self._flush_import_batch(batch)

        logger.info("LanceDB: imported %d vectors from sqlite-vec", imported)
        return imported

    def _flush_import_batch(
        self, batch: list[tuple[str, list[float], str, str]]
    ) -> int:
        """Write one bounded-memory, single-profile canonical vector batch."""
        by_profile: dict[str, list[tuple[str, list[float], str]]] = {}
        for fact_id, vector, tier, profile_id in batch:
            by_profile.setdefault(profile_id, []).append((fact_id, vector, tier))
        imported = 0
        for profile_id, records in by_profile.items():
            imported += self.add_vectors(
                [item[0] for item in records],
                [item[1] for item in records],
                [item[2] for item in records],
                profile_id,
            )
        return imported

    def _decode_vector_blob(self, blob: bytes) -> list[float]:
        """Decode sqlite-vec BLOB to list of floats.

        F-33: Validates dimension and L2 norm.
        sqlite-vec stores vectors as raw float32 little-endian bytes.
        """
        expected_bytes = 768 * 4  # 3072
        if len(blob) != expected_bytes:
            raise ValueError(
                f"Unexpected vector blob size: {len(blob)} (expected {expected_bytes})"
            )

        vec = list(struct.unpack(f"{768}f", blob))

        # F-33: Validate non-zero
        norm = sum(v * v for v in vec) ** 0.5
        if norm < 1e-10:
            raise ValueError(f"Near-zero L2 norm ({norm}) — verify sqlite-vec format")

        return vec

    # ------------------------------------------------------------------
    # Tier Update
    # ------------------------------------------------------------------

    def update_tier(self, fact_id: str, new_tier: str) -> None:
        """Update tier for a single fact."""
        try:
            self._table.update(
                where=f"fact_id = '{fact_id}'",
                values={"tier": new_tier},
            )
        except Exception as exc:
            logger.warning("LanceDB tier update failed for %s: %s", fact_id, exc)

    def bulk_update_tiers_from_sqlite(self, conn: sqlite3.Connection) -> int:
        """Batch update tiers by rebuilding from SQLite.

        More efficient than per-row updates for nightly rebalance (F-19).
        """
        try:
            rows = conn.execute(
                "SELECT fact_id, lifecycle FROM atomic_facts WHERE profile_id = 'default'"
            ).fetchall()

            updated = 0
            for fact_id, tier in rows:
                try:
                    self._table.update(
                        where=f"fact_id = '{fact_id}'",
                        values={"tier": tier},
                    )
                    updated += 1
                except Exception:
                    pass  # Fact may not be in LanceDB yet
            return updated
        except Exception as exc:
            logger.warning("LanceDB bulk tier update failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    def rebuild_from_sqlite(self, conn: sqlite3.Connection) -> int:
        """Drop and rebuild from SQLite."""
        try:
            self._db.drop_table("embeddings")
        except Exception:
            pass
        self._table = self._open_or_create_table()
        return self.bulk_import_from_sqlite(conn)

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Return health status."""
        try:
            count = self._table.count_rows()
            return {
                "status": "active",
                "vectors": count,
                "db_path": self._db_path,
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "db_path": self._db_path,
            }
