# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3

"""VectorStore -- sqlite-vec backed KNN search with profile isolation.

Replaces full-table-scan in SemanticChannel with native vec0 KNN.
Falls back to ANNIndex if sqlite-vec is unavailable (Rule 03).
Implements ANNSearchable protocol for GraphBuilder compatibility (Rule 07).

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: MIT
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)  # Rule 10
class VectorStoreConfig:
    """Configuration for VectorStore."""
    dimension: int = 768
    binary_quantization_threshold: int = 100_000  # L4 fix
    model_name: str = "nomic-embed-text-v1.5"
    enabled: bool = False  # Rule 12: feature flag, default off


class VectorStore:
    """sqlite-vec backed vector store with profile-scoped KNN search.

    - Loads sqlite-vec extension on init (try/except, Rule 03)
    - Creates vec0 virtual table with profile_id PARTITION KEY
    - Maps string fact_ids to integer rowids via embedding_metadata
    - Implements ANNSearchable protocol (Rule 07)
    - Thread-safe via lock on mutations

    If sqlite-vec is unavailable, self.available is False and all
    methods are no-ops (caller uses ANNIndex fallback).
    """

    def __init__(self, db_path: Path, config: VectorStoreConfig) -> None:
        self._db_path = Path(db_path)
        self._config = config
        self._lock = threading.Lock()
        self._available = False

        if not config.enabled:
            logger.debug("VectorStore disabled by config (enabled=False)")
            return

        self._available = self._try_load_extension()
        if self._available:
            self._ensure_vec0_table()

    @property
    def available(self) -> bool:
        """True if sqlite-vec is loaded and vec0 table exists."""
        return self._available

    # -- Extension loading (Rule 03) ----------------------------------------

    def _try_load_extension(self) -> bool:
        """Attempt to load sqlite-vec. Returns True on success.

        Catches ImportError, AttributeError, and any other exception.
        """
        try:
            import sqlite_vec  # noqa: F401
            conn = self._connect()
            conn.close()
            return True
        except ImportError:
            logger.debug("sqlite-vec not installed. VectorStore unavailable.")
            return False
        except AttributeError:
            logger.debug(
                "enable_load_extension not available (macOS default Python). "
                "VectorStore unavailable."
            )
            return False
        except Exception as exc:
            logger.debug("sqlite-vec load failed: %s", exc)
            return False

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with sqlite-vec loaded.

        Every connection loads the extension fresh (per-call model).
        """
        import sqlite_vec

        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        # FK enforcement is OFF here because VectorStore operates on its own
        # tables (fact_embeddings + embedding_metadata). The store pipeline
        # guarantees fact/profile exist before calling upsert.
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    # -- Table creation -----------------------------------------------------

    def _ensure_vec0_table(self) -> None:
        """Create the vec0 virtual table and embedding_metadata if not exist."""
        dim = self._config.dimension
        vec0_ddl = (
            f"CREATE VIRTUAL TABLE IF NOT EXISTS fact_embeddings USING vec0("
            f"profile_id TEXT PARTITION KEY, "
            f"embedding float[{dim}] distance_metric=cosine"
            f")"
        )
        meta_ddl = (
            "CREATE TABLE IF NOT EXISTS embedding_metadata ("
            "vec_rowid INTEGER PRIMARY KEY, "
            "fact_id TEXT NOT NULL UNIQUE, "
            "profile_id TEXT NOT NULL DEFAULT 'default', "
            "model_name TEXT NOT NULL DEFAULT '', "
            "dimension INTEGER NOT NULL DEFAULT 768, "
            "created_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        )
        meta_idx_fact = (
            "CREATE INDEX IF NOT EXISTS idx_embmeta_fact "
            "ON embedding_metadata (fact_id)"
        )
        meta_idx_profile = (
            "CREATE INDEX IF NOT EXISTS idx_embmeta_profile "
            "ON embedding_metadata (profile_id)"
        )
        try:
            conn = self._connect()
            conn.execute(vec0_ddl)
            conn.execute(meta_ddl)
            conn.execute(meta_idx_fact)
            conn.execute(meta_idx_profile)
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.debug("vec0 table creation failed: %s", exc)
            self._available = False

    # -- Serialization ------------------------------------------------------

    @staticmethod
    def _serialize_f32(vector: list[float]) -> bytes:
        """Serialize float list to raw bytes for sqlite-vec."""
        return np.array(vector, dtype=np.float32).tobytes()

    # -- CRUD Operations ----------------------------------------------------

    def upsert(
        self,
        fact_id: str,
        profile_id: str,
        embedding: list[float],
        model_name: str = "",
    ) -> bool:
        """Insert or update a vector in the vec0 table.

        Thread-safe: acquires self._lock.
        Returns True on success, False on failure or if unavailable.
        """
        if not self._available:
            return False

        if len(embedding) != self._config.dimension:
            logger.debug(
                "Dimension mismatch: got %d, expected %d",
                len(embedding), self._config.dimension,
            )
            return False

        vec_bytes = self._serialize_f32(embedding)

        with self._lock:
            try:
                conn = self._connect()
                # Check if fact_id already exists in metadata
                row = conn.execute(
                    "SELECT vec_rowid FROM embedding_metadata "
                    "WHERE fact_id = ?",
                    (fact_id,),
                ).fetchone()

                if row is not None:
                    # UPDATE existing
                    rowid = row["vec_rowid"]
                    conn.execute(
                        "UPDATE fact_embeddings SET embedding = ? "
                        "WHERE rowid = ?",
                        (vec_bytes, rowid),
                    )
                else:
                    # INSERT new
                    conn.execute(
                        "INSERT INTO fact_embeddings(profile_id, embedding) "
                        "VALUES (?, ?)",
                        (profile_id, vec_bytes),
                    )
                    rowid = conn.execute(
                        "SELECT last_insert_rowid()"
                    ).fetchone()[0]
                    conn.execute(
                        "INSERT INTO embedding_metadata "
                        "(vec_rowid, fact_id, profile_id, model_name, dimension) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (rowid, fact_id, profile_id,
                         model_name or self._config.model_name,
                         self._config.dimension),
                    )

                conn.commit()
                conn.close()
                return True
            except Exception as exc:
                logger.debug("upsert failed for fact_id=%s: %s", fact_id, exc)
                return False

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 30,
        profile_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """KNN search. Returns [(fact_id, similarity_score)].

        Score is cosine similarity (1.0 - distance).
        Returns empty list if unavailable, dim mismatch, or error.
        """
        if not self._available:
            return []

        if len(query_embedding) != self._config.dimension:
            return []

        vec_bytes = self._serialize_f32(query_embedding)

        try:
            conn = self._connect()

            if profile_id is not None:
                rows = conn.execute(
                    "SELECT rowid, distance "
                    "FROM fact_embeddings "
                    "WHERE embedding MATCH ? "
                    "AND profile_id = ? "
                    "AND k = ?",
                    (vec_bytes, profile_id, top_k),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT rowid, distance "
                    "FROM fact_embeddings "
                    "WHERE embedding MATCH ? "
                    "AND k = ?",
                    (vec_bytes, top_k),
                ).fetchall()

            if not rows:
                conn.close()
                return []

            # Map rowids -> fact_ids via embedding_metadata
            rowids = [r["rowid"] for r in rows]
            dist_map = {r["rowid"]: r["distance"] for r in rows}

            placeholders = ",".join("?" for _ in rowids)
            meta_rows = conn.execute(
                f"SELECT vec_rowid, fact_id FROM embedding_metadata "
                f"WHERE vec_rowid IN ({placeholders})",
                rowids,
            ).fetchall()

            conn.close()

            results: list[tuple[str, float]] = []
            for mr in meta_rows:
                rid = mr["vec_rowid"]
                fid = str(mr["fact_id"])
                similarity = max(0.0, 1.0 - dist_map[rid])
                results.append((fid, similarity))

            results.sort(key=lambda x: x[1], reverse=True)
            return results

        except Exception as exc:
            logger.debug("search failed: %s", exc)
            return []

    def delete(self, fact_id: str) -> bool:
        """Remove a vector from vec0 and metadata.

        Thread-safe: acquires self._lock.
        Returns True if deleted, False if not found or error.
        """
        if not self._available:
            return False

        with self._lock:
            try:
                conn = self._connect()
                row = conn.execute(
                    "SELECT vec_rowid FROM embedding_metadata "
                    "WHERE fact_id = ?",
                    (fact_id,),
                ).fetchone()

                if row is None:
                    conn.close()
                    return False

                rowid = row["vec_rowid"]
                conn.execute(
                    "DELETE FROM fact_embeddings WHERE rowid = ?",
                    (rowid,),
                )
                conn.execute(
                    "DELETE FROM embedding_metadata WHERE vec_rowid = ?",
                    (rowid,),
                )
                conn.commit()
                conn.close()
                return True
            except Exception as exc:
                logger.debug("delete failed for fact_id=%s: %s", fact_id, exc)
                return False

    def count(self, profile_id: str | None = None) -> int:
        """Count vectors in the store.

        Returns 0 if unavailable.
        """
        if not self._available:
            return 0

        try:
            conn = self._connect()
            if profile_id is not None:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM embedding_metadata "
                    "WHERE profile_id = ?",
                    (profile_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM embedding_metadata",
                ).fetchone()
            conn.close()
            return int(row["c"]) if row else 0
        except Exception as exc:
            logger.debug("count failed: %s", exc)
            return 0

    def rebuild_from_facts(
        self,
        facts: list[tuple[str, str, list[float]]],
    ) -> int:
        """Migrate existing facts from JSON TEXT embeddings to vec0.

        Args:
            facts: List of (fact_id, profile_id, embedding) tuples.

        Returns:
            Number of vectors successfully migrated.
        """
        count = 0
        for fact_id, profile_id, embedding in facts:
            if self.upsert(fact_id, profile_id, embedding):
                count += 1
        return count

    def needs_binary_quantization(self, profile_id: str) -> bool:
        """Check if BQ should be enabled (count >= 100K threshold)."""
        return self.count(profile_id) >= self._config.binary_quantization_threshold
