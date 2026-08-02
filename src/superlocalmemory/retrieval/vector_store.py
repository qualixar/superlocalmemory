# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""VectorStore -- sqlite-vec backed KNN search with profile isolation.

Replaces full-table-scan in SemanticChannel with native vec0 KNN.
Falls back to ANNIndex if sqlite-vec is unavailable (Rule 03).
Implements ANNSearchable protocol for GraphBuilder compatibility (Rule 07).

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np

from superlocalmemory.storage.write_lock import get_write_lock

logger = logging.getLogger(__name__)


@dataclass(frozen=True)  # Rule 10
class VectorStoreConfig:
    """Configuration for VectorStore."""

    dimension: int = 768
    binary_quantization_threshold: int = 100_000  # L4 fix
    model_name: str = "nomic-embed-text-v1.5"
    enabled: bool = True  # Ships enabled by default


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

    @contextmanager
    def _managed_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Close every sqlite-vec connection, rolling back abandoned writes.

        sqlite-vec can reject a commit after its virtual-table shadow rows have
        already opened a SQLite write transaction.  A fail-soft caller must not
        return while that connection still owns the WAL writer lock: the next
        canonical/BM25 write would then wait behind an unreachable transaction.
        """
        conn = self._connect()
        try:
            yield conn
        finally:
            if conn.in_transaction:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
            conn.close()

    # -- Table creation -----------------------------------------------------

    @staticmethod
    def _read_stored_dimension(conn: sqlite3.Connection) -> int | None:
        """Return the embedding dimension of the existing vec0 table, or None.

        Reads the CREATE VIRTUAL TABLE DDL from sqlite_master and extracts the
        float[N] declaration.  Falls back to embedding_metadata.dimension if
        the DDL is absent or unparseable.  Returns None when the table does not
        yet exist, which means no rebuild is needed.
        """
        try:
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'fact_embeddings'"
            ).fetchone()
            if row is not None:
                sql = row["sql"] or row[0]
                if sql:
                    m = re.search(r'float\[(\d+)\]', sql, re.IGNORECASE)
                    if m:
                        return int(m.group(1))
        except Exception:
            pass

        # Fallback: read dimension from the most recent metadata row.
        try:
            row = conn.execute(
                "SELECT dimension FROM embedding_metadata LIMIT 1"
            ).fetchone()
            if row is not None:
                return int(row["dimension"])
        except sqlite3.OperationalError:
            pass  # metadata table does not exist yet

        return None

    def _ensure_vec0_table(self) -> None:
        """Create or rebuild the vec0 virtual table and embedding_metadata.

        If the existing vec0 table was built at a different embedding dimension
        than self._config.dimension, both tables are dropped and recreated at
        the new dimension.  Old embeddings are lost; callers that want to
        re-populate should call rebuild_from_facts() afterward.

        Same-dimension opens are a no-op (IF NOT EXISTS guard).
        """
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
            "CREATE INDEX IF NOT EXISTS idx_embmeta_fact ON embedding_metadata (fact_id)"
        )
        meta_idx_profile = (
            "CREATE INDEX IF NOT EXISTS idx_embmeta_profile ON embedding_metadata (profile_id)"
        )
        row_map_ddl = (
            "CREATE TABLE IF NOT EXISTS vector_row_map ("
            "fact_id TEXT NOT NULL PRIMARY KEY, "
            "profile_id TEXT NOT NULL, "
            "vec_rowid INTEGER NOT NULL"
            ")"
        )
        row_map_idx = (
            "CREATE INDEX IF NOT EXISTS idx_vector_row_map_profile "
            "ON vector_row_map (profile_id)"
        )
        try:
            with self._managed_connection() as conn:
                stored_dim = self._read_stored_dimension(conn)
                if stored_dim is not None and stored_dim != dim:
                    logger.info(
                        "Embedding dimension changed %d→%d: rebuilding vector index",
                        stored_dim,
                        dim,
                    )
                    # Drop metadata first (no FK enforcement, but cleaner ordering).
                    # Indexes on embedding_metadata are dropped automatically.
                    conn.execute("DROP TABLE IF EXISTS embedding_metadata")
                    conn.execute("DROP TABLE IF EXISTS vector_row_map")
                    conn.execute("DROP TABLE IF EXISTS fact_embeddings")
                    # Commit the drops before recreating so that the virtual-table
                    # shadow tables are fully removed before the new CREATE runs.
                    conn.commit()

                conn.execute(vec0_ddl)
                conn.execute(meta_ddl)
                conn.execute(meta_idx_fact)
                conn.execute(meta_idx_profile)
                conn.execute(row_map_ddl)
                conn.execute(row_map_idx)
                conn.commit()
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
                len(embedding),
                self._config.dimension,
            )
            return False

        # Embedding bytes computed OUTSIDE the lock — serialisation must not
        # cover slow computation, only the sqlite3 write transaction itself.
        vec_bytes = self._serialize_f32(embedding)

        # Acquire the process-level write lock for this db file BEFORE opening
        # the sqlite3 connection.  This is the OUTERMOST lock (see write_lock.py
        # ordering rule).  self._lock is INNER and acquired inside.  The write
        # lock is the same RLock that DatabaseManager._lock references for this
        # db_path, so the self-heal backfill pattern
        #   with db._lock: vs.upsert()
        # simply re-enters the RLock (same thread — always safe).
        _wl = get_write_lock(self._db_path)
        with _wl:  # OUTER: serialises all memory.db writers
            with self._lock:  # INNER: VectorStore per-instance state
                try:
                    with self._managed_connection() as conn:
                        # Reserve SQLite's cross-process writer before reading
                        # either side of the row-id allocator.  The Python
                        # RLocks above coordinate threads only; BEGIN IMMEDIATE
                        # plus the connection's bounded busy_timeout serializes
                        # independent MCP/agent processes as well.
                        conn.execute("BEGIN IMMEDIATE")
                        # Check if fact_id already exists in metadata
                        row = conn.execute(
                            "SELECT vec_rowid, profile_id "
                            "FROM embedding_metadata "
                            "WHERE fact_id = ?",
                            (fact_id,),
                        ).fetchone()

                        if row is not None:
                            rowid = row["vec_rowid"]
                            vector_row = conn.execute(
                                "SELECT profile_id FROM fact_embeddings WHERE rowid = ?",
                                (rowid,),
                            ).fetchone()
                            pair_matches_profile = (
                                vector_row is not None
                                and str(row["profile_id"]) == profile_id
                                and str(vector_row["profile_id"]) == profile_id
                            )
                            if pair_matches_profile:
                                conn.execute(
                                    "UPDATE fact_embeddings SET embedding = ? WHERE rowid = ?",
                                    (vec_bytes, rowid),
                                )
                            else:
                                # Older self-heal code could insert metadata
                                # before sqlite-vec, or row-id drift could point
                                # metadata at another profile's vector.  Neither
                                # is a valid projection pair.  Remove the stale
                                # pointer and rebuild at a fresh rowid; never
                                # overwrite the other profile's payload.
                                conn.execute(
                                    "DELETE FROM embedding_metadata WHERE fact_id = ?",
                                    (fact_id,),
                                )
                                # The old vec0 row is orphaned unless another
                                # projection pair still references it. Reclaim it
                                # so drift-repair never abandons raw payload.
                                still_referenced = conn.execute(
                                    "SELECT 1 FROM embedding_metadata "
                                    "WHERE vec_rowid = ? LIMIT 1",
                                    (rowid,),
                                ).fetchone()
                                if still_referenced is None:
                                    conn.execute(
                                        "DELETE FROM fact_embeddings WHERE rowid = ?",
                                        (rowid,),
                                    )
                                conn.execute(
                                    "DELETE FROM vector_row_map WHERE fact_id = ?",
                                    (fact_id,),
                                )
                                row = None

                        if row is None:
                            # Allocate from both sides of the projection pair.
                            # Mature databases can contain orphaned vec0 rows
                            # or metadata rows after older fail-soft releases.
                            # sqlite-vec's implicit last_insert_rowid() only
                            # considers the virtual table, so it can reuse a
                            # rowid that is still owned by embedding_metadata
                            # and make every later projection fail UNIQUE.
                            rowid = conn.execute(
                                "SELECT COALESCE(MAX(candidate), 0) + 1 "
                                "FROM ("
                                "SELECT MAX(rowid) AS candidate "
                                "FROM fact_embeddings "
                                "UNION ALL "
                                "SELECT MAX(vec_rowid) AS candidate "
                                "FROM embedding_metadata"
                                ")"
                            ).fetchone()[0]
                            conn.execute(
                                "INSERT INTO fact_embeddings"
                                "(rowid, profile_id, embedding) "
                                "VALUES (?, ?, ?)",
                                (rowid, profile_id, vec_bytes),
                            )
                            conn.execute(
                                "INSERT INTO embedding_metadata "
                                "(vec_rowid, fact_id, profile_id, model_name, dimension) "
                                "VALUES (?, ?, ?, ?, ?)",
                                (
                                    rowid,
                                    fact_id,
                                    profile_id,
                                    model_name or self._config.model_name,
                                    self._config.dimension,
                                ),
                            )

                        conn.execute(
                            "INSERT INTO vector_row_map (fact_id, profile_id, vec_rowid) "
                            "VALUES (?, ?, ?) "
                            "ON CONFLICT(fact_id) DO UPDATE SET "
                            "profile_id = excluded.profile_id, "
                            "vec_rowid = excluded.vec_rowid",
                            (fact_id, profile_id, rowid),
                        )
                        conn.commit()
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
            with self._managed_connection() as conn:
                if top_k <= 0:
                    return []
                if profile_id is not None:
                    sql = (
                        "SELECT fe.rowid, fe.distance, em.fact_id "
                        "FROM fact_embeddings AS fe "
                        "JOIN embedding_metadata AS em "
                        "ON em.vec_rowid = fe.rowid "
                        "AND em.profile_id = fe.profile_id "
                        "WHERE fe.embedding MATCH ? "
                        "AND fe.profile_id = ? "
                        "AND fe.k = ?"
                    )
                    base_params: tuple[object, ...] = (vec_bytes, profile_id)
                    count_sql = "SELECT COUNT(*) AS c FROM fact_embeddings WHERE profile_id = ?"
                    count_params: tuple[object, ...] = (profile_id,)
                else:
                    sql = (
                        "SELECT fe.rowid, fe.distance, em.fact_id "
                        "FROM fact_embeddings AS fe "
                        "JOIN embedding_metadata AS em "
                        "ON em.vec_rowid = fe.rowid "
                        "AND em.profile_id = fe.profile_id "
                        "WHERE fe.embedding MATCH ? "
                        "AND fe.k = ?"
                    )
                    base_params = (vec_bytes,)
                    count_sql = "SELECT COUNT(*) AS c FROM fact_embeddings"
                    count_params = ()

                # vec0 applies k before the relational join.  A legacy orphan
                # can therefore occupy a nearest-neighbour slot and then be
                # discarded by the profile-safe join. Expand only when that
                # happens, doubling until top_k valid pairs are found or the
                # profile's vector population is exhausted.
                search_k = top_k
                rows = conn.execute(
                    sql,
                    (*base_params, search_k),
                ).fetchall()
                if len(rows) < top_k:
                    count_row = conn.execute(
                        count_sql,
                        count_params,
                    ).fetchone()
                    total_vectors = int(count_row["c"]) if count_row else 0
                    while len(rows) < top_k and search_k < total_vectors:
                        search_k = min(total_vectors, max(search_k + 1, search_k * 2))
                        rows = conn.execute(
                            sql,
                            (*base_params, search_k),
                        ).fetchall()

            results: list[tuple[str, float]] = []
            for row in rows[:top_k]:
                fid = str(row["fact_id"])
                similarity = max(0.0, 1.0 - row["distance"])
                results.append((fid, similarity))

            results.sort(key=lambda x: x[1], reverse=True)
            return results

        except Exception as exc:
            logger.debug("search failed: %s", exc)
            return []

    def delete(self, fact_id: str) -> bool:
        """Remove a vector from vec0 and metadata.

        Thread-safe: acquires the process-level write lock then self._lock.
        Returns True if deleted, False if not found or error.
        """
        if not self._available:
            return False

        _wl = get_write_lock(self._db_path)
        with _wl:  # OUTER: process-level write serialisation
            with self._lock:  # INNER: VectorStore per-instance state
                try:
                    with self._managed_connection() as conn:
                        row = conn.execute(
                            "SELECT vec_rowid, profile_id "
                            "FROM embedding_metadata "
                            "WHERE fact_id = ?",
                            (fact_id,),
                        ).fetchone()

                        if row is None:
                            return False

                        rowid = row["vec_rowid"]
                        vector_row = conn.execute(
                            "SELECT profile_id FROM fact_embeddings WHERE rowid = ?",
                            (rowid,),
                        ).fetchone()
                        if vector_row is not None and str(vector_row["profile_id"]) == str(
                            row["profile_id"]
                        ):
                            conn.execute(
                                "DELETE FROM fact_embeddings WHERE rowid = ?",
                                (rowid,),
                            )
                        conn.execute(
                            "DELETE FROM embedding_metadata WHERE vec_rowid = ?",
                            (rowid,),
                        )
                        conn.execute(
                            "DELETE FROM vector_row_map WHERE fact_id = ?",
                            (fact_id,),
                        )
                        conn.commit()
                    return True
                except Exception as exc:
                    logger.debug("delete failed for fact_id=%s: %s", fact_id, exc)
                    return False

    def raw_vector_present(self, fact_id: str) -> bool:
        if not self._available:
            try:
                conn = sqlite3.connect(str(self._db_path))
                try:
                    row = conn.execute(
                        "SELECT 1 FROM vector_row_map WHERE fact_id = ? LIMIT 1",
                        (fact_id,),
                    ).fetchone()
                    return row is not None
                finally:
                    conn.close()
            except Exception:
                return False
        try:
            with self._managed_connection() as conn:
                row = conn.execute(
                    "SELECT 1 FROM vector_row_map vrm "
                    "WHERE vrm.fact_id = ? "
                    "AND EXISTS (SELECT 1 FROM fact_embeddings fe "
                    "WHERE fe.rowid = vrm.vec_rowid)",
                    (fact_id,),
                ).fetchone()
                return row is not None
        except Exception:
            try:
                conn2 = sqlite3.connect(str(self._db_path))
                try:
                    row = conn2.execute(
                        "SELECT 1 FROM vector_row_map WHERE fact_id = ? LIMIT 1",
                        (fact_id,),
                    ).fetchone()
                    return row is not None
                finally:
                    conn2.close()
            except Exception:
                return False

    def count(self, profile_id: str | None = None) -> int:
        """Count complete metadata/vector pairs in the store.

        Returns 0 if unavailable.
        """
        if not self._available:
            return 0

        try:
            with self._managed_connection() as conn:
                if profile_id is not None:
                    row = conn.execute(
                        "SELECT COUNT(*) AS c "
                        "FROM embedding_metadata em "
                        "JOIN fact_embeddings fe "
                        "ON fe.rowid = em.vec_rowid "
                        "AND fe.profile_id = em.profile_id "
                        "WHERE em.profile_id = ?",
                        (profile_id,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT COUNT(*) AS c "
                        "FROM embedding_metadata em "
                        "JOIN fact_embeddings fe "
                        "ON fe.rowid = em.vec_rowid "
                        "AND fe.profile_id = em.profile_id",
                    ).fetchone()
            return int(row["c"]) if row else 0
        except Exception as exc:
            logger.debug("count failed: %s", exc)
            return 0

    def indexed_fact_ids(self, profile_id: str) -> set[str]:
        """Return fact IDs backed by both metadata and a vec0 payload."""
        if not self._available:
            return set()
        try:
            with self._managed_connection() as conn:
                rows = conn.execute(
                    "SELECT em.fact_id "
                    "FROM embedding_metadata em "
                    "JOIN fact_embeddings fe "
                    "ON fe.rowid = em.vec_rowid "
                    "AND fe.profile_id = em.profile_id "
                    "WHERE em.profile_id = ?",
                    (profile_id,),
                ).fetchall()
            return {str(row["fact_id"]) for row in rows}
        except Exception as exc:
            logger.debug("indexed_fact_ids failed: %s", exc)
            return set()

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
