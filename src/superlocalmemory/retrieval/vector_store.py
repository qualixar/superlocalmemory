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

import enum
import logging
import re
import sqlite3
import threading
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from superlocalmemory.retrieval import vector_presence
from superlocalmemory.storage.embedding_codec import encode_embedding
from superlocalmemory.storage.erasure_visibility import erasure_clause_for_connection
from superlocalmemory.storage.write_lock import get_write_lock

logger = logging.getLogger(__name__)

#: Searches asking for this many or fewer share one scan (#304). Hopfield asks
#: for 500, semantic for 100, spreading activation for 10; on a 37k-vector store
#: k=500 measured 113 ms against 93-113 ms for each of the three separately.
SHARED_SEARCH_K = 512


class VectorDelete(enum.Enum):
    """The three honest answers to "delete this fact's vector".

    ``delete`` used to answer ``False`` both for "there was nothing to delete"
    and for "the delete failed", and logged the failure at debug level, so an
    erasure that left a vector behind looked exactly like one that had nothing
    to do. A caller that erases must be able to tell them apart.

    Truthiness keeps the old meaning ("a vector was removed") so existing
    ``if store.delete(fid):`` callers behave as before; erasure callers read
    :attr:`failed`.
    """

    DELETED = "deleted"
    ABSENT = "absent"
    FAILED = "failed"

    def __bool__(self) -> bool:
        return self is VectorDelete.DELETED

    @property
    def failed(self) -> bool:
        return self is VectorDelete.FAILED


def classify_delete(result: object) -> VectorDelete:
    """A store's answer as a :class:`VectorDelete`.

    A store written before the typed answer returns a bare bool, where ``False``
    cannot be told from a failure; that is read as a failure, never as success.
    """
    if isinstance(result, VectorDelete):
        return result
    return VectorDelete.DELETED if result else VectorDelete.FAILED


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
        self._flights_lock = threading.Lock()
        self._flights: dict[tuple, Future] = {}
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
        try:
            conn.execute("PRAGMA busy_timeout = 10000")
            # FK enforcement is OFF here because VectorStore operates on its own
            # tables (fact_embeddings + embedding_metadata). The store pipeline
            # guarantees fact/profile exist before calling upsert.
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except BaseException:
            # Setup failed: close it now, or the file stays open (Windows
            # then cannot delete or replace it) for as long as the error lives.
            conn.close()
            raise
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
            logger.warning("vec0 table creation failed: %s", exc)
            self._available = False

    # -- Serialization ------------------------------------------------------

    @staticmethod
    def _serialize_f32(vector: list[float]) -> bytes:
        """Serialize float list to raw bytes for sqlite-vec.

        Delegates to the shared embedding codec so that a format change in
        that codec propagates here automatically.
        """
        result = encode_embedding(vector)
        if result is None:
            # encode_embedding returns None only for None input; _serialize_f32
            # is never called with None, so this branch is unreachable in
            # production.  Return empty bytes rather than raising so the caller
            # sees an empty query result rather than an unhandled exception.
            return b""
        return result

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

    def _has_quarantine_column(self) -> bool:
        """Whether atomic_facts carries ``quarantined`` in this database.

        Cached once True; re-probed while absent so a schema pass that lands
        later is picked up. Mirrors DatabaseManager._has_quarantine_column --
        an unmigrated store must degrade to the old query rather than raise on
        every semantic search.
        """
        cached = getattr(self, "_quarantine_col", None)
        if cached is True:
            return True
        try:
            with self._managed_connection() as conn:
                present = any(
                    row[1] == "quarantined"
                    for row in conn.execute("PRAGMA table_info(atomic_facts)")
                )
        except Exception:  # noqa: BLE001 -- a probe must never break search
            return False
        if present:
            self._quarantine_col = True
        return present

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

        if top_k <= 0:
            return []
        vec_bytes = self._serialize_f32(query_embedding)
        # #304: semantic, hopfield and spreading activation ask with the same
        # vector at the same moment, and vec0 scans every vector whatever k is.
        # A small k asks for the shared k and cuts (the rows are ranked, so its
        # head is exactly what a smaller k returns), and an identical search
        # already running answers the others. Only searches in flight are
        # shared: one that has finished is never reused, so writes are seen.
        k = max(top_k, SHARED_SEARCH_K)
        key = (profile_id, vec_bytes, k)
        with self._flights_lock:
            flight = self._flights.get(key)
            leader = flight is None
            if leader:
                flight = self._flights[key] = Future()
        if leader:
            try:
                flight.set_result(self._search_rows(vec_bytes, k, profile_id))
            except BaseException as exc:  # noqa: BLE001 -- the waiters get it too
                flight.set_exception(exc)
            finally:
                with self._flights_lock:
                    self._flights.pop(key, None)
        return flight.result()[:top_k]

    def _search_rows(
        self, vec_bytes: bytes, top_k: int, profile_id: str | None,
    ) -> list[tuple[str, float]]:
        """The ranked nearest ``top_k`` (fact_id, cosine) pairs; ``[]`` on error."""
        try:
            with self._managed_connection() as conn:
                # A withheld fact must not occupy a nearest-neighbour slot.
                #
                # Its vector stays in the index — quarantine is reversible and
                # deleting the projection would cost a re-embed to undo — so it
                # is filtered here instead.
                #
                # Measured on the author's store: 1,192 of 5,086 projections
                # (23.4%) belong to withheld rows, and because model-written
                # summaries of the same clusters land close together in
                # embedding space they crowd each other. Searching with a vector
                # taken from one of them returned 50 of 50 neighbours withheld
                # without this join, and 0 of 50 with it. So for any query near
                # that cluster the semantic channel was contributing nothing at
                # all — every slot spent on a candidate that hydration would
                # discard — while looking like it had answered.
                #
                # The expansion loop below already exists for exactly this
                # shape of problem (orphaned vec0 rows losing a slot to the
                # relational join) and compensates automatically: it doubles k
                # until top_k surviving pairs are found. Nothing new is needed
                # to make the slots back.
                # LEFT, not INNER. An inner join would make a projection
                # depend on its corpus row still existing, so a legacy orphan
                # metadata row would stop being returned at all -- a behaviour
                # change well beyond quarantine, and one the existing
                # vector-store tests caught immediately by building a store
                # with no matching facts. LEFT leaves af.quarantined NULL for a
                # missing row, and COALESCE keeps it.
                quarantine_join = (
                    " LEFT JOIN atomic_facts AS af ON af.fact_id = em.fact_id "
                    if self._has_quarantine_column() else ""
                )
                quarantine_filter = (
                    " AND COALESCE(af.quarantined, 0) = 0 "
                    if self._has_quarantine_column() else ""
                )
                quarantine_filter += erasure_clause_for_connection(conn, "em")
                if profile_id is not None:
                    # The profile is BOUND on the metadata side, not joined to
                    # fe.profile_id: reading the vec0 partition column runs a
                    # nested statement per candidate row, and its allocations
                    # take a process-wide SQLite lock that every other read in
                    # the daemon then waits on. fe.profile_id = ? already holds,
                    # so the rows are identical
                    # (tests/test_retrieval/test_vector_search_binds_the_profile.py).
                    sql = (
                        "SELECT fe.rowid, fe.distance, em.fact_id "
                        "FROM fact_embeddings AS fe "
                        "JOIN embedding_metadata AS em "
                        "ON em.vec_rowid = fe.rowid "
                        "AND em.profile_id = ? "
                        + quarantine_join +
                        "WHERE fe.embedding MATCH ? "
                        "AND fe.profile_id = ? "
                        + quarantine_filter +
                        "AND fe.k = ?"
                    )
                    base_params: tuple[object, ...] = (profile_id, vec_bytes, profile_id)
                    count_sql = "SELECT COUNT(*) AS c FROM fact_embeddings WHERE profile_id = ?"
                    count_params: tuple[object, ...] = (profile_id,)
                else:
                    sql = (
                        "SELECT fe.rowid, fe.distance, em.fact_id "
                        "FROM fact_embeddings AS fe "
                        "JOIN embedding_metadata AS em "
                        "ON em.vec_rowid = fe.rowid "
                        "AND em.profile_id = fe.profile_id "
                        + quarantine_join +
                        "WHERE fe.embedding MATCH ? "
                        + quarantine_filter +
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

            # Rank the whole candidate list, THEN cut it. The expansion above
            # can leave `rows` longer than top_k, and the SQL has no ORDER BY —
            # vec0 applies k, then a relational join emits the survivors in an
            # order SQLite is free to choose. Cutting first therefore cut by
            # position and only ranked what survived, which can drop a nearer
            # fact in favour of a farther one.
            #
            # Measured on the 0.95 GB archive: 29 of 60 queries do take the
            # expansion path (orphan vectors are common), and the produced set
            # matched the true nearest-k in all 29 — the join does emit in
            # distance order in practice. So this was latent, not active. It is
            # fixed because "the planner happens to" is not a guarantee: add an
            # index, change SQLite, or change the join and the answer moves.
            #
            # The tie-break on fact_id makes the cut total, so two facts at
            # equal distance cannot swap across the top_k boundary between runs.
            results: list[tuple[str, float]] = [
                (str(row["fact_id"]), max(0.0, 1.0 - row["distance"]))
                for row in rows
            ]
            results.sort(key=lambda x: (-x[1], x[0]))
            return results[:top_k]

        except Exception as exc:
            logger.debug("search failed: %s", exc)
            return []

    def delete(self, fact_id: str) -> VectorDelete:
        """Remove a vector from vec0 and metadata.

        Thread-safe: acquires the process-level write lock then self._lock.
        Returns :attr:`VectorDelete.DELETED` when a vector was removed,
        ``ABSENT`` when there was none (or the store is not in use), and
        ``FAILED`` when it could not be removed or is still there afterwards.
        A failure is logged at WARNING: it can leave a person's data behind.
        """
        if not self._available:
            return VectorDelete.ABSENT

        _wl = get_write_lock(self._db_path)
        with _wl:  # OUTER: process-level write serialisation
            with self._lock:  # INNER: VectorStore per-instance state
                try:
                    with self._managed_connection() as conn:
                        return self._delete_locked(conn, fact_id)
                except Exception as exc:
                    logger.warning("vector delete failed for fact_id=%s: %s", fact_id, exc)
                    return VectorDelete.FAILED

    @staticmethod
    def _delete_locked(conn: sqlite3.Connection, fact_id: str) -> VectorDelete:
        # A model switch's staged and previous copies go with it.
        from superlocalmemory.storage.embedding_spaces import purge_fact
        purge_fact(conn, fact_id)
        conn.commit()
        row = conn.execute(
            "SELECT vec_rowid, profile_id FROM embedding_metadata WHERE fact_id = ?",
            (fact_id,),
        ).fetchone()
        if row is None:
            # Metadata-less orphan: resolve the rowid via the fact-addressable
            # map so the raw vec0 row is still removable by fact_id.
            row = conn.execute(
                "SELECT vec_rowid, profile_id FROM vector_row_map WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
        if row is None:
            return VectorDelete.ABSENT
        rowid, owner_profile = row["vec_rowid"], str(row["profile_id"])
        vector_row = conn.execute(
            "SELECT profile_id FROM fact_embeddings WHERE rowid = ?", (rowid,),
        ).fetchone()
        owned = vector_row is not None and str(vector_row["profile_id"]) == owner_profile
        if owned:
            conn.execute("DELETE FROM fact_embeddings WHERE rowid = ?", (rowid,))
        conn.execute("DELETE FROM embedding_metadata WHERE vec_rowid = ?", (rowid,))
        conn.execute("DELETE FROM vector_row_map WHERE fact_id = ?", (fact_id,))
        conn.commit()
        if owned and conn.execute(
                "SELECT 1 FROM fact_embeddings WHERE rowid = ?", (rowid,)).fetchone():
            return VectorDelete.FAILED
        return VectorDelete.DELETED

    # Read-only presence and inventory queries live in vector_presence so
    # this module stays within the file-size limit.
    def raw_vector_present(self, fact_id: str) -> bool:
        return vector_presence.raw_vector_present(self, fact_id)

    def is_searchable_by_meaning(
        self,
        fact_id: str,
        profile_id: str | None = None,
    ) -> bool:
        return vector_presence.is_searchable_by_meaning(self, fact_id, profile_id)

    def count(self, profile_id: str | None = None) -> int:
        return vector_presence.count(self, profile_id)

    def indexed_fact_ids(self, profile_id: str) -> set[str]:
        return vector_presence.indexed_fact_ids(self, profile_id)

    def indexed_among(self, profile_id: str, fact_ids) -> set[str]:
        return vector_presence.indexed_among(self, profile_id, fact_ids)

    def gc_orphaned_vectors(self, profile_id: str | None = None) -> int:
        """Physically remove vec0 rows not referenced by any fact-addressable
        mapping (neither embedding_metadata nor vector_row_map).

        Such rows are unreachable by fact_id and would otherwise persist forever.
        Returns the count removed. No-op when the vector backend is unavailable.
        """
        if not self._available:
            return 0

        _wl = get_write_lock(self._db_path)
        with _wl:
            with self._lock:
                try:
                    with self._managed_connection() as conn:
                        conn.execute("BEGIN IMMEDIATE")
                        orphans = self._orphan_rowids(conn, profile_id)
                        for rowid in orphans:
                            conn.execute(
                                "DELETE FROM fact_embeddings WHERE rowid = ?",
                                (rowid,),
                            )
                        conn.commit()
                    return len(orphans)
                except Exception as exc:
                    logger.debug("gc_orphaned_vectors failed: %s", exc)
                    return 0

    @staticmethod
    def _orphan_rowids(conn: sqlite3.Connection, profile_id: str | None) -> set[int]:
        """vec0 rowids not referenced by embedding_metadata or vector_row_map."""
        if profile_id is not None:
            all_rowids = {r[0] for r in conn.execute(
                "SELECT rowid FROM fact_embeddings WHERE profile_id = ?",
                (profile_id,)).fetchall()}
            ref_meta = {r[0] for r in conn.execute(
                "SELECT vec_rowid FROM embedding_metadata WHERE profile_id = ?",
                (profile_id,)).fetchall()}
            ref_map = {r[0] for r in conn.execute(
                "SELECT vec_rowid FROM vector_row_map WHERE profile_id = ?",
                (profile_id,)).fetchall()}
        else:
            all_rowids = {r[0] for r in conn.execute(
                "SELECT rowid FROM fact_embeddings").fetchall()}
            ref_meta = {r[0] for r in conn.execute(
                "SELECT vec_rowid FROM embedding_metadata").fetchall()}
            ref_map = {r[0] for r in conn.execute(
                "SELECT vec_rowid FROM vector_row_map").fetchall()}
        return all_rowids - ref_meta - ref_map

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
