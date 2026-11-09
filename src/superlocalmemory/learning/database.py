# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Learning Database.

Persistent storage for the adaptive ranker: feedback signals, feature
vectors, engagement metrics, and serialized model state. Uses direct
sqlite3 connections (independent of V3 DatabaseManager) for isolation.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS learning_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    query TEXT NOT NULL,
    fact_id TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    value REAL DEFAULT 1.0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_signals_profile
    ON learning_signals(profile_id);
CREATE INDEX IF NOT EXISTS idx_signals_fact
    ON learning_signals(fact_id);

CREATE TABLE IF NOT EXISTS learning_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    query_id TEXT NOT NULL,
    fact_id TEXT NOT NULL,
    features_json TEXT NOT NULL,
    label REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_features_profile
    ON learning_features(profile_id);

CREATE TABLE IF NOT EXISTS learning_model_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL UNIQUE,
    state_bytes BLOB NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS engagement_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    value REAL NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_engagement_profile_metric
    ON engagement_metrics(profile_id, metric_type);
"""



from superlocalmemory.learning.signal_kinds import FEEDBACK_ONLY_SQL  # noqa: E402 - late import, order kept


class LearningDatabase:
    """Persistent storage for the adaptive ranker's training pipeline.

    Owns its own sqlite3 connection — independent of the main DB manager.
    Thread-safe writes via a lock. WAL mode for concurrent reads.

    Args:
        db_path: Path to the learning SQLite database file.
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_schema()

    @property
    def path(self) -> str:
        """Read-only path to the learning SQLite database.

        S8-ARC-02 (v3.4.22): public alternative to the underscore-private
        ``_db_path``. Callers that need a raw connection for specialised
        read patterns should prefer :meth:`ro_connection` over building
        one themselves so WAL + busy_timeout pragmas are consistent.
        """
        return self._db_path

    def ro_connection(self, *, timeout: float = 5.0) -> sqlite3.Connection:
        """Return a read-only-shaped connection with WAL/timeout pragmas set.

        Callers outside this class previously opened raw
        ``sqlite3.connect(lrn_db._db_path, ...)`` connections without the
        WAL/busy_timeout pragmas, making them vulnerable to ``database is
        locked`` errors under concurrent writer activity. This helper
        produces a configured connection they can use instead.
        """
        conn = sqlite3.connect(self._db_path, timeout=timeout)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
        except BaseException:
            # Setup failed: close it now, or the file stays open (Windows
            # then cannot delete or replace it) for as long as the error lives.
            conn.close()
            raise
        return conn

    def _connect(self) -> sqlite3.Connection:
        """Create a configured connection to the learning database."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
        except BaseException:
            # Setup failed: close it now, or the file stays open (Windows
            # then cannot delete or replace it) for as long as the error lives.
            conn.close()
            raise
        return conn

    def _init_schema(self) -> None:
        """Create tables and indexes if they do not exist."""
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def store_signal(
        self,
        profile_id: str,
        query: str,
        fact_id: str,
        signal_type: str,
        value: float = 1.0,
    ) -> int:
        """Record a feedback signal (e.g. recall_hit, recall_miss).

        Args:
            profile_id: Owning profile.
            query: The user query that produced this signal.
            fact_id: The fact that was matched (or missed).
            signal_type: Label for the signal kind.
            value: Numeric value (1.0 = strong positive, 0.0 = negative).

        Returns:
            Row ID of the inserted signal.
        """
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "INSERT INTO learning_signals "
                    "(profile_id, query, fact_id, signal_type, value, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (profile_id, query, fact_id, signal_type, value, self._now()),
                )
                conn.commit()
                return cur.lastrowid  # type: ignore[return-value]
            except sqlite3.Error as exc:
                conn.rollback()
                logger.error("store_signal failed: %s", exc)
                raise
            finally:
                conn.close()

    def get_signal_count(self, profile_id: str) -> int:
        """Count feedback signals for a profile.

        Used to determine the adaptive ranker phase:
        phase 1 (<20 signals) = defaults, phase 2 (20-200) = heuristic,
        phase 3 (>200) = LightGBM.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM learning_signals "
                f"WHERE profile_id = ?{FEEDBACK_ONLY_SQL}",
                (profile_id,),
            ).fetchone()
            return int(row["cnt"]) if row else 0
        finally:
            conn.close()

    def store_features(
        self,
        profile_id: str,
        query_id: str,
        fact_id: str,
        features: dict[str, float],
        label: float,
    ) -> int:
        """Store a labeled feature vector for LightGBM training.

        Args:
            profile_id: Owning profile.
            query_id: Identifier for the query (hash or UUID).
            fact_id: Identifier for the candidate fact.
            features: Feature dict (e.g. semantic_score, bm25_score, ...).
            label: Relevance label (1.0 = relevant, 0.0 = irrelevant).

        Returns:
            Row ID of the inserted record.
        """
        features_json = json.dumps(features, sort_keys=True)
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "INSERT INTO learning_features "
                    "(profile_id, query_id, fact_id, features_json, label, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (profile_id, query_id, fact_id, features_json, label, self._now()),
                )
                conn.commit()
                return cur.lastrowid  # type: ignore[return-value]
            except sqlite3.Error as exc:
                conn.rollback()
                logger.error("store_features failed: %s", exc)
                raise
            finally:
                conn.close()

    def get_training_data(self, profile_id: str, limit: int = 5000) -> list[dict[str, Any]]:
        """Retrieve labeled feature vectors for model training.

        Returns newest examples first.  Each dict contains:
        query_id, fact_id, features (dict), label, created_at.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT query_id, fact_id, features_json, label, created_at "
                "FROM learning_features WHERE profile_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (profile_id, limit),
            ).fetchall()
            results: list[dict[str, Any]] = []
            for row in rows:
                d = dict(row)
                d["features"] = json.loads(d.pop("features_json"))
                results.append(d)
            return results
        finally:
            conn.close()

    def store_model_state(self, profile_id: str, state_bytes: bytes) -> None:
        """Persist serialized model weights for a profile.

        Uses INSERT OR REPLACE so only one state row per profile exists.

        Args:
            profile_id: Owning profile.
            state_bytes: Serialized model bytes (from joblib or similar).
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO learning_model_state "
                    "(profile_id, state_bytes, updated_at) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(profile_id) DO UPDATE SET "
                    "state_bytes = excluded.state_bytes, "
                    "updated_at = excluded.updated_at",
                    (profile_id, state_bytes, self._now()),
                )
                conn.commit()
            except sqlite3.Error as exc:
                conn.rollback()
                logger.error("store_model_state failed: %s", exc)
                raise
            finally:
                conn.close()

    def load_model_state(self, profile_id: str) -> Optional[bytes]:
        """Load serialized model weights for a profile.

        Returns:
            The stored bytes, or None if no model has been persisted.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT state_bytes FROM learning_model_state WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            return bytes(row["state_bytes"]) if row else None
        finally:
            conn.close()

    def record_engagement(
        self,
        profile_id: str,
        metric_type: str,
        value: float = 1.0,
    ) -> None:
        """Increment (or create) an engagement counter.

        Examples of metric_type: recall_count, store_count,
        search_count, feedback_count.

        Args:
            profile_id: Owning profile.
            metric_type: The metric name.
            value: Amount to add (default 1).
        """
        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    "SELECT id, value FROM engagement_metrics "
                    "WHERE profile_id = ? AND metric_type = ?",
                    (profile_id, metric_type),
                ).fetchone()

                if existing:
                    new_value = float(existing["value"]) + value
                    conn.execute(
                        "UPDATE engagement_metrics SET value = ?, updated_at = ? WHERE id = ?",
                        (new_value, self._now(), existing["id"]),
                    )
                else:
                    conn.execute(
                        "INSERT INTO engagement_metrics "
                        "(profile_id, metric_type, value, updated_at) "
                        "VALUES (?, ?, ?, ?)",
                        (profile_id, metric_type, value, self._now()),
                    )
                conn.commit()
            except sqlite3.Error as exc:
                conn.rollback()
                logger.error("record_engagement failed: %s", exc)
                raise
            finally:
                conn.close()

    def get_engagement_stats(self, profile_id: str) -> dict[str, float]:
        """Get all engagement counters for a profile.

        Returns:
            Dict mapping metric_type to cumulative value.
            Empty dict if no engagement data exists.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT metric_type, value FROM engagement_metrics WHERE profile_id = ?",
                (profile_id,),
            ).fetchall()
            return {row["metric_type"]: float(row["value"]) for row in rows}
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # LLD-02 §4.8 — v3.4.22 writer surface
    # ------------------------------------------------------------------

    def count_signals(self, profile_id: str) -> int:
        """Count FEEDBACK rows in ``learning_signals`` for ``profile_id``.

        Used by ``_compute_ranker_phase`` + consolidation_worker training
        gate. Pure SELECT — thread-safe without lock.

        Exposure rows are excluded. This counted every row until 4.1.0, which
        on a live store meant 5,352 instead of 2 — a 2,675x inflation that
        held the ranker in Phase 3 on two feedback events. See
        ``learning/signal_kinds.py`` for why the predicate excludes exposures
        rather than naming feedback kinds.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM learning_signals "
                f"WHERE profile_id = ?{FEEDBACK_ONLY_SQL}",
                (profile_id,),
            ).fetchone()
            return int(row["cnt"]) if row else 0
        finally:
            conn.close()

    def persist_model(
        self,
        *,
        profile_id: str,
        state_bytes: bytes,
        bytes_sha256: str,
        feature_names: list[str],
        trained_on_count: int,
        metrics: dict,
        model_version: str = "3.4.23",
    ) -> int:
        """Persist a newly trained model and flip the active flag.

        LLD-02 §4.8 — single TX:
            1. UPDATE existing active row → is_active = 0.
            2. INSERT new row with is_active = 1.

        Requires M002 (columns ``bytes_sha256``, ``feature_names``,
        ``metrics_json``, ``trained_on_count``, ``is_active``). Raises if
        M002 hasn't been applied.

        Returns the new row id.
        """
        if not isinstance(state_bytes, (bytes, bytearray)):
            raise TypeError("state_bytes must be bytes")
        if not bytes_sha256 or len(bytes_sha256) != 64:
            raise ValueError("bytes_sha256 must be 64 hex chars")
        names_json = json.dumps(list(feature_names), separators=(",", ":"))
        metrics_json = json.dumps(dict(metrics), separators=(",", ":"))
        now = self._now()

        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "UPDATE learning_model_state "
                    "SET is_active = 0 "
                    "WHERE profile_id = ? AND is_active = 1",
                    (profile_id,),
                )
                cur = conn.execute(
                    "INSERT INTO learning_model_state "
                    "(profile_id, model_version, state_bytes, bytes_sha256, "
                    " trained_on_count, feature_names, metrics_json, "
                    " is_active, trained_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                    (
                        profile_id,
                        model_version,
                        bytes(state_bytes),
                        bytes_sha256.lower(),
                        int(trained_on_count),
                        names_json,
                        metrics_json,
                        now,
                        now,
                    ),
                )
                conn.commit()
                return int(cur.lastrowid or 0)
            except sqlite3.Error as exc:
                conn.rollback()
                logger.error("persist_model failed: %s", exc)
                raise
            finally:
                conn.close()

    def load_active_model(self, profile_id: str) -> Optional[dict]:
        """Return the active model row as a dict, or ``None`` if none.

        Post-M002 schema. Keys: ``state_bytes``, ``bytes_sha256``,
        ``feature_names`` (JSON str), ``trained_at``, ``model_version``.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT state_bytes, bytes_sha256, feature_names, trained_at, "
                "       model_version "
                "FROM learning_model_state "
                "WHERE profile_id = ? AND is_active = 1 "
                "LIMIT 1",
                (profile_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "state_bytes": bytes(row["state_bytes"]),
                "bytes_sha256": row["bytes_sha256"],
                "feature_names": row["feature_names"],
                "trained_at": row["trained_at"],
                "model_version": row["model_version"],
            }
        except sqlite3.Error as exc:
            logger.error("load_active_model failed: %s", exc)
            return None
        finally:
            conn.close()

    # --- training-row fetch (version-gated on M006) --------------------

    _SQL_POSITION_ONLY = (
        "SELECT s.id AS signal_id, s.query_id, s.fact_id, s.position, "
        "       s.created_at, f.features_json, NULL AS outcome_reward "
        "FROM learning_signals s "
        "JOIN learning_features f "
        "  ON f.signal_id = s.id AND f.profile_id = s.profile_id "
        "WHERE s.profile_id = ? "
        "  AND s.signal_type IN ('candidate', 'shown', 'legacy_feedback') "
        "  AND f.is_synthetic = 0 "
        "ORDER BY s.created_at DESC "
        "LIMIT ?"
    )

    def _migration_applied(self, name: str) -> bool:
        """Return True if ``name`` is recorded complete in migration_log.

        M006 (action_outcomes.reward) lands in v3.4.22. When absent, we
        fall back to the position-only training query.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT status FROM migration_log WHERE name = ?",
                (name,),
            ).fetchone()
        except sqlite3.Error:
            return False
        finally:
            conn.close()
        if row is None:
            return False
        return row["status"] == "complete"

    def fetch_training_examples(
        self,
        *,
        profile_id: str,
        limit: int = 2000,
        min_outcome_age_sec: int = 60,
        include_synthetic: bool = False,
        memory_db: Optional[Union[str, Path]] = None,
    ) -> list[dict]:
        """Fetch labelled rows for LightGBM lambdarank training.

        Rows come from :func:`training_rows.fetch_labelled_rows`: exposure
        features joined to the outcomes that name the recall and the memory,
        across learning.db and memory.db (``memory_db``, default the one beside
        this file). Judged queries only; position is never a label.
        The old join looked for ``action_outcomes`` in learning.db (it lives in
        memory.db) and gated on M006 in learning.db's migration log (M006 is
        logged in memory.db), so production training never saw a real label.
        With no reachable outcome table the legacy position rows remain.
        Newest-first; the caller regroups by ``query_id``.
        """
        from superlocalmemory.learning.training_rows import (
            fetch_labelled_rows,
            outcome_source,
        )

        mem = Path(memory_db) if memory_db else Path(self._db_path).parent / "memory.db"
        conn = self._connect()
        try:
            table = outcome_source(conn, mem)
            if table is None:
                return self._fetch_position_only(profile_id, limit,
                                                 include_synthetic)
            return fetch_labelled_rows(
                conn, table, profile_id=profile_id, limit=limit,
                min_outcome_age_sec=min_outcome_age_sec,
                include_synthetic=include_synthetic,
            )
        except sqlite3.Error as exc:
            logger.warning("fetch_training_examples failed: %s", exc)
            return []
        finally:
            conn.close()

    def _fetch_position_only(
        self, profile_id: str, limit: int, include_synthetic: bool,
    ) -> list[dict]:
        sql = self._SQL_POSITION_ONLY
        if include_synthetic:
            sql = sql.replace(" AND f.is_synthetic = 0 ", " ")
        conn = self._connect()
        try:
            try:
                rows = conn.execute(sql, (profile_id, int(limit))).fetchall()
            except sqlite3.Error as exc:
                logger.warning("fetch_training_examples (no M006): %s", exc)
                return []
            out: list[dict] = []
            for row in rows:
                d = dict(row)
                try:
                    d["features"] = json.loads(d.pop("features_json") or "{}")
                except (ValueError, TypeError):
                    d["features"] = {}
                out.append(d)
            return out
        finally:
            conn.close()

    @staticmethod
    def _other_profile_tables(conn: sqlite3.Connection, already: list[str]) -> list[str]:
        """Every other table of learning.db that carries a profile (a producer added later).

        ``channel_credits`` and ``co_retrieval_edges`` (written by ordinary recall)
        were not in the list above, so an erasure left the ids of the memories a
        person's queries co-retrieved. Receipts and the markers that refuse a late
        write are decided by ``profile_fold_sidecars`` and are not touched here.
        """
        from superlocalmemory.storage.profile_fold_sidecars import DELETE, LEARNING_DECISIONS

        extra: list[str] = []
        for (name,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            if name in already or name.startswith("sqlite_"):
                continue
            try:
                cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{name}")')}
            except sqlite3.OperationalError:
                continue
            if "profile_id" in cols and LEARNING_DECISIONS.get(name, DELETE) == DELETE:
                extra.append(name)
        return extra

    def reset(self, profile_id: Optional[str] = None) -> None:
        """Delete learning data. GDPR Article 17 handler.

        Args:
            profile_id: If provided, only erase data for that profile.
                        If None, erase ALL learning data.
        """
        if profile_id:
            from superlocalmemory.storage.agent_experience import purge_profile_receipts

            purge_profile_receipts(self._db_path, profile_id, close_profile=False)
        with self._lock:
            conn = self._connect()
            try:
                tables = [
                    "learning_signals",
                    "learning_features",
                    "learning_model_state",
                    "engagement_metrics",
                    # bandit_plays holds shown_fact_ids — the identifiers of the
                    # memories a person was actually shown — and bandit_arms is
                    # a derived behavioural profile. Both are personal data and
                    # neither was erased: this list was hardcoded, so
                    # forget_profile() reported success and left a row reading
                    # ["alice-private-memory-1", ...] in place. Reproduced
                    # before fixing.
                    #
                    # The retention sweep is not a substitute. It deletes only
                    # SETTLED plays older than the horizon, so an unsettled row
                    # would have survived indefinitely.
                    "bandit_plays",
                    "bandit_arms",
                    # Which arm/model served each of the profile's recalls,
                    # and the reward each earned, in a ranker's live test.
                    "shadow_assignments",
                    "shadow_observations",
                ]
                receipt_tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name IN ('agent_experiences', 'cognitive_turn_receipts', "
                        "'external_evidence_receipts', 'execution_learning_receipts', "
                        "'execution_learning_events')"
                    )
                }
                if profile_id is None:
                    tables.extend(sorted(receipt_tables))
                present = {
                    row[0] for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                if profile_id:
                    tables = [*tables, *self._other_profile_tables(conn, tables)]
                for table in tables:
                    # A store predating any of these tables must not abort the
                    # erasure of the rest — an Article 17 request that fails
                    # halfway is worse than one that skips an absent table.
                    if table not in present:
                        continue
                    if profile_id:
                        conn.execute(
                            f"DELETE FROM {table} WHERE profile_id = ?",
                            (profile_id,),
                        )
                    else:
                        conn.execute(f"DELETE FROM {table}")
                conn.commit()
                # The in-process ranking counter holds this profile's recent
                # winners. Ephemeral, but an erasure must not leave them in a
                # live process for the rest of its lifetime.
                try:
                    from superlocalmemory.learning.pcos import RECENT_TOPS

                    RECENT_TOPS.forget(profile_id or "")
                except Exception:  # pragma: no cover — advisory
                    pass
                # The per-session working set is this profile's data too.
                # Cleared here as well as in the compliance path, because this
                # is reached directly — by the transaction owners and by tests —
                # and a residue left behind goes on biasing any later session
                # that reuses one of the erased profile's session ids.
                try:
                    from superlocalmemory.core.working_memory import (
                        discard_profile,
                    )

                    discard_profile(profile_id or "")
                except Exception:  # pragma: no cover — never block an erasure
                    pass
                logger.info(
                    "Learning data reset%s",
                    f" for profile {profile_id}" if profile_id else " (all)",
                )
            except sqlite3.Error as exc:
                conn.rollback()
                logger.error("reset failed: %s", exc)
                raise
            finally:
                conn.close()
