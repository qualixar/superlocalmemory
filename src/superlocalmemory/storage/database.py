# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Database Manager.

SQLite with WAL, profile-scoped CRUD, FTS5 search, BM25 persistence.
Concurrent-safe: WAL mode + busy_timeout + retry on SQLITE_BUSY.
Multiple processes (MCP, CLI, integrations) can read/write safely.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""
from __future__ import annotations

import json, logging, sqlite3, threading, time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Generator

from superlocalmemory.storage.models import (
    AtomicFact, CanonicalEntity, ConsolidationAction, ConsolidationActionType,
    EdgeType, EntityAlias, EntityProfile, FactType, GraphEdge,
    MemoryLifecycle, MemoryRecord, MemoryScene, SignalType, TemporalEvent,
    TrustScore,
)

logger = logging.getLogger(__name__)

def _jl(raw: Any, default: Any = None) -> Any:
    """JSON-load a value, returning *default* on None/empty."""
    if raw is None or raw == "":
        return default if default is not None else []
    return json.loads(raw)

def _jd(val: Any) -> str | None:
    """JSON-dump a list/dict, or return None."""
    return json.dumps(val) if val is not None else None


_BUSY_TIMEOUT_MS = 10_000   # 10 seconds — wait for other writers
_MAX_RETRIES = 5            # retry on transient SQLITE_BUSY
_RETRY_BASE_DELAY = 0.1    # seconds — exponential backoff base


class DatabaseManager:
    """Concurrent-safe SQLite manager with WAL, profile isolation, and FTS5.

    Designed for multi-process access: MCP server, CLI, LangChain, CrewAI,
    and other integrations can all read/write the same database safely.

    Concurrency model:
    - WAL mode: readers never block writers, writers never block readers
    - busy_timeout: writers wait up to 10s for other writers instead of failing
    - Retry with backoff: transient SQLITE_BUSY errors are retried automatically
    - Per-call connections: no shared state between processes
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._txn_conn: sqlite3.Connection | None = None
        self._enable_wal()

    def _enable_wal(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.commit()
        finally:
            conn.close()

    def initialize(self, schema_module: ModuleType) -> None:
        """Create all tables. *schema_module* must expose ``create_all_tables(conn)``."""
        conn = self._connect()
        try:
            schema_module.create_all_tables(conn)
            conn.commit()
            logger.info("Schema initialized at %s", self.db_path)
        finally:
            conn.close()

    def close(self) -> None:
        """No-op for per-call connection model."""

    def __enter__(self) -> DatabaseManager:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=_BUSY_TIMEOUT_MS / 1000)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Atomic transaction. All writes commit or rollback together."""
        with self._lock:
            conn = self._connect()
            self._txn_conn = conn
            try:
                yield
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                self._txn_conn = None
                conn.close()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Execute SQL with automatic retry on SQLITE_BUSY.

        Uses shared conn inside transaction, else per-call with retry.
        """
        if self._txn_conn is not None:
            return self._txn_conn.execute(sql, params).fetchall()

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
                conn.commit()
                return rows
            except sqlite3.OperationalError as exc:
                last_error = exc
                if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.debug(
                        "DB busy (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, _MAX_RETRIES, delay, exc,
                    )
                    time.sleep(delay)
                    continue
                raise
            finally:
                conn.close()

        logger.warning("DB operation failed after %d retries: %s", _MAX_RETRIES, last_error)
        raise last_error  # type: ignore[misc]

    def store_memory(self, record: MemoryRecord) -> str:
        """Persist a raw memory record. Returns memory_id."""
        self.execute(
            """INSERT OR REPLACE INTO memories
               (memory_id, profile_id, content, session_id, speaker,
                role, session_date, created_at, metadata_json)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (record.memory_id, record.profile_id, record.content,
             record.session_id, record.speaker, record.role,
             record.session_date, record.created_at,
             json.dumps(record.metadata)),
        )
        return record.memory_id

    def update_memory_summary(self, memory_id: str, summary: str) -> None:
        """Store a generated summary for a memory record."""
        try:
            self.execute(
                "UPDATE memories SET metadata_json = json_set("
                "  COALESCE(metadata_json, '{}'), '$.summary', ?"
                ") WHERE memory_id = ?",
                (summary, memory_id),
            )
        except Exception:
            pass  # Non-critical — summary is enhancement only

    def get_memory_summary(self, memory_id: str) -> str:
        """Retrieve stored summary for a memory, or empty string."""
        try:
            rows = self.execute(
                "SELECT json_extract(metadata_json, '$.summary') as s "
                "FROM memories WHERE memory_id = ?",
                (memory_id,),
            )
            if rows:
                return dict(rows[0]).get("s") or ""
        except Exception:
            pass
        return ""

    def store_fact(self, fact: AtomicFact) -> str:
        """Persist an atomic fact. Returns fact_id."""
        self.execute(
            """INSERT OR REPLACE INTO atomic_facts
               (fact_id, memory_id, profile_id, content, fact_type,
                entities_json, canonical_entities_json,
                observation_date, referenced_date, interval_start, interval_end,
                confidence, importance, evidence_count, access_count,
                source_turn_ids_json, session_id,
                embedding, fisher_mean, fisher_variance,
                lifecycle, langevin_position,
                emotional_valence, emotional_arousal, signal_type, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (fact.fact_id, fact.memory_id, fact.profile_id, fact.content,
             fact.fact_type.value,
             json.dumps(fact.entities), json.dumps(fact.canonical_entities),
             fact.observation_date, fact.referenced_date,
             fact.interval_start, fact.interval_end,
             fact.confidence, fact.importance, fact.evidence_count, fact.access_count,
             json.dumps(fact.source_turn_ids), fact.session_id,
             _jd(fact.embedding), _jd(fact.fisher_mean), _jd(fact.fisher_variance),
             fact.lifecycle.value, _jd(fact.langevin_position),
             fact.emotional_valence, fact.emotional_arousal,
             fact.signal_type.value, fact.created_at),
        )
        return fact.fact_id

    def _row_to_fact(self, row: sqlite3.Row) -> AtomicFact:
        """Deserialize a row into AtomicFact."""
        d = dict(row)
        return AtomicFact(
            fact_id=d["fact_id"], memory_id=d["memory_id"],
            profile_id=d["profile_id"], content=d["content"],
            fact_type=FactType(d["fact_type"]),
            entities=_jl(d.get("entities_json")),
            canonical_entities=_jl(d.get("canonical_entities_json")),
            observation_date=d.get("observation_date"),
            referenced_date=d.get("referenced_date"),
            interval_start=d.get("interval_start"),
            interval_end=d.get("interval_end"),
            confidence=d["confidence"], importance=d["importance"],
            evidence_count=d["evidence_count"], access_count=d["access_count"],
            source_turn_ids=_jl(d.get("source_turn_ids_json")),
            session_id=d.get("session_id", ""),
            embedding=_jl(d.get("embedding"), None),
            fisher_mean=_jl(d.get("fisher_mean"), None),
            fisher_variance=_jl(d.get("fisher_variance"), None),
            lifecycle=MemoryLifecycle(d["lifecycle"]) if d.get("lifecycle") else MemoryLifecycle.ACTIVE,
            langevin_position=_jl(d.get("langevin_position"), None),
            emotional_valence=d.get("emotional_valence", 0.0),
            emotional_arousal=d.get("emotional_arousal", 0.0),
            signal_type=SignalType(d["signal_type"]) if d.get("signal_type") else SignalType.FACTUAL,
            created_at=d["created_at"],
        )

    def get_all_facts(self, profile_id: str) -> list[AtomicFact]:
        """All facts for a profile, newest first."""
        rows = self.execute(
            "SELECT * FROM atomic_facts WHERE profile_id = ? ORDER BY created_at DESC",
            (profile_id,),
        )
        return [self._row_to_fact(r) for r in rows]

    def get_facts_by_entity(self, entity_id: str, profile_id: str) -> list[AtomicFact]:
        """Facts whose canonical_entities JSON array contains *entity_id*."""
        rows = self.execute(
            "SELECT * FROM atomic_facts WHERE profile_id = ? AND canonical_entities_json LIKE ? "
            "ORDER BY created_at DESC",
            (profile_id, f'%"{entity_id}"%'),
        )
        return [self._row_to_fact(r) for r in rows]

    def get_facts_by_type(self, fact_type: FactType, profile_id: str) -> list[AtomicFact]:
        """All facts of a given type for a profile."""
        rows = self.execute(
            "SELECT * FROM atomic_facts WHERE profile_id = ? AND fact_type = ? "
            "ORDER BY created_at DESC",
            (profile_id, fact_type.value),
        )
        return [self._row_to_fact(r) for r in rows]

    # Allowed columns for partial updates (prevents SQL injection via dict keys)
    _UPDATABLE_FACT_COLUMNS: frozenset[str] = frozenset({
        "content", "fact_type", "entities_json", "canonical_entities_json",
        "observation_date", "referenced_date", "interval_start", "interval_end",
        "confidence", "importance", "evidence_count", "access_count",
        "source_turn_ids_json", "session_id", "embedding",
        "fisher_mean", "fisher_variance", "lifecycle", "langevin_position",
        "emotional_valence", "emotional_arousal", "signal_type",
    })

    def update_fact(self, fact_id: str, updates: dict[str, Any]) -> None:
        """Partial update on a fact. JSON-serializes list/dict values."""
        if not updates:
            raise ValueError("updates dict must not be empty")
        bad_keys = set(updates) - self._UPDATABLE_FACT_COLUMNS
        if bad_keys:
            raise ValueError(f"Disallowed column(s): {bad_keys}")
        clean: dict[str, Any] = {}
        for k, v in updates.items():
            if isinstance(v, (list, dict)):
                clean[k] = json.dumps(v)
            elif isinstance(v, (MemoryLifecycle, FactType, SignalType)):
                clean[k] = v.value
            else:
                clean[k] = v
        set_clause = ", ".join(f"{k} = ?" for k in clean)
        self.execute(
            f"UPDATE atomic_facts SET {set_clause} WHERE fact_id = ?",
            (*clean.values(), fact_id),
        )

    def delete_fact(self, fact_id: str) -> None:
        """Hard-delete a fact."""
        self.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (fact_id,))

    def get_fact_count(self, profile_id: str) -> int:
        """Total fact count for a profile."""
        rows = self.execute(
            "SELECT COUNT(*) AS c FROM atomic_facts WHERE profile_id = ?", (profile_id,),
        )
        return int(rows[0]["c"]) if rows else 0

    def store_entity(self, entity: CanonicalEntity) -> str:
        """Persist a canonical entity. Returns entity_id."""
        self.execute(
            """INSERT OR REPLACE INTO canonical_entities
               (entity_id, profile_id, canonical_name, entity_type,
                first_seen, last_seen, fact_count)
               VALUES (?,?,?,?,?,?,?)""",
            (entity.entity_id, entity.profile_id, entity.canonical_name,
             entity.entity_type, entity.first_seen, entity.last_seen,
             entity.fact_count),
        )
        return entity.entity_id

    def get_entity_by_name(self, name: str, profile_id: str) -> CanonicalEntity | None:
        """Look up entity by name (case-insensitive)."""
        rows = self.execute(
            "SELECT * FROM canonical_entities WHERE profile_id = ? AND LOWER(canonical_name) = LOWER(?)",
            (profile_id, name),
        )
        if not rows:
            return None
        d = dict(rows[0])
        return CanonicalEntity(
            entity_id=d["entity_id"], profile_id=d["profile_id"],
            canonical_name=d["canonical_name"], entity_type=d["entity_type"],
            first_seen=d["first_seen"], last_seen=d["last_seen"],
            fact_count=d["fact_count"],
        )

    def store_alias(self, alias: EntityAlias) -> str:
        """Persist an entity alias. Returns alias_id."""
        self.execute(
            "INSERT OR REPLACE INTO entity_aliases "
            "(alias_id, entity_id, alias, confidence, source) VALUES (?,?,?,?,?)",
            (alias.alias_id, alias.entity_id, alias.alias,
             alias.confidence, alias.source),
        )
        return alias.alias_id

    def get_aliases_for_entity(self, entity_id: str) -> list[EntityAlias]:
        """All aliases for a canonical entity."""
        rows = self.execute(
            "SELECT * FROM entity_aliases WHERE entity_id = ?", (entity_id,),
        )
        return [
            EntityAlias(**{k: dict(r)[k] for k in ("alias_id", "entity_id", "alias", "confidence", "source")})
            for r in rows
        ]

    def get_memory_content_batch(self, memory_ids: list[str]) -> dict[str, str]:
        """Batch-fetch original memory text. Returns {memory_id: content}."""
        if not memory_ids:
            return {}
        unique_ids = list(set(memory_ids))
        ph = ','.join('?' * len(unique_ids))
        rows = self.execute(
            f"SELECT memory_id, content FROM memories WHERE memory_id IN ({ph})",
            tuple(unique_ids),
        )
        return {dict(r)["memory_id"]: dict(r)["content"] for r in rows}

    def get_facts_by_memory_id(
        self, memory_id: str, profile_id: str,
    ) -> list[AtomicFact]:
        """Get all atomic facts for a given memory_id."""
        rows = self.execute(
            "SELECT * FROM atomic_facts WHERE memory_id = ? AND profile_id = ? "
            "ORDER BY confidence DESC",
            (memory_id, profile_id),
        )
        return [self._row_to_fact(r) for r in rows]

    def store_edge(self, edge: GraphEdge) -> str:
        """Persist a graph edge. Returns edge_id."""
        self.execute(
            """INSERT OR REPLACE INTO graph_edges
               (edge_id, profile_id, source_id, target_id, edge_type, weight, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (edge.edge_id, edge.profile_id, edge.source_id, edge.target_id,
             edge.edge_type.value, edge.weight, edge.created_at),
        )
        return edge.edge_id

    def get_edges_for_node(self, node_id: str, profile_id: str) -> list[GraphEdge]:
        """All edges where node_id is source or target."""
        rows = self.execute(
            "SELECT * FROM graph_edges WHERE profile_id = ? "
            "AND (source_id = ? OR target_id = ?)",
            (profile_id, node_id, node_id),
        )
        return [
            GraphEdge(
                edge_id=(d := dict(r))["edge_id"], profile_id=d["profile_id"],
                source_id=d["source_id"], target_id=d["target_id"],
                edge_type=EdgeType(d["edge_type"]), weight=d["weight"],
                created_at=d["created_at"],
            )
            for r in rows
        ]

    def store_temporal_event(self, event: TemporalEvent) -> str:
        """Persist a temporal event. Returns event_id."""
        self.execute(
            """INSERT OR REPLACE INTO temporal_events
               (event_id, profile_id, entity_id, fact_id,
                observation_date, referenced_date, interval_start, interval_end,
                description)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (event.event_id, event.profile_id, event.entity_id, event.fact_id,
             event.observation_date, event.referenced_date,
             event.interval_start, event.interval_end, event.description),
        )
        return event.event_id

    def get_temporal_events(self, entity_id: str, profile_id: str) -> list[TemporalEvent]:
        """All temporal events for an entity, newest first."""
        rows = self.execute(
            "SELECT * FROM temporal_events WHERE profile_id = ? AND entity_id = ? "
            "ORDER BY observation_date DESC",
            (profile_id, entity_id),
        )
        return [
            TemporalEvent(
                event_id=(d := dict(r))["event_id"], profile_id=d["profile_id"],
                entity_id=d["entity_id"], fact_id=d["fact_id"],
                observation_date=d.get("observation_date"),
                referenced_date=d.get("referenced_date"),
                interval_start=d.get("interval_start"),
                interval_end=d.get("interval_end"),
                description=d.get("description", ""),
            )
            for r in rows
        ]

    def store_bm25_tokens(self, fact_id: str, profile_id: str, tokens: list[str]) -> None:
        """Persist BM25 tokens for a fact (survives restart)."""
        self.execute(
            "INSERT OR REPLACE INTO bm25_tokens (fact_id, profile_id, tokens) VALUES (?,?,?)",
            (fact_id, profile_id, json.dumps(tokens)),
        )

    def get_all_bm25_tokens(self, profile_id: str) -> dict[str, list[str]]:
        """Load full BM25 index: fact_id -> token list."""
        rows = self.execute(
            "SELECT fact_id, tokens FROM bm25_tokens WHERE profile_id = ?",
            (profile_id,),
        )
        return {dict(r)["fact_id"]: json.loads(dict(r)["tokens"]) for r in rows}

    def search_facts_fts(self, query: str, profile_id: str, limit: int = 20) -> list[AtomicFact]:
        """Full-text search via FTS5, joined to facts table for reconstruction."""
        rows = self.execute(
            """SELECT f.* FROM atomic_facts_fts AS fts
               JOIN atomic_facts AS f ON f.fact_id = fts.fact_id
               WHERE fts.atomic_facts_fts MATCH ? AND f.profile_id = ?
               ORDER BY fts.rank LIMIT ?""",
            (query, profile_id, limit),
        )
        return [self._row_to_fact(r) for r in rows]

    def list_tables(self) -> set[str]:
        """All table names in the database."""
        rows = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return {dict(r)["name"] for r in rows}

    def get_config(self, key: str) -> str | None:
        """Read a config value by key."""
        rows = self.execute("SELECT value FROM config WHERE key = ?", (key,))
        return str(rows[0]["value"]) if rows else None

    def set_config(self, key: str, value: str) -> None:
        """Write a config value (upsert)."""
        from datetime import UTC, datetime
        self.execute(
            "INSERT OR REPLACE INTO config (key, value, updated_at) VALUES (?,?,?)",
            (key, value, datetime.now(UTC).isoformat()),
        )

    # ------------------------------------------------------------------
    # Phase 0.6: Missing methods (BLOCKER / CRITICAL / HIGH)
    # ------------------------------------------------------------------

    def get_fact(self, fact_id: str) -> AtomicFact | None:
        """Get a single fact by ID."""
        rows = self.execute(
            "SELECT * FROM atomic_facts WHERE fact_id = ?", (fact_id,),
        )
        return self._row_to_fact(rows[0]) if rows else None

    def get_facts_by_ids(
        self, fact_ids: list[str], profile_id: str,
    ) -> list[AtomicFact]:
        """Get multiple facts by their IDs, scoped to a profile."""
        if not fact_ids:
            return []
        placeholders = ",".join("?" for _ in fact_ids)
        rows = self.execute(
            f"SELECT * FROM atomic_facts WHERE fact_id IN ({placeholders}) "
            f"AND profile_id = ? ORDER BY created_at DESC",
            (*fact_ids, profile_id),
        )
        return [self._row_to_fact(r) for r in rows]

    def store_entity_profile(self, ep: EntityProfile) -> str:
        """Persist an entity profile. Returns profile_entry_id."""
        self.execute(
            """INSERT OR REPLACE INTO entity_profiles
               (profile_entry_id, entity_id, profile_id,
                knowledge_summary, fact_ids_json, last_updated)
               VALUES (?,?,?,?,?,?)""",
            (ep.profile_entry_id, ep.entity_id, ep.profile_id,
             ep.knowledge_summary, json.dumps(ep.fact_ids), ep.last_updated),
        )
        return ep.profile_entry_id

    def get_entity_profiles_by_entity(
        self, entity_id: str, profile_id: str,
    ) -> list[EntityProfile]:
        """All profile entries for an entity within a profile scope."""
        rows = self.execute(
            "SELECT * FROM entity_profiles WHERE entity_id = ? AND profile_id = ? "
            "ORDER BY last_updated DESC",
            (entity_id, profile_id),
        )
        return [
            EntityProfile(
                profile_entry_id=(d := dict(r))["profile_entry_id"],
                entity_id=d["entity_id"], profile_id=d["profile_id"],
                knowledge_summary=d["knowledge_summary"],
                fact_ids=_jl(d.get("fact_ids_json")),
                last_updated=d["last_updated"],
            )
            for r in rows
        ]

    def store_scene(self, scene: MemoryScene) -> str:
        """Persist a memory scene. Returns scene_id."""
        self.execute(
            """INSERT OR REPLACE INTO memory_scenes
               (scene_id, profile_id, theme, fact_ids_json,
                entity_ids_json, created_at, last_updated)
               VALUES (?,?,?,?,?,?,?)""",
            (scene.scene_id, scene.profile_id, scene.theme,
             json.dumps(scene.fact_ids), json.dumps(scene.entity_ids),
             scene.created_at, scene.last_updated),
        )
        return scene.scene_id

    def _row_to_scene(self, row: sqlite3.Row) -> MemoryScene:
        """Deserialize a row into MemoryScene."""
        d = dict(row)
        return MemoryScene(
            scene_id=d["scene_id"], profile_id=d["profile_id"],
            theme=d.get("theme", ""),
            fact_ids=_jl(d.get("fact_ids_json")),
            entity_ids=_jl(d.get("entity_ids_json")),
            created_at=d["created_at"], last_updated=d["last_updated"],
        )

    def get_scene(self, scene_id: str) -> MemoryScene | None:
        """Get a scene by ID."""
        rows = self.execute(
            "SELECT * FROM memory_scenes WHERE scene_id = ?", (scene_id,),
        )
        return self._row_to_scene(rows[0]) if rows else None

    def get_all_scenes(self, profile_id: str) -> list[MemoryScene]:
        """All scenes for a profile, newest first."""
        rows = self.execute(
            "SELECT * FROM memory_scenes WHERE profile_id = ? "
            "ORDER BY last_updated DESC",
            (profile_id,),
        )
        return [self._row_to_scene(r) for r in rows]

    def get_scenes_for_fact(
        self, fact_id: str, profile_id: str,
    ) -> list[MemoryScene]:
        """All scenes whose fact_ids JSON array contains *fact_id*."""
        rows = self.execute(
            "SELECT * FROM memory_scenes WHERE profile_id = ? "
            "AND fact_ids_json LIKE ? ORDER BY last_updated DESC",
            (profile_id, f'%"{fact_id}"%'),
        )
        return [self._row_to_scene(r) for r in rows]

    def increment_entity_fact_count(self, entity_id: str) -> None:
        """Atomically increment fact_count for a canonical entity."""
        self.execute(
            "UPDATE canonical_entities SET fact_count = fact_count + 1 "
            "WHERE entity_id = ?",
            (entity_id,),
        )

    def store_trust_score(self, ts: TrustScore) -> str:
        """Persist a trust score. Returns trust_id."""
        self.execute(
            """INSERT OR REPLACE INTO trust_scores
               (trust_id, profile_id, target_type, target_id,
                trust_score, evidence_count, last_updated)
               VALUES (?,?,?,?,?,?,?)""",
            (ts.trust_id, ts.profile_id, ts.target_type, ts.target_id,
             ts.trust_score, ts.evidence_count, ts.last_updated),
        )
        return ts.trust_id

    def get_trust_score(
        self, target_type: str, target_id: str, profile_id: str,
    ) -> TrustScore | None:
        """Look up trust score for a specific target."""
        rows = self.execute(
            "SELECT * FROM trust_scores WHERE target_type = ? "
            "AND target_id = ? AND profile_id = ?",
            (target_type, target_id, profile_id),
        )
        if not rows:
            return None
        d = dict(rows[0])
        return TrustScore(
            trust_id=d["trust_id"], profile_id=d["profile_id"],
            target_type=d["target_type"], target_id=d["target_id"],
            trust_score=d["trust_score"], evidence_count=d["evidence_count"],
            last_updated=d["last_updated"],
        )

    def store_consolidation_action(self, action: ConsolidationAction) -> str:
        """Log a consolidation decision. Returns action_id."""
        self.execute(
            """INSERT OR REPLACE INTO consolidation_log
               (action_id, profile_id, action_type, new_fact_id,
                existing_fact_id, reason, timestamp)
               VALUES (?,?,?,?,?,?,?)""",
            (action.action_id, action.profile_id,
             action.action_type.value, action.new_fact_id,
             action.existing_fact_id, action.reason, action.timestamp),
        )
        return action.action_id

    def get_temporal_events_by_range(
        self, profile_id: str, start_date: str, end_date: str,
    ) -> list[TemporalEvent]:
        """Temporal events within a date range (inclusive)."""
        rows = self.execute(
            "SELECT * FROM temporal_events WHERE profile_id = ? "
            "AND (referenced_date BETWEEN ? AND ? "
            "     OR observation_date BETWEEN ? AND ?) "
            "ORDER BY observation_date DESC",
            (profile_id, start_date, end_date, start_date, end_date),
        )
        return [
            TemporalEvent(
                event_id=(d := dict(r))["event_id"],
                profile_id=d["profile_id"],
                entity_id=d["entity_id"], fact_id=d["fact_id"],
                observation_date=d.get("observation_date"),
                referenced_date=d.get("referenced_date"),
                interval_start=d.get("interval_start"),
                interval_end=d.get("interval_end"),
                description=d.get("description", ""),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Phase 2: fact_context CRUD (Auto-Invoke Engine)
    # ------------------------------------------------------------------

    def store_fact_context(
        self,
        fact_id: str,
        profile_id: str,
        contextual_description: str,
        keywords: str,
        generated_by: str = "rules",
    ) -> None:
        """Store or replace contextual description for a fact."""
        self.execute(
            "INSERT OR REPLACE INTO fact_context "
            "(fact_id, profile_id, contextual_description, keywords, generated_by) "
            "VALUES (?, ?, ?, ?, ?)",
            (fact_id, profile_id, contextual_description, keywords, generated_by),
        )

    def get_fact_context(self, fact_id: str) -> dict | None:
        """Get contextual description for a fact."""
        rows = self.execute(
            "SELECT * FROM fact_context WHERE fact_id = ?", (fact_id,),
        )
        return dict(rows[0]) if rows else None

    def get_all_fact_contexts(self, profile_id: str) -> list[dict]:
        """Get all contextual descriptions for a profile."""
        rows = self.execute(
            "SELECT * FROM fact_context WHERE profile_id = ?", (profile_id,),
        )
        return [dict(r) for r in rows]

    def delete_fact_context(self, fact_id: str) -> None:
        """Delete contextual description for a fact."""
        self.execute("DELETE FROM fact_context WHERE fact_id = ?", (fact_id,))

    # ------------------------------------------------------------------
    # Phase 3: Association Graph CRUD (Rule 15)
    # ------------------------------------------------------------------

    def store_association_edge(self, edge: dict) -> None:
        """Persist an association edge."""
        self.execute(
            "INSERT OR IGNORE INTO association_edges "
            "(edge_id, profile_id, source_fact_id, target_fact_id, "
            " association_type, weight, co_access_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (edge["edge_id"], edge["profile_id"],
             edge["source_fact_id"], edge["target_fact_id"],
             edge["association_type"], edge["weight"],
             edge.get("co_access_count", 0)),
        )

    def get_association_edges(
        self, fact_id: str, profile_id: str,
    ) -> list[dict]:
        """All association edges where fact_id is source or target."""
        rows = self.execute(
            "SELECT * FROM association_edges WHERE profile_id = ? "
            "AND (source_fact_id = ? OR target_fact_id = ?)",
            (profile_id, fact_id, fact_id),
        )
        return [dict(r) for r in rows]

    def get_all_association_edges(self, profile_id: str) -> list[dict]:
        """All association edges for a profile."""
        rows = self.execute(
            "SELECT * FROM association_edges WHERE profile_id = ?",
            (profile_id,),
        )
        return [dict(r) for r in rows]

    def delete_association_edges(self, profile_id: str) -> int:
        """Delete all association edges for a profile. Returns count."""
        before = self.execute(
            "SELECT COUNT(*) AS c FROM association_edges WHERE profile_id = ?",
            (profile_id,),
        )
        count = int(before[0]["c"]) if before else 0
        self.execute(
            "DELETE FROM association_edges WHERE profile_id = ?",
            (profile_id,),
        )
        return count

    def store_activation_cache(self, entry: dict) -> None:
        """Persist an activation cache entry."""
        self.execute(
            "INSERT OR REPLACE INTO activation_cache "
            "(cache_id, profile_id, query_hash, node_id, activation_value, "
            " iteration, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now', '+1 hour'))",
            (entry["cache_id"], entry["profile_id"],
             entry["query_hash"], entry["node_id"],
             entry["activation_value"], entry["iteration"]),
        )

    def get_activation_cache(
        self, query_hash: str, profile_id: str,
    ) -> list[dict]:
        """Get cached activation results (non-expired)."""
        rows = self.execute(
            "SELECT node_id, activation_value FROM activation_cache "
            "WHERE profile_id = ? AND query_hash = ? "
            "AND expires_at > datetime('now') "
            "ORDER BY activation_value DESC",
            (profile_id, query_hash),
        )
        return [dict(r) for r in rows]

    def cleanup_activation_cache(self) -> int:
        """Delete expired cache entries. Returns count deleted."""
        before = self.execute(
            "SELECT COUNT(*) AS c FROM activation_cache "
            "WHERE expires_at < datetime('now')"
        )
        count = int(before[0]["c"]) if before else 0
        self.execute(
            "DELETE FROM activation_cache WHERE expires_at < datetime('now')"
        )
        return count

    def store_fact_importance(self, entry: dict) -> None:
        """Persist fact importance scores."""
        self.execute(
            "INSERT OR REPLACE INTO fact_importance "
            "(fact_id, profile_id, pagerank_score, community_id, "
            " degree_centrality, computed_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            (entry["fact_id"], entry["profile_id"],
             entry["pagerank_score"], entry.get("community_id"),
             entry.get("degree_centrality", 0.0)),
        )

    def get_fact_importance(
        self, fact_id: str, profile_id: str,
    ) -> dict | None:
        """Get importance scores for a fact."""
        rows = self.execute(
            "SELECT * FROM fact_importance "
            "WHERE fact_id = ? AND profile_id = ?",
            (fact_id, profile_id),
        )
        return dict(rows[0]) if rows else None

    def get_top_facts_by_pagerank(
        self, profile_id: str, top_k: int = 20,
    ) -> list[dict]:
        """Top facts by PageRank score."""
        rows = self.execute(
            "SELECT * FROM fact_importance "
            "WHERE profile_id = ? "
            "ORDER BY pagerank_score DESC LIMIT ?",
            (profile_id, top_k),
        )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Phase 4: Temporal Intelligence CRUD (Rule 15)
    # ------------------------------------------------------------------

    def store_temporal_validity(
        self, fact_id: str, profile_id: str,
        valid_from: str | None = None,
        valid_until: str | None = None,
    ) -> None:
        """Create temporal validity record for a fact."""
        self.execute(
            "INSERT OR IGNORE INTO fact_temporal_validity "
            "(fact_id, profile_id, valid_from, valid_until) "
            "VALUES (?, ?, ?, ?)",
            (fact_id, profile_id, valid_from, valid_until),
        )

    def get_temporal_validity(self, fact_id: str) -> dict | None:
        """Get temporal validity record for a fact."""
        rows = self.execute(
            "SELECT * FROM fact_temporal_validity WHERE fact_id = ?",
            (fact_id,),
        )
        return dict(rows[0]) if rows else None

    def get_all_temporal_validity(self, profile_id: str) -> list[dict]:
        """Get all temporal validity records for a profile."""
        rows = self.execute(
            "SELECT * FROM fact_temporal_validity WHERE profile_id = ?",
            (profile_id,),
        )
        return [dict(r) for r in rows]

    def invalidate_fact_temporal(
        self, fact_id: str, invalidated_by: str,
        invalidation_reason: str,
    ) -> None:
        """Set valid_until and system_expired_at for a fact.

        BOTH timestamps set atomically (BI-TEMPORAL INTEGRITY).
        Never deletes the fact (Rule 17: immutability).
        """
        from datetime import UTC, datetime as _dt
        now = _dt.now(UTC).isoformat()
        self.execute(
            "UPDATE fact_temporal_validity "
            "SET valid_until = ?, system_expired_at = ?, "
            "    invalidated_by = ?, invalidation_reason = ? "
            "WHERE fact_id = ?",
            (now, now, invalidated_by, invalidation_reason, fact_id),
        )

    def get_valid_facts(self, profile_id: str) -> list[str]:
        """Get fact_ids that are currently valid (not expired).

        Returns facts that either have no temporal record (assumed valid)
        or have valid_until IS NULL and system_expired_at IS NULL.
        """
        rows = self.execute(
            "SELECT f.fact_id FROM atomic_facts f "
            "LEFT JOIN fact_temporal_validity tv ON f.fact_id = tv.fact_id "
            "WHERE f.profile_id = ? "
            "  AND (tv.fact_id IS NULL OR tv.valid_until IS NULL) "
            "  AND (tv.fact_id IS NULL OR tv.system_expired_at IS NULL)",
            (profile_id,),
        )
        return [dict(r)["fact_id"] for r in rows]

    def delete_temporal_validity(self, fact_id: str) -> None:
        """Delete temporal validity record (for testing/rollback only)."""
        self.execute(
            "DELETE FROM fact_temporal_validity WHERE fact_id = ?",
            (fact_id,),
        )

    # ------------------------------------------------------------------
    # Phase 5: Core Memory Blocks CRUD (Rule 15)
    # ------------------------------------------------------------------

    def store_core_block(
        self,
        block_id: str,
        profile_id: str,
        block_type: str,
        content: str,
        source_fact_ids: str = "[]",
        char_count: int = 0,
        version: int = 1,
        compiled_by: str = "rules",
    ) -> None:
        """Store or replace a Core Memory block.

        Uses INSERT OR REPLACE on UNIQUE(profile_id, block_type)
        to guarantee idempotency (L18).
        """
        self.execute(
            "INSERT OR REPLACE INTO core_memory_blocks "
            "(block_id, profile_id, block_type, content, source_fact_ids, "
            " char_count, version, compiled_by, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))",
            (block_id, profile_id, block_type, content,
             source_fact_ids, char_count, version, compiled_by),
        )

    def get_core_blocks(self, profile_id: str) -> list[dict]:
        """Get all Core Memory blocks for a profile."""
        rows = self.execute(
            "SELECT * FROM core_memory_blocks "
            "WHERE profile_id = ? ORDER BY block_type",
            (profile_id,),
        )
        return [dict(r) for r in rows]

    def get_core_block(self, profile_id: str, block_type: str) -> dict | None:
        """Get a single Core Memory block by profile and type."""
        rows = self.execute(
            "SELECT * FROM core_memory_blocks "
            "WHERE profile_id = ? AND block_type = ?",
            (profile_id, block_type),
        )
        return dict(rows[0]) if rows else None

    def delete_core_blocks(self, profile_id: str) -> None:
        """Delete all Core Memory blocks for a profile."""
        self.execute(
            "DELETE FROM core_memory_blocks WHERE profile_id = ?",
            (profile_id,),
        )
