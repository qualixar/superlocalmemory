# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Database Manager.

SQLite with WAL, profile-scoped CRUD, FTS5 search, BM25 persistence.
All connections use try/finally. Only ``except sqlite3.Error``.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""
from __future__ import annotations

import json, logging, sqlite3, threading
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


class DatabaseManager:
    """Thread-safe SQLite manager with WAL, profile isolation, and FTS5.

    Per-call connections outside transactions; shared connection inside
    a ``transaction()`` block. Thread-safe via threading.Lock.
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
            conn.execute("PRAGMA foreign_keys=ON")
            conn.commit()
        finally:
            conn.close()

    def initialize(self, schema_module: ModuleType) -> None:
        """Create all tables. *schema_module* must expose ``create_all_tables(conn)``."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA foreign_keys=ON")
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
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
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
        """Execute SQL. Uses shared conn inside transaction, else per-call."""
        if self._txn_conn is not None:
            return self._txn_conn.execute(sql, params).fetchall()
        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
            conn.commit()
            return rows
        finally:
            conn.close()

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
