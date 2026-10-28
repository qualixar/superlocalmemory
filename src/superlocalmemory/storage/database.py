# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Database Manager.

SQLite with WAL, profile-scoped CRUD, FTS5 search, BM25 persistence.
Concurrent-safe: WAL mode + busy_timeout + retry on SQLITE_BUSY.
Multiple processes (MCP, CLI, integrations) can read/write safely.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""
from __future__ import annotations

import json
import logging
import os
import platform
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Generator, NoReturn

from superlocalmemory.storage import erasure_derived, projection_outbox, store_signature
from superlocalmemory.storage.correction_cases import SAVED_SUCCESSOR_PARAMS, SAVED_SUCCESSOR_SQL
from superlocalmemory.storage.embedding_codec import (
    decode_embedding,
    decode_float_vector,
    encode_embedding,
    encode_float_vector,
)
from superlocalmemory.storage.event_time import valid_from as _event_start
from superlocalmemory.storage.erasure_visibility import (
    aggregate_erasure_clause_for_db,
    erasure_clause,
    has_tombstones,
    memory_erasure_clause_for_db,
)
from superlocalmemory.storage.memory_kinds import KIND_COLUMNS
from superlocalmemory.storage.models import (
    AtomicFact,
    CanonicalEntity,
    ConsolidationAction,
    EdgeType,
    EntityAlias,
    EntityProfile,
    FactType,
    GraphEdge,
    MemoryLifecycle,
    MemoryRecord,
    MemoryScene,
    SignalType,
    TemporalEvent,
    TrustScore,
)
from superlocalmemory.storage.read_connection import read_only_snapshot
from superlocalmemory.storage.write_lock import get_write_lock

from superlocalmemory.storage import write_generation as _write_generation

logger = logging.getLogger(__name__)

_MISSING = object()

# ---------------------------------------------------------------------------
# Memory-kind upsert SQL (4.1.19 / M052) — generated once at import time.
#
# The CASE predicate below is the single authority guard every kind write
# shares (LLD §4.4, risk R4): a confirmed (user/caller) kind is never
# overwritten by anything except a `user` value, and a `caller` value is
# allowed to replace an older `caller` value (a later write of the same
# authority wins) but not an existing `user` value. Everything below
# confirmed (model suggestions, rules, legacy) always yields to a new write.
# `fact_type` is kept in lock-step with whichever kind value survives, so a
# confirmed row's `fact_type` only ever changes together with its kind (I2).
# ---------------------------------------------------------------------------
#
# L1-13: the third clause used to compare `atomic_facts.memory_kind_source`
# and `excluded.memory_kind_source` to the literal 'caller' with no COALESCE.
# SQL's three-valued logic makes `NULL = 'caller'` evaluate to NULL rather
# than FALSE, so a re-store with no kind source at all (a restore-reimport
# row, or any write that simply omits the column) turned the whole AND-chain
# NULL instead of TRUE — and `CASE WHEN NULL` takes the ELSE branch, the same
# as FALSE. The kind columns' own ELSE (`COALESCE(excluded.col,
# atomic_facts.col)`) happened to fall back to the old value anyway, masking
# the bug there, but `fact_type`'s ELSE (`excluded.fact_type`, no COALESCE)
# does not — so fact_type moved to the incoming row's value even while the
# kind stayed put, breaking the "fact_type travels with its kind" invariant
# (I2). Every comparison below is now COALESCE'd on both sides.
_KIND_KEEP_OLD_PREDICATE = (
    "COALESCE(atomic_facts.memory_kind_source, '') IN ('user', 'caller') "
    "AND COALESCE(excluded.memory_kind_source, '') NOT IN ('user') "
    "AND NOT (COALESCE(atomic_facts.memory_kind_source, '') = 'caller' "
    "AND COALESCE(excluded.memory_kind_source, '') = 'caller')"
)


def _kind_case_sql(column: str) -> str:
    return (
        f"{column} = CASE WHEN {_KIND_KEEP_OLD_PREDICATE} "
        f"THEN atomic_facts.{column} ELSE COALESCE(excluded.{column}, atomic_facts.{column}) END"
    )


_FACT_TYPE_KIND_AWARE_SQL = (
    f"fact_type = CASE WHEN {_KIND_KEEP_OLD_PREDICATE} "
    "THEN atomic_facts.fact_type ELSE excluded.fact_type END"
)

# ---------------------------------------------------------------------------
# L1-06: the same authority guard, for `update_fact`'s plain UPDATE.
#
# `_KIND_KEEP_OLD_PREDICATE` above reads the candidate value off an
# `excluded` row, which only exists inside an upsert's ON CONFLICT clause.
# `update_fact` has no `excluded` row to read, so the candidate
# `memory_kind_source` is bound as an ordinary parameter instead (twice,
# once per place the predicate names it). Without this guard, enrichment —
# `core.kind_assignment.assign_kinds` suggesting a kind for an
# already-materialized fact, then writing it back through
# `kind_update_columns` + `update_fact` — could silently downgrade a kind a
# person or a trusted caller had already confirmed, because `update_fact`
# always overwrote every column it was given with no regard for what was
# already stored.
# ---------------------------------------------------------------------------
_UPDATE_KIND_KEEP_OLD_PREDICATE = (
    "COALESCE(memory_kind_source, '') IN ('user', 'caller') "
    "AND COALESCE(?, '') NOT IN ('user') "
    "AND NOT (COALESCE(memory_kind_source, '') = 'caller' AND COALESCE(?, '') = 'caller')"
)


def _guarded_update_set_clause(clean: dict[str, Any]) -> tuple[str, list[Any]]:
    """The ``SET`` clause and its bound params for ``update_fact``.

    Plain ``column = ?`` for every column, exactly as before M052, UNLESS
    ``clean`` carries at least one kind column (``KIND_COLUMNS``) — in which
    case every kind column present, and ``fact_type`` if it is also present,
    get the authority-aware CASE guard instead of a bare assignment. An
    update that touches no kind column is byte-for-byte the pre-guard SQL:
    the overwhelming majority of `update_fact` calls (embeddings, lifecycle,
    access_count, scope, ...) never go near a kind column and pay nothing
    for this.
    """
    kind_keys = set(clean) & set(KIND_COLUMNS)
    guard_fact_type = bool(kind_keys) and "fact_type" in clean
    new_source = clean.get("memory_kind_source")

    parts: list[str] = []
    params: list[Any] = []
    for key, value in clean.items():
        if key in kind_keys or (key == "fact_type" and guard_fact_type):
            parts.append(
                f"{key} = CASE WHEN {_UPDATE_KIND_KEEP_OLD_PREDICATE} "
                f"THEN {key} ELSE ? END"
            )
            params.extend([new_source, new_source, value])
        else:
            parts.append(f"{key} = ?")
            params.append(value)
    return ", ".join(parts), params


#: Column order for the fact upsert/insert, exactly as the pre-4.1.19
#: statement declared it. Shared by both the legacy and kind-aware variants
#: below so the two can never drift apart on anything but the kind columns.
_FACT_UPSERT_COLUMNS: tuple[str, ...] = (
    "fact_id", "memory_id", "profile_id", "content", "fact_type",
    "entities_json", "canonical_entities_json",
    "observation_date", "referenced_date", "interval_start", "interval_end",
    "confidence", "importance", "evidence_count", "access_count",
    "source_turn_ids_json", "session_id",
    "embedding", "fisher_mean", "fisher_variance",
    "lifecycle", "langevin_position",
    "emotional_valence", "emotional_arousal", "signal_type", "created_at",
    "scope", "shared_with",
)
_FACT_UPSERT_SET_SIMPLE: tuple[str, ...] = tuple(
    c for c in _FACT_UPSERT_COLUMNS if c not in ("fact_id", "profile_id", "created_at", "fact_type")
)


def _build_fact_upsert_sql(*, with_kind: bool) -> str:
    columns = list(_FACT_UPSERT_COLUMNS)
    set_parts = [f"{c} = excluded.{c}" for c in _FACT_UPSERT_SET_SIMPLE]
    if with_kind:
        columns += list(KIND_COLUMNS)
        set_parts += [_kind_case_sql(c) for c in KIND_COLUMNS]
        set_parts.append(_FACT_TYPE_KIND_AWARE_SQL)
    else:
        set_parts.append("fact_type = excluded.fact_type")
    col_sql = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(columns))
    set_sql = ",\n                   ".join(set_parts)
    return (
        "INSERT INTO atomic_facts\n"
        f"           ({col_sql})\n"
        f"           VALUES ({placeholders})\n"
        "           ON CONFLICT(fact_id) DO UPDATE SET\n"
        f"                   {set_sql}\n"
        "           WHERE atomic_facts.profile_id = excluded.profile_id\n"
        "           RETURNING profile_id"
    )


def _build_fact_insert_sql(*, with_kind: bool) -> str:
    columns = list(_FACT_UPSERT_COLUMNS) + (list(KIND_COLUMNS) if with_kind else [])
    col_sql = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(columns))
    return f"INSERT INTO atomic_facts\n           ({col_sql})\n           VALUES ({placeholders})"


#: Without the kind columns, byte-for-byte the pre-4.1.19 statement (modulo
#: whitespace): no behaviour changes on a store M052 has not reached.
_FACT_UPSERT_SQL_LEGACY = _build_fact_upsert_sql(with_kind=False)
_FACT_UPSERT_SQL_WITH_KIND = _build_fact_upsert_sql(with_kind=True)
_FACT_INSERT_SQL_LEGACY = _build_fact_insert_sql(with_kind=False)
_FACT_INSERT_SQL_WITH_KIND = _build_fact_insert_sql(with_kind=True)


class ProfileOwnershipConflict(ValueError):
    """A store tried to hand one profile's existing row to another profile.

    ``memories.memory_id`` and ``atomic_facts.fact_id`` are bare
    ``TEXT PRIMARY KEY`` — global, not per-profile — while every row carries a
    ``profile_id``. So two profiles on one store can be handed the same
    caller-chosen id (an importer keyed on an external record id syncing one
    source into two workspaces does exactly this), and the upsert would
    otherwise rewrite the owner and the content: the first profile's memory
    silently became the second's.
    """


def _jl(raw: Any, default: Any = _MISSING) -> Any:
    """JSON-load a value, returning *default* on None/empty.

    _jl(raw)       -> [] when raw is None/empty  (list fields)
    _jl(raw, None) -> None when raw is None/empty (optional fields)
    """
    if raw is None or raw == "":
        return [] if default is _MISSING else default
    return json.loads(raw)

def _jd(val: Any) -> str | None:
    """JSON-dump a list/dict, or return None."""
    return json.dumps(val) if val is not None else None


def _env_int(name: str, default: int) -> int:
    """Read a positive int from the environment, falling back on bad/absent."""
    try:
        val = int(os.environ.get(name, "").strip())
        return val if val > 0 else default
    except (ValueError, AttributeError):
        return default


def _env_float(name: str, default: float) -> float:
    """Read a positive float from the environment, falling back on bad/absent."""
    try:
        val = float(os.environ.get(name, "").strip())
        return val if val > 0 else default
    except (ValueError, AttributeError):
        return default


# SQLite endurance tuning. Defaults preserve prior hard-coded behaviour exactly;
# operators on slow/contended I/O can raise them via env (issue #53) without a
# code change. Unset env => byte-identical to the previous constants.
_BUSY_TIMEOUT_MS = _env_int("SLM_DB_BUSY_TIMEOUT_MS", 10_000)   # wait for writers
_MAX_RETRIES = _env_int("SLM_DB_MAX_RETRIES", 5)                # retry on SQLITE_BUSY
_RETRY_BASE_DELAY = _env_float("SLM_DB_RETRY_BASE_DELAY", 0.1)  # backoff base (s)

# Warn once per process, not once per connection, when the WAL close-path
# deadlock guard cannot be installed (Python < 3.12).
_NO_CKPT_WARNED = False


def _unbounded_facts_ceiling() -> int:
    """Hard upper bound applied when a fact fetch is called with limit=None, so
    a large tenant can never materialize the whole table into memory. Tunable
    via SLM_MAX_FACTS_UNBOUNDED (default 50_000)."""
    import os as _os
    try:
        return max(1, int(_os.environ.get("SLM_MAX_FACTS_UNBOUNDED", "50000")))
    except (TypeError, ValueError):
        return 50000


def _scope_where(
    profile_id: str,
    *,
    include_global: bool = False,
    include_shared: bool = False,
    prefix: str = "",
) -> tuple[str, list]:
    """Build scope-filtering WHERE clause for multi-scope retrieval.

    Returns ``(where_clause, params)`` for splicing into SQL queries.

    When ``include_global=True``, facts with ``scope='global'`` are included
    regardless of profile. When ``include_shared=True``, facts explicitly
    shared with this profile (via ``shared_with`` JSON array) are also
    included.

    v3.6.15: defaults are SHARED-OFF (include_global/include_shared=False) so
    any DIRECT caller (search, list_recent, fetch, resources) is private by
    default — shared memory is opt-in. The recall channels pass explicit
    resolved flags, so opt-in recall is unaffected. With both False the clause
    collapses to ``profile_id = ?`` — identical to 3.6.14 isolation.
    """
    table = f"{prefix}." if prefix else ""
    clauses = [f"({table}profile_id = ?)"]
    params: list = [profile_id]

    if include_global:
        clauses.append(f"({table}scope = 'global')")

    if include_shared:
        # Match the profile_id as a quoted JSON-array element. ESCAPE the LIKE
        # metacharacters in profile_id so a profile id containing % or _ cannot
        # false-positive-match another profile's shared_with list.
        _esc = profile_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        clauses.append(
            f"({table}scope = 'shared' AND {table}shared_with LIKE ? ESCAPE '\\')"
        )
        params.append(f'%"{_esc}"%')

    where = "(" + " OR ".join(clauses) + ")"
    return where, params


def _compose_visible_clause(
    prefix: str,
    *,
    has_archive: bool,
    has_quarantine: bool,
    has_non_answer: bool = False,
    exclude_non_answer: bool = False,
    include_quarantined: bool = False,
    has_erasure: bool = False,
) -> str:
    """Build the AND-clause from what the store actually has.

    ``exclude_non_answer`` is opt-in and only for RECALL ANSWER paths: a stored
    stub (operational receipt, session boilerplate) is still a memory every
    list, count, export and the dashboard shows -- flagged -- but never an
    answer (WP-19).

    Split out from ``DatabaseManager.visible_fact_clause`` so that callers
    holding a bare ``sqlite3.Connection`` produce a byte-identical predicate
    rather than a second, drifting copy.
    """
    table = f"{prefix}." if prefix else ""
    clause = ""
    if has_archive:
        clause += f" AND COALESCE({table}archive_status, 'live') != 'archived'"
    if not include_quarantined and has_quarantine:
        clause += f" AND COALESCE({table}quarantined, 0) = 0"
    if has_non_answer and exclude_non_answer:
        clause += f" AND COALESCE({table}non_answer, 0) = 0"
    if has_erasure:
        clause += erasure_clause(prefix)
    return clause


def _compose_current_clause(prefix: str, *, has_temporal: bool) -> str:
    """The AND-clause that excludes a fact the store has already retired.

    Separate from ``_compose_visible_clause`` on purpose. Withholding and
    retirement are different guarantees with different audiences: a withheld
    row is a model's non-answer that nobody should ever be shown, while a
    retired row is a real memory that was true and is now superseded. History,
    audit and export must still reach it. Only the surfaces answering "what do
    I know NOW" must not.

    The predicate is lifted verbatim from ``get_pinned``, which has carried it
    inline since corrections shipped. That one path was right and the other
    nineteen were never told. GitHub #136.

    Alias is ``tv_cur`` rather than ``tv`` because callers splicing this in
    already bind ``tv`` -- ``get_correction_inadmissible_fact_ids`` does.
    """
    if not has_temporal:
        return ""
    table = f"{prefix}." if prefix else ""
    return (
        " AND NOT EXISTS ("
        "    SELECT 1 FROM fact_temporal_validity tv_cur"
        f"    WHERE tv_cur.fact_id = {table}fact_id"
        f"      AND tv_cur.profile_id = {table}profile_id"
        "      AND tv_cur.system_expired_at IS NOT NULL"
        ")"
    )


def _column_name(row: object) -> str:
    """PRAGMA row -> column name, whatever row_factory the caller set.

    Not paranoia: ``server/routes/memories.py`` -- the first caller and the
    route that was leaking -- sets ``row_factory = dict_factory``, so a
    positional ``row[1]`` raises KeyError there, and the except below would
    have swallowed it into "no such column" and silently dropped the filter.
    That is the same shape of failure this function exists to close.
    """
    if isinstance(row, dict):
        return str(row.get("name", ""))
    try:
        return str(row["name"])          # sqlite3.Row
    except (TypeError, IndexError, KeyError):
        pass
    try:
        return str(row[1])               # plain tuple
    except (TypeError, IndexError, KeyError):
        return ""


def _connection_has_column(conn: sqlite3.Connection, column: str) -> bool:
    try:
        rows = list(conn.execute("PRAGMA table_info(atomic_facts)"))
    except sqlite3.Error:
        return False
    return any(_column_name(r) == column for r in rows)


def stub_flag_sql_for_connection(conn: sqlite3.Connection, prefix: str = "") -> str:
    """SELECT expression for "is this memory a stored stub" (0 when unknown).

    Lists and the dashboard show stubs, flagged; only recall excludes them.
    """
    table = f"{prefix}." if prefix else ""
    if _connection_has_column(conn, "non_answer"):
        return f"COALESCE({table}non_answer, 0)"
    return "0"


def visible_fact_clause_for_connection(
    conn: sqlite3.Connection,
    prefix: str = "",
    *,
    include_quarantined: bool = False,
    exclude_non_answer: bool = False,
) -> str:
    """``visible_fact_clause`` for a caller that hand-rolls its SQL.

    WHY THIS EXISTS. 4.0.10 put the withheld-row filter on every
    ``DatabaseManager`` read path and every retrieval channel, and enumerated
    them in a test. The enumeration was of *methods*, and two HTTP routes never
    call one: ``/api/memories`` and ``/api/v3/timeline/`` build their own SQL
    against ``atomic_facts``. Measured against the author's store on 4.0.10,
    ``/api/memories?limit=50`` served **24 withheld rows on page one** and
    reported a total of 5,218 against 3,919 real memories -- the release's
    central claim, false on the dashboard's main list.

    So the rule is not "read paths go through DatabaseManager"; plenty of
    reasonable code will not. The rule is that anything answering "what does my
    memory contain" resolves its predicate from here. Presence-guarded on the
    passed connection, because a route may be pointed at a store the engine has
    never opened, and filtering on an absent column would turn a cosmetic gap
    into a 500 on every request.

    Grep is not a sufficient test for this: ``mcp/tools_active.py`` also selects
    from ``atomic_facts`` directly and was already clean, while these two routes
    were not. The test that matters calls the surface and inspects what it
    served -- tests/test_server/test_a_listed_memory_belongs_to_its_profile.py
    """
    def _has(column: str) -> bool:
        return _connection_has_column(conn, column)

    return _compose_visible_clause(
        prefix,
        has_archive=_has("archive_status"),
        has_quarantine=_has("quarantined"),
        has_non_answer=exclude_non_answer and _has("non_answer"),
        exclude_non_answer=exclude_non_answer,
        has_erasure=has_tombstones(conn),
        include_quarantined=include_quarantined,
    )


def current_fact_clause_for_connection(
    conn: sqlite3.Connection,
    prefix: str = "",
    *,
    include_quarantined: bool = False,
    exclude_non_answer: bool = False,
) -> str:
    """``current_fact_clause`` for a caller that hand-rolls its SQL.

    Exists for the same reason its visible-only sibling does, and for one more:
    ``mcp/tools_active.py`` opens a bare read connection on the degraded
    ``session_init`` path, so it can reach neither a DatabaseManager nor the
    recall engine. That path was outside BOTH guarantees until 4.1.15.
    """
    try:
        has_temporal = any(
            True
            for _ in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='fact_temporal_validity'"
            )
        )
    except sqlite3.Error:
        has_temporal = False
    return visible_fact_clause_for_connection(
        conn, prefix, include_quarantined=include_quarantined,
        exclude_non_answer=exclude_non_answer,
    ) + _compose_current_clause(prefix, has_temporal=has_temporal)


def _date_or_none(value):
    """A blank date is no date (M063): stored as NULL, never as ''."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


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
        # Shared write-serialisation lock for this db_path.
        # get_write_lock() returns the SAME RLock for every caller that
        # passes the same resolved path, so DatabaseManager, VectorStore,
        # adapter_base, consolidation, and all other in-process writers
        # all share ONE lock → zero cross-connection SQLite WAL contention.
        #
        # RLock (re-entrant) is required: the self-heal backfill pattern
        #   with db._lock:        # acquires write lock (count: 1→2)
        #       vs.upsert(...)    # re-acquires same lock (count: 2→3)
        # is safe because the same thread re-enters the RLock.
        self._lock = get_write_lock(self.db_path)
        # Transaction connections are thread-affine in sqlite3. A manager is
        # shared across HTTP, materializer, and worker threads, so a process-
        # global connection slot lets another thread accidentally execute on
        # an uncommitted foreign connection. Keep the active connection local
        # to the thread that owns the transaction.
        self._txn_state = threading.local()
        from superlocalmemory.storage.read_connection_pool import ReadConnectionPool
        self._read_pool = ReadConnectionPool(self.db_path, lambda: self._connect(shared=True))
        self._enable_wal()

    def _enable_wal(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")  # FIRST — so WAL pragma below uses configured timeout
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            # Fix D: synchronous=NORMAL is safe under WAL — the WAL ensures
            # atomicity independently of the fsync level.  Removing the
            # fsync-on-every-commit penalty halves write latency under the
            # 14 000+ edge-write bursts the materialiser produces per pass.
            # Trade-off: on a power loss between COMMIT and WAL-checkpoint
            # the last committed write to memory.db may be lost.  Acceptable
            # for an LLM memory system; NOT acceptable for financial records.
            conn.execute("PRAGMA synchronous=NORMAL")
            # Fix D: reduce WAL auto-checkpoint from the default 1000 to 400
            # frames.  Smaller checkpoints run more frequently and complete
            # faster, preventing the WAL file growing unboundedly during
            # high-ingestion bursts (which triggered checkpoint-starvation
            # amplifying the lock storm).
            conn.execute("PRAGMA wal_autocheckpoint=400")
            conn.commit()
        finally:
            conn.close()
        # C4 encryption-at-rest defense-in-depth: sensitive DB files shipped
        # world-readable (0644). Restrict to owner-only now that the file (and
        # its WAL sidecars) exist. Best-effort, never blocks DB open.
        try:
            from superlocalmemory.core.security_primitives import harden_db_perms
            harden_db_perms(self.db_path)
        except Exception:
            pass

    def initialize(self, schema_module: ModuleType) -> None:
        """Create all tables. *schema_module* must expose ``create_all_tables(conn)``."""
        conn = self._connect()
        try:
            schema_module.create_all_tables(conn)
            from superlocalmemory.storage.source_identity import ensure_indexes
            ensure_indexes(conn)
            conn.commit()
            logger.info("Schema initialized at %s", self.db_path)
        finally:
            conn.close()

    def close(self) -> None:
        """Close every thread's reused read connection (storage/read_connection_pool)."""
        pool = getattr(self, "_read_pool", None)
        if pool is not None:
            pool.close_all()
        from superlocalmemory.storage import snapshot_pool
        snapshot_pool.close_path(self.db_path)

    def __enter__(self) -> DatabaseManager:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _connect(self, *, shared: bool = False) -> sqlite3.Connection:
        # shared: a reused read connection, which close() may close from another thread.
        conn = sqlite3.connect(str(self.db_path), timeout=_BUSY_TIMEOUT_MS / 1000,
                               check_same_thread=not shared)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA foreign_keys=ON")
            if shared:
                # Reused for plain reads only: a write on it fails loudly
                # instead of slipping past the single-writer lock.
                conn.execute("PRAGMA query_only=ON")
            # wal_autocheckpoint is a PER-CONNECTION pragma and is NOT persisted in
            # the database file (unlike journal_mode=WAL).  Setting it only on the
            # short-lived initialisation connection left every working connection
            # on SQLite's default of 1000 frames.  With checkpoint-on-close
            # disabled below, autocheckpoint is the ONLY remaining checkpoint path,
            # so the intended value must be set where the writes actually happen.
            conn.execute("PRAGMA wal_autocheckpoint=400")
            # Deadlock hardening (postmortem 2026-08-13, Option B): WAL close
            # triggers a checkpoint that can wait indefinitely on reader marks
            # pinned by another process/thread — while holding SQLite's
            # process-global VFS mutex, which convoys every later connect().
            # busy_timeout does NOT apply to the close path. NO_CKPT_ON_CLOSE
            # makes close() checkpoint-free so it can never block; normal
            # checkpointing continues via the wal_autocheckpoint set above.
            # Available since Python 3.12 / SQLite 3.31; guarded for portability.
            try:
                conn.setconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, 1)  # type: ignore[attr-defined]
            except (AttributeError, sqlite3.OperationalError):
                # Silent degradation would hide an inactive deadlock guard.
                # requires-python is >=3.12 (Connection.setconfig support
                # starts there), so this branch is a defensive fallback for an
                # interpreter that bypassed that floor, not an expected path.
                # Warn once, not per connection.
                global _NO_CKPT_WARNED
                if not _NO_CKPT_WARNED:
                    _NO_CKPT_WARNED = True
                    logger.warning(
                        "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE unavailable (Python %s, "
                        "SQLite %s); WAL close-path deadlock hardening is INACTIVE. "
                        "Python 3.12+ is required for this protection.",
                        platform.python_version(),
                        sqlite3.sqlite_version,
                    )
        except BaseException:
            # Setup failed: close it now, or the file stays open (Windows
            # then cannot delete or replace it) for as long as the error lives.
            conn.close()
            raise
        return conn

    @contextmanager
    def _bind_coordinator_connection(
        self,
        conn: sqlite3.Connection,
        capability: Any,
    ) -> Generator[None, None, None]:
        """Reuse the coordinator's sole writable connection for one handler.

        This deliberately stays internal: only ``WriteCoordinator`` can issue
        a capability, and that capability is valid only for its worker thread
        and the exact resolved database path.  While bound, ``transaction``
        and ``raw_connection`` become no-op ownership scopes: they may yield
        the connection, but they must never commit, rollback, or close it.
        The coordinator owns the enclosing ``BEGIN IMMEDIATE`` and final
        commit/rollback together with the command receipt.
        """
        from superlocalmemory.storage.write_coordinator import WriteCoordinatorError

        if not isinstance(conn, sqlite3.Connection):
            raise WriteCoordinatorError("coordinator binding requires a sqlite3 connection")
        validate = getattr(capability, "_validate", None)
        if not callable(validate):
            raise WriteCoordinatorError("untrusted coordinator capability")
        try:
            validate(self.db_path.expanduser().resolve())
        except Exception as exc:
            # Import lazily so the storage manager retains its legacy import
            # surface when the coordinator is not used.
            if isinstance(exc, WriteCoordinatorError):
                raise
            raise WriteCoordinatorError("untrusted coordinator capability") from exc

        attached = conn.execute("PRAGMA database_list").fetchall()
        main_path = next((row[2] for row in attached if row[1] == "main"), "")
        expected_path = self.db_path.expanduser().resolve()
        if not main_path or Path(main_path).expanduser().resolve() != expected_path:
            raise WriteCoordinatorError("coordinator connection targets a different database")
        if getattr(self._txn_state, "conn", None) is not None:
            raise WriteCoordinatorError("database manager is already bound to a transaction")

        self._txn_state.conn = conn
        self._txn_state.coordinator_bound = True
        try:
            yield
        finally:
            self._txn_state.conn = None
            self._txn_state.coordinator_bound = False
            _write_generation.bump()

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Atomic transaction. All writes commit or rollback together."""
        if getattr(self._txn_state, "coordinator_bound", False):
            # The coordinator has already issued BEGIN IMMEDIATE.  Do not
            # create a nested transaction or steal its commit/close lifecycle.
            yield
            return
        with self._lock:
            conn = self._connect()
            self._txn_state.conn = conn
            try:
                yield
                conn.commit()
                _write_generation.bump()
            except Exception:
                conn.rollback()
                raise
            finally:
                self._txn_state.conn = None
                conn.close()

    @contextmanager
    def raw_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a live sqlite3.Connection for code that needs one directly.

        For callers (e.g. schema migrations) that must hold a real connection
        rather than going through execute(). Commits on success, rolls back on
        error, and always closes — mirroring transaction(). This is the public
        way to obtain a connection; there is no `.conn` attribute.
        """
        coordinator_conn = getattr(self._txn_state, "conn", None)
        if getattr(self._txn_state, "coordinator_bound", False):
            if coordinator_conn is None:  # pragma: no cover - binding invariant
                raise RuntimeError("coordinator binding has no active connection")
            yield coordinator_conn
            return
        with self._lock:
            conn = self._connect()
            self._txn_state.conn = conn
            try:
                yield conn
                conn.commit()
                _write_generation.bump()
            except Exception:
                conn.rollback()
                raise
            finally:
                self._txn_state.conn = None
                conn.close()

    # DML prefixes that require the single-writer lock when executed outside
    # a transaction() context.  Checked case-insensitively against the first
    # word of the stripped SQL statement.
    _DML_PREFIXES: frozenset[str] = frozenset({
        "INSERT", "UPDATE", "DELETE", "REPLACE", "UPSERT",
        "CREATE", "DROP", "ALTER",
    })

    def _execute_one(self, sql: str, params: tuple[Any, ...]) -> list[sqlite3.Row]:
        """Open a per-call connection, execute, commit, close — with retry.

        Never called when inside a transaction() context (that path uses the
        context's existing connection directly).  Factored out of execute() so
        the RLock acquisition logic stays in one place.
        """
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

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Execute SQL with automatic retry on SQLITE_BUSY.

        Fix B — single-writer serialisation:
        • Inside a transaction(): uses the existing connection directly (no
          lock acquisition — the transaction() context manager already holds
          _lock for the duration of the whole transaction).
        • Outside a transaction(), for DML (INSERT/UPDATE/DELETE/…): acquires
          _lock before opening a per-call connection.  This ensures that
          concurrent callers — including background workers that bypass
          transaction() entirely — do not race at the SQLite WAL layer.
        • Outside a transaction(), for SELECTs: no lock needed — WAL mode
          allows concurrent readers without stalling writers.

        ORDERING INVARIANT: the _txn_state.conn check MUST come before the
        lock acquisition.  Reversing the order would deadlock threads that
        call execute() from inside transaction() because threading.RLock is
        re-entrant per-thread but a re-entering thread inside transaction()
        would still try to re-acquire here (lock is already held by the same
        thread, so RLock re-enters safely — but the old threading.Lock would
        have deadlocked; that is exactly why we changed to RLock).
        """
        # Fast path: already inside a transaction — use its connection directly.
        transaction_conn = getattr(self._txn_state, "conn", None)
        if transaction_conn is not None:
            rows = transaction_conn.execute(sql, params).fetchall()
            if transaction_conn.in_transaction:
                _write_generation.bump()
            return rows

        # Determine if this is a write operation that needs serialisation.
        from superlocalmemory.storage import read_connection_pool as _rcp
        first_word = sql.strip().upper().split(None, 1)[0] if sql.strip() else ""
        # A WITH clause can introduce a write (WITH ... DELETE): that is a write.
        if first_word in self._DML_PREFIXES or _rcp.is_write_with(sql):
            with self._lock:
                try:
                    return self._execute_one(sql, params)
                finally:
                    _write_generation.bump()
        else:
            # Read-only path: concurrent reads are safe in WAL mode. A plain
            # read reuses this thread's connection (storage/read_connection_pool).
            pool = getattr(self, "_read_pool", None)
            if pool is not None and _rcp.is_plain_read(sql):
                return _rcp.execute_read(pool, sql, params, self._execute_one,
                                         retries=_MAX_RETRIES, base_delay=_RETRY_BASE_DELAY)
            return self._execute_one(sql, params)

    # The two tables whose primary key is global but whose rows are owned by a
    # profile. Literal, internal, and closed — never built from caller input.
    _OWNED_ROWS: dict[str, str] = {"memories": "memory_id", "atomic_facts": "fact_id"}

    def _refuse_cross_profile_reown(
        self, table: str, row_id: str, profile_id: str,
    ) -> NoReturn:
        """Raise ``ProfileOwnershipConflict``, naming the profile that owns it.

        Called only when an upsert below returned no row. Both statements carry
        ``WHERE <table>.profile_id = excluded.profile_id`` on their
        ``DO UPDATE``, so SQLite skips a conflicting row owned by a different
        profile and ``RETURNING`` yields nothing — which makes the ownership
        check part of the write instead of a read in front of it. That costs no
        extra query on the ordinary path and leaves no window between deciding
        and writing. The lookup here runs only on the refusal, to say whose row
        it is; a refusal nobody can read gets worked around.

        Taking a row away from the profile that owns it is not last-write-wins,
        it is a different tenant's write, so it is refused rather than merged.
        Before this the outcome depended on something unrelated:
        ``scene_fact_members`` carries a composite
        ``(profile_id, fact_id) -> atomic_facts (profile_id, fact_id)`` foreign
        key, so where a scene referenced the fact the ownership change orphaned
        it and SQLite raised a bare ``FOREIGN KEY constraint failed`` — and
        where no scene did, the identical write succeeded in silence and the
        second profile kept the fact. Isolation cannot rest on whether a
        projection happens to exist.
        """
        # Interpolated, not parameterised: SQLite takes no parameter in a table
        # or column position. Both come from _OWNED_ROWS, a closed literal map,
        # and the row id stays bound.
        id_column = self._OWNED_ROWS[table]
        rows = self.execute(
            f"SELECT profile_id FROM {table} WHERE {id_column} = ?",
            (row_id,),
        )
        owner = dict(rows[0])["profile_id"] if rows else "<unknown>"
        raise ProfileOwnershipConflict(
            f"{id_column} {row_id!r} in {table} belongs to profile {owner!r}; "
            f"refusing to re-own it as {profile_id!r}"
        )

    def store_memory(self, record: MemoryRecord) -> str:
        """Persist a raw memory record. Returns memory_id.

        Upserts in place rather than replacing. ``INSERT OR REPLACE`` is a
        DELETE followed by an INSERT, and ``atomic_facts.memory_id`` is a
        foreign key with ``ON DELETE CASCADE`` — so storing a record whose
        ``memory_id`` already existed silently deleted every fact extracted from
        it. Reproduced in isolation: three facts stored, one re-store of the same
        memory_id, zero facts left, no error raised.

        Most callers pass a freshly generated id, which is why this never fired.
        But ``cognitive_consolidator`` supplies its own ``block_id``, and the
        queryable-promotion path in ``run_store`` deliberately avoids calling
        this at all for an existing memory — a rule that has to be remembered
        rather than enforced. ``ON CONFLICT DO UPDATE`` keeps the same
        last-write-wins semantics and takes the loaded gun out of the room.

        ``created_at`` is deliberately not overwritten: the row's first
        observation is a historical fact, and a re-store is not a new one.
        ``profile_id`` is not in the update list either, and a store that would
        change it is refused outright — the ``DO UPDATE`` is conditioned on the
        owner matching, so SQLite skips the row and ``RETURNING`` comes back
        empty. See ``_refuse_cross_profile_reown``.
        """
        _scope = getattr(record, 'scope', None) or 'personal'
        _shared = _jd(getattr(record, 'shared_with', None))
        written = self.execute(
            """INSERT INTO memories
               (memory_id, profile_id, content, session_id, speaker,
                role, session_date, created_at, metadata_json,
                scope, shared_with)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(memory_id) DO UPDATE SET
                   content       = excluded.content,
                   session_id    = excluded.session_id,
                   speaker       = excluded.speaker,
                   role          = excluded.role,
                   session_date  = excluded.session_date,
                   metadata_json = excluded.metadata_json,
                   scope         = excluded.scope,
                   shared_with   = excluded.shared_with
               WHERE memories.profile_id = excluded.profile_id
               RETURNING profile_id""",
            (record.memory_id, record.profile_id, record.content,
             record.session_id, record.speaker, record.role,
             record.session_date, record.created_at,
             json.dumps(record.metadata), _scope, _shared),
        )
        if not written:
            self._refuse_cross_profile_reown(
                "memories", record.memory_id, record.profile_id,
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

    def _atomically(self, work: Any) -> Any:
        """Run ``work`` in one transaction, joining an open one rather than nesting.

        A caller already inside ``transaction()`` must not start a second one:
        the lock is re-entrant but ``_connect`` is not, so a nested attempt opens
        a separate connection to the same file while the first still holds its
        write. ``store_fact`` carried this check inline; it is here because
        three more methods now need the same thing, and a projection intent that
        commits in a different transaction from the row it describes is exactly
        the window this whole mechanism exists to close.
        """
        if getattr(self._txn_state, "conn", None) is not None:
            return work()
        with self.transaction():
            return work()

    def store_fact(self, fact: AtomicFact) -> str:
        """Persist an atomic fact. Returns fact_id.

        v3.6.4 — idempotent on content. If an ACTIVE fact with identical
        content already exists for this profile, reinforce it (bump
        evidence_count + access_count) and return its fact_id instead of
        inserting a duplicate row. The passed fact's ``fact_id`` is rewritten
        to the canonical id so downstream writes keyed on it (embeddings,
        graph edges, context) target the real fact rather than orphaning.

        This enforces the memory-system invariant "storing the same fact
        twice is one fact" — preventing the duplicate explosion that poisons
        importance ranking and core-memory promotion. Empty/whitespace
        content is exempt (handled by placeholder filtering, not dedup).

        The insert below upserts rather than replaces, for the reason
        ``store_memory`` does. ``INSERT OR REPLACE`` is a DELETE followed by an
        INSERT, and eight tables hold
        ``FOREIGN KEY (fact_id) REFERENCES atomic_facts (fact_id) ON DELETE
        CASCADE`` — so re-storing a fact under an occupied id dropped its
        retention row, access history, context and importance, and raised
        nothing.

        The dedup above does not close this: it matches on *content*, so a
        second store of the same id with *different* content falls straight
        through to the insert. ``MemoryEngine.store_fact_direct`` reaches it —
        ``canonical_store_fact`` exists to persist a caller-chosen id and
        raises if that id is not preserved. Within one profile an idempotency
        key of ``prebuilt:<fact_id>`` catches the second store, but that key is
        scoped ``(profile_id, source_type, idempotency_key)`` while
        ``atomic_facts.fact_id`` is a bare ``TEXT PRIMARY KEY``. Reproduced
        across two profiles on one store: the first profile's fact was replaced
        outright — new owner, new content — and its associations were gone.

        ``created_at`` is deliberately not overwritten: the row's first
        observation is a historical fact, and a re-store is not a new one.
        ``pinned`` is not in the column list at all, so the upsert now leaves it
        alone where the replace silently reset it to 0 — pinning is user intent,
        not something a re-store gets to revoke. ``profile_id`` is out of the
        update list for a stronger reason: a store that would change it is
        refused rather than applied, because it is one profile taking another's
        fact rather than a re-store at all. The refusal is a condition on the
        ``DO UPDATE`` itself, so there is no window between checking the owner
        and writing the row.

        ``insert_fact_immutable`` remains the right call for a known-new fact
        that must abort on a collision rather than win it.
        """
        if fact.content and fact.content.strip():
            # Dedup across all LIVE lifecycle zones (active/warm/cold). Excludes
            # 'archived' — that is soft-deleted/forgotten, so re-storing the same
            # content correctly re-learns it as a fresh fact. Matching only
            # 'active' (pre-3.6.4) re-opened the duplication window for every
            # fact that aged to warm/cold (the bulk of the KB).
            #
            # Also excludes a RETIRED fact — one with fact_temporal_validity.
            # system_expired_at set, by a caller replacement (L1-10) or any
            # other invalidation. Matching a retired fact would fold a brand
            # new statement back onto dead history instead of giving it its
            # own current copy, so something the successor's enrichment
            # re-derives would only ever exist as a fact recall never shows.
            from superlocalmemory.storage.source_identity import matching_fact

            existing = matching_fact(self, fact)
            if existing:
                canonical_id = dict(existing[0])["fact_id"]
                from superlocalmemory.encoding.typed_values import has_relative_time
                if has_relative_time(fact.content):
                    self.execute(
                        "UPDATE atomic_facts SET referenced_date=NULL, interval_start=NULL, "
                        "interval_end=NULL WHERE fact_id=? AND COALESCE(observation_date,'')<>?",
                        (canonical_id, fact.observation_date or ''),
                    )
                self.execute(
                    "UPDATE atomic_facts "
                    "SET evidence_count = evidence_count + 1, "
                    "    observation_date = COALESCE(?, observation_date) "
                    "WHERE fact_id = ?",
                    (fact.observation_date, canonical_id),
                )
                # Rewrite caller's id so downstream embedding/graph/context
                # writes target the canonical fact (idempotent), not an
                # orphaned id that was never inserted.
                fact.fact_id = canonical_id
                # Repair the only recoverable interrupted-write state from an
                # early 4.0.2 attempt: a durable fact without its mandatory
                # knowledge-time anchor. Do not manufacture a historical
                # timestamp from ``created_at``; this retry is the earliest
                # trustworthy observation that the anchor was missing.
                if self.get_temporal_validity(canonical_id, fact.profile_id) is None:
                    canonical = self.get_fact(canonical_id, fact.profile_id)
                    self.store_temporal_validity(
                        canonical_id, fact.profile_id,
                        valid_from=_event_start(canonical or fact), fill_only=True,
                    )
                # Re-storing known content is the natural moment to notice a
                # projection that was never written — an upgraded store whose
                # graph predates the migration reaches this branch, not the
                # insert below.
                projection_outbox.enqueue(self, canonical_id, fact.profile_id)
                return canonical_id
        _scope = getattr(fact, 'scope', None) or 'personal'
        _shared = _jd(getattr(fact, 'shared_with', None))
        def _insert_with_knowledge_anchor() -> None:
            use_kind = self.has_memory_kind_columns()
            sql = _FACT_UPSERT_SQL_WITH_KIND if use_kind else _FACT_UPSERT_SQL_LEGACY
            params: list[Any] = [
                fact.fact_id, fact.memory_id, fact.profile_id, fact.content,
                fact.fact_type.value,
                json.dumps(fact.entities), json.dumps(fact.canonical_entities),
                fact.observation_date, fact.referenced_date,
                fact.interval_start, fact.interval_end,
                fact.confidence, fact.importance, fact.evidence_count, fact.access_count,
                json.dumps(fact.source_turn_ids), fact.session_id,
                encode_embedding(fact.embedding),
                encode_float_vector(fact.fisher_mean),
                encode_float_vector(fact.fisher_variance),
                fact.lifecycle.value, _jd(fact.langevin_position),
                fact.emotional_valence, fact.emotional_arousal,
                fact.signal_type.value, fact.created_at, _scope, _shared,
            ]
            if use_kind:
                # A caller-built AtomicFact that predates 4.1.19 has no kind
                # attributes at all; getattr keeps this write from failing on
                # a kind the same way parse_kind never raises on one (I1).
                params.extend((
                    getattr(fact, "memory_kind", None),
                    getattr(fact, "memory_kind_source", None),
                    getattr(fact, "memory_kind_confidence", None),
                    getattr(fact, "memory_kind_recipe", None),
                    getattr(fact, "memory_kind_at", None),
                ))
            written = self.execute(sql, tuple(params))
            if not written:
                self._refuse_cross_profile_reown(
                    "atomic_facts", fact.fact_id, fact.profile_id,
                )
            # Every fact written after 4.0.2 has an explicit transaction-time
            # anchor. Absence deliberately represents pre-4.0.2
            # ``legacy_unknown``; never backfill it from ``created_at``.
            self.store_temporal_validity(fact.fact_id, fact.profile_id,
                                         valid_from=_event_start(fact), fill_only=True)
            # Which facts name an entity is answered from an index; every
            # write path lands here, so this keeps it complete
            # (storage/entity_index.py).
            from superlocalmemory.storage.entity_index import record_fact_entities

            record_fact_entities(
                self, fact.fact_id, fact.profile_id, fact.canonical_entities,
            )
            # The graph and the vectors live in other storage engines, so the
            # intent to project this fact is queued here, in this transaction.
            # Enqueueing in the storage layer rather than at each pipeline call
            # site is deliberate: ingestion, the background materializer, the
            # consolidator and the CLI all write facts through this method, and
            # a call site that forgets would produce a memory that is stored
            # and unrecallable.
            projection_outbox.enqueue(self, fact.fact_id, fact.profile_id)
            # A memory its caller already replaced keeps no current facts, not
            # even ones enrichment derives from it afterwards. Same transaction,
            # same single write path, so no such fact is ever visible un-retired
            # (storage/replaced_memory.py).
            if fact.memory_id:
                from superlocalmemory.storage.replaced_memory import (
                    retire_if_memory_replaced,
                )

                retire_if_memory_replaced(
                    self._txn_state.conn, fact_id=fact.fact_id,
                    memory_id=fact.memory_id, profile_id=fact.profile_id,
                )

        # The fact and its transaction-time anchor are one logical write. Do
        # not open a nested transaction when an owner already holds one.
        if getattr(self._txn_state, "conn", None) is not None:
            _insert_with_knowledge_anchor()
        else:
            with self.transaction():
                _insert_with_knowledge_anchor()
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
            embedding=decode_embedding(d.get("embedding"), fact_id=d.get("fact_id", "<unknown>")),
            fisher_mean=decode_float_vector(
                d.get("fisher_mean"), field="fisher_mean",
                fact_id=d.get("fact_id", "<unknown>")),
            fisher_variance=decode_float_vector(
                d.get("fisher_variance"), field="fisher_variance",
                fact_id=d.get("fact_id", "<unknown>")),
            lifecycle=MemoryLifecycle(d["lifecycle"]) if d.get("lifecycle") else MemoryLifecycle.ACTIVE,
            langevin_position=_jl(d.get("langevin_position"), None),
            emotional_valence=d.get("emotional_valence", 0.0),
            emotional_arousal=d.get("emotional_arousal", 0.0),
            signal_type=SignalType(d["signal_type"]) if d.get("signal_type") else SignalType.FACTUAL,
            pinned=bool(d.get("pinned", 0)),
            non_answer=bool(d.get("non_answer") or 0),
            scope=d.get("scope", "personal"),
            shared_with=_jl(d.get("shared_with"), None),
            # 4.1.19 (M052): absent on a row the migration has not reached —
            # ``dict(row).get(...)`` already returns None for a key ``SELECT
            # *`` never produced, so this is correct whether or not the
            # columns exist (old-style hydration of a new row, and new-style
            # hydration of an old row, both work).
            memory_kind=d.get("memory_kind"),
            memory_kind_source=d.get("memory_kind_source"),
            memory_kind_confidence=d.get("memory_kind_confidence"),
            memory_kind_recipe=d.get("memory_kind_recipe"),
            memory_kind_at=d.get("memory_kind_at"),
            created_at=d["created_at"],
        )

    def insert_fact_immutable(self, fact: AtomicFact) -> str:
        """Insert one known-new fact without content deduplication or replacement.

        Reviewed correction successors require a caller-chosen immutable
        identity.  Unlike normal remember ingestion, equal content must not
        reinforce an existing row, and an occupied ID must abort the enclosing
        transaction rather than overwrite history.
        """
        scope = getattr(fact, "scope", None) or "personal"
        shared = _jd(getattr(fact, "shared_with", None))
        use_kind = self.has_memory_kind_columns()
        params: list[Any] = [
            fact.fact_id,
            fact.memory_id,
            fact.profile_id,
            fact.content,
            fact.fact_type.value,
            json.dumps(fact.entities),
            json.dumps(fact.canonical_entities),
            fact.observation_date,
            fact.referenced_date,
            fact.interval_start,
            fact.interval_end,
            fact.confidence,
            fact.importance,
            fact.evidence_count,
            fact.access_count,
            json.dumps(fact.source_turn_ids),
            fact.session_id,
            encode_embedding(fact.embedding),
            encode_float_vector(fact.fisher_mean),
            encode_float_vector(fact.fisher_variance),
            fact.lifecycle.value,
            _jd(fact.langevin_position),
            fact.emotional_valence,
            fact.emotional_arousal,
            fact.signal_type.value,
            fact.created_at,
            scope,
            shared,
        ]
        if use_kind:
            params.extend((
                getattr(fact, "memory_kind", None),
                getattr(fact, "memory_kind_source", None),
                getattr(fact, "memory_kind_confidence", None),
                getattr(fact, "memory_kind_recipe", None),
                getattr(fact, "memory_kind_at", None),
            ))
        self.execute(
            _FACT_INSERT_SQL_WITH_KIND if use_kind else _FACT_INSERT_SQL_LEGACY,
            tuple(params),
        )
        self.store_temporal_validity(fact.fact_id, fact.profile_id,
                                         valid_from=_event_start(fact), fill_only=True)
        from superlocalmemory.storage.entity_index import record_fact_entities

        record_fact_entities(
            self, fact.fact_id, fact.profile_id, fact.canonical_entities,
        )
        projection_outbox.enqueue(self, fact.fact_id, fact.profile_id)
        return fact.fact_id

    def set_pinned(self, fact_id: str, pinned: bool) -> None:
        """Set or clear the pinned flag on a fact (v3.4.65 core-memory)."""
        self.execute(
            "UPDATE atomic_facts SET pinned = ? WHERE fact_id = ?",
            (1 if pinned else 0, fact_id),
        )

    def get_pinned(
        self, profile_id: str,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> list[AtomicFact]:
        """Return currently admissible pinned facts, highest-importance first.

        Pins are injection priority, not an override for a later correction.
        A system-invalidated fact remains durable and historically queryable,
        but it must not be inserted into the current session context.
        """
        where, params = _scope_where(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
            prefix="f",
        )
        # Pins are injected straight into an agent's context, which makes this
        # the most consequential display path in the class: a withheld row here
        # is not merely shown, it is asserted as background truth.
        rows = self.execute(
            f"SELECT f.* FROM atomic_facts f WHERE {where} AND f.pinned = 1 "
            f"{self.current_fact_clause('f')} "
            "ORDER BY importance DESC",
            (*params,),
        )
        facts = [self._row_to_fact(r) for r in rows]
        if not facts:
            return facts
        try:
            blocked = self.get_nonapplied_correction_successor_ids(
                [fact.fact_id for fact in facts],
                profile_id,
                include_global=include_global,
                include_shared=include_shared,
            )
        except Exception as exc:
            logger.warning("Pinned correction admission lookup failed: %s", exc)
            return []
        if not isinstance(blocked, set):
            logger.warning("Pinned correction admission returned malformed data")
            return []
        return [fact for fact in facts if fact.fact_id not in blocked]

    def _has_archive_status(self) -> bool:
        """Whether atomic_facts carries the M011 ``archive_status`` column.

        M011 is a DEFERRED migration, so the column is absent until it runs;
        callers must not filter on a column that may not exist. Cached once True
        (a column never disappears); re-checked while absent so a later deferred
        migration is picked up.
        """
        if getattr(self, "_archive_col_present", False):
            return True
        present = any(
            dict(row).get("name") == "archive_status"
            for row in self.execute("PRAGMA table_info(atomic_facts)")
        )
        if present:
            self._archive_col_present = True
        return present

    def _has_quarantine_column(self) -> bool:
        """Whether atomic_facts carries the ``quarantined`` column.

        Same shape as ``_has_archive_status``: cached once True (a column never
        disappears), re-checked while absent so a later schema pass is picked
        up. ``storage.schema.create_all_tables`` adds the column at every engine
        init, so on any store the daemon has opened this is True — the guard
        exists for a bare DatabaseManager pointed at a store that engine init
        never touched, where filtering on the column would raise instead of
        returning results.
        """
        if getattr(self, "_quarantine_col_present", False):
            return True
        present = any(
            dict(row).get("name") == "quarantined"
            for row in self.execute("PRAGMA table_info(atomic_facts)")
        )
        if present:
            self._quarantine_col_present = True
        return present

    def has_memory_kind_columns(self) -> bool:
        """Whether ``atomic_facts`` carries the five M052 (4.1.19) kind columns.

        Shaped differently from ``_has_quarantine_column`` on purpose
        (LLD §4.4): True is cached for the life of this instance (a column
        never disappears), but False is re-probed at most once every 5
        seconds rather than on every call. A bare ``DatabaseManager`` built
        before ``apply_all`` runs (the engine-direct path builds its manager
        before migrations run) would otherwise pay a ``PRAGMA`` on every
        single fact write until the eager M052 migration completes — this
        keeps that cost to one probe per 5-second window instead.
        """
        if getattr(self, "_kind_columns_present", False):
            return True
        last_checked = getattr(self, "_kind_columns_checked_at", 0.0)
        now = time.monotonic()
        if last_checked and (now - last_checked) < 5.0:
            return False
        try:
            have = {
                dict(row).get("name")
                for row in self.execute("PRAGMA table_info(atomic_facts)")
            }
        except sqlite3.Error:
            have = set()
        present = set(KIND_COLUMNS).issubset(have)
        self._kind_columns_checked_at = 0.0 if present else now  # after, never mid-probe
        if present:
            self._kind_columns_present = True
        return present

    def _has_temporal_validity_table(self) -> bool:
        """Whether the store carries ``fact_temporal_validity``.

        Same shape and same reason as ``_has_quarantine_column``: cached once
        True, re-checked while absent. A bare DatabaseManager can be pointed at
        a store engine init never touched, and filtering against an absent
        table would turn a cosmetic gap into total read failure.
        """
        if getattr(self, "_temporal_table_present", False):
            return True
        try:
            present = any(
                dict(row).get("name") == "fact_temporal_validity"
                for row in self.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='fact_temporal_validity'"
                )
            )
        except sqlite3.Error:
            return False
        if present:
            self._temporal_table_present = True
        return present

    def _has_projection_tombstones_table(self) -> bool:
        """Cache only schema presence; re-probe absent tables after migrations."""
        if getattr(self, "_tombstone_table_present", False):
            return True
        present = bool(self.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='projection_tombstones'"
        ))
        if present:
            self._tombstone_table_present = True
        return present

    def current_fact_clause(
        self, prefix: str = "", *, include_quarantined: bool = False,
        exclude_non_answer: bool = False,
    ) -> str:
        """``visible_fact_clause`` PLUS "and the store still believes it".

        Use this on any surface that answers "what do I know now": search,
        pins, the degraded recall path, the dashboard. Use the plain
        ``visible_fact_clause`` on history, audit and export, which must keep
        showing a memory that was retired -- "why did my memory change" is a
        question this product has to be able to answer.
        """
        return self.visible_fact_clause(
            prefix, include_quarantined=include_quarantined,
            exclude_non_answer=exclude_non_answer,
        ) + _compose_current_clause(
            prefix, has_temporal=self._has_temporal_validity_table(),
        )

    def visible_fact_clause(
        self, prefix: str = "", *, include_quarantined: bool = False,
        exclude_non_answer: bool = False,
    ) -> str:
        """AND-clause excluding rows no caller should be shown as a memory.

        Erasure intent always wins, including for include_quarantined callers.
        Membership is evaluated by SQLite on every read and is never cached.

        Two other exclusions, one definition: soft-deleted (``archive_status``) and
        withheld (``quarantined``). Both are presence-guarded, because each
        column arrives with a migration and may be absent on a store the engine
        has not opened.

        WHY THIS EXISTS AS A FUNCTION. 4.0.10 first put the quarantine filter in
        ``get_facts_by_ids`` alone, reasoning that every retrieval channel
        re-authorises through it and the engine drops what it cannot hydrate.
        That reasoning was correct and the conclusion was wrong: it covered the
        RECALL pipeline, and ``search``, ``list_recent``, ``fetch``, the MCP
        resources and the dashboard's own search are not the recall pipeline.
        Measured on a copy of the author's store, ``search_facts_fts`` returned
        20 withheld rows out of 50 and ``get_all_facts`` 66 out of 400 — the
        exact defect the design was meant to prevent, in the paths the design
        never looked at.

        There is no single SQL choke point in this codebase; ``_scope_where`` is
        spliced against six other tables and cannot carry a fact column. So the
        honest form of "one place" is one CLAUSE with an enumerable set of call
        sites, and a test that fails when a read path does not use it:
        tests/test_storage/test_no_read_path_shows_a_withheld_row.py

        ``include_quarantined=True`` is for repair, erasure and export — paths
        that must reach a withheld row to act on it.

        ``exclude_non_answer=True`` is for the recall answer paths only (see
        ``_compose_visible_clause``); everything else lists stubs, flagged.
        """
        from superlocalmemory.encoding.non_answer import has_marker

        return _compose_visible_clause(
            prefix,
            has_non_answer=exclude_non_answer and has_marker(self),
            exclude_non_answer=exclude_non_answer,
            has_erasure=self._has_projection_tombstones_table(),
            has_archive=self._has_archive_status(),
            has_quarantine=self._has_quarantine_column(),
            include_quarantined=include_quarantined,
        )

    def get_all_facts(
        self, profile_id: str, limit: int | None = None,
        *,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> list[AtomicFact]:
        """All facts for a profile, newest first.

        memory-bounding-02: optional SQL LIMIT so callers needing only the
        most-recent N (e.g. the Hopfield channel's 5000 cap) don't deserialize
        the entire table into AtomicFact objects. Default (None) = all facts.
        """
        where, params = _scope_where(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        # memory-bounding-02 + perf M-04: an unbounded fetch materializes the
        # whole table (hundreds of MB with embeddings at 50k+ facts). Apply a
        # hard, env-tunable ceiling even when the caller passes limit=None.
        if limit is None:
            limit = _unbounded_facts_ceiling()
        # Soft-deleted and withheld rows are not memories a caller may see.
        rows = self.execute(
            f"SELECT * FROM atomic_facts WHERE {where}"
            f"{self.visible_fact_clause()} "
            "ORDER BY created_at DESC LIMIT ?",
            (*params, int(limit)),
        )
        return [self._row_to_fact(r) for r in rows]

    def get_session_facts(self, profile_id: str, session_id: str) -> list[AtomicFact]:
        """One session's visible facts, newest first, read through idx_facts_session."""
        rows = self.execute(
            "SELECT * FROM atomic_facts WHERE profile_id = ? AND session_id = ?"
            f"{self.visible_fact_clause()} ORDER BY created_at DESC",
            (profile_id, session_id),
        )
        return [self._row_to_fact(r) for r in rows]

    def get_external_visible_facts(
        self,
        profile_id: str,
        *,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> list[AtomicFact]:
        """Cross-profile facts visible to ``profile_id`` under scope policy.

        This is the bounded supplement used by profile-partitioned candidate
        indexes.  It deliberately excludes the requester's own partition so a
        fast local index can merge only the global/authorized-shared rows it
        cannot discover itself.  The canonical scope predicate remains the
        sole authorization rule.
        """
        if not include_global and not include_shared:
            return []
        where, params = _scope_where(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        # Crossing a profile boundary is the last place a withheld row should
        # appear: it would be a model's non-answer presented to somebody else
        # as one of this profile's shared memories.
        rows = self.execute(
            f"SELECT * FROM atomic_facts WHERE {where} AND profile_id != ?"
            f"{self.visible_fact_clause()} ORDER BY created_at DESC",
            (*params, profile_id),
        )
        return [self._row_to_fact(r) for r in rows]

    def get_external_visible_embeddings(
        self,
        profile_id: str,
        *,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> list[tuple[str, Any]]:
        """``get_external_visible_facts`` as ``(fact_id, float32 vector)`` only.

        For channels that score by embedding and read nothing else (#147).
        Implementation and contract: ``storage/embedding_projection.py``.
        """
        from superlocalmemory.storage.embedding_projection import (
            fetch_external_visible_embeddings,
        )
        return fetch_external_visible_embeddings(
            self, profile_id,
            include_global=include_global, include_shared=include_shared,
        )

    def get_fact_embeddings_by_ids(
        self,
        fact_ids: list[str],
        profile_id: str,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> list[tuple[str, Any]]:
        """``get_facts_by_ids`` as ``(fact_id, float32 vector)`` only (#147)."""
        from superlocalmemory.storage.embedding_projection import (
            fetch_fact_embeddings_by_ids,
        )
        return fetch_fact_embeddings_by_ids(
            self, fact_ids, profile_id,
            include_global=include_global, include_shared=include_shared,
        )

    _MAX_FACTS_PER_ENTITY_LOOKUP: int = 100

    def get_facts_by_entity(
        self, entity_id: str, profile_id: str,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> list[AtomicFact]:
        """Facts whose canonical_entities JSON array contains *entity_id*.

        V3.3.14: LIMIT to _MAX_FACTS_PER_ENTITY_LOOKUP (100) to prevent
        unbounded memory growth during ingestion. Previously loaded ALL
        facts for popular entities (500+) causing 17GB+ memory usage.
        Ordered by created_at DESC so newest facts are always included.
        """
        from superlocalmemory.storage.entity_index import facts_for_entity, is_complete
        pairs = facts_for_entity(
            self, entity_id, profile_id, limit=self._MAX_FACTS_PER_ENTITY_LOOKUP,
            indexed=is_complete(self), include_global=include_global,
            include_shared=include_shared,
        )
        facts = self.get_facts_by_ids([fid for fid, _ in pairs], profile_id,
                                    include_global=include_global, include_shared=include_shared)
        by_id = {fact.fact_id: fact for fact in facts}
        return [by_id[fid] for fid, _ in pairs if fid in by_id]

    def get_facts_by_type(
        self, fact_type: FactType, profile_id: str,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> list[AtomicFact]:
        """All facts of a given type for a profile."""
        where, params = _scope_where(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        rows = self.execute(
            f"SELECT * FROM atomic_facts WHERE {where} AND fact_type = ? "
            "ORDER BY created_at DESC",
            (*params, fact_type.value),
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
        # Multi-scope (M016): allow re-scoping a fact after creation so a memory
        # can be shared with a team or made global from the dashboard.
        "scope", "shared_with",
        # 4.1.19 (M052): kind columns. A caller still must not pass these
        # unless ``has_memory_kind_columns()`` is True — this allow-list only
        # says the name is a legitimate column, not that it currently exists.
        *KIND_COLUMNS,
    })

    def update_fact(self, fact_id: str, updates: dict[str, Any],
                    profile_id: str | None = None) -> None:
        """Partial update on a fact. JSON-serializes list/dict values.

        Tenant safety: when ``profile_id`` is supplied the UPDATE is constrained
        to that tenant, so a fact_id belonging to another profile cannot be
        mutated. Authorized routes always pass it.
        """
        if not updates:
            raise ValueError("updates dict must not be empty")
        bad_keys = set(updates) - self._UPDATABLE_FACT_COLUMNS
        if bad_keys:
            raise ValueError(f"Disallowed column(s): {bad_keys}")
        clean: dict[str, Any] = {}
        for k, v in updates.items():
            if k == "embedding":
                # Embeddings are stored in the canonical binary form. Falling
                # through to json.dumps here would write a text row back into a
                # converted store, one fact at a time, undoing the conversion
                # wherever a fact is updated.
                clean[k] = encode_embedding(v) if v is not None else None
            elif k in ("fisher_mean", "fisher_variance"):
                # Same hazard, same answer: these are float vectors of the same
                # width as the embedding and are stored the same way.
                clean[k] = encode_float_vector(v) if v is not None else None
            elif isinstance(v, (list, dict)):
                clean[k] = json.dumps(v)
            elif isinstance(v, (MemoryLifecycle, FactType, SignalType)):
                clean[k] = v.value
            else:
                clean[k] = v
        # L1-06: kind-aware whenever `clean` carries a kind column — see
        # `_guarded_update_set_clause` — so an enrichment write can never
        # downgrade a confirmed kind (or the `fact_type` that travels with
        # it) the same way the `store_fact` upsert already cannot.
        set_clause, set_params = _guarded_update_set_clause(clean)

        def _write() -> None:
            if profile_id is not None:
                self.execute(
                    f"UPDATE atomic_facts SET {set_clause} "
                    "WHERE fact_id = ? AND profile_id = ?",
                    (*set_params, fact_id, profile_id),
                )
            else:
                self.execute(
                    f"UPDATE atomic_facts SET {set_clause} WHERE fact_id = ?",
                    (*set_params, fact_id),
                )
            if "canonical_entities_json" in clean:
                from superlocalmemory.storage import entity_index

                entity_index.record_rewritten_fact(
                    self, fact_id, profile_id, clean["canonical_entities_json"],
                )
            # Only an update that changes something a projection is derived
            # from needs re-projecting. Recall bumps access_count on every hit,
            # so queueing on any update at all would hand the drain worker one
            # row per returned memory per recall, for a column neither Cozo nor
            # Lance holds.
            if set(updates) & projection_outbox.PROJECTED_FACT_COLUMNS:
                if profile_id is not None:
                    projection_outbox.enqueue(self, fact_id, profile_id)
                else:
                    projection_outbox.enqueue_for_fact(self, fact_id)

        self._atomically(_write)
        if "scope" in clean or "shared_with" in clean:
            # Who may see a memory changed: kept picture-search answers are stale.
            from superlocalmemory.media.generation import bump_generation
            bump_generation()

    def delete_fact(self, fact_id: str, profile_id: str | None = None) -> None:
        """Hard-delete a fact.

        DatabaseManager connections enforce FKs (PRAGMA foreign_keys=ON), so
        tables that declare ON DELETE CASCADE go with the fact. Derived rows in
        tables that declare none (BM25 tokens, vector map, ...) are removed by
        storage/fact_dependents.py in this same transaction.

        ``graph_edges`` does NOT cascade, whatever this docstring used to say.
        Its only foreign key is to ``profiles``; there is none to
        ``atomic_facts``, because an edge's endpoints can be entity ids as well
        as fact ids and a single column cannot reference two tables. So the
        edges are deleted here, explicitly. Without that, deleting a memory left
        its connections behind pointing at nothing, and the graph tidy-up pass
        would not reach them until its next run -- during which a search could
        still follow an edge into a memory that no longer exists.

        Tenant safety: when ``profile_id`` is supplied the delete is constrained
        to that tenant (the fact must belong to it), so a fact_id from another
        profile cannot be destroyed. Authorized routes always pass it.
        """
        if profile_id is not None:
            row = self.execute(
                "SELECT 1 FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
                (fact_id, profile_id),
            )
            if not row:
                return  # not this tenant's fact — no-op
        # A forgotten memory that survives in the graph or the vector index is
        # still recallable, which makes an erasure receipt a false statement. So
        # the deletes and the queued removal are one transaction: the process
        # can die immediately afterwards and the projections still get cleaned.
        def _write() -> None:
            # Read the tenant while the fact is still there. After the DELETE
            # there is nothing left to resolve it from, and a queued removal
            # filed under the wrong tenant is a fact id that outlives that
            # tenant's erasure.
            owner = profile_id or projection_outbox.resolve_profile(self, fact_id)
            from superlocalmemory.storage.fact_dependents import delete_derived_rows
            delete_derived_rows(self, fact_id)  # rows no foreign key removes
            if profile_id is not None:
                self.execute(
                    "DELETE FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
                    (fact_id, profile_id),
                )
            else:
                self.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (fact_id,))
            # Both directions: an edge naming this fact at either end is now an
            # edge to nothing.
            self.execute(
                "DELETE FROM graph_edges WHERE source_id = ? OR target_id = ?",
                (fact_id, fact_id),
            )
            projection_outbox.enqueue(
                self, fact_id, owner or "default", projection_outbox.OP_DELETE,
            )

        self._atomically(_write)

    def gc_orphaned_embedding_metadata(self) -> int:
        """Remove embedding_metadata rows whose parent atomic_fact is gone.

        P1-3 (embeddings-vector-02): orphans accumulate when facts are deleted
        through a connection that has FK enforcement OFF (the ON DELETE CASCADE
        never fires). Vector search maps a vec0 rowid → fact_id via this table;
        orphans return stale fact_ids that fail downstream fetch. This
        maintenance sweep removes them regardless of how they were created.
        Returns the number of rows deleted.
        """
        rows = self.execute(
            "SELECT COUNT(*) AS c FROM embedding_metadata "
            "WHERE fact_id NOT IN (SELECT fact_id FROM atomic_facts)"
        )
        n = int(rows[0]["c"]) if rows else 0
        if n:
            self.execute(
                "DELETE FROM embedding_metadata "
                "WHERE fact_id NOT IN (SELECT fact_id FROM atomic_facts)"
            )
        return n

    def get_fact_count(
        self, profile_id: str,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> int:
        """Visible memories (``visible_fact_clause``); personal scope from a warm,
        log-fresh in-memory count when one exists (SQL: 225 ms+ on 2 GB)."""
        warm = None if include_global or include_shared else getattr(self, "visible_count", None)
        if warm is not None and (n := warm(profile_id)) is not None:
            return n
        # Outside a transaction, a count taken since the last commit is still
        # exact (storage/store_signature): recall asked this 2-3 times a call.
        txn = getattr(self, "_txn_state", None)
        in_txn = txn is None or getattr(txn, "conn", None) is not None  # unknown: count
        key = (profile_id, bool(include_global), bool(include_shared))
        sig = None if in_txn else store_signature.of(getattr(self, "db_path", None))
        memo = self.__dict__.setdefault("_fact_count_memo", {})
        hit = memo.get(key)
        if sig is not None and hit is not None and hit[0] == sig:
            return hit[1]
        where, params = _scope_where(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        rows = self.execute(
            f"SELECT COUNT(*) AS c FROM atomic_facts WHERE {where}"
            f"{self.visible_fact_clause()}", (*params,),
        )
        count = int(rows[0]["c"]) if rows else 0
        if sig is not None:
            memo[key] = (sig, count)
        return count

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

    # COLLATE NOCASE, not LOWER(): it is the same ASCII case fold, and it is the
    # one idx_entities_name_lower is built on. LOWER(canonical_name) made every
    # lookup scan the profile's entities (#306: ~3 s per 45 lookups at 38k).
    _ENTITY_BY_NAME_SQL = (
        "SELECT * FROM canonical_entities "
        "WHERE profile_id = ? AND canonical_name = ? COLLATE NOCASE"
    )

    def get_entity_by_name(self, name: str, profile_id: str) -> CanonicalEntity | None:
        """Look up entity by name (case-insensitive)."""
        return self._entity_row(self.execute(self._ENTITY_BY_NAME_SQL, (profile_id, name)))

    def get_entity_by_id_or_name(self, key: str, profile_id: str) -> CanonicalEntity | None:
        """An entity by its ID, else by its name (case-insensitive), in one profile.

        Facts store canonical entity IDs (#306), while older rows may hold
        names; the ID is tried first because it is the primary key.
        """
        rows = self.execute(
            "SELECT * FROM canonical_entities WHERE entity_id = ? AND profile_id = ?",
            (key, profile_id),
        )
        return self._entity_row(rows) or self.get_entity_by_name(key, profile_id)

    @staticmethod
    def _entity_row(rows: list) -> CanonicalEntity | None:
        if not rows:
            return None
        d = dict(rows[0])
        return CanonicalEntity(
            entity_id=d["entity_id"], profile_id=d["profile_id"],
            canonical_name=d["canonical_name"], entity_type=d["entity_type"],
            first_seen=d["first_seen"], last_seen=d["last_seen"],
            fact_count=d["fact_count"],
        )

    def store_alias(self, alias: EntityAlias, profile_id: str) -> str:
        """Persist an entity alias under a profile. Returns alias_id.

        profile_id scopes the alias so the same entity_id appearing in two
        profiles never shares aliases.
        """
        self.execute(
            "INSERT OR REPLACE INTO entity_aliases "
            "(alias_id, profile_id, entity_id, alias, confidence, source) "
            "VALUES (?,?,?,?,?,?)",
            (alias.alias_id, profile_id, alias.entity_id, alias.alias,
             alias.confidence, alias.source),
        )
        return alias.alias_id

    def get_aliases_for_entity(
        self, entity_id: str, profile_id: str,
    ) -> list[EntityAlias]:
        """All aliases for a canonical entity within one profile."""
        rows = self.execute(
            "SELECT * FROM entity_aliases WHERE entity_id = ? AND profile_id = ?",
            (entity_id, profile_id),
        )
        return [
            EntityAlias(**{k: dict(r)[k] for k in ("alias_id", "entity_id", "alias", "confidence", "source")})
            for r in rows
        ]

    def is_memory_erasing(self, memory_id: str, profile_id: str) -> bool:
        """Whether an erasure fence blocks this source, including a deleted parent."""
        clause = memory_erasure_clause_for_db(self, "source")
        if not clause:
            return False
        rows = self.execute(
            "SELECT 1 FROM (SELECT ? AS memory_id, ? AS profile_id) AS source "
            f"WHERE 1=1{clause}", (memory_id, profile_id),
        )
        return not rows

    def get_memory_content_batch(
        self, memory_ids: list[str], profile_id: str,
        include_global: bool = False, include_shared: bool = False,
    ) -> dict[str, str]:
        """Batch-fetch original memory text. Returns {memory_id: content}.

        C4 hardening: this exposes raw memory *content* and is reachable from
        HTTP routes, so it is strictly tenant-scoped — a memory_id belonging to
        another profile is never resolved. Widen only via the scope flags.
        """
        if not memory_ids:
            return {}
        unique_ids = list(set(memory_ids))
        ph = ','.join('?' * len(unique_ids))
        where, sparams = _scope_where(
            profile_id, include_global=include_global, include_shared=include_shared,
        )
        erasure = memory_erasure_clause_for_db(self)
        rows = self.execute(
            f"SELECT memory_id, content FROM memories "
            f"WHERE memory_id IN ({ph}) AND {where}{erasure}",
            (*unique_ids, *sparams),
        )
        return {dict(r)["memory_id"]: dict(r)["content"] for r in rows}

    def get_facts_by_memory_id(
        self, memory_id: str, profile_id: str,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> list[AtomicFact]:
        """Get all atomic facts for a given memory_id."""
        where, params = _scope_where(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        rows = self.execute(
            f"SELECT * FROM atomic_facts WHERE memory_id = ? AND {where}"
            f"{self.visible_fact_clause()} "
            "ORDER BY confidence DESC",
            (memory_id, *params),
        )
        return [self._row_to_fact(r) for r in rows]

    def store_edge(self, edge: GraphEdge) -> str:
        """Persist a graph edge. Returns edge_id.

        graph-integrity-02: dedup on the LOGICAL edge identity
        (profile, source, target, type). The PK is a random edge_id, so
        without this every re-link created a duplicate row, and NetworkX
        builds read last-weight-wins — corrupting PageRank/centrality. On a
        duplicate we keep the MAX weight (strongest association wins) and
        return the existing edge_id.
        """
        # The edge and the re-projection of its endpoints are one logical write.
        # Left as separate statements they were three separate commits, which
        # cost three fsyncs on a path the materializer runs tens of thousands of
        # times, and left a window where the edge was durable and the intent to
        # project it was not.
        def _write() -> str:
            existing = self.execute(
                "SELECT edge_id FROM graph_edges "
                "WHERE profile_id = ? AND source_id = ? AND target_id = ? AND edge_type = ? "
                "LIMIT 1",
                (edge.profile_id, edge.source_id, edge.target_id, edge.edge_type.value),
            )
            if existing:
                canonical_id = dict(existing[0])["edge_id"]
                self.execute(
                    "UPDATE graph_edges SET weight = MAX(weight, ?) WHERE edge_id = ?",
                    (edge.weight, canonical_id),
                )
                self._enqueue_edge_endpoints(edge)
                return canonical_id
            _scope = getattr(edge, 'scope', None) or 'personal'
            _shared = _jd(getattr(edge, 'shared_with', None))
            self.execute(
                """INSERT OR REPLACE INTO graph_edges
                   (edge_id, profile_id, source_id, target_id, edge_type, weight, created_at,
                    scope, shared_with)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (edge.edge_id, edge.profile_id, edge.source_id, edge.target_id,
                 edge.edge_type.value, edge.weight, edge.created_at, _scope, _shared),
            )
            self._enqueue_edge_endpoints(edge)
            return edge.edge_id

        return self._atomically(_write)

    def _enqueue_edge_endpoints(self, edge: GraphEdge) -> None:
        """Re-queue both ends of an edge for projection.

        Ingestion is queryable-first: a fact commits immediately and its edges
        arrive afterwards, from the background materializer. A projection
        queued only when the fact was inserted would therefore be written
        before a single edge existed, leaving the node in the graph with none
        of its connections — which is precisely the adjacency the graph is
        consulted for.

        An endpoint may be an entity id rather than a fact id. Those are queued
        too and the drain skips whatever it cannot find as a fact; filtering
        here would mean a lookup per endpoint on every edge write, which the
        materializer does tens of thousands of times.
        """
        projection_outbox.enqueue_many(
            self, (edge.source_id, edge.target_id), edge.profile_id,
        )

    def get_edges_for_node(
        self, node_id: str, profile_id: str,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> list[GraphEdge]:
        """All edges where node_id is source or target."""
        where, params = _scope_where(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        rows = self.execute(
            f"SELECT * FROM graph_edges WHERE {where} "
            "AND (source_id = ? OR target_id = ?)",
            (*params, node_id, node_id),
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
        _scope = getattr(event, 'scope', None) or 'personal'
        _shared = _jd(getattr(event, 'shared_with', None))
        self.execute(
            """INSERT OR REPLACE INTO temporal_events
               (event_id, profile_id, entity_id, fact_id,
                observation_date, referenced_date, interval_start, interval_end,
                description, scope, shared_with)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (event.event_id, event.profile_id, event.entity_id, event.fact_id,
             _date_or_none(event.observation_date), _date_or_none(event.referenced_date),
             _date_or_none(event.interval_start), _date_or_none(event.interval_end),
             event.description, _scope, _shared),
        )
        return event.event_id

    def get_temporal_events(
        self, entity_id: str, profile_id: str,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> list[TemporalEvent]:
        """All temporal events for an entity, newest first."""
        where, params = _scope_where(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        rows = self.execute(
            f"SELECT * FROM temporal_events WHERE {where} AND entity_id = ? "
            "ORDER BY observation_date DESC",
            (*params, entity_id),
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

    def get_all_bm25_tokens(
        self,
        profile_id: str,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> dict[str, list[str]]:
        """Load the visible legacy BM25 index: fact_id -> token list."""
        if not include_global and not include_shared:
            # Preserve the historical token-store contract, including repair
            # tooling that can inspect orphaned token rows before facts exist.
            rows = self.execute(
                "SELECT fact_id, tokens FROM bm25_tokens WHERE profile_id = ?",
                (profile_id,),
            )
            return {
                dict(row)["fact_id"]: json.loads(dict(row)["tokens"])
                for row in rows
            }
        where, params = _scope_where(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
            prefix="af",
        )
        rows = self.execute(
            "SELECT bt.fact_id, bt.tokens FROM bm25_tokens AS bt "
            "JOIN atomic_facts AS af ON af.fact_id = bt.fact_id "
            f"WHERE {where}",
            (*params,),
        )
        return {dict(r)["fact_id"]: json.loads(dict(r)["tokens"]) for r in rows}

    def search_facts_fts(
        self, query: str, profile_id: str, limit: int = 20,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> list[AtomicFact]:
        """Full-text search via FTS5, joined to facts table for reconstruction."""
        from superlocalmemory.storage.fts_terms import search_match_expression

        match_expr = search_match_expression(query)  # safe MATCH syntax
        if not match_expr:
            return []
        where, params = _scope_where(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
            prefix="f",
        )
        # Full-text search is a display path: the dashboard search box, the
        # `search` tool and `fetch` all land here, and none of them go through
        # the recall engine. Before 4.0.10 put the clause here it returned 20
        # withheld rows out of 50 on the author's store.
        rows = self.execute(
            f"""SELECT f.* FROM atomic_facts_fts AS fts
               JOIN atomic_facts AS f ON f.fact_id = fts.fact_id
               WHERE fts.atomic_facts_fts MATCH ? AND {where}
                     {self.current_fact_clause('f')}
               ORDER BY fts.rank LIMIT ?""",
            (match_expr, *params, limit),
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

    def get_fact(self, fact_id: str, profile_id: str | None = None) -> AtomicFact | None:
        """Get a single row by ID, exactly as stored. NOT a display path.

        C4 defense-in-depth: when ``profile_id`` is provided the lookup is
        tenant-scoped so a fact_id from another profile cannot resolve. Left
        optional (fact_id is a random UUID sourced from already-scoped queries)
        to avoid destabilizing the core store/consolidation write path.

        DELIBERATELY UNFILTERED, and this is load-bearing. It applies neither
        ``archive_status`` nor ``quarantined`` because it is the primitive that
        write paths, correction handling and the 4.0.10 repair use to read a row
        they already hold the id of — including a withheld one, which they must
        be able to see in order to act on it. ``visible_fact_clause`` is for the
        paths that answer a question; this one answers "what is in that row".

        A caller taking a fact_id from user input and rendering the result wants
        ``get_facts_by_ids`` instead.
        """
        if profile_id is not None:
            rows = self.execute(
                "SELECT * FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
                (fact_id, profile_id),
            )
        else:
            rows = self.execute(
                "SELECT * FROM atomic_facts WHERE fact_id = ?", (fact_id,),
            )
        return self._row_to_fact(rows[0]) if rows else None

    def get_facts_by_ids(
        self, fact_ids: list[str], profile_id: str,
        include_global: bool = False,
        include_shared: bool = False,
        *,
        include_quarantined: bool = False,
        exclude_non_answer: bool = False,
    ) -> list[AtomicFact]:
        """Get multiple facts by their IDs, scoped to a profile.

        THIS IS THE PLACE QUARANTINE IS ENFORCED, and the only one.

        Every retrieval channel re-authorises its candidates through here
        (``retrieval/scope_policy.py`` — "candidate generators may use caches,
        approximate indexes, or graph stores that are not the authorization
        source of truth"), and the engine hydrates the fused set from here too.
        A fact this method does not return has no content to show, and
        ``retrieval/engine.py`` drops it: ``if fact is None: continue``. So one
        clause here covers bm25, semantic, temporal, entity, hopfield and
        spreading activation, in normal and deep recall alike, whether or not
        the forgetting filter is registered.

        The alternatives were checked and rejected. ``_scope_where`` looks like
        the natural home but is spliced against ``graph_edges``,
        ``temporal_events``, ``memories``, ``bm25_tokens``,
        ``fact_temporal_validity`` and ``correction_cases`` as well as
        ``atomic_facts``, so a column reference there breaks eight call sites.
        ``ForgettingFilter`` is optional (it no-ops when forgetting is
        disabled) and excludes nothing in deep recall.

        ``include_quarantined=True`` is for repair, export and erasure — paths
        that must be able to see a withheld row in order to act on it. It is
        keyword-only and greppable on purpose: every caller that opts in is
        meant to be found in one search.
        """
        if not fact_ids:
            return []
        where, params = _scope_where(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        placeholders = ",".join("?" for _ in fact_ids)
        visible = self.visible_fact_clause(include_quarantined=include_quarantined,
                                           exclude_non_answer=exclude_non_answer)
        rows = self.execute(
            f"SELECT * FROM atomic_facts WHERE fact_id IN ({placeholders}) "
            f"AND {where}"
            f"{visible} "
            "ORDER BY created_at DESC",
            (*fact_ids, *params),
        )
        return [self._row_to_fact(r) for r in rows]

    def visible_fact_ids(
        self, fact_ids: list[str] | tuple[str, ...], profile_id: str,
        include_global: bool = False,
        include_shared: bool = False,
        *,
        include_quarantined: bool = False,
        exclude_non_answer: bool = False,
    ) -> set[str]:
        """Which of *fact_ids* this profile may see. Same rule, no hydration.

        ``get_facts_by_ids`` is the authorization source of truth and every
        retrieval channel re-authorises through it — but a channel deciding
        *which* of its candidates are allowed does not need their content, and
        paying for the content is most of what recall costs.

        Measured on the author's store: the entity channel authorised 3,659
        candidates to return 20, and that single call was **374 ms of a 430 ms
        recall — 87%**. Not the graph walk, which is 3.8 ms. The cost is
        ``_row_to_fact`` decoding a 768-float embedding and two 768-float Fisher
        vectors per row: about 8.4 million floats deserialised to answer a
        yes/no question about 3,659 ids. The same trap is recorded a few
        hundred lines up, where loading full facts "turned one new fact into a
        5-second recall stall".

        The predicate is built by the same two calls as ``get_facts_by_ids``, in
        the same order, so the two cannot answer differently. A test asserts
        that on the same inputs. Batched because the id list is unbounded and a
        single ``IN`` clause is not.
        """
        if not fact_ids:
            return set()
        where, params = _scope_where(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        visible = self.visible_fact_clause(include_quarantined=include_quarantined,
                                           exclude_non_answer=exclude_non_answer)
        unique = list(dict.fromkeys(fact_ids))
        allowed: set[str] = set()
        # Well inside SQLITE_MAX_VARIABLE_NUMBER once the scope parameters are
        # added, on every build this ships against.
        chunk = 800
        for start in range(0, len(unique), chunk):
            batch = unique[start:start + chunk]
            placeholders = ",".join("?" for _ in batch)
            rows = self.execute(
                f"SELECT fact_id FROM atomic_facts WHERE fact_id IN ({placeholders}) "
                f"AND {where}{visible}",
                (*batch, *params),
            )
            allowed.update(dict(row)["fact_id"] for row in rows)
        return allowed

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
        # Not summary-fenced here: the only caller (retrieval/profile_channel)
        # uses fact ids, which it passes through filter_authorized_results, so a
        # mixed profile still yields its retained facts. Summary TEXT is fenced
        # at the entity routes.
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
        erasure = aggregate_erasure_clause_for_db(self, "memory_scenes")
        rows = self.execute(
            f"SELECT * FROM memory_scenes WHERE profile_id = ?{erasure} "
            "ORDER BY last_updated DESC",
            (profile_id,),
        )
        return [self._row_to_scene(r) for r in rows]

    def get_scenes_for_fact(
        self, fact_id: str, profile_id: str, *, include_erasing: bool = False,
    ) -> list[MemoryScene]:
        """All scenes whose fact_ids JSON array contains *fact_id*.

        Scenes naming a fact whose erasure has started are withheld from
        readers. ``include_erasing`` is for the erasure/mutation path itself,
        which must still find the scene in order to remove the fact from it.
        """
        erasure = "" if include_erasing else aggregate_erasure_clause_for_db(
            self, "memory_scenes")
        rows = self.execute(
            "SELECT * FROM memory_scenes WHERE profile_id = ? "
            f"AND fact_ids_json LIKE ?{erasure} ORDER BY last_updated DESC",
            (profile_id, f'%"{fact_id}"%'),
        )
        return [self._row_to_scene(r) for r in rows]

    def get_scenes_for_facts_batch(
        self, fact_ids: list[str], profile_id: str,
    ) -> dict[str, list[MemoryScene]]:
        """v3.5.0: batch scene lookup — one query replaces N individual LIKE scans.

        The 20 individual ``get_scenes_for_fact`` calls in the retrieval engine's
        scene expansion path were the single largest recall latency source (~5.7s).
        This replaces them with a single multi-LIKE OR query. Returns
        ``{fact_id: [scenes]}`` for facts that belong to at least one scene.
        """
        if not fact_ids:
            return {}
        clauses = " OR ".join(
            '(profile_id = ? AND fact_ids_json LIKE ?)' for _ in fact_ids
        )
        params: list[str] = []
        for fid in fact_ids:
            params.extend((profile_id, f'%"{fid}"%'))
        erasure = aggregate_erasure_clause_for_db(self, "memory_scenes")
        rows = self.execute(
            f"SELECT * FROM memory_scenes WHERE ({clauses}){erasure} ORDER BY last_updated DESC",
            tuple(params),
        )
        out: dict[str, list[MemoryScene]] = {}
        for r in rows:
            scene = self._row_to_scene(r)
            for fid in fact_ids:
                if fid in (scene.fact_ids or []):
                    out.setdefault(fid, []).append(scene)
        return out

    def increment_entity_fact_count(self, entity_id: str, profile_id: str = "default") -> None:
        """Atomically increment fact_count for a canonical entity scoped to profile.

        L-01 fix: the original query had no profile_id guard; any entity_id match
        would be updated regardless of owner. The AND profile_id = ? clause prevents
        cross-profile mutations via shared entity_id values.
        """
        self.execute(
            "UPDATE canonical_entities SET fact_count = fact_count + 1 "
            "WHERE entity_id = ? AND profile_id = ?",
            (entity_id, profile_id),
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
        include_global: bool = False,
        include_shared: bool = False,
    ) -> list[TemporalEvent]:
        """Temporal events within a date range (inclusive)."""
        where, params = _scope_where(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        rows = self.execute(
            f"SELECT * FROM temporal_events WHERE {where} "
            "AND (referenced_date BETWEEN ? AND ? "
            "     OR observation_date BETWEEN ? AND ?) "
            "ORDER BY observation_date DESC",
            (*params, start_date, end_date, start_date, end_date),
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

    def get_fact_context(self, fact_id: str, profile_id: str | None = None) -> dict | None:
        """Get contextual description for a fact (C4: optionally tenant-scoped)."""
        if profile_id is not None:
            rows = self.execute(
                "SELECT * FROM fact_context WHERE fact_id = ? AND profile_id = ?",
                (fact_id, profile_id),
            )
        else:
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

    def cleanup_activation_cache(
        self, batch_size: int = 5000, max_batches: int = 500,
    ) -> int:
        """Delete expired activation_cache rows in bounded batches.

        Wired into MaintenanceScheduler.  Historically NEITHER cleanup path was
        ever called, so activation_cache grew without bound — observed 83,518
        rows on a real DB, all expired, oldest ~3.5 months old.  That bloats the
        table and its ``idx_actcache_expires`` index and slows every cache
        INSERT OR REPLACE / lookup.

        Batched so clearing a large backlog never holds the write lock for one
        long DELETE: each batch commits and yields, letting remember/materialize
        writers interleave.  ``idx_actcache_expires`` makes the predicate
        index-backed.  Steady state (30-min cycle) deletes only one cycle's
        worth, so the loop exits after a single small batch.
        """
        total_deleted = 0
        for _ in range(max_batches):
            remaining = self.execute(
                "SELECT COUNT(*) AS c FROM activation_cache "
                "WHERE expires_at < datetime('now')"
            )
            n = int(remaining[0]["c"]) if remaining else 0
            if n <= 0:
                break
            self.execute(
                "DELETE FROM activation_cache WHERE cache_id IN ("
                "  SELECT cache_id FROM activation_cache "
                "  WHERE expires_at < datetime('now') LIMIT ?)",
                (batch_size,),
            )
            total_deleted += min(n, batch_size)
        return total_deleted

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
        *,
        fill_only: bool = False,
    ) -> None:
        """Create or enrich the temporal record for a fact.

        The 4.0.2 fact writer creates the record immediately to anchor
        transaction time. A later temporal extraction may add event-time bounds;
        it must not be discarded merely because the anchor already exists.

        ``fill_only`` is the fact writer's default event-time start (WP-19 A1):
        it fills an empty ``valid_from`` and never replaces one a correction,
        a replacement or an extraction already set.
        """
        from datetime import UTC
        from datetime import datetime as _dt
        system_created_at = _dt.now(UTC).isoformat()
        self.execute(
            "INSERT OR IGNORE INTO fact_temporal_validity "
            "(fact_id, profile_id, valid_from, valid_until, system_created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (fact_id, profile_id, valid_from, valid_until, system_created_at),
        )
        if fill_only and valid_from is not None:
            self.execute(
                "UPDATE fact_temporal_validity SET valid_from = ? "
                "WHERE fact_id = ? AND profile_id = ? AND valid_from IS NULL",
                (valid_from, fact_id, profile_id),
            )
            current = self.execute(
                "SELECT valid_from FROM fact_temporal_validity WHERE fact_id = ? AND profile_id = ?",
                (fact_id, profile_id),
            )
            valid_from = current[0]["valid_from"] if current else valid_from
        elif valid_from is not None or valid_until is not None:
            self.execute(
                "UPDATE fact_temporal_validity "
                "SET valid_from = COALESCE(?, valid_from), "
                "    valid_until = COALESCE(?, valid_until) "
                "WHERE fact_id = ? AND profile_id = ?",
                (valid_from, valid_until, fact_id, profile_id),
            )

        from superlocalmemory.storage.event_time import mirror

        mirror(self, fact_id, profile_id, valid_from)
        from superlocalmemory.encoding.non_answer import mark

        rows = self.execute('SELECT content FROM atomic_facts WHERE fact_id=? AND profile_id=?',
                            (fact_id, profile_id))
        if rows:
            mark(self, fact_id, profile_id, rows[0]['content'])

    def get_temporal_validity(self, fact_id: str, profile_id: str | None = None) -> dict | None:
        """Get temporal validity record for a fact (C4: optionally tenant-scoped)."""
        if profile_id is not None:
            rows = self.execute(
                "SELECT * FROM fact_temporal_validity WHERE fact_id = ? AND profile_id = ?",
                (fact_id, profile_id),
            )
        else:
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
        *,
        event_valid_until: str | None = None,
    ) -> None:
        """Mark a fact as invalidated, preserving bi-temporal independence.

        - valid_until (event-time): changed only when a reviewer supplies an
          independently validated real-world boundary.  Review time, source
          timestamps, and a detector's conclusion are not a valid substitute.
        - system_expired_at (transaction-time): when the system learned the
          fact was invalid — always set to now.

        The two dimensions must remain independent: a fact can be true until
        2020-06-30 in the real world (valid_until) while the system only
        discovers this in 2024 (system_expired_at).

        Never deletes the fact (immutability).
        """
        from datetime import UTC
        from datetime import datetime as _dt
        now = _dt.now(UTC).isoformat()

        self.execute(
            "UPDATE fact_temporal_validity "
            "SET valid_until = COALESCE(?, valid_until), system_expired_at = ?, "
            "    invalidated_by = ?, invalidation_reason = ? "
            "WHERE fact_id = ? AND system_expired_at IS NULL",
            (event_valid_until, now, invalidated_by, invalidation_reason, fact_id),
        )

    def get_valid_facts(self, profile_id: str) -> list[str]:
        """Get fact_ids that are currently valid (not expired).

        Returns facts that either have no temporal record (assumed valid)
        or whose temporal record satisfies:
          - valid_until IS NULL (open-ended) OR valid_until > now() (still in window)
          - system_expired_at IS NULL (not system-invalidated)

        M-02 fix: the original query used ``tv.valid_until IS NULL`` which
        incorrectly excluded future-dated (still valid) facts. The correct
        predicate is a date comparison against the current timestamp.
        """
        rows = self.execute(
            "SELECT f.fact_id FROM atomic_facts f "
            "LEFT JOIN fact_temporal_validity tv ON f.fact_id = tv.fact_id "
            "WHERE f.profile_id = ? "
            "  AND (tv.fact_id IS NULL "
            "       OR ( "
            "           (tv.valid_until IS NULL "
            "            OR tv.valid_until > strftime('%Y-%m-%dT%H:%M:%SZ', 'now')) "
            "           AND tv.system_expired_at IS NULL "
            "       ))",
            (profile_id,),
        )
        return [dict(r)["fact_id"] for r in rows]

    def get_invalidated_fact_ids(
        self,
        fact_ids: list[str],
        profile_id: str,
        as_of: str | None = None,
        *,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> set[str]:
        """Return the subset of ``fact_ids`` that are system-invalidated.

        A fact is system-invalidated when ``system_expired_at`` is set — i.e.
        it was superseded/contradicted by a newer fact (see
        ``invalidate_fact_temporal``). Such facts are wrong/outdated and must be
        excluded from default retrieval (T1, Phase 4).

        Phase 4b — bi-temporal as_of:
          When ``as_of`` is None (default): returns ALL facts with
          ``system_expired_at IS NOT NULL`` — existing behaviour, no regression.

          When ``as_of`` is set (UTC ISO 8601, "+00:00" suffix): returns only
          facts where ``system_expired_at <= as_of`` (transaction-time boundary
          inclusive). This means supersessions that occurred AFTER ``as_of`` are
          excluded — at the historical query point the fact was still valid.

        ``as_of`` MUST be UTC-normalized via ``normalize_as_of()`` before this
        call. The stored ``system_expired_at`` values use Python's
        ``datetime.now(UTC).isoformat()`` format ("...+00:00") so the
        ``normalize_as_of()`` "+00:00" output produces correct lexicographic
        SQL comparisons.

        Bounded + indexed: only the supplied candidate ids are queried (never a
        full-table scan), keyed on the ``fact_id`` PK with the
        ``idx_temporal_system_expired`` index covering the predicate. Chunked to
        stay well under SQLite's ~999 bound-parameter limit. The visibility
        predicate is evaluated on the fact owner's row, so an opted-in global or
        shared fact is checked against *its owner's* temporal record rather than
        incorrectly against the requesting profile. Facts with no temporal
        record — or a record whose ``system_expired_at`` is NULL — are NOT
        returned (treated as valid), so existing DBs need no backfill.

        Event-time expiry (``valid_until`` in the past) is intentionally NOT
        applied here: it is query-scoped (historical queries legitimately want
        expired facts, per ``include_expired_in_history``) and handled by the
        time-window path, not by this blanket admission filter.
        """
        if not fact_ids:
            return set()
        invalid: set[str] = set()
        scope_where, scope_params = _scope_where(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
            prefix="f",
        )
        chunk = 900
        for start in range(0, len(fact_ids), chunk):
            batch = fact_ids[start:start + chunk]
            placeholders = ",".join("?" for _ in batch)
            if as_of is not None:
                # Transaction-time point-in-time: only supersessions that
                # occurred AT OR BEFORE as_of contribute to invalidation.
                # Supersessions after as_of are invisible at this query point.
                rows = self.execute(
                    f"SELECT tv.fact_id FROM fact_temporal_validity tv "
                    f"JOIN atomic_facts f ON f.fact_id = tv.fact_id "
                    f"WHERE tv.fact_id IN ({placeholders}) "
                    f"  AND {scope_where} "
                    f"  AND tv.profile_id = f.profile_id "
                    f"  AND tv.system_expired_at IS NOT NULL "
                    f"  AND tv.system_expired_at <= ?",
                    (*batch, *scope_params, as_of),
                )
            else:
                rows = self.execute(
                    f"SELECT tv.fact_id FROM fact_temporal_validity tv "
                    f"JOIN atomic_facts f ON f.fact_id = tv.fact_id "
                    f"WHERE tv.fact_id IN ({placeholders}) "
                    f"  AND {scope_where} "
                    f"  AND tv.profile_id = f.profile_id "
                    f"  AND tv.system_expired_at IS NOT NULL",
                    (*batch, *scope_params),
                )
            for r in rows:
                invalid.add(dict(r)["fact_id"])
        return invalid

    def get_nonapplied_correction_successor_ids(
        self,
        fact_ids: list[str],
        profile_id: str,
        *,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> set[str]:
        """Return candidate successors that are not current review truth.

        M042 is optional for older databases.  Its absence is safe because
        canonical proposal never commits a successor unless the same
        transaction also writes M042.  Once present, a read failure must reach
        the retrieval fail-closed boundary rather than be converted to empty.
        """
        if not fact_ids:
            return set()
        present = self.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='correction_cases'"
        )
        if not present:
            return set()
        scope_where, scope_params = _scope_where(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
            prefix="f",
        )
        inadmissible: set[str] = set()
        for start in range(0, len(fact_ids), 900):
            batch = fact_ids[start:start + 900]
            placeholders = ",".join("?" for _ in batch)
            # Neither a caller's replacement nor a save-path guess ever
            # withholds the memory the caller saved (correction_cases.py).
            rows = self.execute(
                "SELECT c.successor_fact_id FROM correction_cases c "
                "JOIN atomic_facts f ON f.fact_id=c.successor_fact_id "
                f"WHERE c.successor_fact_id IN ({placeholders}) AND {scope_where} "
                "AND c.profile_id=f.profile_id "
                "AND c.status IN ('proposed', 'rejected', 'rolled_back') "
                "AND " + SAVED_SUCCESSOR_SQL.format(alias="c"),
                (*batch, *scope_params, *SAVED_SUCCESSOR_PARAMS),
            )
            inadmissible.update(str(row["successor_fact_id"]) for row in rows)
        return inadmissible

    def get_correction_inadmissible_fact_ids(
        self,
        fact_ids: list[str],
        profile_id: str,
        as_of: str | None = None,
        *,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> set[str]:
        """Return current-lifecycle exclusions with one bounded SQLite read.

        Recall needs both sides of reviewed correction truth: an expired
        predecessor and a successor whose case is not applied. One read-only,
        uncached connection and one UNION query keep the older helpers'
        profile/scope and ``as_of`` semantics. It takes no write lock: through
        ``raw_connection`` a recall waited 44 s here behind a long write.
        """
        if not fact_ids:
            return set()
        scope_where, scope_params = _scope_where(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
            prefix="f",
        )
        inadmissible: set[str] = set()
        with read_only_snapshot(self) as conn:
            correction_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='correction_cases'"
            ).fetchone()
            for start in range(0, len(fact_ids), 900):
                batch = fact_ids[start:start + 900]
                placeholders = ",".join("?" for _ in batch)
                temporal_conditions = (
                    "AND tv.system_expired_at IS NOT NULL "
                    + ("AND tv.system_expired_at <= ?" if as_of is not None else "")
                )
                temporal_sql = (
                    "SELECT tv.fact_id AS fact_id "
                    "FROM fact_temporal_validity tv "
                    "JOIN atomic_facts f ON f.fact_id=tv.fact_id "
                    f"WHERE tv.fact_id IN ({placeholders}) AND {scope_where} "
                    "AND tv.profile_id=f.profile_id "
                    f"{temporal_conditions}"
                )
                temporal_params: tuple[Any, ...] = (
                    *batch,
                    *scope_params,
                    *((as_of,) if as_of is not None else ()),
                )
                if correction_table is None:
                    rows = conn.execute(temporal_sql, temporal_params).fetchall()
                else:
                    correction_sql = (
                        "SELECT c.successor_fact_id AS fact_id "
                        "FROM correction_cases c "
                        "JOIN atomic_facts f ON f.fact_id=c.successor_fact_id "
                        f"WHERE c.successor_fact_id IN ({placeholders}) AND {scope_where} "
                        "AND c.profile_id=f.profile_id "
                        "AND c.status IN ('proposed', 'rejected', 'rolled_back') "
                        # Never the memory the caller saved (see
                        # get_nonapplied_correction_successor_ids).
                        "AND " + SAVED_SUCCESSOR_SQL.format(alias="c")
                    )
                    rows = conn.execute(
                        f"{temporal_sql} UNION {correction_sql}",
                        (*temporal_params, *batch, *scope_params,
                         *SAVED_SUCCESSOR_PARAMS),
                    ).fetchall()
                inadmissible.update(str(row["fact_id"]) for row in rows)
        return inadmissible

    def get_strict_temporal_inadmissible_fact_ids(
        self,
        fact_ids: list[str],
        profile_id: str,
        *,
        known_as_of: str | None = None,
        valid_at: str | None = None,
        include_unknown: bool = False,
        include_global: bool = False,
        include_shared: bool = False,
    ) -> set[str]:
        """Return candidates excluded by an explicit two-clock query.

        ``known_as_of`` is transaction time: it asks what this SLM instance had
        learned by a timestamp. ``valid_at`` is event time: it asks what was
        true at a timestamp according to the requested knowledge state. The
        axes are independent and may be supplied separately or together.

        A fact with no temporal row predates the 4.0.2 write invariant and is
        ``legacy_unknown``. Strict time-travel excludes it unless the caller
        explicitly asks to include unknown history. This is intentionally
        bounded to the already-retrieved candidate pool.
        """
        if not fact_ids or (known_as_of is None and valid_at is None):
            return set()
        from datetime import datetime as _dt

        from superlocalmemory.retrieval.temporal_utils import normalize_as_of

        def _parse_timestamp(value: object) -> _dt | None:
            normalized = normalize_as_of(value)
            return _dt.fromisoformat(normalized) if normalized is not None else None

        known_boundary = _parse_timestamp(known_as_of) if known_as_of is not None else None
        valid_boundary = _parse_timestamp(valid_at) if valid_at is not None else None
        inadmissible: set[str] = set()
        scope_where, scope_params = _scope_where(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
            prefix="f",
        )
        chunk = 900
        for start in range(0, len(fact_ids), chunk):
            batch = fact_ids[start:start + chunk]
            placeholders = ",".join("?" for _ in batch)
            rows = self.execute(
                f"SELECT f.fact_id, tv.fact_id AS temporal_fact_id, "
                f"tv.system_created_at, tv.system_expired_at, "
                f"tv.valid_from, tv.valid_until FROM atomic_facts f "
                f"LEFT JOIN fact_temporal_validity tv "
                f"  ON tv.fact_id = f.fact_id AND tv.profile_id = f.profile_id "
                f"WHERE f.fact_id IN ({placeholders}) "
                f"  AND {scope_where}",
                (*batch, *scope_params),
            )
            for row in rows:
                values = dict(row)
                has_temporal_record = values["temporal_fact_id"] is not None
                if not has_temporal_record:
                    if not include_unknown:
                        inadmissible.add(values["fact_id"])
                    continue
                unknown = False
                if known_boundary is not None:
                    created = _parse_timestamp(values["system_created_at"])
                    expired = _parse_timestamp(values["system_expired_at"])
                    if created is None:
                        unknown = True
                    elif created > known_boundary:
                        inadmissible.add(values["fact_id"])
                        continue
                    elif expired is not None and expired <= known_boundary:
                        inadmissible.add(values["fact_id"])
                        continue
                if valid_boundary is not None:
                    valid_from = _parse_timestamp(values["valid_from"])
                    valid_until = _parse_timestamp(values["valid_until"])
                    if values["valid_from"] is not None and valid_from is None:
                        unknown = True
                    elif valid_from is not None and valid_from > valid_boundary:
                        inadmissible.add(values["fact_id"])
                        continue
                    if values["valid_until"] is not None and valid_until is None:
                        unknown = True
                    elif valid_until is not None and valid_until <= valid_boundary:
                        inadmissible.add(values["fact_id"])
                        continue
                if unknown and not include_unknown:
                    inadmissible.add(values["fact_id"])
        return inadmissible

    def get_event_time_expired_fact_ids(
        self,
        fact_ids: list[str],
        profile_id: str,
        as_of: str | None = None,
    ) -> set[str]:
        """Return the subset of ``fact_ids`` that are event-time out-of-range.

        Returns fact_ids whose event-time validity window does not encompass
        ``as_of`` (or the current wall-clock time when ``as_of`` is None):

        1. **Already expired** — ``valid_until IS NOT NULL AND valid_until <= ref``
           where ``ref`` is ``as_of`` when provided or the current UTC time.
           The ``<=`` implements the half-open interval ``[valid_from, valid_until)``:
           a fact with ``valid_until == as_of`` has expired at that boundary (Phase 4b fix).
        2. **Not yet valid** — ``valid_from IS NOT NULL AND valid_from > as_of``
           (only when ``as_of`` is provided for explicit point-in-time recall).

        Zero-regression guarantee: facts with ``valid_until = NULL`` are
        assumed open-ended (still valid) and are NEVER returned. Facts with no
        temporal record at all are NEVER returned (assumed valid). Because almost
        all existing facts have ``valid_until = NULL``, the default path
        (``as_of=None``) returns an empty set and causes no demotion.

        Bounded + indexed: only the supplied candidate ids are queried (never a
        full-table scan), keyed on the ``fact_id`` PK. Chunked to stay under
        SQLite's ~999 bound-parameter limit (chunk size 900). The
        ``idx_temporal_valid(profile_id, valid_until)`` index assists the
        ``valid_until <`` range predicate after the PK IN-lookup. The
        ``valid_from > as_of`` branch operates on the same bounded PK row set
        (≤ 900 rows per chunk) — no full-table scan occurs.

        Fail-open: any DB error logs a warning and returns an empty set so
        retrieval can never break because of a validity lookup failure.

        Args:
            fact_ids: Candidate fact IDs to check (bounded retrieval pool).
            profile_id: Current profile — scopes the lookup to one tenant.
            as_of: Optional ISO 8601 datetime string for point-in-time recall.
                When set, facts not yet valid at this time are also returned.
                When None (default), only facts past their ``valid_until`` are
                returned — the standard current-time path.

        Note:
            ``as_of`` must be in the same ISO 8601 format as the stored
            ``valid_until`` / ``valid_from`` values so SQLite's lexicographic
            string comparison correctly orders the timestamps.
        """
        if not fact_ids:
            return set()
        expired: set[str] = set()
        try:
            chunk = 900
            for start in range(0, len(fact_ids), chunk):
                batch = fact_ids[start:start + chunk]
                placeholders = ",".join("?" for _ in batch)
                if as_of is not None:
                    # Time-travel: expired-before-as_of OR not-yet-started-at-as_of.
                    rows = self.execute(
                        f"SELECT fact_id FROM fact_temporal_validity "
                        f"WHERE fact_id IN ({placeholders}) "
                        f"  AND profile_id = ? "
                        f"  AND ("
                        f"    (valid_until IS NOT NULL AND valid_until <= ?) "
                        f"    OR (valid_from IS NOT NULL AND valid_from > ?)"
                        f"  )",
                        (*batch, profile_id, as_of, as_of),
                    )
                else:
                    # Default path: only facts whose valid_until has passed.
                    # idx_temporal_valid(profile_id, valid_until) assists range scan.
                    rows = self.execute(
                        f"SELECT fact_id FROM fact_temporal_validity "
                        f"WHERE fact_id IN ({placeholders}) "
                        f"  AND profile_id = ? "
                        f"  AND valid_until IS NOT NULL "
                        f"  AND valid_until < strftime('%Y-%m-%dT%H:%M:%SZ', 'now')",
                        (*batch, profile_id),
                    )
                for r in rows:
                    expired.add(dict(r)["fact_id"])
        except Exception as exc:
            logger.warning(
                "Event-time expiry lookup failed (fail-open): %s", exc,
            )
            return set()
        return expired

    def get_fact_event_times(
        self, fact_ids: list[str], profile_id: str,
    ) -> dict[str, str]:
        """Map each candidate fact_id to its best-available event time.

        Priority (most specific first): ``referenced_date`` (the date the fact
        is *about*) → ``observation_date`` (when it was observed) →
        ``valid_from`` (bi-temporal event start) → ``created_at`` (storage
        time, always present). Used by time-window recall to prune candidates
        by when the underlying event happened, falling back to capture time for
        undated facts.

        Bounded + indexed (candidate ids only, ``fact_id`` PK), chunked under
        SQLite's bound-parameter limit. Facts absent from the result (unknown
        id / wrong profile) are simply omitted.
        """
        if not fact_ids:
            return {}
        out: dict[str, str] = {}
        chunk = 900
        for start in range(0, len(fact_ids), chunk):
            batch = fact_ids[start:start + chunk]
            placeholders = ",".join("?" for _ in batch)
            rows = self.execute(
                f"SELECT f.fact_id AS fact_id, "
                f"COALESCE(f.referenced_date, f.observation_date, "
                f"         tv.valid_from, f.created_at) AS event_time "
                f"FROM atomic_facts f "
                f"LEFT JOIN fact_temporal_validity tv ON f.fact_id = tv.fact_id "
                f"WHERE f.fact_id IN ({placeholders}) "
                f"AND (f.profile_id = ? OR f.scope = 'global')",
                (*batch, profile_id),
            )
            for r in rows:
                d = dict(r)
                if d.get("event_time"):
                    out[d["fact_id"]] = d["event_time"]
        return out

    def delete_temporal_validity(self, fact_id: str) -> None:
        """Delete temporal validity record (for testing/rollback only)."""
        self.execute(
            "DELETE FROM fact_temporal_validity WHERE fact_id = ?",
            (fact_id,),
        )

    # ------------------------------------------------------------------
    # Phase 4 (T3b): fact-augmented key expansion (BM25 alt-keys)
    # ------------------------------------------------------------------

    def upsert_fact_expansion(self, fact_id: str, alt_keys: str) -> None:
        """Store/replace a fact's alternate keys in ``fact_expansion_fts``.

        Standalone FTS5 (no external-content triggers), so we replace by hand:
        delete any prior row for the fact, then insert the new alt-keys. An
        empty/blank ``alt_keys`` clears the fact's expansion entry. Fail-soft:
        a missing FTS table (legacy DB) never breaks the write path.
        """
        try:
            self.execute(
                "DELETE FROM fact_expansion_fts WHERE fact_id = ?", (fact_id,)
            )
            if alt_keys and alt_keys.strip():
                self.execute(
                    "INSERT INTO fact_expansion_fts (fact_id, alt_keys) "
                    "VALUES (?, ?)",
                    (fact_id, alt_keys.strip()),
                )
        except Exception as exc:  # pragma: no cover — legacy/missing FTS table
            logger.debug("upsert_fact_expansion skipped for %s: %s", fact_id, exc)

    def reset_fact_expansion(self, fact_id: str, alt_keys: str = "") -> None:
        """Replace the expansion row unconditionally, keeping it alive with new alt_keys.

        Unlike ``upsert_fact_expansion``, this always inserts (even when
        ``alt_keys`` is empty) so the row survives as a cleared placeholder.
        Used by update paths that must guarantee the expansion entry exists but
        holds no stale tokens. Fail-soft: a missing FTS table is a no-op.
        """
        try:
            self.execute(
                "DELETE FROM fact_expansion_fts WHERE fact_id = ?", (fact_id,)
            )
            self.execute(
                "INSERT INTO fact_expansion_fts (fact_id, alt_keys) VALUES (?, ?)",
                (fact_id, alt_keys),
            )
        except Exception as exc:
            logger.debug("reset_fact_expansion skipped for %s: %s", fact_id, exc)

    def update_temporal_event_description(
        self, fact_id: str, description: str
    ) -> None:
        """Update the description column in ``temporal_events`` for a fact.

        Fail-soft: absent table (pre-migration DB) is silently skipped.
        """
        try:
            self.execute(
                "UPDATE temporal_events SET description = ? WHERE fact_id = ?",
                (description, fact_id),
            )
        except Exception as exc:
            logger.debug(
                "update_temporal_event_description skipped for %s: %s", fact_id, exc
            )

    def delete_bm25_tokens_for_fact(self, fact_id: str) -> None:
        """Delete persisted BM25 tokens for a fact from the ``bm25_tokens`` table."""
        try:
            self.execute(
                "DELETE FROM bm25_tokens WHERE fact_id = ?", (fact_id,)
            )
        except Exception as exc:
            logger.debug("delete_bm25_tokens_for_fact skipped for %s: %s", fact_id, exc)

    def delete_graph_edges_for_fact(self, fact_id: str) -> None:
        """Delete all graph edges where this fact is the source or the target."""
        try:
            self.execute(
                "DELETE FROM graph_edges WHERE source_id = ? OR target_id = ?",
                (fact_id, fact_id),
            )
        except Exception as exc:
            logger.debug("delete_graph_edges_for_fact skipped for %s: %s", fact_id, exc)

    def remove_fact_from_scenes(self, fact_id: str, profile_id: str) -> None:
        """Remove a fact_id from every ``memory_scenes`` JSON array in the profile.

        Scenes that become empty after removal are deleted entirely.
        Fail-soft: any exception is logged and ignored.
        """
        try:
            scenes = self.get_scenes_for_fact(fact_id, profile_id, include_erasing=True)
            erased_text = erasure_derived.fact_content(self.execute, profile_id, fact_id)
            for scene in scenes:
                new_ids = [fid for fid in (scene.fact_ids or []) if fid != fact_id]
                if new_ids:
                    # The theme is the anchor fact's first 200 characters, so it
                    # must not outlive the fact (storage/erasure_derived.py).
                    theme = scene.theme
                    if erasure_derived.theme_quotes(
                            theme, scene.fact_ids or [], fact_id, erased_text):
                        theme = erasure_derived.theme_from_members(
                            self.execute, profile_id, new_ids)
                    self.execute(
                        "UPDATE memory_scenes SET fact_ids_json = ?, theme = ? "
                        "WHERE scene_id = ?",
                        (json.dumps(new_ids), theme, scene.scene_id),
                    )
                else:
                    self.execute(
                        "DELETE FROM memory_scenes WHERE scene_id = ?",
                        (scene.scene_id,),
                    )
        except Exception as exc:
            logger.debug("remove_fact_from_scenes skipped for %s: %s", fact_id, exc)

    def delete_memory_for_fact(self, fact_id: str, profile_id: str) -> None:
        """Delete the raw ``memories`` record that sourced this fact.

        Reads the ``memory_id`` from ``atomic_facts`` before the fact row is
        gone, then deletes the memory. Fail-soft: any exception is logged.
        """
        try:
            rows = self.execute(
                "SELECT memory_id FROM atomic_facts "
                "WHERE fact_id = ? AND profile_id = ? LIMIT 1",
                (fact_id, profile_id),
            )
            if rows:
                memory_id = dict(rows[0]).get("memory_id") or ""
                if memory_id:
                    self.execute(
                        "DELETE FROM memories "
                        "WHERE memory_id = ? AND profile_id = ?",
                        (memory_id, profile_id),
                    )
        except Exception as exc:
            logger.debug("delete_memory_for_fact skipped for %s: %s", fact_id, exc)

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

    # ------------------------------------------------------------------
    # Phase A: Fact Retention CRUD (Forgetting Brain)
    # ------------------------------------------------------------------

    def get_retention(self, fact_id: str, profile_id: str) -> dict | None:
        """Get retention data for a single fact.

        Returns dict with column names as keys, or None if not found.
        All SQL parameterized (HR-05).
        """
        rows = self.execute(
            "SELECT fact_id, retention_score, memory_strength, access_count, "
            "       last_accessed_at, lifecycle_zone, last_computed_at "
            "FROM fact_retention WHERE fact_id = ? AND profile_id = ?",
            (fact_id, profile_id),
        )
        return dict(rows[0]) if rows else None

    def batch_get_retention(
        self, fact_ids: list[str], profile_id: str,
    ) -> list[dict]:
        """Get retention data for a batch of facts.

        Uses dynamic ? placeholders for IN clause (never string concat).
        Missing fact_ids are simply absent from results.
        All SQL parameterized (HR-05).
        """
        if not fact_ids:
            return []
        placeholders = ",".join("?" for _ in fact_ids)
        rows = self.execute(
            f"SELECT fact_id, retention_score, lifecycle_zone "
            f"FROM fact_retention "
            f"WHERE fact_id IN ({placeholders}) AND profile_id = ?",
            (*fact_ids, profile_id),
        )
        return [dict(r) for r in rows]

    def upsert_retention(
        self,
        fact_id: str,
        profile_id: str,
        retention_score: float,
        memory_strength: float,
        access_count: int,
        last_accessed_at: str,
        lifecycle_zone: str,
    ) -> None:
        """UPSERT retention data for a fact.

        Retries 3x on SQLITE_BUSY (handled by execute()).
        All SQL parameterized (HR-05).
        """
        from superlocalmemory.core.lifecycle_state import atomic_lifecycle_for

        with self.transaction():
            self._upsert_retention_in_transaction(
                fact_id=fact_id,
                profile_id=profile_id,
                retention_score=retention_score,
                memory_strength=memory_strength,
                access_count=access_count,
                last_accessed_at=last_accessed_at,
                lifecycle_zone=lifecycle_zone,
            )
            self.execute(
                "UPDATE atomic_facts SET lifecycle = ? "
                "WHERE fact_id = ? AND profile_id = ?",
                (atomic_lifecycle_for(lifecycle_zone), fact_id, profile_id),
            )

    def _upsert_retention_in_transaction(
        self,
        *,
        fact_id: str,
        profile_id: str,
        retention_score: float,
        memory_strength: float,
        access_count: int,
        last_accessed_at: str,
        lifecycle_zone: str,
    ) -> None:
        """Write one retention row using the caller's active transaction."""
        self.execute(
            "INSERT INTO fact_retention "
            "(fact_id, profile_id, retention_score, memory_strength, "
            " access_count, last_accessed_at, lifecycle_zone, last_computed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(fact_id) DO UPDATE SET "
            "  retention_score = excluded.retention_score, "
            "  memory_strength = excluded.memory_strength, "
            "  access_count = excluded.access_count, "
            "  lifecycle_zone = excluded.lifecycle_zone, "
            "  last_computed_at = excluded.last_computed_at",
            (fact_id, profile_id, retention_score, memory_strength,
             access_count, last_accessed_at, lifecycle_zone),
        )

    def batch_upsert_retention(
        self, facts: list[dict], profile_id: str,
    ) -> int:
        """Batch UPSERT retention data. Wraps in transaction for atomicity.

        Each dict must contain: fact_id, retention, strength,
        access_count, last_accessed_at, zone.

        Returns count of successfully upserted rows.
        """
        from superlocalmemory.storage.retention_lifecycle import batch_retention

        return batch_retention(self, facts, profile_id)

    def get_facts_needing_decay(self, profile_id: str) -> list[dict]:
        """Get facts that need decay computation (excludes core memory).

        Core memory facts are immune to forgetting (HR-01).
        All SQL parameterized (HR-05).
        """
        rows = self.execute(
            "SELECT f.fact_id, f.created_at, f.profile_id "
            "FROM atomic_facts f "
            "LEFT JOIN fact_retention r ON f.fact_id = r.fact_id "
            "WHERE f.profile_id = ? "
            "AND f.fact_id NOT IN ("
            "  SELECT json_each.value "
            "  FROM core_memory_blocks, json_each(core_memory_blocks.source_fact_ids) "
            "  WHERE core_memory_blocks.profile_id = ?"
            ")",
            (profile_id, profile_id),
        )
        return [dict(r) for r in rows]

    def soft_delete_fact(self, fact_id: str, profile_id: str) -> None:
        """Soft-delete a forgotten fact.

        Sets fact_retention.lifecycle_zone to 'forgotten' and
        atomic_facts.lifecycle to 'archived' (valid enum value).
        Never physically deletes (HR-04).

        Idempotent: if fact not found, logs warning and returns.
        """
        from superlocalmemory.storage.retention_lifecycle import forget_fact

        forget_fact(self, fact_id, profile_id)

    # ------------------------------------------------------------------
    # Phase E: CCQ Consolidated Blocks & Audit CRUD
    # ------------------------------------------------------------------

    def store_ccq_block(
        self,
        block_id: str,
        profile_id: str,
        content: str,
        source_fact_ids: str,
        gist_embedding_rowid: int | None,
        char_count: int,
        cluster_id: str,
    ) -> None:
        """Store a CCQ consolidated block. Parameterized SQL only."""
        self.execute(
            "INSERT INTO ccq_consolidated_blocks "
            "(block_id, profile_id, content, source_fact_ids, "
            " gist_embedding_rowid, char_count, compiled_by, cluster_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'ccq', ?, datetime('now'))",
            (block_id, profile_id, content, source_fact_ids,
             gist_embedding_rowid, char_count, cluster_id),
        )

    def get_ccq_blocks(self, profile_id: str) -> list[dict]:
        """Get all CCQ consolidated blocks for a profile."""
        rows = self.execute(
            "SELECT * FROM ccq_consolidated_blocks "
            "WHERE profile_id = ? ORDER BY created_at DESC",
            (profile_id,),
        )
        return [dict(r) for r in rows]

    def store_ccq_audit(self, entry: dict) -> None:
        """Store a CCQ audit log entry. Parameterized SQL only."""
        self.execute(
            "INSERT INTO ccq_audit_log "
            "(audit_id, profile_id, cluster_id, block_id, fact_ids, fact_count, "
            " gist_text, extraction_mode, bytes_before, bytes_after, "
            " compression_ratio, shared_entities, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (entry["audit_id"], entry["profile_id"], entry["cluster_id"],
             entry["block_id"], entry["fact_ids"], entry["fact_count"],
             entry["gist_text"], entry["extraction_mode"],
             entry["bytes_before"], entry["bytes_after"],
             entry["compression_ratio"], entry["shared_entities"]),
        )

    def get_ccq_audit(self, profile_id: str, limit: int = 50) -> list[dict]:
        """Get CCQ audit log entries for a profile."""
        rows = self.execute(
            "SELECT * FROM ccq_audit_log "
            "WHERE profile_id = ? ORDER BY created_at DESC LIMIT ?",
            (profile_id, limit),
        )
        return [dict(r) for r in rows]
