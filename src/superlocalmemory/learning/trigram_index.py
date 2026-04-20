# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21 — LLD-13 Track C.1

"""Inline trigram entity detection — hot-path Layer A of the two-layer
entity detector defined in ``LLD-13-inline-entity-detection.md``.

Design contract (do NOT improvise):

  - Hot-path ``lookup(text)`` targets **p99 < 2 ms**. Implemented as a
    single parameterised SQLite ``SELECT`` over a pre-built ``entity_trigrams``
    table in ``active_brain_cache.db`` plus a per-session ``@lru_cache``
    (≤200 entries, ≤100 KB total).
  - ``bootstrap()`` builds (or rebuilds) the cache table from
    ``canonical_entities`` + ``entity_aliases`` in ``memory.db``. It
    runs under ``core.ram_lock.ram_reservation('trigram_rebuild',
    required_mb=300)`` per LLD-00 §7.
  - ``memory.db`` is **SACRED** — this module only READS from
    ``canonical_entities`` / ``entity_aliases``. Never writes.
  - ``cache.db`` is **NOT a migration target** (LLD-00 §6). The index
    table is (re)created via ``CREATE TABLE IF NOT EXISTS`` inside
    ``bootstrap()``. ``slm cache clear`` and first-run both hit this
    lazy path.
  - Every SQL call uses parameterised queries (SEC-C-03). The IN-clause
    placeholder count is bounded (≤256 trigrams).
  - SQLite connections open with ``busy_timeout=50`` so a locked DB
    fails fast rather than eating the hook budget.

Stdlib-only imports at module load. The singleton helper
``get_or_none()`` returns a shared ``TrigramIndex`` instance or ``None``
if the cache DB is absent; the hook uses this to fall back silently.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Optional

# Import the RAM semaphore at module scope so tests can monkeypatch
# ``trigram_index.ram_reservation`` to a no-op on CI boxes with tight RAM.
from superlocalmemory.core.ram_lock import ram_reservation


# --------------------------------------------------------------------------
# Module constants
# --------------------------------------------------------------------------

_ACTIVE_PROFILE: str = "default"
_BUSY_TIMEOUT_MS: int = 50
_MAX_IN_CLAUSE: int = 256
_MAX_INPUT_CHARS: int = 500


# H-12/H-P-06: module-level cached connection for the inline lookup
# path. The first cache miss on a fresh session previously paid the
# ``sqlite3.connect`` cost (~1–3 ms warm, blowing the <2 ms p99
# budget). With a shared conn (guarded by ``_CACHE_CONN_LOCK``), every
# lookup pays only the query cost. ``_reset_cache_conn()`` exists so
# tests + ``bootstrap()`` can drop a stale conn after the cache DB is
# rebuilt.
_CACHE_CONN: Optional[sqlite3.Connection] = None
_CACHE_CONN_LOCK = threading.Lock()
_CACHE_CONN_OWNER_PID: int | None = None


def _reset_cache_conn_for_child() -> None:
    """S9-W2 C5 fork safety: wipe the inherited handle in the child.

    Running ``close()`` on a handle the parent still uses would be a
    race; we simply orphan the reference and let the parent keep its
    open fd. The child opens a fresh conn on first ``_get_cache_conn``.
    """
    global _CACHE_CONN, _CACHE_CONN_OWNER_PID
    _CACHE_CONN = None
    _CACHE_CONN_OWNER_PID = os.getpid()


def _get_cache_conn() -> Optional[sqlite3.Connection]:
    """Return a process-cached connection to the trigram cache DB.

    Returns ``None`` if the cache DB is missing or the connect fails.
    Caller holds no lock — every ``execute`` is serialised via
    ``_CACHE_CONN_LOCK``.
    """
    global _CACHE_CONN, _CACHE_CONN_OWNER_PID
    current_pid = os.getpid()
    # S9-W2 C5: pid drift belt-and-suspenders. If a fork path somehow
    # skipped ``register_at_fork``, we still refuse to hand out an
    # inherited handle.
    if _CACHE_CONN is not None and (
        _CACHE_CONN_OWNER_PID is not None
        and _CACHE_CONN_OWNER_PID != current_pid
    ):
        _CACHE_CONN = None
    if _CACHE_CONN is not None:
        return _CACHE_CONN
    with _CACHE_CONN_LOCK:
        if _CACHE_CONN is not None:
            return _CACHE_CONN
        if not TrigramIndex.CACHE_DB_PATH.exists():
            return None
        try:
            conn = sqlite3.connect(
                str(TrigramIndex.CACHE_DB_PATH),
                timeout=0.05,
                isolation_level=None,
                check_same_thread=False,
            )
            conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        except sqlite3.OperationalError:
            return None
        _CACHE_CONN = conn
        _CACHE_CONN_OWNER_PID = current_pid
        return _CACHE_CONN


def _reset_cache_conn() -> None:
    """Drop the cached connection. Called after ``bootstrap()`` swaps
    the cache table so subsequent lookups re-connect to the fresh DB.
    """
    global _CACHE_CONN, _CACHE_CONN_OWNER_PID
    with _CACHE_CONN_LOCK:
        if _CACHE_CONN is not None:
            try:
                _CACHE_CONN.close()
            except sqlite3.Error:  # pragma: no cover — defensive
                pass
            _CACHE_CONN = None
            _CACHE_CONN_OWNER_PID = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_cache_conn_for_child)


# --------------------------------------------------------------------------
# Trigram extraction (stdlib-only, deterministic, NFKD + ASCII-fold)
# --------------------------------------------------------------------------


#: L-P-02: cheap proxy for trigram "commonness". A trigram composed of
#: three ASCII lowercase letters is the most frequent class; any trigram
#: with a digit or with an uncommon starting letter is rarer and thus
#: more discriminative for entity lookup. Lower key == earlier sort ==
#: preferred to keep.
_COMMON_STARTS: frozenset[str] = frozenset("etaoinshrdlucmfgpwby")


def _trigram_rarity_key(t: str) -> int:
    """Return a small-int rarity rank; LOW == rare/kept, HIGH == common."""
    if not t:
        return 3
    has_digit = any(c.isdigit() for c in t)
    starts_common = t[0] in _COMMON_STARTS
    # 0: has a digit (very discriminative, e.g. "sl3", "1st").
    # 1: starts with an uncommon letter.
    # 2: all-letter common trigram (default big bucket).
    if has_digit:
        return 0
    if not starts_common:
        return 1
    return 2


def _trigrams_for(text: str) -> set[str]:
    """Extract 3-gram set from ``text``.

    Pipeline: clamp-to-500-chars -> NFKD normalize -> ASCII-fold ->
    lowercase -> split on non-alphanumeric -> skip tokens < 3 chars ->
    emit overlapping 3-grams per token.

    Matches LLD-13 §4.1 exactly. stdlib-only.
    """
    if not text:
        return set()
    s = unicodedata.normalize("NFKD", text[:_MAX_INPUT_CHARS])
    s = s.encode("ascii", "ignore").decode("ascii").lower()
    s = "".join(c if c.isalnum() else " " for c in s)
    out: set[str] = set()
    for token in s.split():
        if len(token) < 3:
            continue
        for i in range(len(token) - 2):
            out.add(token[i : i + 3])
    return out


# --------------------------------------------------------------------------
# TrigramIndex
# --------------------------------------------------------------------------


class TrigramIndex:
    """Two-layer entity detection — Layer A (hot inline).

    Bootstrap reads ``canonical_entities`` + ``entity_aliases`` from
    the SLM source-of-truth DB and writes a compact
    ``(trigram, entity_id, weight)`` table into the cache DB. The hot
    path does one grouped SELECT per prompt and returns up to 10 ranked
    ``(entity_id, hits)`` candidates.
    """

    CACHE_DB_PATH: Path = Path.home() / ".superlocalmemory" / "active_brain_cache.db"
    MAX_TRIGRAMS: int = 1_000_000
    LOOKUP_LIMIT: int = 10
    LOOKUP_MIN_HITS: int = 2

    # ----------------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------------

    def __init__(self, source_db_path: Path) -> None:
        if not isinstance(source_db_path, Path):
            raise ValueError("source_db_path must be a pathlib.Path")
        self._source_db_path = source_db_path
        # Per-instance LRU wrapper (200 entries, ≤100 KB envelope).
        self._cached_lookup_key = lru_cache(maxsize=200)(self._lookup_raw)

    # ----------------------------------------------------------------------
    # bootstrap() — daemon-side rebuild
    # ----------------------------------------------------------------------

    #: L-P-04: reservation default mirrors LLD-00 §7 (300 MB sized for
    #: ~10k entities × ~5 aliases × ~15 trigrams). On small installs this
    #: over-reserves on tight-RAM laptops; on 500k-entity power users it
    #: under-protects. The env override lets operators right-size per host
    #: without a code change — the fallback stays safe.
    BOOTSTRAP_RAM_MB_DEFAULT: int = 300
    BOOTSTRAP_RAM_MB_ENV: str = "SLM_TRIGRAM_BOOTSTRAP_RAM_MB"

    @classmethod
    def _bootstrap_ram_mb(cls) -> int:
        raw = os.environ.get(cls.BOOTSTRAP_RAM_MB_ENV, "").strip()
        if not raw:
            return cls.BOOTSTRAP_RAM_MB_DEFAULT
        try:
            val = int(raw)
        except ValueError:
            return cls.BOOTSTRAP_RAM_MB_DEFAULT
        if val < 16:
            # Refuse to under-reserve — the minimum keeps semaphore math
            # meaningful even on tiny CI boxes.
            return 16
        return val

    def bootstrap(self) -> None:
        """Read canonical_entities + entity_aliases, recompute trigram
        buckets, atomically swap the cache table.

        Wraps the heavy phase in ``ram_reservation('trigram_rebuild',
        required_mb=<default 300, overridable via
        ``SLM_TRIGRAM_BOOTSTRAP_RAM_MB``>)``. Source DB is opened
        read-only; memory.db is never mutated.
        """
        with ram_reservation(
            "trigram_rebuild",
            required_mb=self._bootstrap_ram_mb(),
        ):
            self._rebuild_index()

    # SEC-M5 — safety cap on rebuild input row count. An adversarial or
    # bloated memory.db with millions of canonical_entities could exceed
    # the 300 MB ``ram_reservation`` block after fast-fail passed, since
    # ``fetchall()`` materialises the entire JOIN into Python memory.
    # ``MAX_TRIGRAMS=1_000_000`` downstream already bounds the final
    # index; capping the source fetch at 5M rows keeps peak RAM within
    # I2 even on pathological inputs.
    _MAX_REBUILD_ROWS: int = 5_000_000

    def _rebuild_index(self) -> None:
        buckets: dict[str, dict[str, float]] = {}
        src = sqlite3.connect(
            f"file:{self._source_db_path}?mode=ro",
            uri=True,
            timeout=1.0,
        )
        try:
            # SEC-M5 — bounded LIMIT + explicit busy_timeout on the
            # source connection so a locked memory.db fails fast rather
            # than blocking the entire timeout.
            src.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            # S9-W3 H-PERF-05: previously ``fetchall()`` materialised the
            # entire 5M-row JOIN as one list in Python, regardless of the
            # ``ram_reservation`` block. On a pathological input this is
            # ~1.5 GB peak RAM — the "SLM_TRIGRAM_BOOTSTRAP_RAM_MB"
            # override looked tuneable but was ornamental. We now iterate
            # the cursor row-by-row (SQLite streams from the prepared
            # statement), so peak Python RAM scales with the bucket
            # dict (bounded by ``MAX_TRIGRAMS``) not the row count.
            cursor = src.execute(
                "SELECT ce.entity_id, ce.canonical_name, "
                "       COALESCE(ea.alias, '') AS alias "
                "FROM canonical_entities ce "
                "LEFT JOIN entity_aliases ea USING (entity_id) "
                "WHERE ce.profile_id = ? "
                "LIMIT ?",
                (_ACTIVE_PROFILE, self._MAX_REBUILD_ROWS),
            )
            rows = cursor  # streamed iteration
            row_iter = iter(rows)
            # Fall through to the bucket loop — ``cursor`` is consumed
            # lazily so we can still close(src) in ``finally``.
        except sqlite3.Error:
            src.close()
            raise
        # Consume the cursor lazily; ``src`` stays open through the
        # buckets loop because sqlite3 cursors hold a reference to it.
        try:
            for entity_id, canonical_name, alias in row_iter:
                for name in (canonical_name, alias):
                    if not name:
                        continue
                    for tri in _trigrams_for(str(name)):
                        buckets.setdefault(tri, {}).setdefault(entity_id, 0.0)
                        buckets[tri][entity_id] += 1.0
        finally:
            src.close()

        # S9-defer H-P-05: stream the flat-list construction through a
        # bounded min-heap of size ``MAX_TRIGRAMS`` instead of
        # materialising the full list and sort-truncating. For a
        # bucket count far above the cap this saves O(N_extra) Python
        # memory AND trades an O(N log N) full-sort for an O(N log K)
        # heap-push pass.
        import heapq
        _cap = int(self.MAX_TRIGRAMS)
        _heap: list[tuple[float, str, str]] = []
        for tri, d in buckets.items():
            for eid, w in d.items():
                # heapq is a min-heap so pushing (w, ...) keeps the
                # LOWEST-weight row at the root; we evict it whenever a
                # higher-weight row arrives. Net effect: the heap holds
                # the top-``_cap`` rows by weight at any given time.
                if len(_heap) < _cap:
                    heapq.heappush(_heap, (float(w), tri, eid))
                else:
                    heapq.heappushpop(_heap, (float(w), tri, eid))
        flat: list[tuple[str, str, float]] = [
            (tri, eid, w) for (w, tri, eid) in _heap
        ]
        # ``buckets`` is no longer needed; release its memory before
        # opening the writer connection.
        buckets = {}

        # Write to cache DB via atomic shadow-table swap.
        self.CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        # S9-W2 C5: close the shared reader connection BEFORE opening the
        # writer. Previously `_reset_cache_conn()` ran AFTER the swap,
        # which meant concurrent `lookup()` calls (same process, other
        # threads) held the old conn through the DROP TABLE window and
        # saw partial/empty rowsets or SQLITE_BUSY retries. Closing up
        # front forces every subsequent lookup to wait for the writer's
        # ALTER TABLE (serialised by SQLite's own locking) and then open
        # a fresh conn against the post-swap schema.
        _reset_cache_conn()

        conn = sqlite3.connect(str(self.CACHE_DB_PATH), timeout=2.0)
        try:
            conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS entity_trigrams (
                    trigram   TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    weight    REAL NOT NULL DEFAULT 1.0,
                    PRIMARY KEY (trigram, entity_id)
                ) WITHOUT ROWID;
                CREATE INDEX IF NOT EXISTS idx_trigram_lookup
                    ON entity_trigrams (trigram);
                CREATE TABLE IF NOT EXISTS entity_trigrams_meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                DROP TABLE IF EXISTS entity_trigrams_shadow;
                CREATE TABLE entity_trigrams_shadow (
                    trigram   TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    weight    REAL NOT NULL DEFAULT 1.0,
                    PRIMARY KEY (trigram, entity_id)
                ) WITHOUT ROWID;
                """
            )
            with conn:
                conn.executemany(
                    "INSERT INTO entity_trigrams_shadow (trigram, entity_id, weight) "
                    "VALUES (?, ?, ?)",
                    flat,
                )
                conn.execute("DROP TABLE entity_trigrams")
                conn.execute(
                    "ALTER TABLE entity_trigrams_shadow "
                    "RENAME TO entity_trigrams"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trigram_lookup "
                    "ON entity_trigrams (trigram)"
                )
                conn.execute(
                    "INSERT OR REPLACE INTO entity_trigrams_meta (key, value) "
                    "VALUES (?, ?)",
                    ("entity_count", str(len(flat))),
                )
        finally:
            conn.close()

        # Bust the per-instance LRU — stale entries would point at now-
        # dropped rows. The module-level cached conn was already dropped
        # BEFORE the writer ran (see C5 fix above); nothing to do here.
        self._cached_lookup_key.cache_clear()

    # ----------------------------------------------------------------------
    # lookup() — hot path
    # ----------------------------------------------------------------------

    def lookup(self, text: str) -> list[tuple[str, int]]:
        """Return up to ``LOOKUP_LIMIT`` ``(entity_id, hits)`` matches,
        ordered by hit count DESC, weight DESC.

        Returns ``[]`` on any failure (missing table, locked DB, empty
        trigram set). Target p99 < 2 ms.
        """
        if not text:
            return []
        trigrams = _trigrams_for(text)
        if not trigrams:
            return []
        if len(trigrams) > _MAX_IN_CLAUSE:
            # L-P-02: alphabetical ``sorted(trigrams)[:256]`` threw away
            # the discriminative tail of the signature. Switch to a
            # rarity-weighted selection that prefers trigrams with at
            # least one digit or non-common prefix — those are IDF-rich
            # relative to plain ASCII letter trigrams. The selection is
            # still deterministic (stable secondary sort on the trigram
            # itself) so the LRU key remains repeatable across identical
            # prompts.
            ranked = sorted(trigrams, key=lambda t: (_trigram_rarity_key(t), t))
            trigrams = set(ranked[:_MAX_IN_CLAUSE])
        key = frozenset(trigrams)
        try:
            return list(self._cached_lookup_key(key))
        except Exception:
            # Any failure: self-heal cache + fall back to a direct query.
            try:
                self._cached_lookup_key.cache_clear()
            except Exception:
                pass
            try:
                return list(self._lookup_raw(key))
            except Exception:
                return []

    def _lookup_raw(self, trigrams: frozenset[str]) -> tuple[tuple[str, int], ...]:
        """SQLite-backed lookup. Returns a tuple (hashable for LRU)."""
        if not trigrams:
            return ()
        if not self.CACHE_DB_PATH.exists():
            return ()

        params = tuple(trigrams)
        placeholders = ",".join("?" * len(params))
        sql = (
            "SELECT entity_id, COUNT(*) AS hits, SUM(weight) AS score "
            "FROM entity_trigrams "
            f"WHERE trigram IN ({placeholders}) "
            "GROUP BY entity_id "
            "HAVING hits >= ? "
            "ORDER BY hits DESC, score DESC "
            "LIMIT ?"
        )
        bound = params + (self.LOOKUP_MIN_HITS, self.LOOKUP_LIMIT)

        # H-12/H-P-06: use the module-cached connection; fall back to a
        # fresh connect only when the cache is empty (first-lookup-in-
        # process or post-rebuild). ``_CACHE_CONN_LOCK`` serialises
        # access because ``check_same_thread=False`` lets worker threads
        # share the conn with the hot path.
        conn = _get_cache_conn()
        if conn is not None:
            try:
                with _CACHE_CONN_LOCK:
                    rows = conn.execute(sql, bound).fetchall()
            except sqlite3.OperationalError as exc:
                # S9-W2 H-PERF-04: only evict the cached conn when the
                # error signals a SCHEMA change (table dropped/rebuilt).
                # A transient SQLITE_BUSY does NOT require re-connecting
                # — that triggered an eviction storm on concurrent slm
                # doctor runs and blew the <2 ms p99 budget on 10-30% of
                # lookups. We let busy errors fall through to the
                # one-shot fresh-connect fallback without touching the
                # shared cache.
                msg = str(exc).lower()
                if "schema" in msg or "no such table" in msg:
                    _reset_cache_conn()
                conn = None
        if conn is None:
            try:
                fresh = sqlite3.connect(
                    str(self.CACHE_DB_PATH),
                    timeout=0.05,  # 50 ms connection timeout
                    isolation_level=None,
                )
            except sqlite3.OperationalError:
                return ()
            try:
                fresh.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
                try:
                    rows = fresh.execute(sql, bound).fetchall()
                except sqlite3.OperationalError:
                    return ()
            finally:
                fresh.close()
        return tuple((eid, int(hits)) for (eid, hits, _score) in rows)


# --------------------------------------------------------------------------
# Singleton accessor used by the hook
# --------------------------------------------------------------------------


_SINGLETON: Optional[TrigramIndex] = None


def get_or_none() -> Optional[TrigramIndex]:
    """Return a process-local ``TrigramIndex`` if the cache DB exists,
    else ``None`` so the hook can fall back to the regex-only signature.

    Test fixtures monkeypatch this module-level function directly.
    """
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON
    if not TrigramIndex.CACHE_DB_PATH.exists():
        return None
    default_source = Path.home() / ".superlocalmemory" / "memory.db"
    if not default_source.exists():
        return None
    _SINGLETON = TrigramIndex(source_db_path=default_source)
    return _SINGLETON


__all__ = ("TrigramIndex", "get_or_none")
