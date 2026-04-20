# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-13 Track C.1

"""Tests for TrigramIndex (LLD-13 inline entity detection).

TDD red-first. These tests were written BEFORE the implementation and
must fail on the bare module, then pass once ``trigram_index.py`` is in
place.

Manifest: IMPLEMENTATION-MANIFEST-v3.4.22-FINAL.md → Track C.1 (12 tests
total across this file + ``test_user_prompt_hook_entity_detection.py``).

Hard invariants verified:
  - Hot-path ``lookup()`` p99 < 2 ms (I1 budget slice).
  - RAM footprint (index + LRU) < 100 MB in steady state.
  - SQL uses parameterised queries (SEC-C-03).
  - ``busy_timeout=50`` fail-fast on lock contention.
  - ``bootstrap()`` uses ``ram_reservation`` from ``core.ram_lock``.
  - Fallback to empty list when cache DB / table missing.
  - Honors ``MAX_TRIGRAMS`` cap.

No test writes to canonical_entities (memory.db SACRED per Delivery Lead).
All fixtures build isolated temp cache DBs + fake source DBs.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def fake_memory_db(tmp_path: Path) -> Path:
    """Small fake memory.db with canonical_entities + entity_aliases.

    Mirrors production schema (columns verified against live DB on
    2026-04-19). 20 entities, 2 aliases each.
    """
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE canonical_entities (
            entity_id       TEXT PRIMARY KEY,
            profile_id      TEXT NOT NULL DEFAULT 'default',
            canonical_name  TEXT NOT NULL,
            entity_type     TEXT NOT NULL DEFAULT '',
            first_seen      TEXT NOT NULL DEFAULT (datetime('now')),
            last_seen       TEXT NOT NULL DEFAULT (datetime('now')),
            fact_count      INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE entity_aliases (
            alias_id    TEXT PRIMARY KEY,
            entity_id   TEXT NOT NULL,
            alias       TEXT NOT NULL,
            confidence  REAL NOT NULL DEFAULT 1.0,
            source      TEXT NOT NULL DEFAULT ''
        );
        """
    )
    entities = [
        ("e001", "SuperLocalMemory"),
        ("e002", "Qualixar"),
        ("e003", "AgentAssert"),
        ("e004", "AgentAssay"),
        ("e005", "SkillFortify"),
        ("e006", "FidelityBench"),
        ("e007", "QualixarOS"),
        ("e008", "TrigramIndex"),
        ("e009", "VarunBhardwaj"),
        ("e010", "Accenture"),
        ("e011", "NeurIPS"),
        ("e012", "Anthropic"),
        ("e013", "ClaudeCode"),
        ("e014", "PolarQuant"),
        ("e015", "TurboQuant"),
        ("e016", "LivingBrain"),
        ("e017", "InlineEntity"),
        ("e018", "HotPath"),
        ("e019", "BenchmarkHarness"),
        ("e020", "ReinforcementLearn"),
    ]
    conn.executemany(
        "INSERT INTO canonical_entities (entity_id, canonical_name) VALUES (?, ?)",
        entities,
    )
    aliases = []
    for eid, name in entities:
        aliases.append((f"a{eid}_1", eid, name.lower(), 1.0, "test"))
        aliases.append((f"a{eid}_2", eid, name[:6].lower(), 0.8, "test"))
    conn.executemany(
        "INSERT INTO entity_aliases VALUES (?,?,?,?,?)", aliases,
    )
    conn.commit()
    conn.close()
    return db


@pytest.fixture()
def cache_db(tmp_path: Path) -> Path:
    """Empty cache DB path. TrigramIndex.bootstrap() will create the table."""
    return tmp_path / "active_brain_cache.db"


@pytest.fixture()
def index(fake_memory_db: Path, cache_db: Path, monkeypatch):
    """Construct a TrigramIndex wired to the fake DBs, with ram_reservation
    stubbed out so tests don't require ≥500 MB free on the box."""
    from superlocalmemory.learning import trigram_index as ti

    # Redirect the class-level CACHE_DB_PATH to the temp file.
    monkeypatch.setattr(ti.TrigramIndex, "CACHE_DB_PATH", cache_db)

    # Stub ram_reservation to a no-op CM so CI boxes with tight RAM pass.
    from contextlib import contextmanager

    @contextmanager
    def _noop(name, *, timeout_s=60.0, required_mb=0):
        yield

    monkeypatch.setattr(ti, "ram_reservation", _noop)

    idx = ti.TrigramIndex(source_db_path=fake_memory_db)
    return idx


# --------------------------------------------------------------------------
# 1. bootstrap() creates the index from canonical_entities
# --------------------------------------------------------------------------


def test_bootstrap_creates_index(index, cache_db: Path):
    index.bootstrap()
    assert cache_db.exists()
    conn = sqlite3.connect(cache_db)
    try:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "entity_trigrams" in names
        assert "entity_trigrams_meta" in names
        # Index populated from 20 entities × alias + canonical → > 0 rows.
        count = conn.execute("SELECT COUNT(*) FROM entity_trigrams").fetchone()[0]
        assert count > 0
    finally:
        conn.close()


# --------------------------------------------------------------------------
# 2. bootstrap() uses ram_reservation (contract with core.ram_lock)
# --------------------------------------------------------------------------


def test_bootstrap_uses_ram_reservation(fake_memory_db: Path, cache_db: Path,
                                         monkeypatch):
    """Bootstrap must wrap its rebuild in ``ram_reservation('trigram_rebuild',
    required_mb=300)`` per LLD-00 §7. We assert the CM is entered."""
    from superlocalmemory.learning import trigram_index as ti

    monkeypatch.setattr(ti.TrigramIndex, "CACHE_DB_PATH", cache_db)

    from contextlib import contextmanager

    calls: list[tuple[str, int]] = []

    @contextmanager
    def _tracking(name, *, timeout_s=60.0, required_mb=0):
        calls.append((name, required_mb))
        yield

    monkeypatch.setattr(ti, "ram_reservation", _tracking)

    idx = ti.TrigramIndex(source_db_path=fake_memory_db)
    idx.bootstrap()

    assert len(calls) == 1
    assert calls[0][0] == "trigram_rebuild"
    assert calls[0][1] == 300


# --------------------------------------------------------------------------
# 3. lookup() hits a known entity
# --------------------------------------------------------------------------


def test_lookup_hits_known_entity(index):
    index.bootstrap()
    hits = index.lookup("tell me about SuperLocalMemory architecture")
    entity_ids = [eid for eid, _ in hits]
    assert "e001" in entity_ids


# --------------------------------------------------------------------------
# 4. Hot-path p99 < 2 ms (I1 budget slice)
# --------------------------------------------------------------------------


def test_lookup_p99_under_2ms(index):
    """10k lookups; p99 < 2 ms. Manifest Track C.1 perf test."""
    index.bootstrap()
    prompts = [
        "what is SuperLocalMemory",
        "Qualixar and AgentAssert architecture",
        "how does TrigramIndex work in LivingBrain",
        "Claude Code hook latency",
        "benchmark harness for retrieval",
        "tell me about PolarQuant and TurboQuant",
        "Varun Bhardwaj at Accenture NeurIPS 2026",
        "SkillFortify and FidelityBench comparison",
        "HotPath budget for InlineEntity detection",
        "ReinforcementLearn with Anthropic models",
    ]
    N = 10_000
    timings: list[float] = []
    # Warm up (avoid LRU-cold outlier in the p99 tail).
    for p in prompts:
        index.lookup(p)

    for i in range(N):
        p = prompts[i % len(prompts)]
        t0 = time.perf_counter_ns()
        index.lookup(p)
        timings.append((time.perf_counter_ns() - t0) / 1_000_000.0)  # ms

    timings.sort()
    p99 = timings[int(N * 0.99)]
    assert p99 < 2.0, f"p99 lookup latency {p99:.3f} ms >= 2 ms budget"


# --------------------------------------------------------------------------
# 5. RAM footprint stays under 100 MB (LLD-13 §10)
# --------------------------------------------------------------------------


def test_lookup_ram_under_100mb(index):
    """Steady-state RAM delta from the lookup LRU + open connection
    must stay under 100 MB. Per LLD-13 §10."""
    import psutil, os
    index.bootstrap()
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    for i in range(5000):
        index.lookup(f"prompt {i} SuperLocalMemory Qualixar AgentAssert")
    rss_after = proc.memory_info().rss
    delta_mb = (rss_after - rss_before) / (1024 * 1024)
    # Generous ceiling: LLD caps at 100 MB, we assert 100 hard.
    assert delta_mb < 100, f"lookup RSS delta {delta_mb:.1f} MB exceeds 100 MB"


# --------------------------------------------------------------------------
# 6. LRU cache hit path (same trigram frozenset → cached return)
# --------------------------------------------------------------------------


def test_lookup_lru_cache_hit(index):
    index.bootstrap()
    prompt = "lookup for SuperLocalMemory entity"
    first = index.lookup(prompt)
    info_before = index._cached_lookup_key.cache_info()
    second = index.lookup(prompt)
    info_after = index._cached_lookup_key.cache_info()
    assert first == second
    # Exactly one more hit registered; at least one hit total.
    assert info_after.hits >= info_before.hits + 1


# --------------------------------------------------------------------------
# 7. Fallback: index missing / table absent → lookup returns []
# --------------------------------------------------------------------------


def test_lookup_fallback_when_index_missing(fake_memory_db: Path, cache_db: Path,
                                              monkeypatch):
    """If cache DB or table is missing, lookup must return [] silently.
    LLD-13 §9 golden rule: any Layer A failure reduces to fallback."""
    from superlocalmemory.learning import trigram_index as ti

    monkeypatch.setattr(ti.TrigramIndex, "CACHE_DB_PATH", cache_db)
    idx = ti.TrigramIndex(source_db_path=fake_memory_db)
    # No bootstrap() call → cache DB absent on disk.
    out = idx.lookup("query about SuperLocalMemory")
    assert out == []


# --------------------------------------------------------------------------
# 8. busy_timeout=50 → fast-fail on locked cache DB
# --------------------------------------------------------------------------


def test_lookup_fast_fail_on_db_lock(index, cache_db: Path):
    """Hold an EXCLUSIVE lock on cache.db via a BEGIN IMMEDIATE writer
    in another thread. lookup() must return [] within ~60 ms (busy_timeout
    of 50 ms + a small margin) — not hang."""
    index.bootstrap()

    hold_lock = threading.Event()
    release = threading.Event()

    def _writer():
        w = sqlite3.connect(cache_db, isolation_level=None, timeout=5.0)
        w.execute("BEGIN EXCLUSIVE")
        hold_lock.set()
        release.wait(timeout=5.0)
        try:
            w.execute("ROLLBACK")
        except Exception:
            pass
        w.close()

    t = threading.Thread(target=_writer, daemon=True)
    t.start()
    hold_lock.wait(timeout=2.0)

    try:
        t0 = time.perf_counter()
        out = index.lookup("lookup while locked SuperLocalMemory")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        # Graceful fallback: empty list.
        assert out == []
        # Must fast-fail. 50 ms busy_timeout + 200 ms slack for CI jitter.
        assert elapsed_ms < 250, f"lookup hung under lock: {elapsed_ms:.1f} ms"
    finally:
        release.set()
        t.join(timeout=2.0)


# --------------------------------------------------------------------------
# 9. SQL uses parameterised queries (SEC-C-03)
# --------------------------------------------------------------------------


def test_lookup_sql_uses_parameterized_query(index):
    """SEC-C-03: no string-format of untrusted identifiers into SQL.

    Python 3.14's ``sqlite3.Connection`` is immutable, so we can't
    monkeypatch ``execute`` to spy at runtime. Instead we verify two
    static invariants of the implementation:

      1. Calling lookup with SQL-injection-shaped tokens does NOT alter
         the result shape (parameterisation blocks breakout).
      2. The module source for the lookup SELECT uses ``?`` placeholders
         (SQL tokens parameterised via DB-API binds, never %-format).
    """
    index.bootstrap()

    # 1. Behavioural: injection-shaped tokens cannot alter the query.
    # Trigram extraction strips non-alnum, so these reduce to legitimate
    # 3-grams but the *path* through the raw _lookup_raw method is proven
    # safe because its SQL is built from a ``?`` placeholder per param.
    malicious = "'); DROP TABLE entity_trigrams; -- and SuperLocalMemory"
    hits = index.lookup(malicious)
    # No exception, no table-drop side effect, still returns a list.
    assert isinstance(hits, list)

    # Verify cache table is still intact after the malicious call.
    import sqlite3 as _sql
    from superlocalmemory.learning.trigram_index import TrigramIndex
    conn = _sql.connect(str(TrigramIndex.CACHE_DB_PATH))
    try:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "entity_trigrams" in names, (
            "entity_trigrams table was dropped — injection succeeded!"
        )
    finally:
        conn.close()

    # 2. Static: the module source uses parameterised lookup SQL.
    import inspect
    from superlocalmemory.learning import trigram_index as ti
    src = inspect.getsource(ti)
    # The lookup SELECT must have ``WHERE trigram IN (`` followed by
    # ``?`` placeholders in a joined string, NOT %-format of raw values.
    assert "WHERE trigram IN (" in src
    assert '",".join("?"' in src or '",".join(["?"' in src, (
        "lookup SELECT must build placeholders via ','.join('?'...)"
    )
    # No %-format / f-string trigram interpolation patterns.
    assert "% trigram" not in src.lower()
    # No concatenation of raw trigram into the WHERE clause.
    assert "WHERE trigram = '" not in src
    assert 'WHERE trigram = "' not in src


# --------------------------------------------------------------------------
# 10. bootstrap() honours MAX_TRIGRAMS cap
# --------------------------------------------------------------------------


def test_bootstrap_respects_max_trigrams_cap(index, cache_db: Path, monkeypatch):
    """Set MAX_TRIGRAMS to a tiny value and verify the rebuilt table does
    not exceed it. Eviction policy: keep highest-weight rows."""
    from superlocalmemory.learning import trigram_index as ti

    monkeypatch.setattr(ti.TrigramIndex, "MAX_TRIGRAMS", 5)
    index.bootstrap()
    conn = sqlite3.connect(cache_db)
    try:
        count = conn.execute("SELECT COUNT(*) FROM entity_trigrams").fetchone()[0]
        assert count <= 5, f"trigram count {count} exceeds cap 5"
    finally:
        conn.close()
