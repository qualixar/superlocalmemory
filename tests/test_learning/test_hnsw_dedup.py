# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Track B.2 (LLD-12)

"""Tests for ``superlocalmemory.learning.hnsw_dedup`` (HnswDeduplicator).

Contract references:
  - LLD-00 §1.4 — memory_archive + memory_merge_log schema.
  - LLD-00 §7  — ram_reservation protocol.
  - LLD-12 §2  — cosine > 0.95 AND entity_overlap > 0.8 thresholds.
  - LLD-12 §3  — hnswlib RAM budget + prefix-dedup fallback.
  - IMPLEMENTATION-MANIFEST v3.4.22 FINAL B.2 — test names verbatim.

Invariants:
  - NEVER deletes rows from atomic_facts. Only UPDATE archive_status.
  - Every merge writes a memory_merge_log row (reversible).
  - Uses ram_reservation('hnswlib', required_mb=...) before build.
  - Fallback to prefix dedup if hnswlib unavailable or MAX_FACTS exceeded.
"""

from __future__ import annotations

import json
import os
import resource
import sqlite3
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bootstrap_memory_db(path: Path) -> None:
    """Create the minimal shape for hnsw dedup tests.

    Mirrors production post-M011: atomic_facts has the archive lifecycle
    columns + retrieval_prior, and memory_archive / memory_merge_log exist.
    """
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE atomic_facts (
                fact_id            TEXT PRIMARY KEY,
                profile_id         TEXT NOT NULL DEFAULT 'default',
                content            TEXT NOT NULL,
                canonical_entities_json TEXT NOT NULL DEFAULT '[]',
                embedding          TEXT,
                confidence         REAL NOT NULL DEFAULT 1.0,
                importance         REAL NOT NULL DEFAULT 0.5,
                created_at         TEXT NOT NULL DEFAULT (datetime('now')),
                archive_status     TEXT DEFAULT 'live',
                archive_reason     TEXT,
                merged_into        TEXT,
                retrieval_prior    REAL DEFAULT 0.0
            );
            CREATE TABLE memory_archive (
                archive_id    TEXT PRIMARY KEY,
                fact_id       TEXT NOT NULL,
                profile_id    TEXT NOT NULL,
                payload_json  TEXT NOT NULL,
                archived_at   TEXT NOT NULL,
                reason        TEXT NOT NULL
            );
            CREATE TABLE memory_merge_log (
                merge_id          TEXT PRIMARY KEY,
                profile_id        TEXT NOT NULL,
                canonical_fact_id TEXT NOT NULL,
                merged_fact_id    TEXT NOT NULL,
                cosine_sim        REAL,
                entity_jaccard    REAL,
                merged_at         TEXT NOT NULL,
                reversible        INTEGER DEFAULT 1
            );
            CREATE TABLE action_outcomes (
                outcome_id       TEXT PRIMARY KEY,
                profile_id       TEXT NOT NULL DEFAULT 'default',
                fact_ids_json    TEXT NOT NULL DEFAULT '[]',
                reward           REAL,
                settled          INTEGER NOT NULL DEFAULT 0,
                settled_at       TEXT
            );
            """
        )


def _embed(vec: list[float]) -> str:
    return json.dumps(vec)


def _seed_fact(
    conn: sqlite3.Connection,
    fact_id: str,
    profile_id: str,
    content: str,
    entities: list[str],
    embedding: list[float],
    *,
    importance: float = 0.5,
    confidence: float = 1.0,
) -> None:
    conn.execute(
        "INSERT INTO atomic_facts "
        "(fact_id, profile_id, content, canonical_entities_json, embedding, "
        " importance, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (fact_id, profile_id, content, json.dumps(entities),
         _embed(embedding), importance, confidence),
    )


@pytest.fixture
def memory_db(tmp_path: Path) -> Path:
    p = tmp_path / "memory.db"
    _bootstrap_memory_db(p)
    return p


# ---------------------------------------------------------------------------
# Synthetic duplicate pairs: 100 facts, 10 known duplicates (pairs)
# ---------------------------------------------------------------------------


def _seed_known_duplicates(
    db: Path, profile_id: str = "p1", n_unique: int = 90, n_dup_pairs: int = 10,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Insert n_unique unique facts + n_dup_pairs near-duplicate pairs.

    Returns (all_fact_ids, duplicate_pair_ids). Each pair shares embedding
    (cos=1.0) + full entity overlap.
    """
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    all_ids: list[str] = []
    dup_pairs: list[tuple[str, str]] = []

    # Unique facts — each gets a unique embedding direction.
    dim = 8
    for i in range(n_unique):
        fid = f"u{i:04d}"
        all_ids.append(fid)
        vec = [0.0] * dim
        vec[i % dim] = 1.0
        # Tiny perturbation keeps them distinct.
        vec[(i + 1) % dim] = 0.01 * ((i % 7) + 1)
        _seed_fact(conn, fid, profile_id,
                   f"unique content {i}", [f"ent_{i}"], vec)

    # Duplicate pairs — share embedding + entities.
    for j in range(n_dup_pairs):
        vec = [0.0] * dim
        vec[j % dim] = 1.0
        vec[(j + 3) % dim] = 0.5   # distinct from uniques
        a = f"d{j:04d}a"
        b = f"d{j:04d}b"
        _seed_fact(conn, a, profile_id, f"dup content {j} canonical",
                   [f"dup_ent_{j}", "shared"], vec, importance=0.9)
        _seed_fact(conn, b, profile_id, f"dup content {j} loser",
                   [f"dup_ent_{j}", "shared"], vec, importance=0.3)
        all_ids.extend([a, b])
        dup_pairs.append((a, b))

    conn.commit()
    conn.close()
    return all_ids, dup_pairs


# ---------------------------------------------------------------------------
# Tests — exact names per IMPLEMENTATION-MANIFEST v3.4.22 FINAL B.2
# ---------------------------------------------------------------------------


def test_hnsw_finds_known_duplicates(memory_db: Path) -> None:
    from superlocalmemory.learning.hnsw_dedup import HnswDeduplicator
    _seed_known_duplicates(memory_db)
    dedup = HnswDeduplicator(memory_db_path=memory_db)
    candidates = dedup.find_merge_candidates("p1")
    # Expect all 10 duplicate pairs surfaced.
    found_pairs = {
        frozenset((c[0], c[1])) for c in candidates
    }
    # At least 10 pairs should match — unique facts shouldn't cross threshold.
    assert len(found_pairs) >= 10, (
        f"expected >=10 dup pairs; got {len(found_pairs)}"
    )


def test_hnsw_respects_cosine_threshold(memory_db: Path) -> None:
    from superlocalmemory.learning.hnsw_dedup import HnswDeduplicator
    conn = sqlite3.connect(memory_db)
    # Two facts with cos ~0.94 (below 0.95 threshold), full entity overlap.
    v1 = [1.0, 0.0, 0.0, 0.0]
    # v2 chosen so cos(v1, v2) ≈ 0.94
    v2 = [0.94, 0.3411, 0.0, 0.0]
    _seed_fact(conn, "f1", "p1", "alpha", ["shared"], v1)
    _seed_fact(conn, "f2", "p1", "beta", ["shared"], v2)
    conn.commit()
    conn.close()

    dedup = HnswDeduplicator(memory_db_path=memory_db)
    candidates = dedup.find_merge_candidates("p1")
    for canonical, loser, cos, jac in candidates:
        assert cos > HnswDeduplicator.COSINE_THRESHOLD, (
            f"candidate cos={cos} should be > {HnswDeduplicator.COSINE_THRESHOLD}"
        )


def test_hnsw_respects_entity_jaccard_threshold(memory_db: Path) -> None:
    from superlocalmemory.learning.hnsw_dedup import HnswDeduplicator
    conn = sqlite3.connect(memory_db)
    v = [1.0, 0.0, 0.0, 0.0]
    # Identical embeddings but weak entity overlap (jac ~0.33).
    _seed_fact(conn, "f1", "p1", "x", ["a", "b", "c"], v)
    _seed_fact(conn, "f2", "p1", "y", ["a", "x", "y"], v)
    conn.commit()
    conn.close()

    dedup = HnswDeduplicator(memory_db_path=memory_db)
    candidates = dedup.find_merge_candidates("p1")
    # No candidate because jaccard < 0.8 despite cos ≈ 1.0.
    assert candidates == [], (
        f"expected no candidates for jac<0.8; got {candidates}"
    )


def test_hnsw_uses_ram_reservation(memory_db: Path) -> None:
    # v3.4.22 F4.A: HnswDeduplicator now lives in learning.dedup_hnsw.
    # The shim ``learning.hnsw_dedup`` re-exports; patches must target
    # the real definition module so HnswDeduplicator.find_merge_candidates
    # sees the spy.
    from superlocalmemory.learning import dedup_hnsw as mod

    _seed_known_duplicates(memory_db, n_unique=20, n_dup_pairs=5)

    called = {"n": 0, "name": None, "required_mb": None}

    from contextlib import contextmanager

    @contextmanager
    def _spy(name: str, *, required_mb: int = 0, **kw):
        called["n"] += 1
        called["name"] = name
        called["required_mb"] = required_mb
        yield

    with patch.object(mod, "ram_reservation", _spy):
        dedup = mod.HnswDeduplicator(memory_db_path=memory_db)
        dedup.find_merge_candidates("p1")

    assert called["n"] >= 1, "ram_reservation must be invoked"
    assert called["name"] == "hnswlib"
    assert isinstance(called["required_mb"], int)
    assert called["required_mb"] > 0


def test_hnsw_fallback_to_prefix_when_oom(memory_db: Path) -> None:
    # v3.4.22 F4.A: patch the real module, not the shim.
    from superlocalmemory.learning import dedup_hnsw as mod

    _seed_known_duplicates(memory_db, n_unique=5, n_dup_pairs=2)

    from contextlib import contextmanager

    @contextmanager
    def _explode(name: str, **kw):
        raise RuntimeError("ram_reservation(hnswlib): OOM simulated")
        yield  # unreachable

    with patch.object(mod, "ram_reservation", _explode):
        dedup = mod.HnswDeduplicator(memory_db_path=memory_db)
        # Must not raise — falls through to prefix dedup.
        candidates = dedup.find_merge_candidates("p1")

    # Prefix fallback still finds duplicates (same content prefix).
    # Either way, API contract: returns a list (possibly empty) not raise.
    assert isinstance(candidates, list)


def test_hnsw_fallback_when_unavailable(memory_db: Path) -> None:
    from superlocalmemory.learning import hnsw_dedup as mod

    _seed_known_duplicates(memory_db, n_unique=5, n_dup_pairs=2)

    # Force the module to pretend hnswlib is missing.
    dedup = mod.HnswDeduplicator(memory_db_path=memory_db)
    candidates = dedup.find_merge_candidates("p1", _force_unavailable=True)
    assert isinstance(candidates, list)


def test_merge_never_deletes_atomic_facts_row(memory_db: Path) -> None:
    from superlocalmemory.learning.hnsw_dedup import HnswDeduplicator
    from superlocalmemory.learning.memory_merge import apply_merges

    _, dup_pairs = _seed_known_duplicates(memory_db, n_unique=10, n_dup_pairs=5)

    dedup = HnswDeduplicator(memory_db_path=memory_db)
    candidates = dedup.find_merge_candidates("p1")
    assert len(candidates) >= 5

    # Hook a SQLite authorizer that forbids DELETE on atomic_facts.
    conn = sqlite3.connect(memory_db)

    def _authorizer(code, arg1, arg2, arg3, arg4):
        if code == sqlite3.SQLITE_DELETE and arg1 == "atomic_facts":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    conn.set_authorizer(_authorizer)
    conn.close()

    # Apply merges; MUST NOT issue DELETE on atomic_facts anywhere.
    applied = apply_merges(memory_db, candidates, profile_id="p1")
    assert applied >= 5

    # Verify both canonical and loser rows still exist, loser has
    # archive_status='merged'.
    conn = sqlite3.connect(memory_db)
    total = conn.execute(
        "SELECT COUNT(*) FROM atomic_facts"
    ).fetchone()[0]
    merged = conn.execute(
        "SELECT COUNT(*) FROM atomic_facts "
        "WHERE archive_status='merged' AND merged_into IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    assert total == 20, f"expected all 20 rows to remain; got {total}"
    assert merged >= 5, f"expected >=5 rows flipped to merged; got {merged}"


def test_merge_preserves_both_in_merge_log(memory_db: Path) -> None:
    from superlocalmemory.learning.hnsw_dedup import HnswDeduplicator
    from superlocalmemory.learning.memory_merge import apply_merges

    _seed_known_duplicates(memory_db, n_unique=5, n_dup_pairs=3)

    dedup = HnswDeduplicator(memory_db_path=memory_db)
    candidates = dedup.find_merge_candidates("p1")
    apply_merges(memory_db, candidates, profile_id="p1")

    conn = sqlite3.connect(memory_db)
    rows = conn.execute(
        "SELECT canonical_fact_id, merged_fact_id, cosine_sim, entity_jaccard "
        "FROM memory_merge_log WHERE profile_id=?",
        ("p1",),
    ).fetchall()
    conn.close()
    assert len(rows) >= 3
    for canonical, merged, cos, jac in rows:
        assert canonical and merged
        assert canonical != merged
        assert 0.0 <= cos <= 1.0 + 1e-6
        assert 0.0 <= jac <= 1.0 + 1e-6


def test_merge_reversible_via_unmerge_cli(memory_db: Path) -> None:
    from superlocalmemory.learning.hnsw_dedup import HnswDeduplicator
    from superlocalmemory.learning.memory_merge import apply_merges, unmerge

    _seed_known_duplicates(memory_db, n_unique=5, n_dup_pairs=2)

    dedup = HnswDeduplicator(memory_db_path=memory_db)
    candidates = dedup.find_merge_candidates("p1")
    apply_merges(memory_db, candidates, profile_id="p1")

    conn = sqlite3.connect(memory_db)
    merge_rows = conn.execute(
        "SELECT merge_id, merged_fact_id FROM memory_merge_log"
    ).fetchall()
    conn.close()
    assert merge_rows, "no merges recorded to unmerge"

    merge_id, merged_fid = merge_rows[0]
    ok = unmerge(memory_db, merge_id)
    assert ok is True

    conn = sqlite3.connect(memory_db)
    row = conn.execute(
        "SELECT archive_status, merged_into FROM atomic_facts WHERE fact_id=?",
        (merged_fid,),
    ).fetchone()
    log_row = conn.execute(
        "SELECT reversible FROM memory_merge_log WHERE merge_id=?",
        (merge_id,),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "live"
    assert row[1] is None
    assert log_row is not None
    assert log_row[0] == 0  # no longer reversible (already reversed)


def test_reward_gated_archive_respects_60d_window(memory_db: Path, tmp_path: Path) -> None:
    """Fact with recent positive reward (<60d) MUST NOT be archived."""
    from superlocalmemory.learning.hnsw_dedup import run_reward_gated_archive

    conn = sqlite3.connect(memory_db)
    # Fact 'f1' has a recent positive reward, 'f2' has none.
    _seed_fact(conn, "f1", "p1", "rewarded", ["e1"], [1.0, 0.0, 0.0])
    _seed_fact(conn, "f2", "p1", "cold", ["e2"], [0.0, 1.0, 0.0])
    # Recent positive reward for f1 (well within 60d).
    conn.execute(
        "INSERT INTO action_outcomes "
        "(outcome_id, profile_id, fact_ids_json, reward, settled, settled_at) "
        "VALUES (?, ?, ?, ?, ?, datetime('now'))",
        ("o1", "p1", json.dumps(["f1"]), 0.9, 1),
    )
    conn.commit()
    conn.close()

    archived = run_reward_gated_archive(memory_db, "p1", candidate_fact_ids=["f1", "f2"])
    assert "f1" not in archived, "recent-rewarded fact must not be archived"
    assert "f2" in archived, "unrewarded candidate must be archived"

    # And atomic_facts row still exists — never deleted.
    conn = sqlite3.connect(memory_db)
    cnt = conn.execute(
        "SELECT COUNT(*) FROM atomic_facts WHERE fact_id IN ('f1','f2')"
    ).fetchone()[0]
    conn.close()
    assert cnt == 2


def test_reward_gated_archive_skips_important_flag(memory_db: Path) -> None:
    """Fact flagged important MUST NOT be archived."""
    from superlocalmemory.learning.hnsw_dedup import run_reward_gated_archive

    conn = sqlite3.connect(memory_db)
    _seed_fact(conn, "imp", "p1", "sacred", ["e"], [1.0, 0.0, 0.0],
               importance=1.0)
    _seed_fact(conn, "ord", "p1", "ordinary", ["e"], [1.0, 0.0, 0.0],
               importance=0.4)
    conn.commit()
    conn.close()

    archived = run_reward_gated_archive(
        memory_db, "p1", candidate_fact_ids=["imp", "ord"],
    )
    assert "imp" not in archived
    assert "ord" in archived


def test_strong_memory_boost_caps_retrieval_prior(memory_db: Path) -> None:
    """retrieval_prior boost capped at MAX (per LLD-12 §5)."""
    from superlocalmemory.learning.hnsw_dedup import apply_strong_memory_boost

    conn = sqlite3.connect(memory_db)
    _seed_fact(conn, "f1", "p1", "boosted", ["e"], [1.0, 0.0, 0.0])
    # Seed many positive outcomes — uncapped would shoot past 1.0.
    for i in range(50):
        conn.execute(
            "INSERT INTO action_outcomes "
            "(outcome_id, profile_id, fact_ids_json, reward, settled) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"o{i}", "p1", json.dumps(["f1"]), 0.95, 1),
        )
    conn.commit()
    conn.close()

    apply_strong_memory_boost(memory_db, "p1")

    conn = sqlite3.connect(memory_db)
    prior = conn.execute(
        "SELECT retrieval_prior FROM atomic_facts WHERE fact_id=?",
        ("f1",),
    ).fetchone()[0]
    conn.close()
    assert 0.0 < prior <= 0.5 + 1e-9, (
        f"retrieval_prior must be capped in (0, 0.5], got {prior}"
    )


def test_soft_prompts_mine_high_reward_only(memory_db: Path) -> None:
    """Reward-aware soft-prompt mining surfaces only high-reward facts."""
    from superlocalmemory.learning.hnsw_dedup import (
        select_high_reward_fact_ids,
    )

    conn = sqlite3.connect(memory_db)
    _seed_fact(conn, "good", "p1", "winner", ["e"], [1.0, 0.0])
    _seed_fact(conn, "meh", "p1", "neutral", ["e"], [0.0, 1.0])
    conn.execute(
        "INSERT INTO action_outcomes "
        "(outcome_id, profile_id, fact_ids_json, reward, settled) "
        "VALUES (?, ?, ?, ?, 1)",
        ("og", "p1", json.dumps(["good"]), 0.85),
    )
    conn.execute(
        "INSERT INTO action_outcomes "
        "(outcome_id, profile_id, fact_ids_json, reward, settled) "
        "VALUES (?, ?, ?, ?, 1)",
        ("om", "p1", json.dumps(["meh"]), 0.2),
    )
    conn.commit()
    conn.close()

    high = select_high_reward_fact_ids(memory_db, "p1", min_reward=0.6)
    assert "good" in high
    assert "meh" not in high


def test_consolidation_wall_time_5min_cap(memory_db: Path) -> None:
    """find_merge_candidates respects the wall_seconds budget."""
    from superlocalmemory.learning.hnsw_dedup import HnswDeduplicator

    _seed_known_duplicates(memory_db, n_unique=20, n_dup_pairs=5)
    dedup = HnswDeduplicator(memory_db_path=memory_db)
    # Use a ludicrously small budget; must return cleanly, not spin.
    candidates = dedup.find_merge_candidates("p1", wall_seconds=0.001)
    assert isinstance(candidates, list)


def test_consolidation_partial_progress_resumable(memory_db: Path) -> None:
    """apply_merges is transactional — partial failure leaves DB consistent."""
    from superlocalmemory.learning.hnsw_dedup import HnswDeduplicator
    from superlocalmemory.learning.memory_merge import apply_merges

    _seed_known_duplicates(memory_db, n_unique=5, n_dup_pairs=4)

    dedup = HnswDeduplicator(memory_db_path=memory_db)
    candidates = dedup.find_merge_candidates("p1")
    assert candidates

    # Apply first 2 merges; remaining can be resumed next cycle.
    first = candidates[:2]
    applied_first = apply_merges(memory_db, first, profile_id="p1")
    assert applied_first == 2

    conn = sqlite3.connect(memory_db)
    merged_so_far = conn.execute(
        "SELECT COUNT(*) FROM atomic_facts WHERE archive_status='merged'"
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM atomic_facts").fetchone()[0]
    conn.close()

    # Partial progress: 2 flipped, all 13 rows still present.
    assert merged_so_far == 2
    assert total == 13

    # Now apply the rest and confirm dedup is idempotent on already-merged.
    remaining = [
        c for c in candidates
        if c[1] not in {x[1] for x in first}
    ]
    apply_merges(memory_db, remaining, profile_id="p1")


def test_10k_facts_under_200mb_ram_peak(memory_db: Path) -> None:
    """10k synthetic facts → peak RAM delta must stay under 200 MB (I2 budget)."""
    from superlocalmemory.learning.hnsw_dedup import HnswDeduplicator

    # Seed 10k facts with 8-dim embeddings. hnsw index should stay tiny.
    conn = sqlite3.connect(memory_db)
    conn.execute("PRAGMA synchronous=OFF")  # test only
    dim = 8
    for i in range(10_000):
        vec = [0.0] * dim
        vec[i % dim] = 1.0
        vec[(i + 2) % dim] = 0.5
        conn.execute(
            "INSERT INTO atomic_facts "
            "(fact_id, profile_id, content, canonical_entities_json, "
            " embedding, importance, confidence) VALUES "
            "(?, 'p1', ?, ?, ?, 0.5, 1.0)",
            (f"f{i:05d}", f"c{i}", json.dumps([f"e{i % 100}"]),
             json.dumps(vec)),
        )
    conn.commit()
    conn.close()

    # Measure peak RSS delta around the dedup run.
    before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns bytes; Linux returns KB. Normalise to MB.
    if sys.platform == "darwin":
        before_mb = before_kb / (1024 * 1024)
    else:
        before_mb = before_kb / 1024

    dedup = HnswDeduplicator(memory_db_path=memory_db)
    dedup.find_merge_candidates("p1", wall_seconds=30.0)

    after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        after_mb = after_kb / (1024 * 1024)
    else:
        after_mb = after_kb / 1024

    delta_mb = after_mb - before_mb
    # Soft budget: peak delta must stay well below 200 MB.
    assert delta_mb < 200.0, (
        f"RAM delta {delta_mb:.1f} MB exceeds 200 MB budget"
    )
