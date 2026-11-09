# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file

"""Long-run RSS soak — remember + recall unbounded-growth detection.

Invariant I3 for SLM 4.0.6: no memory leaks — long-run soak with RSS
tracking; gate on slope.

EXCLUDED from the default pytest run (addopts = -m 'not slow').
Run explicitly with:
    pytest -m slow tests/test_storage/test_memory_leak_soak.py -s -v

Design
------
Two-phase structure:

  Phase 1 — Warm-up (WARMUP_ITERS iterations, not measured):
    Covers one-time costs that must not be charged to the leak budget:
      (a) Lazy Python module imports: encoding, FTS5 virtual table init,
          graph_builder, entity_resolver, spreading_activation, etc.
          These pull in numpy / scipy / networkx once and stabilise.
          Typically complete within the first 5-10 calls.
      (b) SQLite page cache fill: the first ~10-15 writes force sqlite to
          allocate cache pages (bounded by PRAGMA cache_size=-32768, ~32MB).
          After that, existing pages are reused rather than allocated.
      (c) Python malloc arena pre-warm: on macOS (jemalloc / libgcc) arenas
          grow eagerly and do not return to the OS; the first 10-20 iters
          see ~10-50 KB RSS/iter as the arena reaches its steady shape.
      WHY 25? Conservative sum of all three phases.  If the real 930MB
      embedding subprocess were active, warm-up would need ≥60 iters to
      cover model load.  This test uses the session-level mock that patches
      out WorkerPool and CrossEncoderReranker, so no model is loaded and
      25 iters is sufficient.

  Phase 2 — Steady state (STEADY_ITERS iterations, RSS sampled after
    gc.collect(2) each iteration):
    Linear regression (scipy.stats.linregress) over all STEADY_ITERS
    samples gives the slope in bytes/iteration.  This is compared to
    RSS_SLOPE_THRESHOLD_BYTES_PER_ITER.

Threshold calibration
---------------------
Measured against the mock-backed harness on macOS (ARM, 2026):

  Baseline slope (full 75-sample window): ~10-13 KB/iter
  Source: SQLite WAL page allocation (~1 new 4KB page per 3-4 facts) +
          BM25 in-memory token accumulation + Python dict resizes.

  Threshold = 50 KB/iter = ~4-5× the measured baseline.
  At 75 steady-state iterations this permits up to 3.75 MB of growth
  before failing — i.e. any sustained leak ≥ 40 KB/iter is caught.

Injected-leak proof of teeth
------------------------------
_INJECT_SYNTHETIC_LEAK = True appends _LEAK_PER_ITER_BYTES bytes to a
module-level list each iteration.  At 150 KB/iter × 75 iters = 11.25 MB
total, the slope (~160 KB/iter) clearly exceeds the 50 KB/iter threshold
and the test FAILS.  Set to False (default) for the clean run, which PASSES.

Python-level leak suspects
--------------------------
Beyond RSS, two bounded-growth assertions guard against Python-level leaks
that may not yet pressure RSS:

  (A) sqlite3.Connection count — any path that opens a sqlite3.Connection
      without closing it will grow this count indefinitely.  Only the
      DatabaseManager write connection and optional WAL reader should
      persist.  We track the live count via gc.get_objects() and assert
      no more than MAX_SQLITE_CONN_GROWTH new connections appear.

  (B) GC object count per iteration — the net growth of all live Python
      objects across the steady-state window should not exceed
      GC_OBJECTS_PER_ITER_BUDGET per iteration (set to 5 000 to absorb
      SQLite row dicts cached at the DB layer).
"""

from __future__ import annotations

import gc
import sqlite3
import time

import psutil
import pytest
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Thresholds — calibrated against mock-backed harness on macOS ARM, 2026
# ---------------------------------------------------------------------------

# Maximum linear RSS growth rate (bytes/iteration) over the full steady-state
# window.  At 75 iterations this permits up to 3.75 MB of total growth.
# Baseline clean run: ~10-13 KB/iter (SQLite page alloc + BM25 token index).
# Threshold = 5× baseline for clear headroom vs. CI noise.
_RSS_SLOPE_THRESHOLD_BYTES_PER_ITER: int = 50 * 1024  # 50 KB/iter

# Warm-up iterations excluded from slope computation.
# See module docstring for detailed justification of the 25-iter cutoff.
# NB: if real model workers are present (not mocked), increase this to ≥60.
_WARMUP_ITERS: int = 25

# Steady-state iterations used for regression (= total − warmup).
_STEADY_ITERS: int = 75  # total 100 iters, well within 120s budget

# Maximum allowed growth in live sqlite3.Connection objects over the soak.
# DatabaseManager opens one persistent write connection.  Transient read
# connections (context_cache fast-path, emergency recall fallback) MUST be
# closed inside their respective finally blocks.  More than 2 net new
# connections indicates a handle leak.
_MAX_SQLITE_CONN_GROWTH: int = 2

# Maximum allowed net growth in total gc-tracked objects per steady-state
# iteration.  Each store_fast + recall pair creates and frees ~100-500
# transient objects (MemoryRecord, AtomicFact, RecallResponse, etc.).
# After gc.collect(2), survivors should be near zero.  Budget set at
# 5 000/iter to absorb SQLite row dicts persisted in BM25/entity caches.
_GC_OBJECTS_PER_ITER_BUDGET: int = 5_000


# ---------------------------------------------------------------------------
# Injected-leak harness — teeth-proof mechanism (disabled in CI)
# ---------------------------------------------------------------------------
# Set _INJECT_SYNTHETIC_LEAK = True to enable a deliberate in-process leak
# that MUST trip the slope assertion.  Used ONLY to verify the harness
# itself can detect unbounded growth; must be False in committed code.
#
# Mechanism: _LEAK_BUCKET is a module-level list.  When injection is
# enabled, _maybe_inject_leak() appends _LEAK_PER_ITER_BYTES bytes each
# iteration.  gc.collect(2) does not reclaim these because the list is a
# GC root (module global); RSS climbs proportionally.
_INJECT_SYNTHETIC_LEAK: bool = False

# Bucket size per iteration when injection is active.
# 150 KB/iter × 75 iters = 11.25 MB → slope ≈ 160 KB/iter >> threshold.
_LEAK_PER_ITER_BYTES: int = 150 * 1024

_LEAK_BUCKET: list[bytes] = []


def _maybe_inject_leak() -> None:
    """Append a fixed buffer to the module-level leak list if injection is on."""
    if _INJECT_SYNTHETIC_LEAK:
        _LEAK_BUCKET.append(b"\xAB" * _LEAK_PER_ITER_BYTES)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rss_bytes() -> int:
    """Return process RSS in bytes.

    Called AFTER gc.collect(2) so that reference cycles and generation-2
    garbage are collected before sampling.  This eliminates the ~1-2 MB
    noise spike that appears when the GC defers a collection until Python's
    internal threshold is reached (typically every 100-700 allocations).

    Limitation (documented, not fixed): on macOS, freed Python objects are
    lazily decommitted via madvise(MADV_FREE), so RSS may not drop
    immediately after a large free.  The linear regression over 75 points
    tolerates per-sample noise of ±2 MB; individual outliers do not
    invalidate the slope estimate.
    """
    gc.collect(2)
    return psutil.Process().memory_info().rss


def _live_sqlite_connection_count() -> int:
    """Count sqlite3.Connection objects reachable via the GC graph."""
    gc.collect(2)
    return sum(1 for obj in gc.get_objects() if type(obj) is sqlite3.Connection)


# ---------------------------------------------------------------------------
# Soak test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_remember_recall_no_rss_leak(engine_with_mock_deps) -> None:
    """Remember + recall must not show unbounded RSS growth over 100 iterations.

    Uses the session-scoped ``_prevent_heavy_model_loading`` fixture (from
    ``tests/conftest.py``) which patches out WorkerPool.shared and
    CrossEncoderReranker so the test runs without the 930 MB embedding
    subprocess.  This keeps runtime well under 120 s and isolates the
    store+recall Python/SQLite code path from model-loading memory costs.

    Assertions
    ----------
    (a) Linear RSS slope < RSS_SLOPE_THRESHOLD_BYTES_PER_ITER
        Primary leak gate.  linregress over 75 steady-state samples.
        r² is reported; values < 0.10 indicate high noise (unlikely a real
        leak) — the assertion message calls this out.
    (b) sqlite3.Connection count delta ≤ MAX_SQLITE_CONN_GROWTH
        Catches unclosed DB handles.
    (c) Total GC object growth ≤ GC_OBJECTS_PER_ITER_BUDGET × steady_iters
        Catches reference-cycle accumulation in module-level collections
        (e.g. _behavioral_tracker_cache, _forgetting_scheduler_cache in
        store_pipeline.py, or _pr_cache / _comm_cache in
        spreading_activation.py).
    (d) Elapsed time ≤ 120 s
        CI budget guard.
    """
    engine = engine_with_mock_deps

    # ------------------------------------------------------------------
    # Warm-up phase — excluded from regression
    # ------------------------------------------------------------------
    # Vary content to exercise multiple code branches (entity extraction
    # regex, BM25 tokenizer, different topic_sig signatures).
    warmup_contents = [
        f"warm-up fact {i}: The agent learned about topic-{i % 10} during "
        f"session warmup-{i // 5}. Entity-{i % 6} performed action {i}."
        for i in range(_WARMUP_ITERS)
    ]
    for i, content in enumerate(warmup_contents):
        engine.store_fast(content, metadata={"session_id": f"warmup-{i // 5}"})
        engine.recall(
            f"topic-{i % 10} entity-{i % 6}",
            fast=True,
            limit=5,  # cap results to prevent O(n²) recall growth masking leaks
        )

    # Baseline measurements taken AFTER warm-up, BEFORE steady state.
    # gc.collect(2) is implicit in the helpers below.
    baseline_rss: int = _rss_bytes()
    baseline_conn_count: int = _live_sqlite_connection_count()
    baseline_gc_count: int = len(gc.get_objects())

    # ------------------------------------------------------------------
    # Steady-state phase — RSS sampled after gc.collect(2) each iteration
    # ------------------------------------------------------------------
    rss_samples: list[int] = []
    t_start = time.monotonic()

    for i in range(_STEADY_ITERS):
        # Rotate content across multiple entities, concepts, and sessions
        # to exercise diverse code paths without creating an unbounded
        # growing recall result-set per query.
        content = (
            f"steady fact {i}: entity-{i % 7} observed concept-{i % 5} "
            f"during session soak-{i // 8}. Confidence level {i % 4 + 1}/4."
        )
        engine.store_fast(
            content,
            metadata={"session_id": f"soak-{i // 8}"},
        )
        engine.recall(
            f"entity-{i % 7} concept-{i % 5}",
            fast=True,
            limit=5,  # bounded recall depth prevents recall-load growth
        )

        # Synthetic leak injection (disabled by default — see module header).
        _maybe_inject_leak()

        # Sample RSS after a full GC cycle to eliminate GC timing noise.
        rss_samples.append(_rss_bytes())

    elapsed = time.monotonic() - t_start

    # Final Python-object counts (gc.collect(2) is implicit in helper).
    final_conn_count: int = _live_sqlite_connection_count()
    final_gc_count: int = len(gc.get_objects())
    conn_delta: int = final_conn_count - baseline_conn_count
    gc_growth_total: int = final_gc_count - baseline_gc_count
    gc_growth_per_iter: float = gc_growth_total / _STEADY_ITERS

    # ------------------------------------------------------------------
    # Linear regression over the full steady-state window
    # ------------------------------------------------------------------
    xs = list(range(_STEADY_ITERS))
    slope, _intercept, r_value, p_value, std_err = linregress(xs, rss_samples)

    # ------------------------------------------------------------------
    # Diagnostic output (visible with pytest -s)
    # ------------------------------------------------------------------
    rss_mb = [r / (1024 * 1024) for r in rss_samples]
    total_growth_mb = (rss_samples[-1] - baseline_rss) / (1024 * 1024)
    print(
        f"\n[soak] elapsed={elapsed:.1f}s  iters={_STEADY_ITERS}"
        f"  baseline={baseline_rss // 1024}KB"
        f"  rss_min={min(rss_mb):.1f}MB  rss_max={max(rss_mb):.1f}MB"
        f"  total_growth={total_growth_mb:.2f}MB"
    )
    print(
        f"[soak] slope={slope / 1024:.2f}KB/iter"
        f"  threshold={_RSS_SLOPE_THRESHOLD_BYTES_PER_ITER // 1024}KB/iter"
        f"  r²={r_value ** 2:.3f}  p={p_value:.4f}  stderr={std_err / 1024:.2f}KB"
    )
    print(
        f"[soak] conn_delta={conn_delta}  "
        f"gc_growth={gc_growth_total} ({gc_growth_per_iter:.1f}/iter)"
    )
    if _INJECT_SYNTHETIC_LEAK:
        print(
            f"[soak] *** LEAK INJECTION ACTIVE — {_LEAK_PER_ITER_BYTES // 1024}KB/iter "
            f"× {_STEADY_ITERS} iters = {_LEAK_PER_ITER_BYTES * _STEADY_ITERS // (1024 * 1024)}MB injected ***"
        )

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    # (a) RSS slope gate — primary leak signal.
    #
    # A high r² (≥0.5) means the slope estimate is reliable.  A low r²
    # with a large slope is more likely measurement noise than a genuine
    # leak; the message flags this so engineers can investigate rather than
    # dismiss the failure.
    assert slope < _RSS_SLOPE_THRESHOLD_BYTES_PER_ITER, (
        f"Sustained RSS growth detected.\n"
        f"  slope = {slope / 1024:.1f} KB/iter  "
        f"  threshold = {_RSS_SLOPE_THRESHOLD_BYTES_PER_ITER // 1024} KB/iter\n"
        f"  r² = {r_value ** 2:.3f}  p = {p_value:.4f}\n"
        f"  {'(r² < 0.3: high noise — verify with a longer soak before escalating)' if r_value ** 2 < 0.3 else ''}"
        f"  RSS range: {min(rss_mb):.1f}-{max(rss_mb):.1f} MB  "
        f"  baseline = {baseline_rss // 1024} KB\n"
        f"  Likely suspects: unclosed DB handles, growing module-level dict "
        f"  (_behavioral_tracker_cache, _forgetting_scheduler_cache, "
        f"  _pr_cache, _comm_cache), lru_cache without maxsize, or an "
        f"  event-bus queue that never drains."
    )

    # (b) sqlite3.Connection handle leak gate.
    #
    # Only the DatabaseManager write connection should remain open across
    # the full soak.  context_cache.read_entry_fast() and
    # _sqlite_emergency_recall() open read-only connections but MUST close
    # them in their finally blocks.
    assert conn_delta <= _MAX_SQLITE_CONN_GROWTH, (
        f"sqlite3.Connection count grew by {conn_delta} over {_STEADY_ITERS} "
        f"steady-state iterations (baseline={baseline_conn_count}, "
        f"final={final_conn_count}).  Unclosed DB handle suspected.  "
        f"Check: context_cache.read_entry_fast(), "
        f"_sqlite_emergency_recall(), _verify_ingestion_schema(), "
        f"and any inline sqlite3.connect() calls in the recall pipeline."
    )

    # (c) GC object accumulation gate.
    #
    # Detects Python-level leaks that have not yet pressured RSS: growing
    # module-level lists/dicts/sets that hold references to transient
    # objects, preventing GC collection.
    max_allowed_gc_growth = _GC_OBJECTS_PER_ITER_BUDGET * _STEADY_ITERS
    assert gc_growth_total <= max_allowed_gc_growth, (
        f"GC object count grew by {gc_growth_total} over {_STEADY_ITERS} "
        f"iterations ({gc_growth_per_iter:.1f}/iter), exceeding budget of "
        f"{_GC_OBJECTS_PER_ITER_BUDGET}/iter.  "
        f"Possible unbounded accumulation in a module-level collection "
        f"(check store_pipeline._behavioral_tracker_cache, "
        f"store_pipeline._forgetting_scheduler_cache, "
        f"spreading_activation._pr_cache, or any list that appends "
        f"without eviction)."
    )

    # (d) Runtime guard — ensures CI build does not hang.
    assert elapsed <= 120.0, (
        f"Soak ran for {elapsed:.1f}s (budget 120s).  "
        f"Performance regression or blocking I/O detected.  "
        f"Check whether a real embedding worker subprocess was spawned "
        f"(should be mocked by _prevent_heavy_model_loading)."
    )
