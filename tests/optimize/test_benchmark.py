# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""CI gate for the Optimize benchmark (v3.6.10, plan §7).

Runs the benchmark harness in-process and asserts the gating INVARIANTS so a
regression in cache or compression correctness fails the test suite — not just
a manual benchmark run. No API key / model / network required.

Invariants enforced:
  * exact-cache replay: 100% hit, 100% byte-identical
  * false-hit guard: 0 false hits, 0 near-miss false hits
  * semantic wiring: fallback fires above threshold, guard blocks below
  * compression: off lossless, safe lossless (all types), code unchanged in safe
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The benchmark modules live in benchmarks/optimize (not on the package path).
_BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "optimize"


@pytest.fixture(scope="module", autouse=True)
def _add_bench_to_path():
    sys.path.insert(0, str(_BENCH_DIR))
    yield
    try:
        sys.path.remove(str(_BENCH_DIR))
    except ValueError:
        pass


@pytest.mark.benchmark
def test_cache_exact_replay_all_byte_identical():
    import bench_cache
    r = bench_cache.bench_exact_replay(n=25)
    assert r["hit_rate"] == 1.0, f"expected 100% hit, got {r['hit_rate']}"
    assert r["byte_identical_rate"] == 1.0, "cache hit must be byte-identical to stored response"
    assert r["pass"] is True


@pytest.mark.benchmark
def test_cache_false_hit_is_zero():
    import bench_cache
    r = bench_cache.bench_false_hit(n=25)
    assert r["false_hits"] == 0, "exact cache returned a wrong answer (false hit) — class bug"
    assert r["near_miss_false_hits"] == 0, "1-char-different prompt must not hit"
    assert r["pass"] is True


@pytest.mark.benchmark
def test_cache_semantic_wiring():
    import bench_cache
    r = bench_cache.bench_semantic_wiring()
    assert r["fallback_fires_above_threshold"] is True
    assert r["guard_blocks_below_threshold"] is True
    assert r["disabled_tier_not_consulted"] is True
    assert r["pass"] is True


@pytest.mark.benchmark
def test_compression_safe_is_lossless():
    import bench_compression
    r = bench_compression.run()
    assert r["off_lossless"] is True, "off mode must never mutate the body"
    assert r["safe_lossless"] is True, "safe mode must be lossless for json/code/prose"
    assert r["code_unchanged_in_safe"] is True, "§2.6: code forwarded unchanged in safe mode"
    assert r["pass"] is True


@pytest.mark.benchmark
def test_full_bench_orchestrator_passes(tmp_path, monkeypatch):
    """run_bench.main() returns 0 and writes RESULTS files."""
    import run_bench
    monkeypatch.setattr(run_bench, "_HERE", tmp_path)
    rc = run_bench.main()
    assert rc == 0
    assert (tmp_path / "RESULTS.json").exists()
    assert (tmp_path / "RESULTS.md").exists()
