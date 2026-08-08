#!/usr/bin/env python3
"""
bench_perf.py — SuperLocalMemory V4 Performance Benchmark

Drives the REAL production MemoryEngine to produce latency distributions,
concurrency-vs-latency curves, and RSS time-series. No synthetic timings.

Usage:
    python bench_perf.py --smoke               # quick end-to-end check
    python bench_perf.py --n 300 --duration 60 # full benchmark for paper

Args: --n N (latency ops), --db PATH (source DB copy), --duration D (RSS secs),
      --profile P (profile ID), --smoke (n=5, duration=10)

SAFETY: NEVER touches ~/.superlocalmemory or port 8765. Reference DB is
read-only; writes go to a timestamped throwaway copy under bench_run/.
FAIL-LOUD: raises and exits non-zero if embedding cannot initialise.

CRIT — Three Flaws A Senior Perf Engineer Would Catch
------------------------------------------------------
CRIT-1  COLD-CACHE vs WARM
    First recall pays Python module imports (~50–200ms), Ollama HTTP
    connection setup (~10–50ms), SQLite 959MB page-cache cold start.
    Fix: N_WARMUP=5 ops before every timed section. Warmup excluded from
    all latency figures. JSON field "warmup_ops_before_timing" discloses this.

CRIT-2  GC PAUSES CONTAMINATING LATENCY TAILS
    CPython GC pauses 1–20ms mid-loop, inflating p99/max artificially.
    Fix: gc.disable() immediately before every timed loop; gc.collect() +
    gc.enable() immediately after. Applied to recall loop, store_fast loop,
    and each concurrency sweep point. JSON "gc_disabled_during" discloses this.

CRIT-3  EMBEDDING MODEL LOAD TIME CONTAMINATING FIRST-OP LATENCY
    OllamaEmbedder._available is None at start (lazy). First embed() does
    HTTP discovery + model wake (134–2318ms observed). Falls inside timing
    window → grotesque outlier corrupting p99/max.
    Fix: detect_and_warm_embedder() forces is_available() + one timed embed()
    call OUTSIDE the timing window. Raises if embedding is unavailable.
    embedder_warmup_ms recorded in JSON environment block.

ADDITIONALLY DOCUMENTED (not "fixed" — this IS the production model):
CRIT-4  THREAD/GIL CONCURRENCY SEMANTICS
    ThreadPoolExecutor = 16 Python threads sharing the GIL, not 16 OS
    parallel executors. I/O-bound paths (Ollama HTTP, SQLite WAL writes)
    release the GIL, so I/O parallelism is real. Writes serialise further
    via DatabaseManager.get_write_lock() (process-wide RLock + SQLite WAL).
    This IS the engine's production concurrency model; the benchmark measures
    it honestly without papering over it.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import random
import shutil
import sqlite3
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCRATCHPAD = Path(
    "/private/tmp/claude-501"
    "/-Users-v-pratap-bhardwaj-Documents-varun-world-Agentic-official"
    "/56c17f7c-3d6d-4ab6-b209-54da740ba392/scratchpad"
)
_REAL_DB_REF = _SCRATCHPAD / "realdb" / "memory.db"
_BENCH_BASE = _SCRATCHPAD / "bench_run"
_RESULTS_DIR = Path(__file__).resolve().parent / "results"

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("bench_perf")

N_WARMUP: int = 5   # warm-path ops before every timed section (CRIT-1)
SEED: int = 42      # deterministic query selection


def _pct(data: list[float], p: float) -> float:
    """Linear-interpolation percentile (p in [0,100])."""
    if not data:
        return float("nan")
    s = sorted(data)
    idx = (p / 100.0) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def _summarise(ms: list[float]) -> dict:
    return {
        "n": len(ms),
        "p50_ms": round(_pct(ms, 50), 3),
        "p90_ms": round(_pct(ms, 90), 3),
        "p95_ms": round(_pct(ms, 95), 3),
        "p99_ms": round(_pct(ms, 99), 3),
        "max_ms": round(max(ms), 3) if ms else float("nan"),
        "mean_ms": round(sum(ms) / len(ms), 3) if ms else float("nan"),
    }


def prepare_bench_dir(source_db: Path, bench_dir: Path) -> Path:
    """Copy source_db to bench_dir/memory.db. Raises if source missing."""
    if not source_db.exists():
        raise FileNotFoundError(
            f"FAIL-LOUD: source DB not found: {source_db}\n"
            "Cannot benchmark without the real DB — refusing to fabricate data."
        )
    bench_dir.mkdir(parents=True, exist_ok=True)
    dest = bench_dir / "memory.db"
    print(f"  Copying {source_db} ({source_db.stat().st_size // 1024 // 1024} MB) → {dest}",
          flush=True)
    t0 = time.monotonic()
    shutil.copy2(str(source_db), str(dest))
    print(f"  Copy complete in {time.monotonic() - t0:.1f}s", flush=True)
    return dest


def _load_bench_config(bench_dir: Path, profile_id: str) -> Any:
    """Load production SLMConfig; redirect I/O to bench_dir only."""
    from superlocalmemory.core.config import (
        SLMConfig, ConsolidationConfig, ForgettingConfig,
    )
    config = SLMConfig.load()
    config.base_dir = bench_dir
    config.db_path = bench_dir / "memory.db"
    config.active_profile = profile_id
    # Disable background writers that would skew latency measurements
    config.consolidation = ConsolidationConfig(
        enabled=False, step_count_trigger=10_000_000,
        session_trigger=False, scheduled_sessions=10_000_000,
    )
    config.forgetting = ForgettingConfig(enabled=False)
    return config


def init_engine(config: Any) -> Any:
    from superlocalmemory.core.engine import MemoryEngine
    from superlocalmemory.core.engine_capabilities import Capabilities
    engine = MemoryEngine(config, capabilities=Capabilities.FULL)
    engine.initialize()
    return engine


def detect_and_warm_embedder(engine: Any) -> dict:
    """Force embedder warm-up outside timing window (CRIT-3).

    Raises RuntimeError if embedding is unavailable — we never fabricate
    latencies without the real semantic channel.
    """
    embedder = getattr(engine, "_embedder", None)
    if embedder is None:
        raise RuntimeError(
            "FAIL-LOUD: engine._embedder is None after initialize().\n"
            "Run 'slm doctor'. Refusing to benchmark without semantic channel."
        )
    provider_class = type(embedder).__name__
    is_avail_fn = getattr(embedder, "is_available", None)
    if callable(is_avail_fn):
        available = is_avail_fn()
    else:
        available = bool(getattr(embedder, "_available", True))
    if not available:
        raise RuntimeError(
            f"FAIL-LOUD: {provider_class} reports not available. "
            "Ensure Ollama/sentence-transformers is reachable."
        )
    t0 = time.monotonic()
    probe = embedder.embed("benchmark probe: AI Reliability Engineering")
    warmup_ms = (time.monotonic() - t0) * 1000.0
    if probe is None:
        raise RuntimeError("FAIL-LOUD: embedder.embed() returned None for probe query.")
    cfg = getattr(embedder, "_config", None)
    model_name = (
        getattr(cfg, "model_name", None)
        or getattr(cfg, "ollama_model", None)
        or getattr(embedder, "_model", None)
        or "unknown"
    )
    return {
        "provider_class": provider_class,
        "model_name": model_name,
        "dimension": len(probe) if hasattr(probe, "__len__") else "unknown",
        "warmup_ms": round(warmup_ms, 1),
        "available": available,
    }


def build_query_pool(db_path: Path, profile_id: str, n: int, rng: random.Random) -> list[str]:
    """Real fact contents from DB as recall queries (exercises real retrieval)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT content FROM atomic_facts "
            "WHERE profile_id=? AND length(content)>20 AND length(content)<300 ORDER BY fact_id",
            (profile_id,),
        ).fetchall()
    finally:
        conn.close()
    texts = [r[0] for r in rows] if rows else [f"AI Reliability Engineering query {i}" for i in range(n)]
    while len(texts) < n:
        texts += texts
    rng.shuffle(texts)
    return texts[:n]


def build_write_corpus(n: int, rng: random.Random) -> list[str]:
    """Varied content strings for store_fast() writes."""
    templates = [
        "Project update: {t} completed {d} with {o}.",
        "User preference: use {tool} for {task} in {ctx}.",
        "Decision: {ch} over {alt} — reason: {r}.",
        "Meeting note: discussed {t} with {p}; action: {a}.",
        "Technical: {sys} requires {req} to work.",
        "Research: {paper} shows {res} for {dom}.",
    ]
    topics = ["SLM V4", "AI Reliability Engineering", "recall latency", "BM25",
              "SQLite WAL", "Ollama", "Hopfield", "Ebbinghaus", "entity graph"]
    corpus = []
    for i in range(n):
        tmpl = rng.choice(templates)
        text = tmpl.format(
            t=rng.choice(topics), d=f"2026-{rng.randint(1,8):02d}-{rng.randint(1,28):02d}",
            o=rng.choice(["success", "partial", "blocked"]),
            tool=rng.choice(["Python", "SQLite", "Ollama"]),
            task=rng.choice(["benchmarking", "recall", "storage"]),
            ctx=rng.choice(["production", "research", "dev"]),
            ch=rng.choice(["BM25", "WAL", "Ollama"]), alt=rng.choice(["TF-IDF", "rollback"]),
            r=rng.choice(["performance", "reliability"]), p=rng.choice(["Varun", "team"]),
            a=rng.choice(["benchmark", "review", "test"]),
            sys=rng.choice(["SLM", "SQLite"]), req=rng.choice(["Python 3.13", "WAL mode"]),
            paper=rng.choice(["SLM-paper", "QJL 2025"]),
            res=rng.choice(["improved", "maintained"]), dom=rng.choice(["RAG", "memory"]),
        )
        corpus.append(f"bench_perf write {i}: {text}")
    return corpus


def section_latency(engine: Any, queries: list[str], writes: list[str],
                    n: int, profile_id: str) -> dict:
    """N recall() + N store_fast() with warmup (CRIT-1) and GC disabled (CRIT-2)."""
    print("\n[LATENCY] Warming recall path...", flush=True)
    for i in range(N_WARMUP):
        engine.recall(queries[i % len(queries)], profile_id=profile_id, fast=True)

    print(f"[LATENCY] Timing {n} recall() calls (GC disabled)...", flush=True)
    recall_ms: list[float] = []
    gc.disable()
    try:
        for i in range(n):
            t0 = time.monotonic()
            engine.recall(queries[i % len(queries)], profile_id=profile_id, fast=True)
            recall_ms.append((time.monotonic() - t0) * 1000.0)
    finally:
        gc.collect(); gc.enable()

    print("[LATENCY] Warming store_fast() path...", flush=True)
    for i in range(N_WARMUP):
        engine.store_fast(writes[i % len(writes)])

    print(f"[LATENCY] Timing {n} store_fast() calls (GC disabled)...", flush=True)
    store_ms: list[float] = []
    gc.disable()
    try:
        for i in range(n):
            t0 = time.monotonic()
            engine.store_fast(writes[i % len(writes)])
            store_ms.append((time.monotonic() - t0) * 1000.0)
    finally:
        gc.collect(); gc.enable()

    return {
        "recall": {"raw_ms": [round(v, 3) for v in recall_ms], **_summarise(recall_ms)},
        "store_fast": {"raw_ms": [round(v, 3) for v in store_ms], **_summarise(store_ms)},
        "gc_disabled_for": ["recall_loop", "store_fast_loop"],
        "warmup_ops": N_WARMUP,
    }


def _worker(engine, queries, writes, profile_id, k_ops, wid, results, errors, seed):
    rng = random.Random(seed)
    lats: list[float] = []
    errs = 0
    try:
        for _ in range(k_ops):
            if rng.random() < 0.5:
                content = writes[rng.randint(0, len(writes) - 1)]
                t0 = time.monotonic()
                try:
                    engine.store_fast(content)
                except Exception as e:
                    if "database is locked" in str(e).lower():
                        errs += 1
                    else:
                        raise
            else:
                q = queries[rng.randint(0, len(queries) - 1)]
                t0 = time.monotonic()
                try:
                    engine.recall(q, profile_id=profile_id, fast=True)
                except Exception as e:
                    if "database is locked" in str(e).lower():
                        errs += 1
                    else:
                        raise
            lats.append((time.monotonic() - t0) * 1000.0)
    except Exception as e:
        results.append({"error": str(e), "latencies": lats})
        errors.append(errs)
        return
    results.append({"latencies": lats})
    errors.append(errs)


def section_concurrency(engine, queries, writes, profile_id, k_ops=30) -> dict:
    """Sweep [1,2,4,8,12,16] workers; record throughput, latency, lock errors."""
    import concurrent.futures
    sweep = []
    for n_workers in [1, 2, 4, 8, 12, 16]:
        print(f"[CONCURRENCY] {n_workers} workers × {k_ops} ops...", flush=True)
        buckets: list[list] = [[] for _ in range(n_workers)]
        errs: list[list] = [[] for _ in range(n_workers)]
        gc.disable()
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_worker, engine, queries, writes, profile_id, k_ops,
                                w, buckets[w], errs[w], SEED + w + n_workers * 1000)
                    for w in range(n_workers)]
            concurrent.futures.wait(futs)
        elapsed = time.monotonic() - t0
        gc.collect(); gc.enable()
        for w, f in enumerate(futs):
            if f.exception():
                raise RuntimeError(f"Worker {w}: {f.exception()}") from f.exception()
        all_lats: list[float] = []
        lock_errs = 0
        for b, e in zip(buckets, errs):
            if b and "error" in b[0]:
                raise RuntimeError(f"Worker error at n_workers={n_workers}: {b[0]['error']}")
            elif b:
                all_lats.extend(b[0]["latencies"])
            lock_errs += sum(e)
        total_ops = len(all_lats)
        tps = total_ops / elapsed if elapsed > 0 else 0.0
        row = {
            "n_workers": n_workers, "k_ops_per_worker": k_ops, "total_ops": total_ops,
            "elapsed_s": round(elapsed, 3), "throughput_ops_s": round(tps, 1),
            "median_latency_ms": round(_pct(all_lats, 50), 3) if all_lats else None,
            "p99_latency_ms": round(_pct(all_lats, 99), 3) if all_lats else None,
            "lock_errors": lock_errs,
        }
        sweep.append(row)
        flag = "OK" if lock_errs == 0 else f"LOCK ERRORS: {lock_errs}"
        print(f"  → {tps:.1f} ops/s p50={row['median_latency_ms']}ms "
              f"lock_errors={lock_errs} [{flag}]", flush=True)
    return {
        "sweep": sweep, "k_ops_per_worker": k_ops,
        "concurrency_model": (
            "ThreadPoolExecutor (GIL-shared threads). Writes serialised by "
            "DatabaseManager.get_write_lock() (process-wide RLock) + SQLite WAL. CRIT-4."
        ),
    }


def section_rss(engine, queries, writes, profile_id, duration_s, interval=0.5) -> dict:
    """Sample RSS every interval seconds during sustained mixed load."""
    try:
        import psutil
    except ImportError:
        return {"error": "psutil not available — pip install psutil", "samples": []}
    proc = psutil.Process(os.getpid())
    samples: list[dict] = []
    load_errors: list[str] = []
    stop = threading.Event()

    def _load():
        rng = random.Random(SEED + 99999)
        i = 0
        while not stop.is_set():
            try:
                if rng.random() < 0.4:
                    engine.store_fast(writes[i % len(writes)])
                else:
                    engine.recall(queries[i % len(queries)], profile_id=profile_id, fast=True)
            except Exception as e:
                load_errors.append(str(e))
            i += 1

    t0 = time.monotonic()
    t = threading.Thread(target=_load, daemon=True)
    t.start()
    try:
        while (time.monotonic() - t0) < duration_s:
            samples.append({"t_s": round(time.monotonic() - t0, 2),
                             "rss_mb": round(proc.memory_info().rss / 1048576, 1)})
            time.sleep(interval)
    finally:
        stop.set()
        t.join(timeout=5.0)
    vals = [s["rss_mb"] for s in samples]
    spread = max(vals) - min(vals) if vals else 0
    return {
        "samples": samples, "duration_s": duration_s, "sample_interval_s": interval,
        "rss_min_mb": round(min(vals), 1) if vals else None,
        "rss_max_mb": round(max(vals), 1) if vals else None,
        "rss_final_mb": round(vals[-1], 1) if vals else None,
        "load_errors": load_errors[:20],
        "leak_indicator": (
            f"NONE (spread={spread:.0f}MB <= 50MB)" if spread <= 50
            else f"WARN: RSS spread {spread:.0f}MB during {duration_s}s load"
        ) if vals else "N/A",
    }


def _environment(db_path: Path, emb: dict, config: Any) -> dict:
    import superlocalmemory
    stored_models: list[str] = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        stored_models = [r[0] for r in conn.execute(
            "SELECT DISTINCT model_name FROM embedding_metadata ORDER BY model_name"
        ).fetchall() if r[0]]
        conn.close()
    except Exception:
        pass
    return {
        "package": "superlocalmemory",
        "version": getattr(superlocalmemory, "__version__", "unknown"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "timestamp": datetime.now(UTC).isoformat(),
        "db_path": str(db_path),
        "db_size_bytes": db_path.stat().st_size if db_path.exists() else 0,
        "embedding_mode": emb.get("provider_class", "unknown"),
        "embedding_model_ollama_tag": emb.get("model_name", "unknown"),
        "embedding_model_stored_in_db": stored_models,
        "embedding_dimension": emb.get("dimension", "unknown"),
        "embedder_warmup_ms": emb.get("warmup_ms"),
        "gc_disabled_during": ["recall_loop", "store_fast_loop", "each_concurrency_sweep_point"],
        "warmup_ops_before_timing": N_WARMUP,
        "config_mode": getattr(getattr(config, "mode", None), "value", str(config.mode)),
        "profile_id": getattr(config, "active_profile", "unknown"),
    }


def _print_summary(env: dict, lat: dict, conc: dict, rss: dict) -> None:
    stored = "/".join(env.get("embedding_model_stored_in_db", []))
    print("\n" + "=" * 70)
    print("SuperLocalMemory V4 — Performance Benchmark Summary")
    print("=" * 70)
    print(f"Version       : {env['version']}")
    print(f"Python        : {env['python']}")
    print(f"Platform      : {env['platform']}")
    print(f"Embedding     : {env['embedding_mode']} / {env['embedding_model_ollama_tag']} "
          f"({env['embedding_dimension']}d)" + (f" [stored: {stored}]" if stored else ""))
    print(f"Embedder init : {env['embedder_warmup_ms']} ms (warm-up, excluded from latency timing)")
    print(f"DB size       : {env['db_size_bytes'] // 1024 // 1024} MB")
    print(f"Profile       : {env['profile_id']}")
    print()
    for sec, label in (("recall", "RECALL"), ("store_fast", "STORE_FAST")):
        s = lat[sec]
        print(f"{label} LATENCY (warm, GC disabled)")
        print(f"  n={s['n']}  p50={s['p50_ms']}ms  p90={s['p90_ms']}ms  "
              f"p95={s['p95_ms']}ms  p99={s['p99_ms']}ms  max={s['max_ms']}ms")
    print()
    print("CONCURRENCY SWEEP:")
    print(f"  {'Workers':>8}  {'Ops/s':>10}  {'p50 ms':>8}  {'p99 ms':>8}  {'Lock Errs':>10}")
    for row in conc["sweep"]:
        flag = " OK" if row["lock_errors"] == 0 else f" FAIL:{row['lock_errors']}"
        print(f"  {row['n_workers']:>8}  {row['throughput_ops_s']:>10.1f}  "
              f"{str(row['median_latency_ms']):>8}  {str(row['p99_latency_ms']):>8}  "
              f"{row['lock_errors']:>10}{flag}")
    print()
    if rss.get("samples"):
        print(f"RSS ({rss['duration_s']}s sustained load): "
              f"min={rss['rss_min_mb']} MB  max={rss['rss_max_mb']} MB  "
              f"final={rss['rss_final_mb']} MB")
        print(f"  Leak indicator: {rss['leak_indicator']}")
    else:
        print(f"RSS: {rss.get('error', 'no data')}")
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description="SLM V4 performance benchmark")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--db", type=str, default=str(_REAL_DB_REF))
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--profile", type=str, default="default")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n = 5; args.duration = 10
        print("SMOKE MODE: n=5, duration=10s")

    source_db = Path(args.db)
    run_ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    bench_dir = _BENCH_BASE / run_ts

    print(f"\nSuperLocalMemory V4 Performance Benchmark — {run_ts}")
    print(f"Source DB  : {source_db}")
    print(f"Bench dir  : {bench_dir}")
    print(f"N          : {args.n}  Duration: {args.duration}s")

    print("\n[SETUP] Preparing bench DB copy...")
    bench_db = prepare_bench_dir(source_db, bench_dir)

    print("[SETUP] Loading config + initialising engine...")
    config = _load_bench_config(bench_dir, args.profile)
    t0 = time.monotonic()
    engine = init_engine(config)
    print(f"  Engine initialised in {(time.monotonic()-t0)*1000:.0f}ms")

    print("[SETUP] Detecting and warming embedding path (CRIT-3)...")
    emb_info = detect_and_warm_embedder(engine)
    print(f"  Embedding: {emb_info['provider_class']} / {emb_info['model_name']} "
          f"({emb_info['dimension']}d) — warmup={emb_info['warmup_ms']}ms")

    print("[SETUP] Building query pool and write corpus...")
    rng = random.Random(SEED)
    pool_size = max(args.n + N_WARMUP, 600)
    queries = build_query_pool(bench_db, args.profile, pool_size, rng)
    writes = build_write_corpus(pool_size, rng)
    print(f"  {len(queries)} queries, {len(writes)} write items")

    print("\n[SECTION 1/3] Latency...")
    lat = section_latency(engine, queries, writes, args.n, args.profile)

    k_ops = max(args.n // 10, 5) if not args.smoke else 3
    print(f"\n[SECTION 2/3] Concurrency (k_ops={k_ops}/worker)...")
    conc = section_concurrency(engine, queries, writes, args.profile, k_ops=k_ops)

    print(f"\n[SECTION 3/3] RSS ({args.duration}s)...")
    rss = section_rss(engine, queries, writes, args.profile, args.duration)

    env = _environment(bench_db, emb_info, config)
    output = {
        "environment": env,
        "crit_findings": {
            "CRIT-1": (
                "Cold-cache vs warm: N_WARMUP=5 ops before every timed section. "
                "Warms SQLite page cache and Ollama HTTP connection pool. Warm path only is measured."
            ),
            "CRIT-2": (
                "GC pauses: gc.disable() before every timed loop; gc.collect()+gc.enable() after. "
                "Applied to recall, store_fast, and each concurrency sweep point."
            ),
            "CRIT-3": (
                f"Embedding model load: first embed() took {emb_info['warmup_ms']}ms "
                f"(provider={emb_info['provider_class']}, model={emb_info['model_name']}). "
                "Excluded from all latency measurements. Recorded in environment.embedder_warmup_ms."
            ),
            "CRIT-4": (
                "Thread/GIL semantics: ThreadPoolExecutor = GIL-shared threads. "
                "I/O-bound paths release GIL; writes serialise on process-wide RLock + SQLite WAL. "
                "This IS the real engine concurrency model. Not 'fixed' — documented."
            ),
        },
        "latency": lat,
        "concurrency": conc,
        "rss": rss,
    }

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = _RESULTS_DIR / "bench_perf.json"
    result_path.write_text(json.dumps(output, indent=2, sort_keys=False), encoding="utf-8")
    print(f"\n[OUTPUT] JSON written: {result_path}")

    _print_summary(env, lat, conc, rss)

    total_lock_errs = sum(r["lock_errors"] for r in conc["sweep"])
    if total_lock_errs > 0:
        print(f"\nWARNING: {total_lock_errs} database-lock errors in concurrency sweep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
