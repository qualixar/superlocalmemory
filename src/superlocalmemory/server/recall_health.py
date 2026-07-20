"""Runtime recall-health monitor — keep the full recall path warm,
detect a "warm-but-broken" embedder at runtime, and self-heal it (v3.6.8).

Why this exists
---------------
On a long-running daemon the cold full-fusion recall could exceed the MCP
pool's 30s timeout, and ``session_init`` then silently fell back to FTS5/BM25
("DEGRADED MODE") — observed 7×/2 days in production logs. Two root causes:

1. **Page-cache eviction.** The ~100 MB ``association_edges`` / graph table is
   evicted under memory pressure, so the first recall after idle re-reads it
   from disk (15-24 s+), blowing the pool timeout.
2. **Silent embedder death.** ``OllamaEmbedder.embed`` returns ``None`` on a
   transient Ollama failure, while the boot-set ``_embedding_warm`` flag still
   reports ``True``. With ``q_emb is None`` the engine skips the semantic,
   hopfield and spreading_activation channels — recall silently degrades to
   keyword-only (``semantic`` score ``0.0`` on every result).

The boot warmup (``_warmup_recall``) runs ONCE and never again, so neither
condition is repaired in-life.

The fix — an industry-standard 3-tier monitor, validated against Ollama
keep-alive, Chroma's active heartbeat, LangChain's circuit breaker and the
Kubernetes liveness/readiness split:

* **Tier 1 — RE-WARM.** Fire a real ``engine.recall`` every ``interval_s`` to
  keep the graph page cache hot and nomic-embed resident.
* **Tier 2 — READINESS PROBE.** Assert the semantic channel actually fired
  (``max semantic > 0``). Rows returned with ``semantic == 0`` everywhere is the
  warm-but-broken signature → the embedder is returning ``None``.
* **Tier 3 — CIRCUIT-BREAKER SELF-HEAL.** Reset the embedder's cached
  ``_available`` flag (the "available once, cached forever" bug), re-exercise
  ``embed()``, track consecutive failures, and log **CRITICAL** so the
  degradation is never silent.

All logic is fail-soft: a crash in the monitor never takes down the daemon.
"""
from __future__ import annotations

import logging
import threading
from contextlib import nullcontext
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# A stable probe query. Content-agnostic: we only assert that *some* result
# carries a non-zero semantic score, which proves the embedder + semantic
# channel are alive end-to-end.
DEFAULT_PROBE = "memory recall health probe"
# A distinct string for the heal embed() call so it is obvious in thread dumps
# / Ollama logs and never collides with a real cached query embedding.
HEAL_PROBE = "__recall_health_rewarm__"

# Default cadence. 5 min keeps nomic-embed resident (Ollama default unload is
# 5 min) and the page cache warm without meaningful load.
DEFAULT_INTERVAL_S = 300


@dataclass
class RecallHealth:
    """Mutable health state for the recall path."""

    healthy: bool = True
    consecutive_failures: int = 0
    total_heals: int = 0
    checks: int = 0
    last_semantic_score: float = 0.0
    last_error: str = ""


def _max_semantic(results) -> float:
    """Largest semantic channel score across results (0.0 if none)."""
    best = 0.0
    for r in results:
        cs = getattr(r, "channel_scores", None) or {}
        try:
            best = max(best, float(cs.get("semantic", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
    return best


def _get_embedder(engine):
    """Locate the engine's embedder (full mode: engine._embedder; some paths
    hang it off the retrieval engine)."""
    emb = getattr(engine, "_embedder", None)
    if emb is None:
        re_eng = getattr(engine, "_retrieval_engine", None)
        emb = getattr(re_eng, "_embedder", None) if re_eng is not None else None
    return emb


def _heal_embedder(engine, *, log) -> bool:
    """Tier 3: reset the cached availability flag and re-exercise the embedder.

    ``OllamaEmbedder.is_available`` caches its first result forever, so once
    Ollama blips the flag can stay stale. Clearing ``_available`` forces a
    re-probe; ``embed()`` itself always re-attempts the HTTP call, so a single
    successful embed proves recovery. Returns True iff the embedder produced a
    vector.
    """
    emb = _get_embedder(engine)
    if emb is None:
        log.critical("recall-health: no embedder on engine — cannot self-heal")
        return False
    # Reset the "available once, cached forever" flag if present.
    if hasattr(emb, "_available"):
        try:
            emb._available = None
        except Exception:  # pragma: no cover - defensive
            pass
    try:
        vec = emb.embed(HEAL_PROBE)
    except Exception as exc:
        log.warning("recall-health: heal embed() raised: %s", exc)
        return False
    return vec is not None and bool(vec)


def run_health_tick(engine, state: RecallHealth, *, probe: str = DEFAULT_PROBE,
                    log=logger, runtime=None) -> RecallHealth:
    """One monitor tick: re-warm (Tier 1), probe (Tier 2), self-heal (Tier 3).

    Mutates and returns ``state``. Never raises — a timed-out / failing recall
    marks the path unhealthy instead of propagating.
    """
    state.checks += 1

    # Tier 1: re-warm. A real full-fusion recall keeps the graph page cache hot
    # and the embedder resident.
    try:
        lease = runtime.operation() if runtime is not None else nullcontext()
        with lease:
            resp = engine.recall(probe, limit=3, fast=False)
    except Exception as exc:
        state.healthy = False
        state.consecutive_failures += 1
        state.last_error = f"recall raised: {exc}"
        log.critical(
            "recall-health: re-warm recall FAILED (%s) — recall path unhealthy",
            exc,
        )
        return state

    results = list(getattr(resp, "results", []) or [])
    sem = _max_semantic(results)
    state.last_semantic_score = sem

    # Tier 2: readiness. Rows present but semantic never fired == warm-but-broken.
    # Zero results is NOT this signature (could be an empty/filtered corpus).
    broken = bool(results) and sem <= 0.0
    if not broken:
        if not state.healthy:
            log.warning(
                "recall-health: RECOVERED (semantic=%.3f, %d results)",
                sem, len(results),
            )
        state.healthy = True
        state.consecutive_failures = 0
        state.last_error = ""
        return state

    # Tier 3: self-heal.
    log.critical(
        "recall-health: semantic channel DEAD (%d results, max semantic=0.0) "
        "— embedder returning None; attempting self-heal",
        len(results),
    )
    if _heal_embedder(engine, log=log):
        state.total_heals += 1
        state.healthy = True
        state.consecutive_failures = 0
        state.last_error = ""
        log.warning("recall-health: embedder self-heal SUCCEEDED (re-warmed)")
    else:
        state.healthy = False
        state.consecutive_failures += 1
        state.last_error = "semantic channel dead; embedder heal failed"
        log.critical(
            "recall-health: self-heal FAILED — recall DEGRADED to keyword-only "
            "(consecutive_failures=%d)", state.consecutive_failures,
        )
    return state


def health_monitor_loop(engine, *, interval_s: int, stop_event: threading.Event,
                        state: RecallHealth, probe: str = DEFAULT_PROBE,
                        log=logger, runtime=None) -> None:
    """Background loop. Sleeps ``interval_s`` between ticks; exits promptly when
    ``stop_event`` is set. An initial short delay avoids racing boot warmup."""
    # Initial delay (bounded) so we don't pile onto the boot warmup threads.
    if stop_event.wait(min(interval_s, 60)):
        return
    while not stop_event.is_set():
        try:
            run_health_tick(
                engine, state, probe=probe, log=log, runtime=runtime,
            )
        except Exception as exc:  # pragma: no cover - belt & suspenders
            log.warning("recall-health: tick crashed (non-fatal): %s", exc)
        if stop_event.wait(interval_s):
            break


# Module-level state so /health can surface the latest verdict without holding
# a reference to the thread.
_GLOBAL_STATE = RecallHealth()


def start_recall_health_monitor(engine, *, interval_s: int = DEFAULT_INTERVAL_S,
                                probe: str = DEFAULT_PROBE, log=None,
                                runtime=None):
    """Start the monitor as a daemon thread. Returns ``(thread, stop_event,
    state)``. The state is the shared module-level state read by
    :func:`get_recall_health`."""
    log = log or logger
    state = _GLOBAL_STATE
    stop = threading.Event()
    t = threading.Thread(
        target=health_monitor_loop,
        kwargs=dict(
            engine=engine, interval_s=interval_s, stop_event=stop,
            state=state, probe=probe, log=log, runtime=runtime,
        ),
        daemon=True,
        name="recall-health",
    )
    t.start()
    return t, stop, state


def get_recall_health() -> dict:
    """Snapshot for /health surfacing (visibility — never silent degradation)."""
    s = _GLOBAL_STATE
    return {
        "recall_healthy": s.healthy,
        "consecutive_failures": s.consecutive_failures,
        "total_heals": s.total_heals,
        "checks": s.checks,
        "last_semantic_score": round(s.last_semantic_score, 4),
        "last_error": s.last_error,
    }
