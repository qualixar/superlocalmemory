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
import time
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
# A heal that keeps failing (no model on disk, no network) respawns an embedding
# worker each time and cannot succeed. Failed heals back off: 300 s, 600 s, ...
# and stop after MAX_HEAL_ATTEMPTS until a healthy tick or a restart.
HEAL_BACKOFF_BASE_S = 300
MAX_HEAL_ATTEMPTS = 4


@dataclass
class RecallHealth:
    """Mutable health state for the recall path."""

    healthy: bool = True
    consecutive_failures: int = 0
    total_heals: int = 0
    checks: int = 0
    last_semantic_score: float = 0.0
    last_error: str = ""
    #: When the last tick finished, as a unix timestamp. A tick that finds
    #: nothing wrong logs nothing, which is correct -- a monitor that narrates
    #: every success is a monitor whose real warnings get skimmed past. But it
    #: left no way to tell a monitor that is ticking quietly from a thread that
    #: died or never started, and that ambiguity cost someone an hour of looking
    #: for log lines that were never going to appear. So the fact of the tick is
    #: recorded here and surfaced on /health, where it can be checked instead of
    #: inferred.
    last_tick_at: float = 0.0
    #: Whether the embedder could produce a vector at the last tick.
    embedder_alive: bool = True
    #: The cross-encoder reranker. Not watched before 4.1.18: on a real daemon
    #: it died and stayed dead for seven days while /health reported
    #: recall_healthy. ``reranker_ready`` is None when the configured reranker
    #: has no local model to judge (RemoteReranker).
    reranker_configured: bool = False
    reranker_ready: bool | None = None
    reranker_rearms: int = 0
    #: Self-heal backoff (see HEAL_BACKOFF_BASE_S). ``next_heal_at`` is a unix
    #: timestamp; ``heal_stopped`` stays set until a healthy tick.
    heal_failures_in_row: int = 0
    next_heal_at: float = 0.0
    heal_stopped: bool = False


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


def _get_reranker(engine):
    """The retrieval engine's reranker, or None when none is configured."""
    re_eng = getattr(engine, "_retrieval_engine", None)
    return getattr(re_eng, "_reranker", None) if re_eng is not None else None


def _watch_reranker(engine, state: RecallHealth, *, log) -> None:
    """Report the reranker, and re-arm its warm-up if its model is gone.

    Every way a reranker worker dies — the 500-request recycle, the 30-minute
    idle kill, a crash — leaves no model loaded. The reranker re-warms itself
    when a recall reaches it; this is a second path that does not depend on
    recall traffic, so a quiet daemon recovers too. Starting the warm-up is
    idempotent and never blocks.
    """
    reranker = _get_reranker(engine)
    state.reranker_configured = reranker is not None
    start = getattr(reranker, "_start_background_warmup", None)
    if reranker is None or not callable(start) or not hasattr(reranker, "_model_loaded"):
        state.reranker_ready = None
        return
    state.reranker_ready = bool(reranker._model_loaded)
    if state.reranker_ready or getattr(reranker, "_worker_loading", False):
        return
    try:
        started = start()
    except Exception as exc:
        log.critical("recall-health: could not re-arm the reranker warm-up: %s", exc)
        return
    if started is False:
        return  # refused: the reranker's own backoff decides when to retry
    state.reranker_rearms += 1
    log.warning(
        "recall-health: reranker had no model loaded — re-armed its warm-up "
        "(re-arm #%d)", state.reranker_rearms,
    )


def _embedder_produces_vector(engine) -> bool:
    """Ask the embedder for one vector. True only if it returns a non-empty one."""
    emb = _get_embedder(engine)
    if emb is None:
        return False
    try:
        vec = emb.embed(HEAL_PROBE)
    except Exception:
        return False
    try:
        return vec is not None and len(vec) > 0
    except TypeError:
        return bool(vec)


def _embedder_is_dead(engine) -> bool:
    """Can the embedder produce a vector right now?

    Asked directly, because it cannot be inferred from a recall. The monitor
    used to decide the embedder was fine whenever the probe recall came back
    with no results at all -- and a dead embedder is one of the reasons a recall
    comes back with no results, so the one symptom that should have triggered a
    heal was read as proof that none was needed.

    That is not hypothetical. An idle-timeout kill leaves no worker; the next
    probe finds nothing by meaning, finds nothing by keyword either because the
    probe phrase appears in nobody's memories, and returns zero results. The
    monitor then recorded "healthy", logged nothing, and never respawned the
    worker -- so ``readiness.embedding`` stayed false and the daemon sat in
    ``warming`` until someone restarted it by hand, with not one line in the log
    to say why.

    Fails safe in the opposite direction from before: an embedder this cannot
    reach is reported dead, so the worst case is one unnecessary re-warm rather
    than a silent outage.
    """
    emb = _get_embedder(engine)
    if emb is None:
        return False  # BM25-only by configuration; nothing to heal.
    warm = getattr(emb, "is_warm", None)
    if warm is not None and not warm:
        return True
    if getattr(emb, "_available", True) is False:
        return True
    return False


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


def _reset_heal(state: RecallHealth) -> None:
    """Forget failed heals: the embedder works again."""
    state.heal_failures_in_row = 0
    state.next_heal_at = 0.0
    state.heal_stopped = False


def _note_heal_failed(state: RecallHealth, log) -> None:
    """Count a failed heal, schedule the next one, stop after the cap."""
    state.heal_failures_in_row += 1
    k = state.heal_failures_in_row
    state.next_heal_at = time.time() + HEAL_BACKOFF_BASE_S * 2 ** (k - 1)
    if k >= MAX_HEAL_ATTEMPTS:
        state.heal_stopped = True
        log.warning(
            "recall-health: embedder self-heal stopped after %d failed "
            "attempts; run 'slm warmup' or restart the daemon", k,
        )


def _heal_paused(state: RecallHealth) -> bool:
    return state.heal_stopped or time.time() < state.next_heal_at


def _tier3_heal(engine, state: RecallHealth, dead: bool, n_results: int,
                log) -> RecallHealth:
    """Tier 3: self-heal, unless the backoff says a heal cannot help yet."""
    if _heal_paused(state):
        state.healthy = False
        state.consecutive_failures += 1
        state.last_error = (
            "embedder self-heal stopped after repeated failures"
            if state.heal_stopped else "embedder self-heal paused (backing off)"
        )
        log.debug("recall-health: %s", state.last_error)
        return state
    # The dead-embedder case already said so above; saying "semantic channel
    # DEAD (max semantic=0.0)" as well would be a second, differently-worded
    # CRITICAL about the same tick.
    if not dead:
        log.critical(
            "recall-health: semantic channel DEAD (%d results, max semantic=0.0) "
            "— embedder returning None; attempting self-heal",
            n_results,
        )
    if _heal_embedder(engine, log=log):
        state.total_heals += 1
        state.healthy = True
        state.consecutive_failures = 0
        state.last_error = ""
        _reset_heal(state)
        log.warning("recall-health: embedder self-heal SUCCEEDED (re-warmed)")
        return state
    state.healthy = False
    state.consecutive_failures += 1
    state.last_error = "semantic channel dead; embedder heal failed"
    _note_heal_failed(state, log)
    log.critical(
        "recall-health: self-heal FAILED — recall DEGRADED to keyword-only "
        "(consecutive_failures=%d)", state.consecutive_failures,
    )
    return state


def run_health_tick(engine, state: RecallHealth, *, probe: str = DEFAULT_PROBE,
                    log=logger, runtime=None) -> RecallHealth:
    """One monitor tick: re-warm (Tier 1), probe (Tier 2), self-heal (Tier 3).

    Mutates and returns ``state``. Never raises — a timed-out / failing recall
    marks the path unhealthy instead of propagating.

    Uses ``operation_nowait()`` when a runtime is supplied so that a pending
    profile switch is not blocked by the health-probe recall.  If a transition
    is in progress the tick is skipped entirely — the next scheduled tick will
    re-warm once the switch has committed.
    """
    state.checks += 1
    _watch_reranker(engine, state, log=log)

    # Tier 1: re-warm. A real full-fusion recall keeps the graph page cache hot
    # and the embedder resident.
    # Cooperative preemption: use operation_nowait() so a pending profile switch
    # is not held hostage by a slow full-fusion recall (2–10s with fast=False).
    try:
        if runtime is not None:
            lease = runtime.operation_nowait()
        else:
            lease = nullcontext()
        with lease as _snap:
            if runtime is not None and _snap is None:
                # A profile transition is in progress — skip this tick so we
                # do not hold the drain window.  Health state is unchanged;
                # the next tick fires after the switch commits.
                log.debug(
                    "recall-health: tick skipped — profile transition in progress"
                )
                return state
            # fast=True: a health probe must release its operation lease well
            # within the 5s profile-switch drain window (fast=False is 2-10s and
            # would make every profile switch time out while a tick is in flight).
            from superlocalmemory.core.recall_gate import background_work
            with background_work():
                resp = engine.recall(probe, limit=3, fast=True)
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
    state.last_tick_at = time.time()

    # Tier 2: readiness.
    #
    #   * the embedder cannot produce a vector   -> dead, whatever the recall said
    #   * rows present but none scored by meaning -> ask the embedder directly
    #
    # The first used to be missing, and its absence was load-bearing: zero
    # results was treated as healthy, so the case where the embedder is dead AND
    # the probe matches nothing by keyword -- the normal shape of an idle-timeout
    # kill -- came out as healthy, silently.
    #
    # The second used to be read as an outage on its own. 4.1.18: it is the
    # ordinary outcome for a working embedder -- the probe phrase is content-free
    # and only its top three results come back, so none of them need carry a
    # semantic score. A live daemon logged "semantic channel DEAD" on 1,970
    # consecutive ticks that way while its real queries scored semantic 1.2. So
    # the embedder is asked for a vector, and only a failure counts.
    #
    # What neither signature can catch, and the old one never did either: a
    # vector index that is broken while the embedder still works. That needs a
    # probe whose answer is known to be stored. The old check fired on every
    # tick, so it would have looked exactly the same during a real index failure.
    dead = _embedder_is_dead(engine)
    state.embedder_alive = not dead
    semantic_silent = bool(results) and sem <= 0.0
    if semantic_silent and not dead and _embedder_produces_vector(engine):
        semantic_silent = False
    broken = dead or semantic_silent
    if dead and not _heal_paused(state):
        log.critical(
            "recall-health: embedder cannot produce a vector (%d probe results) "
            "— attempting self-heal",
            len(results),
        )
    if not broken:
        if not state.healthy:
            log.warning(
                "recall-health: RECOVERED (semantic=%.3f, %d results)",
                sem, len(results),
            )
        state.healthy = True
        state.consecutive_failures = 0
        state.last_error = ""
        _reset_heal(state)
        return state

    return _tier3_heal(engine, state, dead, len(results), log)


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
    now = time.time()
    return {
        "recall_healthy": s.healthy,
        "consecutive_failures": s.consecutive_failures,
        "total_heals": s.total_heals,
        "checks": s.checks,
        "last_semantic_score": round(s.last_semantic_score, 4),
        "last_error": s.last_error,
        "reranker_configured": s.reranker_configured,
        "reranker_ready": s.reranker_ready,
        "reranker_rearms": s.reranker_rearms,
        # Proof of life. A tick that finds nothing wrong logs nothing, so there
        # was no way to tell this monitor apart from a thread that never started
        # -- someone spent an hour reading logs for lines that were never going
        # to be written. These two answer that without needing the log at all:
        # if seconds_since_last_tick keeps climbing past the interval, the thread
        # is gone.
        "last_tick_at": round(s.last_tick_at, 3) if s.last_tick_at else None,
        "seconds_since_last_tick": (
            round(now - s.last_tick_at, 1) if s.last_tick_at else None
        ),
        "embedder_alive": s.embedder_alive,
        "embedder_heal_stopped": s.heal_stopped,
        "embedder_heal_retry_in_s": (
            None if s.heal_stopped else round(max(0.0, s.next_heal_at - now), 1)
        ),
    }
