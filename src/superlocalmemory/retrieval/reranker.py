# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Cross-Encoder Reranker (Subprocess-Isolated).

V3.3.3: All PyTorch/ONNX model work runs in a SEPARATE subprocess.
The main process (dashboard, MCP, CLI) NEVER imports torch and stays
at ~60 MB. Same isolation pattern as EmbeddingService.

The worker subprocess auto-kills after 2 minutes idle.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import sys
import threading
import time
import weakref
from pathlib import Path
from typing import Any

from superlocalmemory.infra.data_root import state_path
from superlocalmemory.retrieval._worker_process import (
    readline_with_timeout,
    stop_process,
)
from superlocalmemory.storage.models import AtomicFact

_RERANKER_PID_FILE = None  # test-only override


def _reranker_pid_file() -> Path:
    return _RERANKER_PID_FILE or state_path(".reranker-worker.pid")


def _is_reranker_worker_alive() -> bool:
    """Check if a reranker worker PID is already alive (machine-wide singleton)."""
    try:
        pid_file = _reranker_pid_file()
        if not pid_file.exists():
            return False
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        pid = 0  # PID file unreadable or invalid: treat as stale
    from superlocalmemory.core.platform_utils import is_pid_alive

    if is_pid_alive(pid):
        return True
    _reranker_pid_file().unlink(missing_ok=True)
    return False

# Track all live reranker instances for atexit cleanup
_live_rerankers: set[weakref.ref] = set()

logger = logging.getLogger(__name__)

_IDLE_TIMEOUT_SECONDS = 1800  # V3.8.1: keep interactive sessions warm.
# V3.3.12: Configurable via SLM_RERANKER_IDLE_TIMEOUT env var.
# Low-RAM installations can retain aggressive recycling through the override.
# Set ``SLM_RERANKER_IDLE_TIMEOUT=120`` + ``slm restart`` to revert.
_IDLE_TIMEOUT_SECONDS = int(os.environ.get("SLM_RERANKER_IDLE_TIMEOUT", _IDLE_TIMEOUT_SECONDS))
_SUBPROCESS_RESPONSE_TIMEOUT = 15  # v3.4.52: 15s (was 180s). Long timeout blocked the
# entire FastAPI event loop — a dead reranker subprocess held ALL
# endpoints hostage for 3 minutes. 15s is enough for ONNX inference
# cold start; if the worker can't respond, we fall back to fusion
# scores without reranking.
_WORKER_RECYCLE_AFTER = 500  # Recycle after N requests
# A recall that finds the worker lock held by a recycle's swap waits up to this
# long instead of falling back to unranked order. The swap is a pointer
# exchange plus a PID-file write and an idle-timer reset: milliseconds.
_SWAP_WAIT_SECONDS = 1.0
#: How long a recall waits for the reranker while another recall is using it.
#: It used to skip reranking at once and return fusion order, so two recalls at
#: the same moment could rank the same question differently. One rerank takes
#: ~0.2 s: this lets about five queued recalls through inside the 3 s budget.
_RERANK_QUEUE_WAIT_SECONDS = 1.0

# One-time model load is far heavier than a live rerank request: the child
# process imports torch / sentence-transformers and runs a warmup inference,
# which measured 9-16s on the reference machine.  Sharing the 15s live-request
# timeout (``_SUBPROCESS_RESPONSE_TIMEOUT``) made the load a coin flip — logs
# showed ~half of daemon boots hitting "timed out after 15s", killing the
# worker, and leaving recall on FALLBACK scoring for the entire daemon
# lifetime (a silent quality regression, not a transient one).  The load gets
# its own generous budget, and the background warmup RETRIES with backoff so a
# transient slow/failed load self-heals instead of permanently degrading
# recall quality.  Live rerank requests keep the tight 15s cap so a recall
# never blocks on a sick subprocess.
_WARMUP_LOAD_TIMEOUT = int(os.environ.get("SLM_RERANKER_WARMUP_TIMEOUT", "90"))
_WARMUP_MAX_ATTEMPTS = int(os.environ.get("SLM_RERANKER_WARMUP_ATTEMPTS", "5"))
_WARMUP_RETRY_BACKOFF_S = float(os.environ.get("SLM_RERANKER_WARMUP_BACKOFF", "3"))
# A whole failed round (all attempts) is followed by a cooldown that doubles
# each round; after _WARMUP_MAX_ROUNDS failed rounds automatic retries stop.
_WARMUP_COOLDOWN_S = float(os.environ.get("SLM_RERANKER_WARMUP_COOLDOWN", "300"))
_WARMUP_MAX_ROUNDS = int(os.environ.get("SLM_RERANKER_WARMUP_ROUNDS", "4"))


# Substrings that mark a load failure as a configuration problem rather than a
# transient one. Retrying these can never succeed, so the warmup aborts on the
# first occurrence instead of spending _WARMUP_MAX_ATTEMPTS × backoff on them.
_PERMANENT_LOAD_ERROR_MARKERS = (
    "unknown backend",
    "sentence-transformers is not installed",
)


def _is_permanent_load_error(error: str) -> bool:
    """True when a worker load error cannot be fixed by retrying."""
    lowered = (error or "").lower()
    return any(m in lowered for m in _PERMANENT_LOAD_ERROR_MARKERS)


class CrossEncoderReranker:
    """Rerank candidate facts using a local cross-encoder model.

    V3.3.3: SUBPROCESS-ISOLATED. The main process never imports
    sentence_transformers or torch. All model work runs in a child
    process via JSON over stdin/stdout.

    Non-blocking first-use: triggers background worker spawn, returns
    fallback scores until worker is ready.

    Args:
        model_name: HuggingFace cross-encoder model identifier.
        backend: Inference backend. "onnx" for ONNX Runtime (light),
            "" for PyTorch (heavy). Default: "onnx".
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        backend: str = "onnx",
    ) -> None:
        self._model_name = model_name
        self._backend = backend
        self._worker_proc: subprocess.Popen | None = None
        self._model_loaded = False  # True once worker confirms model is ready
        self._worker_loading = False  # True while background warmup in progress
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        # Warm-up backoff: see _warmup_round_failed.
        self._warmup_rounds_failed = 0
        self._warmup_not_before = 0.0  # time.monotonic()
        self._warmup_stopped = False
        self._warmup_thread: threading.Thread | None = None
        self._idle_timer: threading.Timer | None = None
        self._request_count: int = 0
        # Makes "is a warm-up already running?" and "start one" one step, so
        # two cold recalls cannot each start a warm-up thread.
        self._warmup_guard = threading.Lock()
        # Recycling (see _begin_replacement). _generation changes whenever the
        # serving worker is dropped or replaced, so a replacement that finishes
        # warming after that is discarded instead of swapped in.
        self._replacing = False
        self._generation = 0
        self._recycle_thread: threading.Thread | None = None
        self._replacement_proc: subprocess.Popen | None = None
        # Set while _replace_worker holds (or is about to take) the lock for
        # the swap, so recall waits for it rather than going unranked.
        self._swap_pending = threading.Event()

        # Register for atexit cleanup (prevent orphaned workers)
        ref = weakref.ref(self, _live_rerankers.discard)
        _live_rerankers.add(ref)

        # Start background warmup immediately — worker loads model
        # while the rest of init continues. First recall gets instant
        # fallback; second recall uses the warm model.
        self._start_background_warmup()

    def __del__(self) -> None:
        """Kill worker subprocess when reranker is garbage-collected."""
        try:
            self.shutdown(timeout=0.1)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Background warmup (non-blocking model load)
    # ------------------------------------------------------------------

    def _warmup_reset(self) -> None:
        """Forget earlier failed rounds (success, or an explicit retry)."""
        self._warmup_rounds_failed = 0
        self._warmup_not_before = 0.0
        self._warmup_stopped = False

    def _warmup_round_failed(self, permanent: bool = False) -> None:
        """Account for a failed warm-up round: free the worker, back off.

        A worker that holds no model is ~800 MB of nothing, so it is killed
        under the lock. The next round may not start before a cooldown that
        doubles each round; after ``_WARMUP_MAX_ROUNDS`` rounds (or a
        permanent error) automatic retries stop.
        """
        with self._lock:
            self._kill_worker()
        rounds = getattr(self, "_warmup_rounds_failed", 0) + 1
        self._warmup_rounds_failed = rounds
        wait = _WARMUP_COOLDOWN_S * 2 ** (rounds - 1)
        self._warmup_not_before = time.monotonic() + wait
        if permanent or rounds >= _WARMUP_MAX_ROUNDS:
            self._warmup_stopped = True
            if not permanent:
                logger.warning(
                    "Reranker warmup failed %d rounds; automatic retries "
                    "stopped until the daemon restarts or 'slm warmup' runs; "
                    "recall continues with fusion scores.", rounds,
                )
            return
        logger.warning(
            "Reranker warmup round %d/%d failed; the next round may start "
            "in %.0fs (after a recall). Run 'slm doctor' for diagnostics.",
            rounds, _WARMUP_MAX_ROUNDS, wait,
        )

    def _warmup_gate_open(self) -> bool:
        """True when the backoff allows a new automatic round now."""
        if getattr(self, "_warmup_stopped", False):
            return False
        return time.monotonic() >= getattr(self, "_warmup_not_before", 0.0)

    def _start_background_warmup(self, force: bool = False) -> bool:
        """Start worker and load model in background thread.

        Returns whether a warm-up was started. Without ``force`` it refuses
        while stopped or inside the cooldown (see ``_warmup_round_failed``).

        V3.3.16: Uses _send_request (lock-protected) instead of raw
        stdin/stdout access. Previous code wrote to stdin without the
        lock, creating a race where the warmup's readline thread could
        steal responses meant for _send_request → deadlock → timeout.
        """
        guard = getattr(self, "_warmup_guard", None) or threading.Lock()
        with guard:
            if (self._shutdown_event.is_set() or self._worker_loading
                    or self._model_loaded):
                return False
            if not force and not self._warmup_gate_open():
                return False
            self._worker_loading = True

        def _warmup() -> None:
            try:
                for attempt in range(1, _WARMUP_MAX_ATTEMPTS + 1):
                    if self._shutdown_event.is_set() or self._model_loaded:
                        return
                    # Spawn under the same lock the request path holds when IT
                    # spawns (_send_request). Without it both could pass the
                    # "no live worker" check at once and each start one; the
                    # second assignment then dropped the first worker, which
                    # kept running -- a model-sized process nobody owned.
                    try:
                        with self._lock:
                            self._ensure_worker()
                    except Exception as exc:
                        logger.warning(
                            "Reranker warmup attempt %d/%d: worker spawn "
                            "raised: %s", attempt, _WARMUP_MAX_ATTEMPTS, exc,
                        )
                        with self._lock:
                            self._worker_proc = None

                    if self._worker_proc is None:
                        # Either the spawn failed, or another process already
                        # owns the machine-wide singleton worker.  If a sibling
                        # worker is alive this instance will use it on demand —
                        # stop retrying quietly rather than spinning.
                        if _is_reranker_worker_alive():
                            logger.debug(
                                "Reranker warmup: worker owned by another "
                                "process; this instance uses it on demand",
                            )
                            return
                    else:
                        # Give the ONE-TIME model load a generous budget — it is
                        # far heavier than a live rerank request. On timeout
                        # _send_request kills the worker, so the next attempt
                        # respawns cleanly (no stale-response race).
                        resp = None
                        try:
                            resp = self._send_request({
                                "cmd": "load",
                                "model_name": self._model_name,
                                "backend": self._backend,
                            }, timeout=_WARMUP_LOAD_TIMEOUT)
                        except Exception as exc:
                            logger.warning(
                                "Reranker warmup attempt %d/%d: load request "
                                "raised: %s",
                                attempt, _WARMUP_MAX_ATTEMPTS, exc,
                            )
                        if resp and resp.get("ok"):
                            self._model_loaded = True
                            self._warmup_reset()
                            logger.info(
                                "Reranker worker warm (attempt %d/%d, "
                                "backend=%s, warmup_inference=%s)",
                                attempt, _WARMUP_MAX_ATTEMPTS,
                                resp.get("backend", "?"),
                                resp.get("warmup_inference", False),
                            )
                            return
                        # v3.8.11 (issue #103): this used to report
                        # "(timeout=90s)" for EVERY failure, including loads
                        # that failed instantly. A user whose retries were 3s
                        # apart was told each one timed out after 90s, and the
                        # worker's actual error was never printed at all.
                        # Distinguish the two cases and surface the real cause.
                        if resp is None:
                            logger.warning(
                                "Reranker warmup attempt %d/%d: no response "
                                "from worker within %ds; retrying",
                                attempt, _WARMUP_MAX_ATTEMPTS,
                                _WARMUP_LOAD_TIMEOUT,
                            )
                        else:
                            load_error = (
                                resp.get("error")
                                or "worker reported not-ready without an error"
                            )
                            # A misconfiguration cannot fix itself. Retrying a
                            # bad backend name or a missing dependency four
                            # more times burns ~7.5 minutes of daemon startup
                            # to reach the same answer (issue #103). Fail fast
                            # and say exactly what to change.
                            if _is_permanent_load_error(load_error):
                                logger.error(
                                    "Reranker disabled — configuration error: "
                                    "%s. Not retrying. Fix the config or set "
                                    "retrieval.use_cross_encoder=false; recall "
                                    "continues with fusion scores.",
                                    load_error,
                                )
                                self._warmup_round_failed(permanent=True)
                                return
                            logger.warning(
                                "Reranker warmup attempt %d/%d failed: %s; "
                                "retrying",
                                attempt, _WARMUP_MAX_ATTEMPTS, load_error,
                            )

                    if attempt < _WARMUP_MAX_ATTEMPTS and not self._model_loaded:
                        if self._shutdown_event.wait(
                            min(_WARMUP_RETRY_BACKOFF_S * attempt, 15.0),
                        ):
                            return

                if not self._model_loaded:
                    logger.warning(
                        "Reranker warmup exhausted %d attempts; recall uses "
                        "fallback scoring.", _WARMUP_MAX_ATTEMPTS,
                    )
                    self._warmup_round_failed()
            except Exception as exc:
                logger.debug("Background reranker warmup failed: %s", exc)
            finally:
                self._worker_loading = False

        self._warmup_thread = threading.Thread(target=_warmup, daemon=True, name="ce-warmup")
        self._warmup_thread.start()
        return True

    def warmup_sync(self, timeout: float = 120.0) -> bool:
        """Block until reranker model is loaded. Returns True if ready.

        V3.3.12: Critical for benchmarks and first-recall quality.
        Without this, first 30-60s of recalls get no reranking (-30.7pp).
        """
        if self._model_loaded:
            return True
        if (
            not self._shutdown_event.is_set()
            and not self._worker_loading
            and not self._model_loaded
        ):
            self._warmup_reset()
            self._start_background_warmup(force=True)
        t = getattr(self, '_warmup_thread', None)
        if t is not None:
            t.join(timeout=timeout)
        return self._model_loaded

    # ------------------------------------------------------------------
    # Worker management (mirrors EmbeddingService pattern)
    # ------------------------------------------------------------------

    def _ensure_worker(self) -> None:
        """Spawn worker subprocess if not running. Machine-wide singleton.

        v3.4.13: Checks PID file before spawning — only ONE reranker worker
        can exist at a time on the machine.
        """
        if self._shutdown_event.is_set():
            return
        if self._worker_proc is not None and self._worker_proc.poll() is None:
            return
        self._worker_proc = None
        self._worker_ready = False

        # v3.4.13: Machine-wide singleton guard
        if _is_reranker_worker_alive():
            logger.debug("Reranker worker already alive (PID file), skipping spawn")
            return

        try:
            self._worker_proc = self._spawn_process()
            # v3.4.13: Register PID for machine-wide singleton
            pid_file = _reranker_pid_file()
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            pid_file.write_text(str(self._worker_proc.pid), encoding="utf-8")
            logger.info(
                "Reranker worker spawned (PID %d)", self._worker_proc.pid,
            )
            # v3.4.51: Detect immediate subprocess crash (e.g. ONNX segfault on
            # Python 3.14 before official ONNX Runtime support). Poll after 1s;
            # if the process already exited, disable reranking rather than
            # letting the broken worker linger and corrupt scores.
            time.sleep(1.0)
            if self._worker_proc.poll() is not None:
                rc = self._worker_proc.returncode
                logger.warning(
                    "Reranker worker exited immediately (returncode=%d). "
                    "ONNX Runtime may be unsupported on this Python version (%s). "
                    "Reranking disabled — recall will use fusion scores only.",
                    rc, sys.version,
                )
                self._worker_proc = None
                return
            self._worker_ready = True
        except Exception as exc:
            logger.warning("Failed to spawn reranker worker: %s", exc)
            self._worker_proc = None

    def _worker_argv(self) -> list[str]:
        """The command that starts one worker process."""
        return [sys.executable, "-m", "superlocalmemory.core.reranker_worker"]

    def _spawn_process(self) -> subprocess.Popen:
        """Start a worker process. No bookkeeping: callers own what it means."""
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
            "PYTORCH_MPS_MEM_LIMIT": "0",
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_DEVICE": "cpu",
            "ORT_DISABLE_COREML": "1",
            # Restore parallel OpenMP. The package caps OMP_NUM_THREADS
            # globally to avoid a torch+lightgbm libomp SIGSEGV in the
            # main process. This worker loads torch but never lightgbm,
            # so there is no collision risk and full parallelism is safe.
            "OMP_NUM_THREADS": str(os.cpu_count() or 4),
        }
        from superlocalmemory.core.platform_utils import popen_platform_kwargs
        return subprocess.Popen(
            self._worker_argv(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
            **popen_platform_kwargs(),
        )

    def _send_request(self, req: dict, timeout: float | None = None,
                      block: bool = True, wait: float = 0.0) -> dict | None:
        """Send JSON request to worker, get response. Thread-safe.

        Uses a short timeout (10s) for rerank requests since the model
        should already be loaded by the background warmup. Uses the full
        timeout only for explicit load/ping commands.

        v3.4.52: when ``block=False``, uses ``try_lock`` instead of
        ``lock.acquire()``. If another thread is already using the
        reranker subprocess, returns ``None`` immediately (the caller
        falls back to fusion scores without reranking). This prevents
        concurrent recall requests from serialising on the lock.
        """
        if self._shutdown_event.is_set():
            return None
        effective_timeout = timeout or _SUBPROCESS_RESPONSE_TIMEOUT

        acquired = self._lock.acquire(blocking=block)
        if not acquired and wait > 0:
            acquired = self._lock.acquire(timeout=wait)  # queue behind another recall
        if not acquired and self._swap_in_progress():
            # Not another recall's inference but a recycle swapping workers:
            # bounded wait, so the swap never leaves a recall unranked.
            acquired = self._lock.acquire(timeout=_SWAP_WAIT_SECONDS)
        if not acquired:
            return None  # another request is using the subprocess
        try:
            if (self._request_count >= _WORKER_RECYCLE_AFTER
                    and self._worker_proc is not None and self._model_loaded):
                # Blue/green: the replacement warms in the background while
                # this worker keeps answering, and is swapped in only once it
                # is ready. Killing first (4.1.18) left recall unranked for a
                # whole cold load -- 22 to 63 s in the daemon logs -- every 500
                # requests. The cost is memory, not quality: two workers exist
                # for the length of one load.
                self._begin_replacement()

            # Ensure worker is alive (re-spawn if crashed)
            if self._worker_proc is None or self._worker_proc.poll() is not None:
                self._ensure_worker()
            if self._worker_proc is None:
                return None

            msg = json.dumps(req) + "\n"
            self._worker_proc.stdin.write(msg)
            self._worker_proc.stdin.flush()

            resp_line = self._readline_with_timeout(
                self._worker_proc.stdout,
                effective_timeout,
            )
            if not resp_line:
                logger.warning("Reranker worker timed out after %ds", effective_timeout)
                self._kill_worker()
                self._model_loaded = False
                return None

            resp = json.loads(resp_line)
            self._reset_idle_timer()
            self._request_count += 1
            return resp
        except (BrokenPipeError, OSError, json.JSONDecodeError) as exc:
            logger.warning("Reranker worker communication failed: %s", exc)
            self._kill_worker()
            self._model_loaded = False
            return None
        finally:
            self._lock.release()

    # Pipe I/O lives in _worker_process; kept as methods so tests can patch them.
    _readline_with_timeout = staticmethod(readline_with_timeout)
    _stop_process = staticmethod(stop_process)

    def _kill_worker(self, timeout: float = 3.0) -> None:
        """Terminate the worker and close every owned pipe exactly once."""
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

        proc = self._worker_proc
        if proc is not None:
            # Detach first so re-entrant/finalizer cleanup is idempotent.
            self._worker_proc = None
            self._worker_ready = False
            self._generation = getattr(self, "_generation", 0) + 1
            # Invariant: a dead worker has no loaded model. Enforcing this in
            # ONE place (not just the recycle/timeout callers) means the idle
            # timer's kill also clears the flag, so the recall path sees the
            # gap and triggers a background re-warmup instead of sending a
            # rerank to a cold worker and risking a 15s-timeout churn.
            self._model_loaded = False
            self._stop_process(proc, timeout)

    # ------------------------------------------------------------------
    # Recycling: warm the replacement first, then swap (blue/green)
    # ------------------------------------------------------------------

    def _begin_replacement(self) -> None:
        """Start warming a replacement worker. Caller holds ``self._lock``."""
        if getattr(self, "_replacing", False) or self._shutdown_event.is_set():
            return
        self._replacing = True
        logger.info(
            "Recycling reranker worker after %d requests: warming its "
            "replacement first; the current worker keeps answering",
            self._request_count,
        )
        thread = threading.Thread(
            target=self._replace_worker, args=(self._generation,),
            daemon=True, name="ce-recycle",
        )
        self._recycle_thread = thread
        try:
            thread.start()
        except Exception as exc:  # noqa: BLE001 -- recycling is best-effort
            self._replacing = False
            logger.warning("Reranker recycle not started: %s", exc)

    def _replace_worker(self, generation: int) -> None:
        """Spawn and load a new worker; swap it in only if it is ready."""
        new = None
        try:
            new = self._spawn_process()
            self._replacement_proc = new
            if not self._load_into(new):
                logger.warning(
                    "Reranker recycle: the replacement did not load; the "
                    "current worker keeps serving and recycling is retried "
                    "after another %d requests", _WORKER_RECYCLE_AFTER,
                )
                with self._lock:
                    self._request_count = 0
                return
            self._swap_pending.set()
            try:
                with self._lock:
                    if (self._shutdown_event.is_set() or self._worker_proc is None
                            or self._generation != generation):
                        return  # the worker it was meant to replace is gone
                    old, self._worker_proc = self._worker_proc, new
                    new = None
                    self._replacement_proc = None
                    self._generation += 1
                    self._request_count = 0
                    self._model_loaded = True
                    self._worker_ready = True
                    self._record_worker_pid(self._worker_proc.pid)
                    self._reset_idle_timer()
            finally:
                self._swap_pending.clear()
            logger.info(
                "Reranker worker recycled (PID %d -> %d) with no unranked window",
                old.pid, self._worker_proc.pid if self._worker_proc else -1,
            )
            self._stop_process(old)
        except Exception as exc:  # noqa: BLE001 -- the old worker still serves
            logger.warning("Reranker recycle failed: %s", exc)
        finally:
            self._replacement_proc = None
            if new is not None:
                self._stop_process(new, timeout=1.0)
            self._replacing = False

    def _swap_in_progress(self) -> bool:
        pending = getattr(self, "_swap_pending", None)
        return pending is not None and pending.is_set()

    def _load_into(self, proc: Any) -> bool:
        """Load the model into a worker nobody else is using yet."""
        try:
            proc.stdin.write(json.dumps({
                "cmd": "load", "model_name": self._model_name,
                "backend": self._backend,
            }) + "\n")
            proc.stdin.flush()
            line = self._readline_with_timeout(proc.stdout, _WARMUP_LOAD_TIMEOUT)
            resp = json.loads(line) if line else None
        except (BrokenPipeError, OSError, ValueError) as exc:
            logger.debug("Reranker replacement load failed: %s", exc)
            return False
        return bool(resp and resp.get("ok"))

    @staticmethod
    def _record_worker_pid(pid: int) -> None:
        try:
            pid_file = _reranker_pid_file()
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            pid_file.write_text(str(pid), encoding="utf-8")
        except OSError as exc:
            logger.debug("Reranker PID file not updated: %s", exc)

    def _reset_idle_timer(self) -> None:
        """Reset idle timer — kills worker after 2 min inactivity."""
        if self._shutdown_event.is_set():
            return
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        self._idle_timer = threading.Timer(
            _IDLE_TIMEOUT_SECONDS, self.unload,
        )
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def unload(self) -> None:
        """Kill the worker subprocess to free all memory."""
        with self._lock:
            self._kill_worker()
            logger.info("CrossEncoderReranker: worker killed (idle timeout)")

    def shutdown(self, timeout: float = 3.0) -> None:
        """Cancel warmup, terminate the child, and join owned background work."""
        shutdown_event = getattr(self, "_shutdown_event", None)
        if shutdown_event is not None:
            shutdown_event.set()
        self._kill_worker(timeout=min(max(0.0, timeout), 1.0))
        replacement = getattr(self, "_replacement_proc", None)
        if replacement is not None:
            # Ending it also ends the recycle thread's wait for its load.
            self._stop_process(replacement, timeout=min(max(0.0, timeout), 1.0))
        for name in ("_warmup_thread", "_recycle_thread"):
            thread = getattr(self, name, None)
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[tuple[AtomicFact, float]],
        top_k: int = 10,
    ) -> list[tuple[AtomicFact, float]]:
        """Rerank candidates by cross-encoder relevance.

        NON-BLOCKING: If the worker is still loading the model
        (background warmup), returns candidates by existing score
        immediately. Once the worker is warm, subsequent calls use
        the cross-encoder. This means CLI first-call gets instant
        results (without reranking), and MCP gets reranked results
        (worker stays warm between calls).
        """
        results, _, _ = self.rerank_with_status(query, candidates, top_k=top_k)
        return results

    def rerank_with_status(
        self,
        query: str,
        candidates: list[tuple[AtomicFact, float]],
        top_k: int = 10,
    ) -> tuple[list[tuple[AtomicFact, float]], bool, str]:
        """Return results plus whether cross-encoder inference actually ran."""
        if not candidates:
            return [], False, "no_candidates"

        # Non-blocking: if the model isn't loaded, return fallback AND kick a
        # background (re)warmup so the reranker SELF-HEALS.  Without this, a
        # worker that was recycled (every 500 reqs), idle-killed (30 min), or
        # crashed left ``_model_loaded`` False with nothing to reload it — every
        # subsequent recall degraded to fallback scoring until the daemon was
        # restarted (a silent, sustained quality regression).  The warmup is
        # guarded (no-op if already loading/loaded), retried, and never blocks
        # this recall — it returns fallback now and full quality resumes within
        # seconds once the model is warm again.
        if not self._model_loaded:
            if not self._shutdown_event.is_set() and not self._worker_loading:
                self._start_background_warmup()
            sorted_cands = sorted(candidates, key=lambda x: (-x[1], x[0].fact_id))
            return sorted_cands[:top_k], False, "fallback_not_ready"

        documents = [fact.content for fact, _ in candidates]

        # v3.4.53: block=False — if another recall is using the reranker
        # subprocess, skip reranking and return fusion scores directly.
        # This prevents concurrent recalls from serialising on the lock.
        # 15s timeout (was 180s) — warm ONNX inference takes ~100ms; if
        # the worker can't respond in 15s it's dead and we fall back.
        resp = self._send_request({
            "cmd": "rerank",
            "query": query,
            "documents": documents,
        }, timeout=15.0, block=False, wait=_RERANK_QUEUE_WAIT_SECONDS)

        if resp is None or not resp.get("ok"):
            # Fallback: return by existing score
            sorted_cands = sorted(candidates, key=lambda x: (-x[1], x[0].fact_id))
            return sorted_cands[:top_k], False, "fallback_busy_or_unavailable"

        scores = resp["scores"]
        scored: list[tuple[AtomicFact, float]] = [
            (fact, float(score))
            for (fact, _), score in zip(candidates, scores)
        ]
        scored.sort(key=lambda x: (-x[1], x[0].fact_id))
        return scored[:top_k], True, "applied"

    def score_pair(self, query: str, document: str) -> float:
        """Score a single (query, document) pair."""
        resp = self._send_request({
            "cmd": "score",
            "query": query,
            "document": document,
            "model_name": self._model_name,
            "backend": self._backend,
        })

        if resp is None or not resp.get("ok"):
            return 0.0
        return float(resp.get("score", 0.0))

    @property
    def is_available(self) -> bool:
        """Whether the cross-encoder worker can be spawned."""
        resp = self._send_request({"cmd": "ping"})
        if resp is None:
            return False
        return resp.get("ok", False)


# ---------------------------------------------------------------------------
# Module-level atexit: kill ALL reranker workers on process exit
# ---------------------------------------------------------------------------

def _cleanup_all_rerankers() -> None:
    """Kill all reranker worker subprocesses on interpreter exit.

    Prevents orphaned 1.3 GB ONNX/PyTorch workers surviving after
    parent exits (especially during test runs with parallel agents).
    """
    for ref in list(_live_rerankers):
        reranker = ref()
        if reranker is not None:
            try:
                reranker.shutdown()
            except Exception:
                pass
    _live_rerankers.clear()


atexit.register(_cleanup_all_rerankers)
