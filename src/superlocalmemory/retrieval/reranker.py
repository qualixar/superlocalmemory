# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the Elastic License 2.0 - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Cross-Encoder Reranker (Subprocess-Isolated).

V3.3.3: All PyTorch/ONNX model work runs in a SEPARATE subprocess.
The main process (dashboard, MCP, CLI) NEVER imports torch and stays
at ~60 MB. Same isolation pattern as EmbeddingService.

The worker subprocess auto-kills after 2 minutes idle.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: Elastic-2.0
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
from typing import Any

from superlocalmemory.storage.models import AtomicFact

# Track all live reranker instances for atexit cleanup
_live_rerankers: set[weakref.ref] = set()

logger = logging.getLogger(__name__)

_IDLE_TIMEOUT_SECONDS = 120  # 2 min → kill worker
# V3.3.12: Configurable via SLM_RERANKER_IDLE_TIMEOUT env var
_IDLE_TIMEOUT_SECONDS = int(os.environ.get("SLM_RERANKER_IDLE_TIMEOUT", _IDLE_TIMEOUT_SECONDS))
_SUBPROCESS_RESPONSE_TIMEOUT = 180  # V3.3.12: 180s (was 120s) for stressed system respawns
_WORKER_RECYCLE_AFTER = 500  # Recycle after N requests


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
        self._idle_timer: threading.Timer | None = None
        self._request_count: int = 0

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
            self._kill_worker()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Background warmup (non-blocking model load)
    # ------------------------------------------------------------------

    def _start_background_warmup(self) -> None:
        """Start worker and load model in background thread.

        V3.3.16: Uses _send_request (lock-protected) instead of raw
        stdin/stdout access. Previous code wrote to stdin without the
        lock, creating a race where the warmup's readline thread could
        steal responses meant for _send_request → deadlock → timeout.
        """
        if self._worker_loading or self._model_loaded:
            return
        self._worker_loading = True

        def _warmup() -> None:
            try:
                self._ensure_worker()
                if self._worker_proc is None:
                    return
                resp = self._send_request({
                    "cmd": "load",
                    "model_name": self._model_name,
                    "backend": self._backend,
                }, timeout=_SUBPROCESS_RESPONSE_TIMEOUT)
                if resp and resp.get("ok"):
                    self._model_loaded = True
                    logger.info(
                        "Reranker worker warm (backend=%s, warmup_inference=%s)",
                        resp.get("backend", "?"),
                        resp.get("warmup_inference", False),
                    )
            except Exception as exc:
                logger.debug("Background reranker warmup failed: %s", exc)
            finally:
                self._worker_loading = False

        self._warmup_thread = threading.Thread(target=_warmup, daemon=True, name="ce-warmup")
        self._warmup_thread.start()

    def warmup_sync(self, timeout: float = 120.0) -> bool:
        """Block until reranker model is loaded. Returns True if ready.

        V3.3.12: Critical for benchmarks and first-recall quality.
        Without this, first 30-60s of recalls get no reranking (-30.7pp).
        """
        if self._model_loaded:
            return True
        if not self._worker_loading and not self._model_loaded:
            self._start_background_warmup()
        t = getattr(self, '_warmup_thread', None)
        if t is not None:
            t.join(timeout=timeout)
        return self._model_loaded

    # ------------------------------------------------------------------
    # Worker management (mirrors EmbeddingService pattern)
    # ------------------------------------------------------------------

    def _ensure_worker(self) -> None:
        """Spawn worker subprocess if not running. Non-blocking."""
        if self._worker_proc is not None and self._worker_proc.poll() is None:
            return
        self._worker_proc = None
        self._worker_ready = False

        worker_module = "superlocalmemory.core.reranker_worker"
        try:
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
                "PYTORCH_MPS_MEM_LIMIT": "0",
                "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TORCH_DEVICE": "cpu",
            }
            kwargs = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            else:
                kwargs["start_new_session"] = True

            self._worker_proc = subprocess.Popen(
                [sys.executable, "-m", worker_module],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=env,
                **kwargs
            )
            logger.info(
                "Reranker worker spawned (PID %d)", self._worker_proc.pid,
            )
            self._worker_ready = True
        except Exception as exc:
            logger.warning("Failed to spawn reranker worker: %s", exc)
            self._worker_proc = None

    def _send_request(self, req: dict, timeout: float | None = None) -> dict | None:
        """Send JSON request to worker, get response. Thread-safe.

        Uses a short timeout (10s) for rerank requests since the model
        should already be loaded by the background warmup. Uses the full
        timeout only for explicit load/ping commands.
        """
        effective_timeout = timeout or _SUBPROCESS_RESPONSE_TIMEOUT

        with self._lock:
            if self._request_count >= _WORKER_RECYCLE_AFTER and self._worker_proc is not None:
                logger.info("Recycling reranker worker after %d requests", self._request_count)
                self._kill_worker()
                self._model_loaded = False
                self._request_count = 0

            # Ensure worker is alive (re-spawn if crashed)
            if self._worker_proc is None or self._worker_proc.poll() is not None:
                self._ensure_worker()
            if self._worker_proc is None:
                return None

            try:
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

    @staticmethod
    def _readline_with_timeout(stream: Any, timeout_seconds: float) -> str:
        """Read a line from stream with timeout. Returns '' on timeout."""
        result_container: list[str] = []
        error_container: list[Exception] = []

        def _read() -> None:
            try:
                result_container.append(stream.readline())
            except Exception as exc:
                error_container.append(exc)

        reader = threading.Thread(target=_read, daemon=True)
        reader.start()
        reader.join(timeout=timeout_seconds)

        if reader.is_alive():
            return ""
        if error_container:
            raise error_container[0]
        return result_container[0] if result_container else ""

    def _kill_worker(self) -> None:
        """Terminate worker subprocess."""
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
        if self._worker_proc is not None:
            try:
                self._worker_proc.stdin.write('{"cmd":"quit"}\n')
                self._worker_proc.stdin.flush()
                self._worker_proc.wait(timeout=3)
            except Exception:
                try:
                    self._worker_proc.kill()
                except Exception:
                    pass
            self._worker_proc = None
            self._worker_ready = False

    def _reset_idle_timer(self) -> None:
        """Reset idle timer — kills worker after 2 min inactivity."""
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
        if not candidates:
            return []

        # Non-blocking: if model isn't loaded yet, return fallback
        if not self._model_loaded:
            sorted_cands = sorted(candidates, key=lambda x: x[1], reverse=True)
            return sorted_cands[:top_k]

        documents = [fact.content for fact, _ in candidates]

        # V3.3.16: Timeout 180s — ONNX CoreML compilation can take 30-60s on
        # first inference even after model load. The warmup_inference in the
        # worker should prevent this, but 180s is a safety net.
        resp = self._send_request({
            "cmd": "rerank",
            "query": query,
            "documents": documents,
        }, timeout=180.0)

        if resp is None or not resp.get("ok"):
            # Fallback: return by existing score
            sorted_cands = sorted(candidates, key=lambda x: x[1], reverse=True)
            return sorted_cands[:top_k]

        scores = resp["scores"]
        scored: list[tuple[AtomicFact, float]] = [
            (fact, float(score))
            for (fact, _), score in zip(candidates, scores)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

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
                reranker._kill_worker()
            except Exception:
                pass
    _live_rerankers.clear()


atexit.register(_cleanup_all_rerankers)
