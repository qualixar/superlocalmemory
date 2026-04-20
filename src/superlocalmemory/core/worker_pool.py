# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Recall worker pool — manages subprocess lifecycle for all callers.

Single shared worker process handles requests from dashboard, MCP, CLI.
Serializes concurrent requests via a threading lock (one at a time to
avoid interleaved stdout). Worker auto-kills after idle timeout.

Usage:
    pool = WorkerPool.shared()
    result = pool.recall("what is X?", limit=10)
    result = pool.store("some content", metadata={})

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time

logger = logging.getLogger(__name__)

_IDLE_TIMEOUT = 120   # 2 min — kill worker after idle
_REQUEST_TIMEOUT = 120  # 120 sec per request (V3.3.2: ONNX cold start can take 30-60s)
_WARMUP_TIMEOUT = 180  # 3 min — first cold start: engine + ONNX export + models
_WORKER_RECYCLE_AFTER = 1000  # Recycle worker after N requests (C++ fragmentation prevention)


class WorkerPool:
    """Manages a single recall_worker subprocess with idle auto-kill.

    Thread-safe: concurrent callers are serialized via lock.
    The worker subprocess holds all heavy memory (PyTorch, engine).
    The calling process stays at ~60 MB.
    """

    _instance: WorkerPool | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._idle_timer: threading.Timer | None = None
        self._last_used: float = 0.0
        self._request_count: int = 0

    @classmethod
    def shared(cls) -> WorkerPool:
        """Get or create the singleton worker pool."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recall(
        self, query: str, limit: int = 10, session_id: str = "",
    ) -> dict:
        """Run recall in worker subprocess. Returns result dict.

        S9-DASH-02: ``session_id`` threads through to ``engine.recall``
        so the outcome-queue gets a pending_outcomes row for this
        recall. Without it, hook-based signals have no outcome to
        attach to.
        """
        return self._send({
            "cmd": "recall", "query": query, "limit": limit,
            "session_id": session_id or "",
        })

    def store(self, content: str, metadata: dict | None = None) -> dict:
        """Run store in worker subprocess. Returns result dict."""
        return self._send({
            "cmd": "store", "content": content,
            "metadata": metadata or {},
        })

    def delete_memory(self, fact_id: str, agent_id: str = "system") -> dict:
        """Delete a specific memory by fact_id. Logged for audit."""
        return self._send({"cmd": "delete_memory", "fact_id": fact_id, "agent_id": agent_id})

    def update_memory(self, fact_id: str, content: str, agent_id: str = "system") -> dict:
        """Update content of a specific memory. Logged for audit."""
        return self._send({"cmd": "update_memory", "fact_id": fact_id, "content": content, "agent_id": agent_id})

    def get_memory_facts(self, memory_id: str) -> dict:
        """Get original memory text + child atomic facts."""
        return self._send({"cmd": "get_memory_facts", "memory_id": memory_id})

    def summarize(self, texts: list[str]) -> dict:
        """Generate summary from texts (heuristic in A, LLM in B/C)."""
        return self._send({"cmd": "summarize", "texts": texts})

    def synthesize(self, query: str, facts: list[dict]) -> dict:
        """Generate synthesized answer from query + facts."""
        return self._send({"cmd": "synthesize", "query": query, "facts": facts})

    def status(self) -> dict:
        """Get engine status from worker."""
        return self._send({"cmd": "status"})

    def shutdown(self) -> None:
        """Gracefully kill the worker."""
        with self._lock:
            self._kill()

    def warmup(self) -> None:
        """Pre-spawn and warm up the worker in a background thread.

        Spawns the recall_worker subprocess so that PyTorch, models, and
        the engine are all loaded BEFORE the first user request. This
        amortizes the 30s cold-start at dashboard/MCP startup time.

        Call from startup events — non-blocking, runs in background.
        """
        def _do_warmup() -> None:
            logger.info("Worker warmup starting (background)...")
            try:
                result = self._send_with_timeout(
                    {"cmd": "warmup"}, timeout=_WARMUP_TIMEOUT,
                )
                if result.get("ok"):
                    logger.info("Worker warmup complete (engine + models ready)")
                else:
                    logger.warning("Worker warmup returned: %s", result)
            except Exception as exc:
                logger.warning("Worker warmup failed: %s", exc)

        t = threading.Thread(target=_do_warmup, daemon=True, name="worker-warmup")
        t.start()

    @property
    def worker_pid(self) -> int | None:
        """PID of the worker process, or None if not running."""
        if self._proc and self._proc.poll() is None:
            return self._proc.pid
        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _send(self, request: dict) -> dict:
        """Send request to worker and get response. Thread-safe.

        Auto-retries once on worker death (idle timeout, crash).
        """
        resp = self._send_with_timeout(request, timeout=_REQUEST_TIMEOUT)
        if not resp.get("ok") and "Worker" in resp.get("error", ""):
            logger.info("Auto-restarting worker after failure, retrying request")
            resp = self._send_with_timeout(request, timeout=_REQUEST_TIMEOUT)
        return resp

    def _send_with_timeout(self, request: dict, timeout: float) -> dict:
        """Send request with configurable timeout. Thread-safe."""
        with self._lock:
            # Worker recycling: restart after N requests to prevent
            # C++ allocator fragmentation over long-running sessions.
            if self._request_count >= _WORKER_RECYCLE_AFTER and self._proc is not None:
                logger.info("Recycling recall worker after %d requests", self._request_count)
                self._kill()
                self._request_count = 0

            self._ensure_worker()
            if self._proc is None:
                return {"ok": False, "error": "Worker failed to start"}

            req_line = json.dumps(request) + "\n"
            try:
                self._proc.stdin.write(req_line)
                self._proc.stdin.flush()

                # Read response with timeout using a thread.
                # selectors/select do NOT work with pipes on Windows,
                # so we use the same thread-based approach as EmbeddingService.
                resp_line = self._readline_with_timeout(
                    self._proc.stdout, timeout,
                )

                if not resp_line:
                    logger.warning("Worker returned empty, restarting. Run 'slm doctor' to diagnose.")
                    self._kill()
                    return {"ok": False, "error": "Worker died"}

                self._reset_idle_timer()
                self._request_count += 1
                return json.loads(resp_line)

            except (BrokenPipeError, OSError, json.JSONDecodeError) as exc:
                logger.warning("Worker communication failed: %s. Run 'slm doctor' to diagnose.", exc)
                self._kill()
                return {"ok": False, "error": str(exc)}

    @staticmethod
    def _readline_with_timeout(stream, timeout_seconds: float) -> str:
        """Read one line from *stream* with a timeout.

        Uses a daemon thread so the call never blocks the main thread
        indefinitely. This is the cross-platform replacement for
        ``selectors`` which fails on Windows pipes.

        Returns the line read, or ``""`` on timeout / error.
        """
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

    def _ensure_worker(self) -> None:
        """Spawn worker if not running."""
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = None
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
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "superlocalmemory.core.recall_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=True,  # Prevent terminal signals bleeding to worker
            )
            logger.info("Recall worker spawned (PID %d)", self._proc.pid)
        except Exception as exc:
            logger.error("Failed to spawn recall worker: %s. Run 'slm doctor' to diagnose. Python: %s", exc, sys.executable)
            self._proc = None

    def _kill(self) -> None:
        """Terminate worker. ALL memory freed to OS."""
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
        if self._proc is not None:
            pid = self._proc.pid
            try:
                self._proc.stdin.write('{"cmd":"quit"}\n')
                self._proc.stdin.flush()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=2)
                except Exception:
                    pass
            self._proc = None
            logger.info("Recall worker killed (PID %s)", pid)

    def _reset_idle_timer(self) -> None:
        """Kill worker after 2 min of no requests."""
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        self._idle_timer = threading.Timer(_IDLE_TIMEOUT, self._idle_kill)
        self._idle_timer.daemon = True
        self._idle_timer.start()
        self._last_used = time.time()

    def _idle_kill(self) -> None:
        """Called by idle timer — kill worker to free memory."""
        with self._lock:
            if self._proc is not None:
                logger.info("Idle timeout — killing recall worker")
                self._kill()
