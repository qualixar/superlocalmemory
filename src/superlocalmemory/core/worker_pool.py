# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
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

_IDLE_TIMEOUT = 120  # 2 min — kill worker after idle
_REQUEST_TIMEOUT = 60  # 60 sec max per request


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

    def recall(self, query: str, limit: int = 10) -> dict:
        """Run recall in worker subprocess. Returns result dict."""
        return self._send({"cmd": "recall", "query": query, "limit": limit})

    def store(self, content: str, metadata: dict | None = None) -> dict:
        """Run store in worker subprocess. Returns result dict."""
        return self._send({
            "cmd": "store", "content": content,
            "metadata": metadata or {},
        })

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
        """Send request to worker and get response. Thread-safe."""
        with self._lock:
            self._ensure_worker()
            if self._proc is None:
                return {"ok": False, "error": "Worker failed to start"}

            req_line = json.dumps(request) + "\n"
            try:
                self._proc.stdin.write(req_line)
                self._proc.stdin.flush()

                # Read response with timeout
                import selectors
                sel = selectors.DefaultSelector()
                sel.register(self._proc.stdout, selectors.EVENT_READ)
                ready = sel.select(timeout=_REQUEST_TIMEOUT)
                sel.close()

                if not ready:
                    logger.error("Worker timed out after %ds", _REQUEST_TIMEOUT)
                    self._kill()
                    return {"ok": False, "error": "Worker timed out"}

                resp_line = self._proc.stdout.readline()
                if not resp_line:
                    logger.warning("Worker returned empty, restarting")
                    self._kill()
                    return {"ok": False, "error": "Worker died"}

                self._reset_idle_timer()
                return json.loads(resp_line)

            except (BrokenPipeError, OSError, json.JSONDecodeError) as exc:
                logger.warning("Worker communication failed: %s", exc)
                self._kill()
                return {"ok": False, "error": str(exc)}

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
            )
            logger.info("Recall worker spawned (PID %d)", self._proc.pid)
        except Exception as exc:
            logger.error("Failed to spawn recall worker: %s", exc)
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
