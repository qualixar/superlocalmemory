# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the Elastic License 2.0 - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Embedding Service (Subprocess-Isolated).

All PyTorch/model work runs in a SEPARATE subprocess. The main process
(dashboard, MCP, CLI) never imports torch and stays at ~60 MB.

The worker subprocess auto-kills after 2 minutes idle, returning all
memory to the OS. It respawns on next embed call (~3 sec cold start).

Part of Qualixar | Author: Varun Pratap Bhardwaj
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
from typing import TYPE_CHECKING

import numpy as np

# Track all live embedding services for atexit cleanup
_live_embedding_services: set[weakref.ref] = set()

if TYPE_CHECKING:
    from numpy.typing import NDArray

from superlocalmemory.core.config import EmbeddingConfig

logger = logging.getLogger(__name__)

# Fisher variance constants
_FISHER_VAR_MIN = 0.05
_FISHER_VAR_MAX = 2.0
_FISHER_VAR_RANGE = _FISHER_VAR_MAX - _FISHER_VAR_MIN


class DimensionMismatchError(RuntimeError):
    """Raised when the actual embedding dimension differs from config."""


# ---------------------------------------------------------------------------
# V3.3.28: System-wide concurrency guard for embedding workers.
#
# The memory blast incident (April 7, 2026) was caused by 20+ concurrent
# `slm observe` CLI processes each spawning their own embedding_worker
# subprocess (1.4 GB each). This file lock ensures only MAX_CONCURRENT
# embedding workers can exist across ALL processes on the machine.
#
# Primary defense: daemon routing (cmd_observe → daemon → singleton engine).
# This lock is the secondary safety net for when the daemon isn't available.
# ---------------------------------------------------------------------------

_EMBEDDING_LOCK_FILE = Path.home() / ".superlocalmemory" / ".embedding.lock"
_MAX_CONCURRENT_WORKERS = int(os.environ.get("SLM_MAX_EMBEDDING_WORKERS", 2))
_embedding_lock_fd: int | None = None


def acquire_embedding_lock(timeout: float = 5.0) -> bool:
    """Acquire system-wide embedding worker lock.

    Uses fcntl.flock on Unix. On Windows, falls back to allowing (no lock).
    Returns True if lock acquired, False if timed out (another worker active).
    """
    global _embedding_lock_fd
    if sys.platform == "win32":
        return True  # No file locking on Windows — daemon routing is primary defense

    import fcntl
    _EMBEDDING_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        _embedding_lock_fd = os.open(str(_EMBEDDING_LOCK_FILE), os.O_CREAT | os.O_RDWR)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                fcntl.flock(_embedding_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except (BlockingIOError, OSError):
                time.sleep(0.2)
        # Timeout — another worker holds the lock
        os.close(_embedding_lock_fd)
        _embedding_lock_fd = None
        return False
    except Exception:
        return True  # On error, allow through (don't block functionality)


def release_embedding_lock() -> None:
    """Release system-wide embedding worker lock."""
    global _embedding_lock_fd
    if _embedding_lock_fd is not None:
        try:
            import fcntl
            fcntl.flock(_embedding_lock_fd, fcntl.LOCK_UN)
            os.close(_embedding_lock_fd)
        except Exception:
            pass
        _embedding_lock_fd = None


_IDLE_TIMEOUT_SECONDS = 120  # 2 minutes — kill worker after idle
# V3.3.12: Configurable via SLM_EMBED_IDLE_TIMEOUT env var (seconds)
_IDLE_TIMEOUT_SECONDS = int(os.environ.get("SLM_EMBED_IDLE_TIMEOUT", _IDLE_TIMEOUT_SECONDS))
# V3.3.21: Configurable response timeout — 180s default, but batch ingestion
# (2-turn chunks across 10 conversations) needs 600s+ to survive cold-start
# model downloads and ARM64 ONNX compilation pauses.
_SUBPROCESS_RESPONSE_TIMEOUT = int(os.environ.get("SLM_EMBED_RESPONSE_TIMEOUT", 180))
# V3.3.21: Increase recycle threshold to 5000 (was 1000). With 2-turn chunks,
# a single conversation produces ~50-80 store calls. 10 conversations = 500-800.
# Recycling at 1000 caused mid-ingestion worker death → timeout cascade.
_WORKER_RECYCLE_AFTER = int(os.environ.get("SLM_EMBED_RECYCLE_AFTER", 5000))


class EmbeddingService:
    """Subprocess-isolated embedding service.

    All model inference runs in a child process. The main process never
    imports torch/sentence-transformers, keeping its memory at ~60 MB.

    The worker auto-kills after 2 min idle. First embed after idle takes
    ~3 sec (model reload). Subsequent embeds are instant (<100ms).
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._worker_proc: subprocess.Popen | None = None
        self._available = True
        self._last_used: float = 0.0
        self._idle_timer: threading.Timer | None = None
        self._worker_ready = False
        self._request_count: int = 0

        # Register for atexit cleanup (prevent orphaned workers)
        ref = weakref.ref(self, _live_embedding_services.discard)
        _live_embedding_services.add(ref)

    def __del__(self) -> None:
        """Kill worker subprocess when service is garbage-collected."""
        try:
            self._kill_worker()
        except Exception:
            pass

    @property
    def is_available(self) -> bool:
        """Check if embedding service can produce embeddings."""
        if self._config.is_cloud:
            return bool(self._config.api_endpoint and self._config.api_key)
        return self._available

    @property
    def dimension(self) -> int:
        return self._config.dimension

    def unload(self) -> None:
        """Kill the worker subprocess to free all memory."""
        with self._lock:
            self._kill_worker()
            logger.info("EmbeddingService: worker killed (idle timeout)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float] | None:
        """Embed a single text string. Returns list of floats or None."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        if self._config.is_cloud:
            return self._cloud_embed_single(text)
        result = self._subprocess_embed([text])
        if result is None:
            return None
        vec = result[0]
        self._validate_dimension(np.asarray(vec))
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Embed a batch of texts."""
        if not texts:
            raise ValueError("Cannot embed empty batch")
        if self._config.is_cloud:
            return self._cloud_embed_batch(texts)
        result = self._subprocess_embed(texts)
        if result is None:
            return [None] * len(texts)
        for vec in result:
            if vec is not None:
                self._validate_dimension(np.asarray(vec))
        return result

    def compute_fisher_params(
        self, embedding: list[float],
    ) -> tuple[list[float], list[float]]:
        """Compute Fisher-Rao parameters from a raw embedding."""
        arr = np.asarray(embedding, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
        if norm < 1e-10:
            mean = np.zeros(len(arr), dtype=np.float64)
            variance = np.full(len(arr), _FISHER_VAR_MAX, dtype=np.float64)
            return mean.tolist(), variance.tolist()
        mean = arr / norm
        abs_mean = np.abs(mean)
        max_val = float(np.max(abs_mean)) + 1e-10
        signal_strength = abs_mean / max_val
        variance = _FISHER_VAR_MAX - _FISHER_VAR_RANGE * signal_strength
        variance = np.clip(variance, _FISHER_VAR_MIN, _FISHER_VAR_MAX)
        return mean.tolist(), variance.tolist()

    # ------------------------------------------------------------------
    # Subprocess worker management
    # ------------------------------------------------------------------

    def _subprocess_embed(self, texts: list[str]) -> list[list[float]] | None:
        """Send texts to worker subprocess, get embeddings back.

        Includes a timeout (_SUBPROCESS_RESPONSE_TIMEOUT seconds) so the CLI
        never hangs indefinitely on cold model loads or network issues.
        """
        with self._lock:
            # Worker recycling: restart after N requests to prevent
            # C++ allocator fragmentation over long-running sessions.
            if self._request_count >= _WORKER_RECYCLE_AFTER and self._worker_proc is not None:
                logger.info("Recycling embedding worker after %d requests", self._request_count)
                self._kill_worker()
                self._request_count = 0

            self._ensure_worker()
            if self._worker_proc is None:
                return None

            req = json.dumps({
                "cmd": "embed",
                "texts": texts,
                "model_name": self._config.model_name,
                "dimension": self._config.dimension,
            }) + "\n"

            try:
                self._worker_proc.stdin.write(req)
                self._worker_proc.stdin.flush()
                resp_line = self._readline_with_timeout(
                    self._worker_proc.stdout,
                    _SUBPROCESS_RESPONSE_TIMEOUT,
                )
                if not resp_line:
                    logger.warning(
                        "Embedding worker timed out after %ds. "
                        "Run 'slm setup' to download models and verify installation.",
                        _SUBPROCESS_RESPONSE_TIMEOUT,
                    )
                    # Print to stderr so CLI users see this even without logging
                    print(
                        f"\n⚠ Embedding worker did not respond within "
                        f"{_SUBPROCESS_RESPONSE_TIMEOUT}s.\n"
                        f"  Run: slm setup   (download models + verify)\n"
                        f"  Run: slm doctor  (diagnose issues)\n",
                        file=sys.stderr,
                    )
                    self._kill_worker()
                    return None
                resp = json.loads(resp_line)
                if not resp.get("ok"):
                    logger.warning("Worker error: %s", resp.get("error"))
                    return None
                self._reset_idle_timer()
                self._request_count += 1
                return resp["vectors"]
            except (BrokenPipeError, OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Embedding worker communication failed: %s — respawning.",
                    exc,
                )
                self._kill_worker()
                # V3.3.16: Auto-retry once after worker death (RSS watchdog
                # or crash). Respawn + re-send instead of returning None.
                try:
                    self._ensure_worker()
                    if self._worker_proc is not None:
                        self._worker_proc.stdin.write(req)
                        self._worker_proc.stdin.flush()
                        resp_line = self._readline_with_timeout(
                            self._worker_proc.stdout,
                            _SUBPROCESS_RESPONSE_TIMEOUT,
                        )
                        if resp_line:
                            resp = json.loads(resp_line)
                            if resp.get("ok"):
                                self._reset_idle_timer()
                                self._request_count = 1
                                return resp["vectors"]
                except Exception:
                    self._kill_worker()
                return None

    @staticmethod
    def _readline_with_timeout(stream, timeout_seconds: float) -> str:
        """Read a line from stream with a timeout. Returns '' on timeout."""
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
            logger.warning(
                "Embedding worker did not respond within %ds", timeout_seconds,
            )
            return ""
        if error_container:
            raise error_container[0]
        return result_container[0] if result_container else ""

    @staticmethod
    def _check_memory_pressure() -> bool:
        """Check if system has enough memory to spawn a worker.

        V3.3.28: Prevents spawning embedding workers (1.4 GB each) when
        the system is already under memory pressure. Returns True if safe.
        """
        min_available_gb = float(os.environ.get("SLM_MIN_AVAILABLE_MEMORY_GB", "2.0"))
        try:
            if sys.platform == "darwin":
                # macOS: use vm_stat to get free + inactive pages
                import subprocess as _sp
                result = _sp.run(["vm_stat"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.split("\n")
                    page_size = 16384  # default on Apple Silicon
                    free_pages = 0
                    for line in lines:
                        if "page size of" in line:
                            try:
                                page_size = int(line.split()[-2])
                            except (ValueError, IndexError):
                                pass
                        if "Pages free" in line or "Pages inactive" in line:
                            try:
                                free_pages += int(line.split()[-1].rstrip("."))
                            except (ValueError, IndexError):
                                pass
                    available_gb = (free_pages * page_size) / (1024 ** 3)
                    if available_gb < min_available_gb:
                        logger.warning(
                            "Low memory (%.1f GB available, need %.1f GB) — "
                            "deferring embedding worker spawn",
                            available_gb, min_available_gb,
                        )
                        return False
            else:
                # Linux/other: use /proc/meminfo or psutil
                try:
                    with open("/proc/meminfo") as f:
                        for line in f:
                            if line.startswith("MemAvailable:"):
                                available_kb = int(line.split()[1])
                                available_gb = available_kb / (1024 * 1024)
                                if available_gb < min_available_gb:
                                    logger.warning(
                                        "Low memory (%.1f GB available) — "
                                        "deferring embedding worker spawn",
                                        available_gb,
                                    )
                                    return False
                                break
                except FileNotFoundError:
                    pass  # Not Linux, allow through
        except Exception:
            pass  # On error, allow through (don't block functionality)
        return True

    def _ensure_worker(self) -> None:
        """Spawn worker subprocess if not running."""
        if self._worker_proc is not None and self._worker_proc.poll() is None:
            return
        self._worker_proc = None

        # V3.3.28: Check memory pressure before spawning
        if not self._check_memory_pressure():
            logger.warning("Skipping embedding worker spawn due to memory pressure")
            self._available = False
            return

        worker_module = "superlocalmemory.core.embedding_worker"
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
            self._worker_proc = subprocess.Popen(
                [sys.executable, "-m", worker_module],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=True,  # Prevent terminal signals bleeding to worker
            )
            logger.info("Embedding worker spawned (PID %d)", self._worker_proc.pid)
            self._worker_ready = True
        except Exception as exc:
            logger.warning(
                "Failed to spawn embedding worker: %s. "
                "Run 'slm doctor' to verify your Python environment. "
                "Using Python: %s",
                exc, sys.executable,
            )
            self._available = False
            self._worker_proc = None

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
        self._last_used = time.time()

    # ------------------------------------------------------------------
    # Cloud embedding (no subprocess needed — just HTTP)
    # ------------------------------------------------------------------

    def _cloud_embed_single(self, text: str) -> list[float]:
        vecs = self._cloud_embed_batch([text])
        return vecs[0]

    def _cloud_embed_batch(
        self, texts: list[str], *, max_retries: int = 3,
    ) -> list[list[float]]:
        """Encode via Azure OpenAI embedding API with retry."""
        import httpx
        url = (
            f"{self._config.api_endpoint.rstrip('/')}/openai/deployments/"
            f"{self._config.deployment_name}/embeddings"
            f"?api-version={self._config.api_version}"
        )
        headers = {
            "Content-Type": "application/json",
            "api-key": self._config.api_key,
        }
        body = {"input": texts, "model": self._config.deployment_name}
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
                    resp = client.post(url, headers=headers, json=body)
                    resp.raise_for_status()
                data = resp.json()
                results = []
                for item in sorted(data["data"], key=lambda d: d["index"]):
                    results.append(item["embedding"])
                return results
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Cloud embedding failed: {last_error}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_dimension(self, vec: NDArray) -> None:
        actual = len(vec)
        if actual != self._config.dimension:
            raise DimensionMismatchError(
                f"Embedding dimension {actual} != expected {self._config.dimension}"
            )


# ---------------------------------------------------------------------------
# Module-level atexit: kill ALL embedding workers on process exit
# ---------------------------------------------------------------------------

def _cleanup_all_embedding_services() -> None:
    """Kill all embedding worker subprocesses on interpreter exit.

    Prevents orphaned 500-800 MB sentence-transformer workers surviving
    after parent exits (especially during test runs with parallel agents).
    """
    for ref in list(_live_embedding_services):
        svc = ref()
        if svc is not None:
            try:
                svc._kill_worker()
            except Exception:
                pass
    _live_embedding_services.clear()


atexit.register(_cleanup_all_embedding_services)
