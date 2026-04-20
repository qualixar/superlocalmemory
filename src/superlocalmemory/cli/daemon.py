# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLM Daemon — client functions for communicating with the unified daemon.

The unified daemon (server/unified_daemon.py) runs as a single FastAPI/uvicorn
process on port 8765, with port 8767 as a backward-compat TCP redirect.

This module contains CLIENT functions used by CLI commands:
  - is_daemon_running(): check if daemon is alive
  - ensure_daemon(): start daemon if not running
  - stop_daemon(): gracefully stop the daemon
  - daemon_request(): send HTTP request to daemon

The actual daemon server code is in server/unified_daemon.py.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import threading
from threading import Thread

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8765  # v3.4.3: unified daemon on 8765 (was 8767)
_LEGACY_PORT = 8767   # backward-compat redirect
_DEFAULT_IDLE_TIMEOUT = 0  # v3.4.3: 24/7 default (was 1800)
_PID_FILE = Path.home() / ".superlocalmemory" / "daemon.pid"
_PORT_FILE = Path.home() / ".superlocalmemory" / "daemon.port"


# ---------------------------------------------------------------------------
# Client: check if daemon running + send requests
# ---------------------------------------------------------------------------

def _is_pid_alive(pid: int) -> bool:
    """Cross-platform check if a process with given PID exists."""
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False


def is_daemon_running() -> bool:
    """Check if daemon is alive via PID file + HTTP health check.

    v3.4.4 FIX: If PID is alive, returns True EVEN IF health check fails.
    This prevents starting duplicate daemons when the existing one is
    warming up (Ollama processing, model download, embedding init).

    Priority:
      1. PID file exists AND process alive → True (daemon warming up or ready)
      2. No PID file → try health check on known ports (MCP/hook started daemon)
      3. PID file stale (process dead) → clean up, return False
    """
    if _PID_FILE.exists():
        try:
            pid = int(_PID_FILE.read_text().strip())
            if _is_pid_alive(pid):
                # PID alive = daemon exists. Don't check health — it might be warming up.
                # This is the critical fix: NEVER start a second daemon if PID is alive.
                return True
            else:
                # Process died — clean up stale PID file
                _PID_FILE.unlink(missing_ok=True)
                _PORT_FILE.unlink(missing_ok=True)
        except (ValueError, OSError):
            _PID_FILE.unlink(missing_ok=True)

    # No PID file — maybe daemon was started by MCP/hook without PID file.
    # Try health check on known ports as last resort.
    for try_port in (_DEFAULT_PORT, _LEGACY_PORT):
        try:
            import urllib.request
            resp = urllib.request.urlopen(
                f"http://127.0.0.1:{try_port}/health", timeout=2,
            )
            if resp.status == 200:
                # Daemon running without PID file — write one for future checks
                try:
                    import json as _json
                    data = _json.loads(resp.read().decode())
                    pid = data.get("pid")
                    if pid:
                        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
                        _PID_FILE.write_text(str(pid))
                        _PORT_FILE.write_text(str(try_port))
                except Exception:
                    pass
                return True
        except Exception:
            continue
    return False


def _get_port() -> int:
    if _PORT_FILE.exists():
        try:
            return int(_PORT_FILE.read_text().strip())
        except ValueError:
            pass
    return _DEFAULT_PORT


def daemon_request(method: str, path: str, body: dict | None = None) -> dict | None:
    """Send request to daemon. Returns parsed JSON or None on failure."""
    port = _get_port()
    try:
        import urllib.request
        url = f"http://127.0.0.1:{port}{path}"
        data = json.dumps(body).encode() if body else None
        headers = {"Content-Type": "application/json"} if data else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read().decode())
    except Exception:
        return None


_LOCK_FILE = Path.home() / ".superlocalmemory" / "daemon.lock"


def ensure_daemon() -> bool:
    """Start daemon if not running. Returns True if daemon is ready.

    v3.4.4 BULLETPROOF:
      1. If PID alive → return True immediately (even if warming up)
      2. File lock prevents two callers from starting concurrent daemons
      3. After starting, waits for PID file (not health check) — fast detection
      4. Cross-platform: macOS + Windows + Linux
    """
    if is_daemon_running():
        return True

    # File lock — prevent concurrent starts from multiple CLI/MCP calls
    lock_fd = None
    try:
        _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = open(_LOCK_FILE, "w")

        # Cross-platform file locking
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
            except (IOError, OSError):
                # Another process is starting the daemon — just wait for it
                lock_fd.close()
                return _wait_for_daemon(timeout=60)
        else:
            import fcntl
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, OSError):
                lock_fd.close()
                return _wait_for_daemon(timeout=60)

        # Re-check after acquiring lock (another process may have started it)
        if is_daemon_running():
            return True

        # Start unified daemon in background
        import subprocess
        cmd = [sys.executable, "-m", "superlocalmemory.server.unified_daemon", "--start"]
        log_dir = Path.home() / ".superlocalmemory" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "daemon.log"

        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True

        with open(log_file, "a") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, **kwargs)

        # Write PID immediately so other callers see it during warmup
        _PID_FILE.write_text(str(proc.pid))
        _PORT_FILE.write_text(str(_DEFAULT_PORT))

        return _wait_for_daemon(timeout=60)

    except Exception as exc:
        logger.debug("ensure_daemon error: %s", exc)
        return False
    finally:
        if lock_fd:
            try:
                lock_fd.close()
            except Exception:
                pass
            try:
                _LOCK_FILE.unlink(missing_ok=True)
            except Exception:
                pass


def _wait_for_daemon(timeout: int = 60) -> bool:
    """Wait for daemon to become reachable. Checks PID alive first (fast),
    then health endpoint (confirms HTTP server is bound)."""
    for _ in range(timeout * 2):  # check every 0.5s
        time.sleep(0.5)
        if is_daemon_running():
            # PID is alive — now optionally check if HTTP is ready
            port = _get_port()
            try:
                import urllib.request
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                return True  # HTTP is ready
            except Exception:
                # PID alive but HTTP not ready — daemon is warming up, that's OK
                return True
    return False


def stop_daemon() -> bool:
    """Stop ALL SLM daemon processes and their workers.

    v3.4.7: Nuclear cleanup — finds and kills ALL processes matching
    superlocalmemory.server.unified_daemon, embedding_worker, recall_worker,
    reranker_worker. Not just the PID file daemon. Multiple daemons can
    accumulate from rapid restarts, MCP warmups, and concurrent sessions.
    """
    killed = 0

    try:
        import psutil
        my_pid = os.getpid()
        targets = [
            "superlocalmemory.server.unified_daemon",
            "superlocalmemory.core.embedding_worker",
            "superlocalmemory.core.recall_worker",
            "superlocalmemory.core.reranker_worker",
        ]

        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                if proc.pid == my_pid:
                    continue
                cmdline = " ".join(proc.info.get("cmdline") or [])
                if any(t in cmdline for t in targets):
                    # Kill children first, then process
                    for child in proc.children(recursive=True):
                        try:
                            child.kill()
                            killed += 1
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    proc.kill()
                    killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    except ImportError:
        # Fallback: pkill by pattern
        try:
            import subprocess as _sp
            for pattern in [
                "superlocalmemory.server.unified_daemon",
                "superlocalmemory.core.embedding_worker",
                "superlocalmemory.core.recall_worker",
                "superlocalmemory.core.reranker_worker",
            ]:
                result = _sp.run(
                    ["pkill", "-9", "-f", pattern],
                    capture_output=True, timeout=5,
                )
                if result.returncode == 0:
                    killed += 1
        except Exception:
            pass

    # Clean up PID/port files + worker PID files
    _PID_FILE.unlink(missing_ok=True)
    _PORT_FILE.unlink(missing_ok=True)
    # v3.4.13: Clean worker PID files (singleton guards)
    for pidfile in (".embedding-worker.pid", ".reranker-worker.pid"):
        (Path.home() / ".superlocalmemory" / pidfile).unlink(missing_ok=True)

    # v3.4.13: Wait for ALL workers to actually die before returning.
    # Without this, `slm restart` starts a new daemon before old workers exit,
    # causing duplicate embedding_workers (1.6GB each).
    if killed:
        logger.info("Stopped %d SLM processes, waiting for exit...", killed)
        _wait_for_workers_dead(timeout=10)

    return True


def _wait_for_workers_dead(timeout: int = 10) -> None:
    """Wait until no SLM worker processes remain alive."""
    targets = [
        "superlocalmemory.server.unified_daemon",
        "superlocalmemory.core.embedding_worker",
        "superlocalmemory.core.recall_worker",
        "superlocalmemory.core.reranker_worker",
    ]
    my_pid = os.getpid()
    deadline = time.time() + timeout

    while time.time() < deadline:
        alive = False
        try:
            import psutil
            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    if proc.pid == my_pid:
                        continue
                    cmdline = " ".join(proc.info.get("cmdline") or [])
                    if any(t in cmdline for t in targets):
                        alive = True
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except ImportError:
            # No psutil — just wait a fixed time
            time.sleep(3)
            return

        if not alive:
            return
        time.sleep(0.5)

    logger.warning("Some SLM workers still alive after %ds timeout", timeout)


# ---------------------------------------------------------------------------
# Server: HTTP request handler with engine singleton
# ---------------------------------------------------------------------------

_engine = None
_last_activity = time.monotonic()

# ---------------------------------------------------------------------------
# V3.3.28: Observation debounce buffer.
#
# When 20+ file edits arrive in quick succession (from parallel AI agents,
# git checkout, or batch sed), we buffer observations for _OBSERVE_DEBOUNCE_SEC
# seconds and deduplicate by content hash. This reduces 20 observations → 1-3
# batches, each processed by the singleton engine (1 embedding worker).
# ---------------------------------------------------------------------------

_OBSERVE_DEBOUNCE_SEC = float(os.environ.get("SLM_OBSERVE_DEBOUNCE_SEC", "3.0"))
_observe_buffer: list[str] = []
_observe_seen: set[str] = set()  # content hashes for dedup within window
_observe_lock = threading.Lock()
_observe_timer: threading.Timer | None = None


def _flush_observe_buffer() -> None:
    """Process all buffered observations as a single batch."""
    global _observe_timer
    with _observe_lock:
        if not _observe_buffer:
            return
        batch = list(_observe_buffer)
        _observe_buffer.clear()
        _observe_seen.clear()
        _observe_timer = None

    # Process each unique observation (already deduped)
    engine = _get_engine()
    from superlocalmemory.hooks.auto_capture import AutoCapture
    auto = AutoCapture(engine=engine)

    for content in batch:
        try:
            decision = auto.evaluate(content)
            if decision.capture:
                auto.capture(content, category=decision.category)
        except Exception:
            pass  # Don't let one bad observation kill the batch

    logger.info("Observe debounce: processed %d observations (from buffer)", len(batch))


def _enqueue_observation(content: str) -> dict:
    """Add an observation to the debounce buffer. Returns immediate response."""
    global _observe_timer
    import hashlib
    content_hash = hashlib.md5(content.encode()).hexdigest()

    with _observe_lock:
        if content_hash in _observe_seen:
            return {"captured": False, "reason": "duplicate within debounce window"}

        _observe_seen.add(content_hash)
        _observe_buffer.append(content)
        buf_size = len(_observe_buffer)

        # Reset debounce timer
        if _observe_timer is not None:
            _observe_timer.cancel()
        _observe_timer = threading.Timer(_OBSERVE_DEBOUNCE_SEC, _flush_observe_buffer)
        _observe_timer.daemon = True
        _observe_timer.start()

    return {"captured": True, "queued": True, "buffer_size": buf_size,
            "debounce_sec": _OBSERVE_DEBOUNCE_SEC}


def _get_engine():
    global _engine
    if _engine is None:
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.engine import MemoryEngine

        config = SLMConfig.load()
        _engine = MemoryEngine(config)
        _engine.initialize()

        # Force reranker warmup (blocking — daemon can afford to wait)
        retrieval_eng = getattr(_engine, '_retrieval_engine', None)
        if retrieval_eng:
            reranker = getattr(retrieval_eng, '_reranker', None)
            if reranker and hasattr(reranker, 'warmup_sync'):
                reranker.warmup_sync(timeout=120)

        logger.info("Daemon engine initialized and warm")
    return _engine


class DaemonHandler(BaseHTTPRequestHandler):
    """Lightweight HTTP handler for daemon requests."""

    def log_message(self, format, *args):
        """Suppress default access logging."""
        pass

    def _send_json(self, status: int, data: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode())

    def do_GET(self) -> None:
        global _last_activity
        _last_activity = time.monotonic()

        if self.path == "/health":
            self._send_json(200, {"status": "ok", "pid": os.getpid()})
            return

        if self.path.startswith("/recall"):
            try:
                # Parse query from URL params
                from urllib.parse import urlparse, parse_qs
                params = parse_qs(urlparse(self.path).query)
                query = params.get("q", [""])[0]
                limit = int(params.get("limit", ["20"])[0])

                # S9-DASH-02: session_id for outcome-queue enqueue.
                # Priority: ?session_id= query arg > X-SLM-Session-Id
                # header > synthetic "cli:<ts>". Without any of these
                # the recall still works — it just doesn't produce a
                # pending_outcome (hook-based signals can't match).
                session_id = params.get("session_id", [""])[0]
                if not session_id:
                    session_id = self.headers.get("X-SLM-Session-Id", "")
                if not session_id:
                    import time as _t
                    session_id = f"http:{int(_t.time() * 1000)}"

                engine = _get_engine()
                response = engine.recall(
                    query, limit=limit, session_id=session_id,
                )
                results = [
                    {"content": r.fact.content, "score": round(r.score, 4),
                     "fact_type": getattr(r.fact.fact_type, 'value', str(r.fact.fact_type)),
                     "fact_id": r.fact.fact_id}
                    for r in response.results
                ]
                self._send_json(200, {
                    "results": results, "count": len(results),
                    "query_type": response.query_type,
                    "retrieval_time_ms": round(response.retrieval_time_ms, 1),
                })
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        if self.path == "/list":
            try:
                engine = _get_engine()
                facts = engine.list_facts(limit=50)
                items = [
                    {"content": f.content[:100], "fact_type": getattr(f.fact_type, 'value', str(f.fact_type)),
                     "created_at": (f.created_at or "")[:19], "fact_id": f.fact_id}
                    for f in facts
                ]
                self._send_json(200, {"results": items, "count": len(items)})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        if self.path == "/status":
            engine = _get_engine()
            uptime = time.monotonic() - _server_start_time
            self._send_json(200, {
                "status": "running", "pid": os.getpid(),
                "uptime_s": round(uptime),
                "mode": engine._config.mode.value,
                "fact_count": engine.fact_count,
                "idle_s": round(time.monotonic() - _last_activity),
            })
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        global _last_activity
        _last_activity = time.monotonic()

        if self.path == "/remember":
            try:
                body = self._read_body()
                content = body.get("content", "")
                tags = body.get("tags", "")
                if not content:
                    self._send_json(400, {"error": "content required"})
                    return

                engine = _get_engine()
                metadata = {"tags": tags} if tags else {}
                fact_ids = engine.store(content, metadata=metadata)
                self._send_json(200, {"fact_ids": fact_ids, "count": len(fact_ids)})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        if self.path == "/observe":
            try:
                body = self._read_body()
                content = body.get("content", "")
                if not content:
                    self._send_json(400, {"error": "content required"})
                    return

                # V3.3.28: Debounced observation processing.
                # Buffers observations for 3s, deduplicates, processes as batch.
                # Returns immediately — the actual capture happens asynchronously
                # via the debounce timer, using the singleton engine.
                result = _enqueue_observation(content)
                self._send_json(200, result)
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        if self.path == "/stop":
            self._send_json(200, {"status": "stopping"})
            Thread(target=_shutdown_server, daemon=True).start()
            return

        self._send_json(404, {"error": "not found"})


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

_server: HTTPServer | None = None
_server_start_time = time.monotonic()


def _shutdown_server() -> None:
    global _engine, _server
    # V3.3.28: Flush any buffered observations before shutdown
    try:
        _flush_observe_buffer()
    except Exception:
        pass
    time.sleep(0.5)
    if _engine is not None:
        try:
            _engine.close()
        except Exception:
            pass
        _engine = None
    if _server is not None:
        _server.shutdown()
    _PID_FILE.unlink(missing_ok=True)
    _PORT_FILE.unlink(missing_ok=True)


def _idle_watchdog(timeout: int) -> None:
    """Auto-shutdown after idle timeout."""
    global _last_activity
    while True:
        time.sleep(30)
        idle = time.monotonic() - _last_activity
        if idle > timeout:
            logger.info("Daemon idle for %ds, shutting down", int(idle))
            _shutdown_server()
            os._exit(0)


def start_server(port: int = _DEFAULT_PORT, idle_timeout: int | None = None) -> None:
    """Start the daemon HTTP server. Blocks until stopped."""
    global _server, _server_start_time, _last_activity

    idle_timeout = idle_timeout or int(os.environ.get(
        "SLM_DAEMON_IDLE_TIMEOUT", str(_DEFAULT_IDLE_TIMEOUT),
    ))

    # Write PID + port files
    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(os.getpid()))
    _PORT_FILE.write_text(str(port))

    # Handle SIGTERM for graceful shutdown
    signal.signal(signal.SIGTERM, lambda *_: _shutdown_server() or os._exit(0))

    # Pre-warm engine (this is the cold start — daemon absorbs it once)
    logger.info("Daemon starting — warming engine...")
    _get_engine()
    logger.info("Engine warm. Daemon ready on port %d (idle timeout: %ds)", port, idle_timeout)

    _server_start_time = time.monotonic()
    _last_activity = time.monotonic()

    # Start idle watchdog
    Thread(target=_idle_watchdog, args=(idle_timeout,), daemon=True, name="idle-watchdog").start()

    # Start HTTP server
    # SO_REUSEADDR must be set on the class BEFORE __init__ calls bind()
    HTTPServer.allow_reuse_address = True
    _server = HTTPServer(("127.0.0.1", port), DaemonHandler)
    try:
        _server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown_server()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if "--start" in sys.argv:
        start_server()
    elif "--stop" in sys.argv:
        stop_daemon()
    else:
        print("Usage: python -m superlocalmemory.cli.daemon --start|--stop")
