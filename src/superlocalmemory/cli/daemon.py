# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the Elastic License 2.0 - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLM Daemon — keeps engine warm for instant CLI/MCP response.

Problem: CLI cold start is 23s (embedding worker spawn + model load).
Solution: Background daemon keeps MemoryEngine warm. CLI commands route
requests through the daemon via localhost HTTP (~10ms overhead).

Architecture:
  slm serve       → starts daemon (engine init, workers warm, ~600MB RAM)
  slm remember X  → HTTP POST to daemon → instant (no cold start)
  slm recall X    → HTTP GET from daemon → instant
  slm serve stop  → graceful shutdown, workers killed, RAM freed

Auto-start: if daemon not running on CLI use, starts it automatically.
Auto-shutdown: after 30 min idle (configurable via SLM_DAEMON_IDLE_TIMEOUT).

Memory safety:
  - RSS watchdog on embedding worker (2.5GB cap)
  - Worker recycling every 5000 requests
  - Parent watchdog kills workers if daemon dies
  - SQLite WAL mode for concurrent access

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: Elastic-2.0
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

_DEFAULT_PORT = 8767
_DEFAULT_IDLE_TIMEOUT = 1800  # 30 min
_PID_FILE = Path.home() / ".superlocalmemory" / "daemon.pid"
_PORT_FILE = Path.home() / ".superlocalmemory" / "daemon.port"


# ---------------------------------------------------------------------------
# Client: check if daemon running + send requests
# ---------------------------------------------------------------------------

def is_daemon_running() -> bool:
    """Check if daemon is alive via PID file + HTTP health check."""
    if not _PID_FILE.exists():
        return False
    try:
        pid = int(_PID_FILE.read_text().strip())
        os.kill(pid, 0)  # Check if process exists
    except (ValueError, ProcessLookupError, PermissionError):
        _PID_FILE.unlink(missing_ok=True)
        return False

    # PID exists — verify HTTP health
    port = _get_port()
    try:
        import urllib.request
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=2,
        )
        return resp.status == 200
    except Exception:
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


def ensure_daemon() -> bool:
    """Start daemon if not running. Returns True if daemon is ready."""
    if is_daemon_running():
        return True

    # Start daemon in background
    import subprocess
    cmd = [sys.executable, "-m", "superlocalmemory.cli.daemon", "--start"]
    log_dir = Path.home() / ".superlocalmemory" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "daemon.log"

    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True

    with open(log_file, "a") as lf:
        subprocess.Popen(
            cmd, stdout=lf, stderr=lf,
            **kwargs
        )

    # Wait for daemon to become ready (max 30s for cold start)
    for _ in range(60):
        time.sleep(0.5)
        if is_daemon_running():
            return True

    return False


def stop_daemon() -> bool:
    """Stop the running daemon gracefully."""
    if not _PID_FILE.exists():
        return True
    try:
        pid = int(_PID_FILE.read_text().strip())
        if sys.platform == "win32":
            import subprocess
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(pid)])
        else:
            os.kill(pid, signal.SIGTERM)

        # Wait for cleanup
        for _ in range(20):
            time.sleep(0.5)
            try:
                if sys.platform == "win32":
                    import ctypes
                    # Using OpenProcess to check if PID is alive on Windows
                    # PROCESS_QUERY_INFORMATION = 0x0400
                    handle = ctypes.windll.kernel32.OpenProcess(0x0400, False, pid)
                    if handle == 0:
                        break
                    ctypes.windll.kernel32.CloseHandle(handle)
                else:
                    os.kill(pid, 0)
            except ProcessLookupError:
                break
        _PID_FILE.unlink(missing_ok=True)
        _PORT_FILE.unlink(missing_ok=True)
        return True
    except Exception:
        return False


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

                engine = _get_engine()
                response = engine.recall(query, limit=limit)
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
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, lambda *_: _shutdown_server() or os._exit(0))
    else:
        # On Windows, we can use SIGBREAK if available
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, lambda *_: _shutdown_server() or os._exit(0))

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
