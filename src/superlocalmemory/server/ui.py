#!/usr/bin/env python3
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""
SuperLocalMemory V3 - FastAPI UI Server
App initialization, middleware, static mount, and router registration.

All route handlers live in routes/ directory:
    routes/memories.py  -- /api/memories, /api/graph, /api/search, /api/clusters
    routes/stats.py     -- /api/stats, /api/timeline, /api/patterns
    routes/profiles.py  -- /api/profiles (CRUD + switch)
    routes/backup.py    -- /api/backup (status, create, configure, list)
    routes/data_io.py   -- /api/export, /api/import
    routes/events.py    -- /events/stream (SSE), /api/events [v2.5]
    routes/agents.py    -- /api/agents, /api/trust [v2.5]
    routes/ws.py        -- /ws/updates (WebSocket)
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


from superlocalmemory.server.routes.helpers import SLM_VERSION  # noqa: E402

_script_dir = str(Path(__file__).parent.resolve())
sys.path = [p for p in sys.path if p not in ("", _script_dir)]

try:
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    import uvicorn
except ImportError:
    raise ImportError(
        "FastAPI dependencies not installed. "
        "Install with: pip install 'fastapi[all]' uvicorn websockets"
    )

from superlocalmemory.server.security_middleware import SecurityHeadersMiddleware

# V3 Paths (migrated from ~/.claude-memory to ~/.superlocalmemory)
MEMORY_DIR = Path.home() / ".superlocalmemory"
DB_PATH = MEMORY_DIR / "memory.db"
# V3.3.21: UI shipped inside the package for pip/npm installs.
# Check package location first, then fall back to repo root for dev mode.
_PKG_UI = Path(__file__).resolve().parent.parent / "ui"
_REPO_UI = Path(__file__).resolve().parent.parent.parent.parent / "ui"
UI_DIR = _PKG_UI if (_PKG_UI / "index.html").exists() else _REPO_UI


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="SuperLocalMemory V3 UI Server",
        description="Memory Dashboard with V3 Engine, Trust, Learning, and Compliance",
        version=SLM_VERSION,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # Middleware (order matters: security headers should be outermost)
    application.add_middleware(SecurityHeadersMiddleware)
    application.add_middleware(GZipMiddleware, minimum_size=1000)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8765", "http://127.0.0.1:8765",
            "http://localhost:8417", "http://127.0.0.1:8417",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-SLM-API-Key"],
    )

    # Rate limiting (graceful)
    try:
        from superlocalmemory.infra.rate_limiter import RateLimiter
        _write_limiter = RateLimiter(max_requests=30, window_seconds=60)
        _read_limiter = RateLimiter(max_requests=120, window_seconds=60)

        @application.middleware("http")
        async def rate_limit_middleware(request, call_next):
            client_ip = request.client.host if request.client else "unknown"
            is_write = request.method in ("POST", "PUT", "DELETE", "PATCH")
            limiter = _write_limiter if is_write else _read_limiter
            allowed, remaining = limiter.is_allowed(client_ip)
            if not allowed:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=429,
                    content={"error": "Too many requests. Please slow down."},
                    headers={"Retry-After": str(limiter.window)},
                )
            response = await call_next(request)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            return response
    except (ImportError, Exception):
        pass

    # Auth middleware (graceful)
    try:
        from superlocalmemory.infra.auth_middleware import check_api_key

        @application.middleware("http")
        async def auth_middleware(request, call_next):
            is_write = request.method in ("POST", "PUT", "DELETE", "PATCH")
            headers = dict(request.headers)
            if not check_api_key(headers, is_write=is_write):
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid or missing API key."},
                )
            response = await call_next(request)
            return response
    except (ImportError, Exception):
        pass

    # Mount static files (UI directory)
    UI_DIR.mkdir(exist_ok=True)
    application.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

    # ========================================================================
    # Register Route Modules
    # ========================================================================
    from superlocalmemory.server.routes.memories import router as memories_router
    from superlocalmemory.server.routes.stats import router as stats_router
    from superlocalmemory.server.routes.profiles import router as profiles_router
    from superlocalmemory.server.routes.backup import router as backup_router
    from superlocalmemory.server.routes.data_io import router as data_io_router
    from superlocalmemory.server.routes.events import router as events_router, register_event_listener
    from superlocalmemory.server.routes.agents import router as agents_router
    from superlocalmemory.server.routes.ws import router as ws_router, manager as ws_manager

    application.include_router(memories_router)
    application.include_router(stats_router)
    application.include_router(profiles_router)
    application.include_router(backup_router)
    application.include_router(data_io_router)
    application.include_router(events_router)
    application.include_router(agents_router)
    application.include_router(ws_router)

    # V3 API endpoints (dashboard, mode, trust, math, etc.)
    from superlocalmemory.server.routes.v3_api import router as v3_router
    application.include_router(v3_router)

    # v3.4.1: Chat SSE + Insights + Timeline endpoints
    for _module_name_v341 in ("chat",):
        try:
            _mod_v341 = __import__(f"superlocalmemory.server.routes.{_module_name_v341}", fromlist=["router"])
            application.include_router(_mod_v341.router)
        except (ImportError, Exception) as _exc:
            logger.warning("Optional router %s failed: %s", _module_name_v341, _exc)

    # Graceful optional routers
    for _module_name in ("learning", "lifecycle", "behavioral", "compliance"):
        try:
            _mod = __import__(f"superlocalmemory.server.routes.{_module_name}", fromlist=["router"])
            application.include_router(_mod.router)
        except (ImportError, Exception):
            pass

    # Wire WebSocket manager into routes that need broadcast capability
    import superlocalmemory.server.routes.profiles as _profiles_mod
    import superlocalmemory.server.routes.data_io as _data_io_mod
    _profiles_mod.ws_manager = ws_manager
    _data_io_mod.ws_manager = ws_manager

    # ========================================================================
    # Basic Routes (root page + health check)
    # ========================================================================

    @application.get("/", response_class=HTMLResponse)
    async def root():
        """Serve main UI page."""
        index_path = UI_DIR / "index.html"
        if not index_path.exists():
            return (
                "<!DOCTYPE html><html><head>"
                "<title>SuperLocalMemory V3</title></head>"
                "<body style='font-family:Arial;padding:40px'>"
                "<h1>SuperLocalMemory V3 UI Server Running</h1>"
                "<p>UI not found. Check ui/index.html</p>"
                "<p><a href='/api/docs'>API Documentation</a></p>"
                "</body></html>"
            )
        return index_path.read_text()

    @application.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": SLM_VERSION,
            "database": "connected" if DB_PATH.exists() else "missing",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ========================================================================
    # Startup Events
    # ========================================================================

    @application.on_event("startup")
    async def startup_event():
        """Initialize event bus and warm up worker subprocess.

        Engine runs in subprocess worker (never in this process).
        Background warmup pre-loads PyTorch + models so first recall is fast.
        """
        application.state.engine = None
        logger.info("Dashboard started (~60 MB, engine runs in subprocess worker)")
        register_event_listener()

        # Background warmup: pre-spawn worker and load all models.
        # This runs in a daemon thread — dashboard is responsive immediately.
        # Worker will be ready by the time user does first search (~10-30s).
        try:
            from superlocalmemory.core.worker_pool import WorkerPool
            WorkerPool.shared().warmup()
            logger.info("Worker warmup initiated (background)")
        except Exception as exc:
            logger.warning("Worker warmup failed to start: %s", exc)

    @application.on_event("shutdown")
    async def shutdown_event():
        """Kill worker subprocess on dashboard shutdown."""
        try:
            from superlocalmemory.core.worker_pool import WorkerPool
            WorkerPool.shared().shutdown()
        except Exception:
            pass

    return application


app = create_app()

# ============================================================================
# Server Startup
# ============================================================================

if __name__ == "__main__":
    import argparse
    import socket

    parser = argparse.ArgumentParser(description="SuperLocalMemory V3 - Web Dashboard")
    parser.add_argument("--port", type=int, default=8765, help="Port (default 8765)")
    parser.add_argument("--profile", type=str, default=None, help="Memory profile")
    args = parser.parse_args()

    def find_available_port(preferred: int) -> int:
        for port in [preferred] + list(range(preferred + 1, preferred + 20)):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", port))
                    return port
            except OSError:
                continue
        return preferred

    ui_port = find_available_port(args.port)
    if ui_port != args.port:
        print(f"\n  Port {args.port} in use -- using {ui_port} instead\n")

    print("=" * 70)
    print("  SuperLocalMemory V3 - Web Dashboard")
    print("  Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar")
    print("=" * 70)
    print(f"  Database:  {DB_PATH}")
    print(f"  UI:        {UI_DIR}")
    print("=" * 70)
    print(f"\n  Dashboard:   http://localhost:{ui_port}")
    print(f"  API Docs:    http://localhost:{ui_port}/api/docs")
    print(f"  Health:      http://localhost:{ui_port}/health")
    print(f"  SSE Stream:  http://localhost:{ui_port}/events/stream")
    print(f"  WebSocket:   ws://localhost:{ui_port}/ws/updates")
    print("\n  Press Ctrl+C to stop\n")

    uvicorn.run(app, host="127.0.0.1", port=ui_port, log_level="info", access_log=True)
