#!/usr/bin/env python3
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""
SuperLocalMemory V3 - FastAPI API Server
Provides REST endpoints for memory visualization and exploration.
Uses V3 MemoryEngine for all operations.

v3.7.8 (WS3 F4): ``create_app()`` in this module is a standalone/legacy app
factory. The running daemon serves ``superlocalmemory.server.unified_daemon:
create_app`` (see ``unified_daemon.py``'s uvicorn config); THIS factory is
reachable only via direct ``python -m superlocalmemory.server.api`` /
programmatic use and the tests that exercise it directly
(``tests/test_api/test_api_lifespan_contract.py``,
``tests/test_security/test_rate_limit_e2e.py``). Its ``auth_middleware``
below intentionally keeps the older, simpler ``check_api_key``-only gate
(unconditional per-write check, no daemon-capability / install-token
identity layer) because it predates and is independent of the unified
daemon's richer ``write_identity.require_http_mutation_actor`` boundary and
the ``SLM_REQUIRE_API_KEY_LOOPBACK`` opt-in (see
``infra/auth_middleware.py``). Do not treat this module's auth posture as
the production write boundary -- that is ``unified_daemon.py``'s
``auth_middleware``. ``UI_DIR`` below is still imported by
``unified_daemon.py`` and must not be removed.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from superlocalmemory.core.config import CANONICAL_RECALL_LIMIT
from superlocalmemory.infra.data_root import DynamicStatePath
from superlocalmemory.server.routes.helpers import SLM_VERSION
from superlocalmemory.server.security_middleware import SecurityHeadersMiddleware

logger = logging.getLogger("superlocalmemory.api_server")

# V3 paths
MEMORY_DIR = DynamicStatePath()
DB_PATH = DynamicStatePath("memory.db")
# V3.3.21+: the dashboard ships inside the package (superlocalmemory/ui) for
# every install path. The legacy repo-root ui/ dev fallback was retired in
# v3.8.0 when that copy was deleted; the packaged directory is authoritative.
UI_DIR = Path(__file__).resolve().parent.parent / "ui"


# ============================================================================
# Request/Response Models
# ============================================================================

class SearchRequest(BaseModel):
    query: str
    limit: int = CANONICAL_RECALL_LIMIT
    min_score: float = 0.3


class MemoryFilter(BaseModel):
    category: Optional[str] = None
    project_name: Optional[str] = None
    cluster_id: Optional[int] = None
    min_importance: Optional[int] = None


# ============================================================================
# V3 Engine Initialization (lifespan context)
# ============================================================================

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize V3 engine on startup, cleanup on shutdown."""
    try:
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.engine import MemoryEngine

        config = SLMConfig.load()
        engine = MemoryEngine(config)
        engine.initialize()
        application.state.engine = engine
        application.state.config = config
        logger.info("V3 MemoryEngine initialized: mode=%s", config.mode.value)
    except Exception as exc:
        logger.warning("V3 engine init failed (API will use fallback): %s", exc)
        application.state.engine = None
        application.state.config = None

    # Event fan-out belongs to the same application lifecycle as the engine.
    # Registering it through FastAPI.on_event created a second, deprecated
    # startup path and made TestClient initialization emit warnings.
    from superlocalmemory.server.routes.events import register_event_listener

    register_event_listener()
    yield

    # Cleanup
    if hasattr(application.state, 'engine') and application.state.engine:
        application.state.engine.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="SuperLocalMemory V4 API",
        description="V4 Memory Engine REST API",
        version=SLM_VERSION,
        lifespan=lifespan,
    )

    # Middleware
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
        from superlocalmemory.core.remote_mode import (
            is_rate_limit_exempt,
            rate_limit_config,
        )
        from superlocalmemory.infra.rate_limiter import RateLimiter
        # v3.6.12 (issue #40): env-tunable thresholds (defaults unchanged).
        _rl_write, _rl_read, _rl_window = rate_limit_config()
        _write_limiter = RateLimiter(max_requests=_rl_write, window_seconds=_rl_window)
        _read_limiter = RateLimiter(max_requests=_rl_read, window_seconds=_rl_window)

        @application.middleware("http")
        async def rate_limit_middleware(request, call_next):
            client_ip = request.client.host if request.client else "unknown"
            if is_rate_limit_exempt(client_ip):
                return await call_next(request)
            is_write = request.method in ("POST", "PUT", "DELETE", "PATCH")
            limiter = _write_limiter if is_write else _read_limiter
            allowed, remaining = limiter.is_allowed(client_ip)
            if not allowed:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=429,
                    content={"error": "Too many requests."},
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

    # Mount static files
    UI_DIR.mkdir(exist_ok=True)
    application.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

    # ========================================================================
    # Register Route Modules
    # ========================================================================
    from superlocalmemory.server.routes.agents import router as agents_router
    from superlocalmemory.server.routes.backup import router as backup_router
    from superlocalmemory.server.routes.data_io import router as data_io_router
    from superlocalmemory.server.routes.events import router as events_router
    from superlocalmemory.server.routes.memories import router as memories_router
    from superlocalmemory.server.routes.profiles import router as profiles_router
    from superlocalmemory.server.routes.stats import router as stats_router
    from superlocalmemory.server.routes.v3_api import router as v3_router
    from superlocalmemory.server.routes.ws import manager as ws_manager
    from superlocalmemory.server.routes.ws import router as ws_router

    application.include_router(memories_router)
    # 4.1.21: the summary routes moved out of memories.py; saved views are new.
    from superlocalmemory.server.routes.summaries import router as summaries_router
    from superlocalmemory.server.routes.views import router as views_router
    application.include_router(summaries_router)
    application.include_router(views_router)
    application.include_router(stats_router)
    application.include_router(profiles_router)
    application.include_router(backup_router)
    from superlocalmemory.server.routes.backup_encryption import router as backup_encryption_router
    application.include_router(backup_encryption_router)
    application.include_router(data_io_router)
    application.include_router(events_router)
    application.include_router(agents_router)
    application.include_router(ws_router)
    application.include_router(v3_router)

    # v3.4.1: Chat SSE endpoint
    for _module_name_v341 in ("chat",):
        try:
            _mod_v341 = __import__(f"superlocalmemory.server.routes.{_module_name_v341}", fromlist=["router"])
            application.include_router(_mod_v341.router)
        except (ImportError, Exception):
            pass

    # Graceful optional routers
    for _module_name in ("learning", "lifecycle", "behavioral", "compliance", "insights", "timeline"):
        try:
            _mod = __import__(f"superlocalmemory.server.routes.{_module_name}", fromlist=["router"])
            application.include_router(_mod.router)
        except (ImportError, Exception):
            pass

    # Wire WebSocket manager
    import superlocalmemory.server.routes.data_io as _data_io_mod
    import superlocalmemory.server.routes.profiles as _profiles_mod
    _profiles_mod.ws_manager = ws_manager
    _data_io_mod.ws_manager = ws_manager

    # ========================================================================
    # Basic Routes
    # ========================================================================

    @application.get("/", response_class=HTMLResponse)
    async def root():
        """Serve main UI page."""
        index_path = UI_DIR / "index.html"
        if not index_path.exists():
            return (
                "<html><head><title>SuperLocalMemory V4</title></head>"
                "<body style='font-family:Arial;padding:40px'>"
                "<h1>SuperLocalMemory V4 API Server Running</h1>"
                "<p><a href='/docs'>API Documentation</a></p>"
                "</body></html>"
            )
        from superlocalmemory import __version__ as _v

        # __SLM_VERSION__ was substituted only by the unified daemon, so the
        # dashboard's upgrade detector did nothing when served from here.
        # Asset versioning is cosmetic. It must never be why this page 500s.
        #
        # The import is deferred (house style, keeps startup lean), which means
        # it resolves at REQUEST time — so when `pip install -e .` replaced the
        # installed package underneath a running daemon, this route began
        # answering "Internal Server Error" on the dashboard while every other
        # endpoint was fine. A stale hand-written version string is a trifle; a
        # blank page is not. Fall back to the file as written.
        try:
            from superlocalmemory.server.asset_versions import render_index

            return render_index(
                index_path, UI_DIR, substitutions={"__SLM_VERSION__": _v},
            )
        except Exception as exc:  # noqa: BLE001 — serve the page regardless
            logger.warning(
                "asset version rewrite unavailable, serving index.html as "
                "written: %s: %s", type(exc).__name__, exc,
            )
            return index_path.read_text(encoding="utf-8").replace("__SLM_VERSION__", _v)

    @application.get("/health")
    async def health_check():
        """Health check."""
        from datetime import datetime, timezone
        engine = application.state.engine
        return {
            "status": "healthy",
            "version": SLM_VERSION,
            "engine": "initialized" if engine else "unavailable",
            "database": "connected" if DB_PATH.exists() else "missing",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Outermost: another computer needs credentials (access_gate), and a
    # proxied request never counts as local (forwarded_guard).
    from superlocalmemory.server.access_gate import NonLocalAccessGate
    from superlocalmemory.server.forwarded_guard import ForwardedLoopbackDemotion
    application.add_middleware(NonLocalAccessGate)
    application.add_middleware(ForwardedLoopbackDemotion)
    return application


# v3.4.3: Module-level app removed to prevent duplicate MemoryEngine.
# The unified daemon (server/unified_daemon.py) imports routes from this
# module via _register_dashboard_routes(). Do NOT create app here.
# For standalone use: python -m superlocalmemory.server.api

if __name__ == "__main__":
    app = create_app()
    print("=" * 60)
    print("SuperLocalMemory V4 - API Server (standalone mode)")
    print("=" * 60)
    print(f"Database: {DB_PATH}")
    print(f"UI Directory: {UI_DIR}")
    print("=" * 60)
    print("\nStarting server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop\n")

    from superlocalmemory.server.forwarded_guard import uvicorn_proxy_options
    uvicorn.run(
        app, host="127.0.0.1", port=8000, log_level="info",
        **uvicorn_proxy_options(),
    )
