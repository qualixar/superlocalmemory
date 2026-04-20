#!/usr/bin/env python3
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""
SuperLocalMemory V3 - FastAPI API Server
Provides REST endpoints for memory visualization and exploration.
Uses V3 MemoryEngine for all operations.
"""

import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import uvicorn

from superlocalmemory.server.security_middleware import SecurityHeadersMiddleware
from superlocalmemory.server.routes.helpers import SLM_VERSION

logger = logging.getLogger("superlocalmemory.api_server")

# V3 paths
MEMORY_DIR = Path.home() / ".superlocalmemory"
DB_PATH = MEMORY_DIR / "memory.db"
# V3.3.21: UI shipped inside the package for pip/npm installs.
_PKG_UI = Path(__file__).resolve().parent.parent / "ui"
_REPO_UI = Path(__file__).resolve().parent.parent.parent.parent / "ui"
UI_DIR = _PKG_UI if (_PKG_UI / "index.html").exists() else _REPO_UI


# ============================================================================
# Request/Response Models
# ============================================================================

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
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

    yield

    # Cleanup
    if hasattr(application.state, 'engine') and application.state.engine:
        application.state.engine.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="SuperLocalMemory V3 API",
        description="V3 Memory Engine REST API",
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
                    content={"error": "Too many requests."},
                    headers={"Retry-After": str(limiter.window_seconds)},
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
    from superlocalmemory.server.routes.memories import router as memories_router
    from superlocalmemory.server.routes.stats import router as stats_router
    from superlocalmemory.server.routes.profiles import router as profiles_router
    from superlocalmemory.server.routes.backup import router as backup_router
    from superlocalmemory.server.routes.data_io import router as data_io_router
    from superlocalmemory.server.routes.events import router as events_router, register_event_listener
    from superlocalmemory.server.routes.agents import router as agents_router
    from superlocalmemory.server.routes.ws import router as ws_router, manager as ws_manager
    from superlocalmemory.server.routes.v3_api import router as v3_router

    application.include_router(memories_router)
    application.include_router(stats_router)
    application.include_router(profiles_router)
    application.include_router(backup_router)
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
    import superlocalmemory.server.routes.profiles as _profiles_mod
    import superlocalmemory.server.routes.data_io as _data_io_mod
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
                "<html><head><title>SuperLocalMemory V3</title></head>"
                "<body style='font-family:Arial;padding:40px'>"
                "<h1>SuperLocalMemory V3 API Server Running</h1>"
                "<p><a href='/docs'>API Documentation</a></p>"
                "</body></html>"
            )
        return index_path.read_text()

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

    @application.on_event("startup")
    async def startup_event():
        register_event_listener()

    return application


# v3.4.3: Module-level app removed to prevent duplicate MemoryEngine.
# The unified daemon (server/unified_daemon.py) imports routes from this
# module via _register_dashboard_routes(). Do NOT create app here.
# For standalone use: python -m superlocalmemory.server.api

if __name__ == "__main__":
    app = create_app()
    print("=" * 60)
    print("SuperLocalMemory V3 - API Server (standalone mode)")
    print("=" * 60)
    print(f"Database: {DB_PATH}")
    print(f"UI Directory: {UI_DIR}")
    print("=" * 60)
    print("\nStarting server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(
        app, host="127.0.0.1", port=8000, log_level="info",
    )
