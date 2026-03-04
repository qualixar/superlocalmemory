#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
SuperLocalMemory V2.5.0 - FastAPI UI Server
App initialization, middleware, static mount, and router registration.

All route handlers live in routes/ directory:
    routes/memories.py  — /api/memories, /api/graph, /api/search, /api/clusters
    routes/stats.py     — /api/stats, /api/timeline, /api/patterns
    routes/profiles.py  — /api/profiles (CRUD + switch)
    routes/backup.py    — /api/backup (status, create, configure, list)
    routes/data_io.py   — /api/export, /api/import
    routes/events.py    — /events/stream (SSE), /api/events [v2.5]
    routes/agents.py    — /api/agents, /api/trust [v2.5]
    routes/ws.py        — /ws/updates (WebSocket)
"""

import sys
from pathlib import Path
from datetime import datetime

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

from security_middleware import SecurityHeadersMiddleware

# Add src/ and routes/ to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Configuration
MEMORY_DIR = Path.home() / ".claude-memory"
DB_PATH = MEMORY_DIR / "memory.db"
UI_DIR = Path(__file__).parent / "ui"

# Initialize FastAPI application
app = FastAPI(
    title="SuperLocalMemory V2.5.0 UI Server",
    description="Real-Time Memory Dashboard with Event Bus, Agent Registry, and Trust Scoring",
    version="2.5.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Middleware (order matters: security headers should be outermost)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8765",    # Dashboard
        "http://127.0.0.1:8765",
        "http://localhost:8417",    # MCP
        "http://127.0.0.1:8417",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-SLM-API-Key"],
)

# Rate limiting (v2.6)
try:
    from rate_limiter import write_limiter, read_limiter

    @app.middleware("http")
    async def rate_limit_middleware(request, call_next):
        client_ip = request.client.host if request.client else "unknown"

        # Determine if this is a write or read endpoint
        is_write = request.method in ("POST", "PUT", "DELETE", "PATCH")
        limiter = write_limiter if is_write else read_limiter

        allowed, remaining = limiter.is_allowed(client_ip)
        if not allowed:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests. Please slow down."},
                headers={"Retry-After": str(limiter.window)}
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

except ImportError:
    pass  # Rate limiter not available — continue without it

# Optional API key authentication (v2.6)
try:
    from auth_middleware import check_api_key

    @app.middleware("http")
    async def auth_middleware(request, call_next):
        is_write = request.method in ("POST", "PUT", "DELETE", "PATCH")
        headers = dict(request.headers)
        if not check_api_key(headers, is_write=is_write):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or missing API key. Set X-SLM-API-Key header."}
            )
        response = await call_next(request)
        return response
except ImportError:
    pass  # Auth middleware not available

# Mount static files (UI directory)
UI_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

# ============================================================================
# Register Route Modules
# ============================================================================

from routes.memories import router as memories_router
from routes.stats import router as stats_router
from routes.profiles import router as profiles_router
from routes.backup import router as backup_router
from routes.data_io import router as data_io_router
from routes.events import router as events_router, register_event_listener
from routes.agents import router as agents_router
from routes.ws import router as ws_router, manager as ws_manager

# v2.7 Learning routes (graceful)
try:
    from routes.learning import router as learning_router
    LEARNING_ROUTES = True
except ImportError:
    LEARNING_ROUTES = False

app.include_router(memories_router)
app.include_router(stats_router)
app.include_router(profiles_router)
app.include_router(backup_router)
app.include_router(data_io_router)
app.include_router(events_router)
app.include_router(agents_router)
app.include_router(ws_router)
if LEARNING_ROUTES:
    app.include_router(learning_router)

# v2.8 routes (graceful — don't fail if engines unavailable)
try:
    from routes.lifecycle import router as lifecycle_router
    app.include_router(lifecycle_router)
except ImportError:
    pass

try:
    from routes.behavioral import router as behavioral_router
    app.include_router(behavioral_router)
except ImportError:
    pass

try:
    from routes.compliance import router as compliance_router
    app.include_router(compliance_router)
except ImportError:
    pass

# Wire WebSocket manager into routes that need broadcast capability
import routes.profiles as _profiles_mod
import routes.data_io as _data_io_mod
_profiles_mod.ws_manager = ws_manager
_data_io_mod.ws_manager = ws_manager

# ============================================================================
# Basic Routes (root page + health check)
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main UI page."""
    index_path = UI_DIR / "index.html"
    if not index_path.exists():
        return """
        <!DOCTYPE html>
        <html>
            <head><title>SuperLocalMemory V2.5.0</title></head>
            <body style="font-family: Arial; padding: 40px;">
                <h1>SuperLocalMemory V2.5.0 UI Server Running</h1>
                <p>UI not found. Check ui/index.html</p>
                <p><a href="/api/docs">API Documentation</a></p>
            </body>
        </html>
        """
    return index_path.read_text()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.5.0",
        "database": "connected" if DB_PATH.exists() else "missing",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Startup Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Register Event Bus listener for SSE bridge on startup."""
    register_event_listener()


# ============================================================================
# Server Startup
# ============================================================================

if __name__ == "__main__":
    import argparse
    import socket

    parser = argparse.ArgumentParser(description="SuperLocalMemory V2 - Web Dashboard")
    parser.add_argument("--port", type=int, default=8765, help="Port to run on (default 8765)")
    parser.add_argument("--profile", type=str, default=None, help="Memory profile to use")
    args = parser.parse_args()

    def find_available_port(preferred):
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
        print(f"\n  Port {args.port} in use — using {ui_port} instead\n")

    print("=" * 70)
    print("  SuperLocalMemory V2.5.0 - Web Dashboard")
    print("  Copyright (c) 2026 Varun Pratap Bhardwaj")
    print("=" * 70)
    print(f"  Database:  {DB_PATH}")
    print(f"  UI:        {UI_DIR}")
    print(f"  Routes:    8 modules, 28 endpoints")
    print("=" * 70)
    print(f"\n  Dashboard:   http://localhost:{ui_port}")
    print(f"  API Docs:    http://localhost:{ui_port}/api/docs")
    print(f"  Health:      http://localhost:{ui_port}/health")
    print(f"  SSE Stream:  http://localhost:{ui_port}/events/stream")
    print(f"  WebSocket:   ws://localhost:{ui_port}/ws/updates")
    print("\n  Press Ctrl+C to stop\n")

    # SECURITY: Bind to localhost only
    uvicorn.run(app, host="127.0.0.1", port=ui_port, log_level="info", access_log=True)
