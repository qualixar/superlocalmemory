#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Part of Qualixar | Author: Varun Pratap Bhardwaj (qualixar.com | varunpratap.com)
"""Security headers middleware for FastAPI servers.

Adds comprehensive security headers to all HTTP responses:
- X-Content-Type-Options: Prevents MIME type sniffing
- X-Frame-Options: Prevents clickjacking attacks
- X-XSS-Protection: Enables browser XSS filters
- Content-Security-Policy: Restricts resource loading
- Referrer-Policy: Controls referrer information leakage
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all HTTP responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and add security headers to response."""
        response = await call_next(request)

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking attacks
        response.headers["X-Frame-Options"] = "DENY"

        # Enable browser XSS filter (legacy, but doesn't hurt)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Content Security Policy (v3.4.22 — vendored assets, no CDN hosts).
        # All Bootstrap/D3/Sigma/graphology/Inter assets ship locally under
        # /static/vendor/, so we drop every CDN host from the allow-list.
        #
        # v3.7.9 (B2): 'unsafe-inline' removed from script-src. Every inline
        # on*= event handler was migrated to data-act-* attributes dispatched
        # by js/event-delegation.js (nonces cannot authorise inline event-
        # handler attributes, so delegation is the only real fix). A stored-XSS
        # payload can no longer execute inline script here.
        # 'unsafe-inline' is retained on style-src only: 82 inline style=
        # attributes remain, style injection is far lower risk than script
        # injection, and inline style attributes cannot be nonce'd either
        # (GitHub/GitLab keep the same script-locked / style-inline posture).
        # img-src drops the https: wildcard now that nothing remote loads.
        csp_directives = [
            "default-src 'self'",
            "script-src 'self'",
            "style-src 'self' 'unsafe-inline'",
            "font-src 'self'",
            "img-src 'self' data:",
            "connect-src 'self' ws://localhost:* ws://127.0.0.1:*",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)

        # Control referrer information leakage
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # v3.4.23: Cache-Control strategy
        # ---------------------------------------------------------------
        # Three classes of paths, three policies:
        #
        #   /api/*        -> no-store (sensitive data, never cache)
        #   index.html    -> no-cache, must-revalidate (always revalidate)
        #   /static/*     -> no-cache, must-revalidate (always revalidate
        #                    with ETag; fast reloads but never stale-after-
        #                    upgrade)
        #
        # Before v3.4.23 only /api/* had cache headers. Browsers then cached
        # JS/CSS/HTML aggressively via default heuristics, and after a daemon
        # upgrade the dashboard showed an infinite spinner because old cached
        # JS was calling endpoints with stale response shapes. "no-cache"
        # (not "no-store") still allows 304s on unchanged files, so reload
        # cost stays low.
        path = request.url.path
        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        elif path == "/" or path.endswith(".html") or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache, must-revalidate"

        return response
