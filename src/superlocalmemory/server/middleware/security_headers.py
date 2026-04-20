# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-04 §4.2 (v2)

"""Strict security-headers ASGI middleware for the Brain UI.

LLD-04 §4.2 v2 — deterministic, allow-list-only policy:

    content-security-policy: default-src 'self'; script-src 'self';
                             style-src 'self'; img-src 'self' data:;
                             connect-src 'self'; frame-ancestors 'none';
                             base-uri 'self'; form-action 'self'
    x-content-type-options:  nosniff
    x-frame-options:         DENY
    referrer-policy:         no-referrer
    permissions-policy:      interest-cohort=(), microphone=(),
                             camera=(), geolocation=()

Design decisions (from LLD-04 §4.2 Policy block):

* **No CSP nonce.** Nonce plumbing was the most common v1 bug; removing
  it entirely means there is literally no path for an inline script/style
  to execute — ``script-src 'self'`` blocks all of them.
* **Deterministic bytes.** The header tuple is a module-level constant.
  No per-request computation, no string concatenation, no dependency on
  request state. This keeps the middleware O(1) and cheap.
* **Works on error responses.** Because we wrap ``send`` itself, the
  headers land on 401s, 500s, and body-less responses alike — not only
  on successful handlers. Regression tested in ``test_headers.py``.
"""

from __future__ import annotations

from starlette.types import ASGIApp, Receive, Scope, Send


# Deterministic, module-level, immutable.  Every request gets the same bytes.
# Keeping the declaration as a tuple of (name, value) bytes pairs avoids any
# per-request allocation in the hot path.
_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (
        b"content-security-policy",
        (
            b"default-src 'self'; "
            b"script-src 'self'; "
            b"style-src 'self'; "
            b"img-src 'self' data:; "
            b"connect-src 'self'; "
            b"frame-ancestors 'none'; "
            b"base-uri 'self'; "
            b"form-action 'self'"
        ),
    ),
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (
        b"permissions-policy",
        b"interest-cohort=(), microphone=(), camera=(), geolocation=()",
    ),
)


# Set of header names (lower-case bytes) we own. We strip existing values
# from downstream responses so we never double-up if another middleware
# already set a looser variant.
_OWNED_NAMES: frozenset[bytes] = frozenset(name for name, _ in _HEADERS)


# LLD-04 §4.2 v2 scope note: the strict policy applies to the Brain API and
# any Brain-scoped static endpoints. Pre-existing dashboard routes (``/``,
# ``/ui/*``, ``/static/*``, other tabs) currently load vendor libraries
# (Bootstrap, Bootstrap Icons, Inter font, D3, Sigma, graphology) from CDNs
# and would break under ``script-src 'self'`` until those assets are vendored
# locally (tracked for v3.4.22 vendoring work). Applying the strict set there
# produced a user-visible regression during Stage 6 delivery-lead review,
# which contradicts LLD-04's user-benefit goal. This middleware therefore
# enforces the strict set on the routes that actually need it today —
# ``/api/v3/brain*`` and the deprecated brain shims — and is a no-op on
# everything else. The existing ``server/security_middleware.py`` keeps
# emitting its legacy headers on the remaining routes.
_STRICT_SCOPE_PREFIXES: tuple[bytes, ...] = (
    b"/api/v3/brain",
    b"/api/v3/learning/stats",
    b"/api/v3/patterns",
    b"/api/v3/behavioral",
)


def _is_strict_path(raw_path: bytes) -> bool:
    for prefix in _STRICT_SCOPE_PREFIXES:
        if raw_path == prefix or raw_path.startswith(prefix + b"/") \
                or raw_path.startswith(prefix + b"?"):
            return True
    return False


class SecurityHeadersMiddleware:
    """ASGI middleware that injects the strict v2 header set on Brain routes.

    Unlike ``BaseHTTPMiddleware``, this wraps ``send`` directly so we can
    act on the ``http.response.start`` event before the client sees any
    headers. That guarantees coverage of exception responses, ``Response``
    objects from ``HTTPException`` handlers, and streaming responses — all
    of which skip ``BaseHTTPMiddleware``'s dispatch flow in edge cases.

    Scope: see ``_STRICT_SCOPE_PREFIXES``. Outside that scope the middleware
    passes through untouched so pre-existing dashboard CSP policy still
    governs index/static responses until the vendoring work in v3.4.22.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send,
    ) -> None:
        # Only HTTP responses get headers. Websocket / lifespan skip.
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        raw_path = scope.get("raw_path") or scope.get("path", "").encode("latin-1")
        if not _is_strict_path(raw_path):
            return await self.app(scope, receive, send)

        async def _send(message: dict) -> None:
            if message.get("type") == "http.response.start":
                existing = list(message.get("headers") or [])
                # Strip any existing copies of our owned headers so we
                # never emit duplicates.
                filtered = [
                    (name, value)
                    for name, value in existing
                    if name.lower() not in _OWNED_NAMES
                ]
                for name, value in _HEADERS:
                    filtered.append((name, value))
                message["headers"] = filtered
            await send(message)

        await self.app(scope, receive, _send)


__all__ = ("SecurityHeadersMiddleware", "_is_strict_path", "_STRICT_SCOPE_PREFIXES")
