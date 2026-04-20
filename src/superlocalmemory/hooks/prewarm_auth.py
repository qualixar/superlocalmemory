# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-01 §4.5

"""Authentication primitives for the /internal/prewarm daemon route.

LLD reference: `.backup/active-brain/lld/LLD-01-context-cache-and-hot-path-hooks.md`
Section 4.5.

Four gates, applied in order, BEFORE any engine work:
  1. Loopback-only — client address must be 127.0.0.1 / ::1.
  2. Origin-header CSRF guard — browsers always send Origin on CORS
     requests; hooks using stdlib urllib do not. Present Origin ⇒ reject.
  3. Install-token match — X-SLM-Hook-Token constant-time compared to
     the bytes stored at ``~/.superlocalmemory/.install_token``.
  4. Body-size cap — requests > ``MAX_BODY_BYTES`` rejected upfront.

Framework-agnostic. The FastAPI route composes these primitives; tests
exercise them without starting an HTTP server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from superlocalmemory.core.security_primitives import verify_install_token

# Headers we consider equivalent for the install-token lookup. Covers the
# common normalizations (exact, lowercase, title-case).
_TOKEN_HEADER_VARIANTS: tuple[str, ...] = (
    "X-SLM-Hook-Token",
    "x-slm-hook-token",
    "X-Slm-Hook-Token",
)

_ORIGIN_HEADER_VARIANTS: tuple[str, ...] = ("Origin", "origin")

# Loopback addresses accepted by LLD-01. ``localhost`` is NOT included per
# SEC-01-02 — we want literal IPs only to avoid DNS-based bypass tricks.
_LOOPBACK_ADDRS: frozenset[str] = frozenset({"127.0.0.1", "::1"})

# Body-size cap: LLD-01 §4.5 step 4 → 8 KB.
MAX_BODY_BYTES: int = 8 * 1024


@dataclass(frozen=True, slots=True)
class AuthDecision:
    """Outcome of ``authorize``.

    - ``allowed`` — True when the request passes every gate.
    - ``status`` — suggested HTTP status when ``allowed`` is False
      (``200`` otherwise).
    - ``reason`` — short machine-readable tag. Never echoes secrets.
    """

    allowed: bool
    status: int
    reason: str = ""


# ---------------------------------------------------------------------------
# Gate 1: Loopback-only
# ---------------------------------------------------------------------------


def is_loopback(client_host: str) -> bool:
    """Return True iff ``client_host`` is an accepted loopback literal."""
    if not isinstance(client_host, str) or not client_host:
        return False
    return client_host in _LOOPBACK_ADDRS


# ---------------------------------------------------------------------------
# Gate 2: Origin CSRF guard
# ---------------------------------------------------------------------------


def is_browser_originated(headers: Mapping[str, str]) -> bool:
    """True if the request carries a non-empty ``Origin`` header.

    Defensive against accidental case variants. We treat an explicit empty
    Origin as non-browser per LLD-01 §4.5 — real browsers always send a
    non-empty origin on cross-origin requests.
    """
    if not headers:
        return False
    for name in _ORIGIN_HEADER_VARIANTS:
        val = headers.get(name)
        if val:
            return True
    return False


# ---------------------------------------------------------------------------
# Gate 3: Install-token
# ---------------------------------------------------------------------------


def _extract_token(headers: Mapping[str, str]) -> str:
    """Return the presented X-SLM-Hook-Token across casing variants."""
    if not headers:
        return ""
    for name in _TOKEN_HEADER_VARIANTS:
        val = headers.get(name)
        if val:
            return val
    return ""


# ---------------------------------------------------------------------------
# Gate 4: Body size
# ---------------------------------------------------------------------------


def check_body_size(body: bytes) -> tuple[bool, str]:
    """Verify request body is within ``MAX_BODY_BYTES``.

    Returns ``(True, "")`` on pass and ``(False, reason)`` on fail.
    """
    if not isinstance(body, (bytes, bytearray)):
        return False, "body must be bytes"
    if len(body) > MAX_BODY_BYTES:
        return False, f"body size {len(body)} exceeds {MAX_BODY_BYTES}"
    return True, ""


# ---------------------------------------------------------------------------
# Composite authorize()
# ---------------------------------------------------------------------------


def authorize(
    *,
    client_host: str,
    headers: Mapping[str, str],
) -> AuthDecision:
    """Run gates 1 → 2 → 3 in order and return the first failure.

    Order rationale:
      - Loopback check runs first so we reject off-host traffic with 403
        before touching any user-supplied header material.
      - Origin check runs second to neutralize browser-driven CSRF even
        when the attacker somehow obtained the install token.
      - Token check runs last; constant-time compared via
        ``verify_install_token``.
    """
    if not is_loopback(client_host):
        return AuthDecision(False, 403, "loopback only")

    if is_browser_originated(headers):
        return AuthDecision(False, 403, "origin header not allowed")

    token = _extract_token(headers)
    if not token:
        return AuthDecision(False, 401, "unauthorized: missing token")
    if not verify_install_token(token):
        return AuthDecision(False, 401, "unauthorized: token mismatch")

    return AuthDecision(True, 200, "")


__all__ = (
    "AuthDecision",
    "MAX_BODY_BYTES",
    "authorize",
    "check_body_size",
    "is_browser_originated",
    "is_loopback",
)
