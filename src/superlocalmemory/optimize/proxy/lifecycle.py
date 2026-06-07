"""lifecycle.py — Hook Protocol interfaces per INTERFACE-CONTRACT §3 (FROZEN).

TYPE NAME CANONICAL RULE:
  ProxyRequest   — request DTO
  CachedResponse — cache hit/miss result
  ProviderResponse — compress result / passthrough body
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol, runtime_checkable


# ─── Data types ──────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class ProxyRequest:
    """Immutable request snapshot passed to every hook.

    headers MUST be redacted before construction (CWE-532 guard).
    """
    provider: str
    method: str
    path: str
    headers: dict
    body: dict
    body_bytes: bytes
    request_id: str
    stream: bool
    has_tools: bool

    def __repr__(self) -> str:
        return (
            f"ProxyRequest(provider={self.provider!r}, method={self.method!r}, "
            f"path={self.path!r}, request_id={self.request_id!r}, "
            f"stream={self.stream}, has_tools={self.has_tools})"
        )


@dataclasses.dataclass
class CachedResponse:
    """Cache check result."""
    hit: bool
    data: bytes | None
    cache_key: str
    ttl_seconds: int


@dataclasses.dataclass
class ProviderResponse:
    """Compress-modified body container / passthrough body shape."""
    modified: bool
    body: dict
    body_bytes: bytes
    tokens_before: int
    tokens_after: int
    strategy: str


# ─── Hook Protocols — INTERFACE-CONTRACT §3 ────────────────────────────────


@runtime_checkable
class CacheHook(Protocol):
    def check(self, req: ProxyRequest) -> CachedResponse | None: ...
    def store(self, req: ProxyRequest, resp: ProviderResponse) -> None: ...
    def on_hit(self, req: ProxyRequest, resp: bytes, tokens_saved: int) -> None: ...
    def on_miss(self, req: ProxyRequest) -> None: ...


@runtime_checkable
class CompressHook(Protocol):
    def compress(self, req: ProxyRequest) -> ProxyRequest: ...
    def on_compress(self, before_tokens: int, after_tokens: int, lossy: bool) -> None: ...


@dataclasses.dataclass
class HookChain:
    cache: CacheHook | None = None
    compress: CompressHook | None = None

    @classmethod
    def empty(cls) -> "HookChain":
        return cls(cache=None, compress=None)


# ─── Lifecycle exports (INTERFACE-CONTRACT v2.2) ───────────────────────────


def ensure_proxy_running() -> bool:
    """Ensure the proxy is configured AND alive (liveness probe).

    Checks two conditions in order:
    1. ``proxy_enabled`` is True in optimize.json (config gate).
    2. The proxy HTTP server responds to GET /health within 1 second (liveness
       gate).  This verifies that ProxyApp.startup() was actually called and
       the httpx.AsyncClient is non-None — a detail the config flag alone
       cannot confirm.

    Returns True only when both gates pass, False otherwise.
    """
    try:
        from superlocalmemory.optimize.config import get_optimize_config
        cfg = get_optimize_config()
        if not cfg.proxy_enabled:
            return False
    except Exception:
        return False

    # Liveness probe — confirm the daemon is actually listening.
    try:
        import urllib.request
        port = proxy_port()
        url = f"http://127.0.0.1:{port}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False


def proxy_port() -> int:
    """Returns the proxy port (always 8765 — shared with SLM daemon)."""
    return 8765
