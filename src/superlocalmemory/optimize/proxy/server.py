"""server.py — ProxyApp and build_proxy_router()."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from fastapi import APIRouter
from fastapi.requests import Request
from fastapi.responses import Response

from superlocalmemory.optimize.config.schema import OptimizeConfig
from superlocalmemory.optimize.proxy.lifecycle import HookChain

logger = logging.getLogger("slm.optimize.proxy")

_PROXY_VERSION = "3.6.3"
_REQUEST_TIMEOUT_S = 300.0
_CONNECT_TIMEOUT_S = 10.0
_MAX_CONNECTIONS = 100
_MAX_KEEPALIVE = 20
_MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024  # 10 MB


class ProxyApp:
    """Core proxy. Holds httpx client + hook chain.

    Lifecycle:
      create_app() → ProxyApp() → application.state.optimize_proxy = proxy
      lifespan startup → await proxy.startup()
      lifespan shutdown → await proxy.shutdown()
    """

    def __init__(self, config: OptimizeConfig) -> None:
        self.config = config
        self.hooks: HookChain = HookChain.empty()
        self.http_client: httpx.AsyncClient | None = None
        self._request_counter: int = 0
        self._counter_lock = asyncio.Lock()

    async def startup(self) -> None:
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=_CONNECT_TIMEOUT_S,
                read=_REQUEST_TIMEOUT_S,
                write=_REQUEST_TIMEOUT_S,
                pool=_CONNECT_TIMEOUT_S,
            ),
            limits=httpx.Limits(
                max_connections=_MAX_CONNECTIONS,
                max_keepalive_connections=_MAX_KEEPALIVE,
            ),
            follow_redirects=False,
        )
        self.hooks = _load_hooks(self.config)
        logger.info(
            "slm.optimize.proxy started version=%s port=8765 "
            "cache_hook=%s compress_hook=%s",
            _PROXY_VERSION,
            type(self.hooks.cache).__name__ if self.hooks.cache else "None",
            type(self.hooks.compress).__name__ if self.hooks.compress else "None",
        )

    async def shutdown(self) -> None:
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        logger.info("slm.optimize.proxy shut down")

    async def next_request_id(self) -> str:
        async with self._counter_lock:
            self._request_counter += 1
            return f"slm_{int(time.monotonic() * 1000)}_{self._request_counter:06d}"

    def reload_from_config(self, config: OptimizeConfig) -> None:
        """Hot-swap cache/compress behavior when optimize.json changes (v3.6.10).

        Called by the ConfigStore change-callback (UI save → immediate; external
        file/CLI edit → within the 2s watchdog poll). Rebuilds the HookChain so
        ``cache_enabled`` and ``compress_enabled`` can be toggled INDEPENDENTLY
        at runtime with no daemon restart. Note: ``proxy_enabled`` (whether the
        proxy claims /v1/* at all) is a startup decision and is NOT changed here.
        """
        self.config = config
        self.hooks = _load_hooks(config)
        logger.info(
            "slm.optimize.proxy reloaded (config v%s): cache=%s compress=%s",
            getattr(config, "config_version", "?"),
            type(self.hooks.cache).__name__ if self.hooks.cache else "None",
            type(self.hooks.compress).__name__ if self.hooks.compress else "None",
        )


def build_proxy_router(proxy: ProxyApp) -> APIRouter:
    """Build and return the FastAPI router for all proxy surfaces."""
    from superlocalmemory.optimize.proxy.anthropic_surface import (
        handle_count_tokens,
        handle_messages,
        handle_models,
    )
    from superlocalmemory.optimize.proxy.gemini_surface import (
        handle_gemini_native,
        handle_gemini_openai_compat,
    )
    from superlocalmemory.optimize.proxy.openai_surface import (
        handle_chat_completions,
        handle_embeddings,
    )

    router = APIRouter(tags=["slm-optimize-proxy"])

    @router.post("/v1/messages")
    async def messages_route(request: Request) -> Response:
        return await handle_messages(proxy, request)

    @router.post("/v1/messages/count_tokens")
    async def count_tokens_route(request: Request) -> Response:
        return await handle_count_tokens(proxy, request)

    @router.get("/v1/models")
    async def models_route(request: Request) -> Response:
        return await handle_models(proxy, request)

    @router.post("/v1/chat/completions")
    async def chat_completions_route(request: Request) -> Response:
        return await handle_chat_completions(proxy, request)

    @router.post("/v1/embeddings")
    async def embeddings_route(request: Request) -> Response:
        return await handle_embeddings(proxy, request)

    @router.post("/v1beta/models/{model_and_method:path}")
    async def gemini_native_route(
        request: Request, model_and_method: str
    ) -> Response:
        return await handle_gemini_native(proxy, request, model_and_method)

    @router.post("/v1beta/openai/chat/completions")
    async def gemini_openai_post_route(request: Request) -> Response:
        return await handle_gemini_openai_compat(proxy, request)

    @router.get("/v1beta/openai/models")
    async def gemini_openai_models_route(request: Request) -> Response:
        return await handle_gemini_openai_compat(proxy, request)

    return router


def _load_hooks(config: OptimizeConfig) -> HookChain:
    # v3.6.10 shadow-capture (plan §7): capture mode is PURE passthrough — no
    # cache, no compression — so the corpus records only authentic upstream
    # exchanges. This is defense-in-depth alongside the per-surface guard.
    from superlocalmemory.optimize.proxy.capture import capture_enabled
    if capture_enabled():
        logger.info(
            "slm.optimize.proxy: SLM_OPTIMIZE_CAPTURE on — cache/compress "
            "DISABLED, recording exchanges to optimize_capture.jsonl"
        )
        return HookChain.empty()

    cache_hook = None
    compress_hook = None

    if config.cache_enabled:
        try:
            from superlocalmemory.optimize.cache.manager import CacheManager
            cache_hook = CacheManager.get_instance()
        except Exception as exc:
            logger.warning(
                "cache hook load failed (proxy continues without cache): %s", exc
            )

    if config.compress_enabled:
        try:
            from superlocalmemory.optimize.compress.router import CompressRouter
            from superlocalmemory.optimize.metrics.counters import MetricsCollector
            compress_hook = CompressRouter.get_instance()
            compress_hook.set_metrics(MetricsCollector.get_instance())
        except Exception as exc:
            logger.warning(
                "compress hook load failed (proxy continues without compress): %s", exc
            )

    return HookChain(cache=cache_hook, compress=compress_hook)
