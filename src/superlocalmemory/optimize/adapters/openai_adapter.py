"""SLM OpenAI SDK adapter."""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from superlocalmemory.optimize.cache.manager import CacheManager
    from superlocalmemory.optimize.config.schema import OptimizeConfig

logger = logging.getLogger(__name__)


def _detect_async_openai(client: Any) -> bool:
    try:
        from openai import AsyncOpenAI
        return isinstance(client, AsyncOpenAI)
    except ImportError:
        return False


class SLMOpenAIAdapter:
    """Wraps openai.OpenAI (or AsyncOpenAI) with SLM cache hooks."""

    def __init__(
        self,
        client: Any,
        cache_manager: "CacheManager",
        config: "OptimizeConfig",
        tenant_id: str = "default",
    ) -> None:
        self._original = client
        self._cache_manager = cache_manager
        self._cache_view = cache_manager.for_tenant(tenant_id)
        self._config = config
        self._tenant_id = tenant_id
        self._is_async: bool = _detect_async_openai(client)
        self.chat = _ChatCompletionsProxy(self)

    def _build_cache_key(self, kwargs: dict) -> str:
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        exclude = {"model", "messages", "stream", "user", "request_id"}
        params = {k: v for k, v in kwargs.items() if k not in exclude}
        canonical = json.dumps(
            {"messages": messages, "params": params},
            sort_keys=True, default=str,
        )
        return self._cache_manager.build_key(
            {"model": model, "messages": messages, "params": params},
            self._tenant_id,
        )

    def _should_cache(self, kwargs: dict) -> bool:
        if kwargs.get("stream", False):
            return False
        if kwargs.get("tools") or kwargs.get("functions"):
            return False
        return True

    def _chat_create_sync(self, **kwargs: Any) -> Any:
        if not self._config.cache_enabled or not self._should_cache(kwargs):
            return self._original.chat.completions.create(**kwargs)
        key = self._build_cache_key(kwargs)
        if key is None:
            return self._original.chat.completions.create(**kwargs)
        hit = self._cache_view.get(key)
        if hit is not None:
            try:
                import json as _json
                return _json.loads(hit.decode("utf-8"))
            except Exception:
                return self._original.chat.completions.create(**kwargs)
        response = self._original.chat.completions.create(**kwargs)
        try:
            import json as _json
            encoded = _json.dumps(response, default=str).encode("utf-8")
            self._cache_view.set(key, encoded)
        except Exception as exc:
            logger.warning("SLMOpenAIAdapter: failed to cache response: %s", exc)
        return response

    async def _chat_create_async(self, **kwargs: Any) -> Any:
        if not self._config.cache_enabled or not self._should_cache(kwargs):
            return await self._original.chat.completions.create(**kwargs)
        key = self._build_cache_key(kwargs)
        if key is None:
            return await self._original.chat.completions.create(**kwargs)
        hit = self._cache_view.get(key)
        if hit is not None:
            try:
                import json as _json
                return _json.loads(hit.decode("utf-8"))
            except Exception:
                return await self._original.chat.completions.create(**kwargs)
        response = await self._original.chat.completions.create(**kwargs)
        try:
            import json as _json
            encoded = _json.dumps(response, default=str).encode("utf-8")
            self._cache_view.set(key, encoded)
        except Exception as exc:
            logger.warning("SLMOpenAIAdapter: failed to cache response: %s", exc)
        return response


class _ChatCompletionsProxy:
    """Mimics openai.resources.chat.Chat interface: `client.chat.completions.create`."""

    def __init__(self, adapter: SLMOpenAIAdapter) -> None:
        self._adapter = adapter
        self.completions = self._Completions(adapter)

    class _Completions:
        def __init__(self, adapter: SLMOpenAIAdapter) -> None:
            self._adapter = adapter

        def create(self, **kwargs: Any) -> Any:
            if self._adapter._is_async:
                return self._adapter._chat_create_async(**kwargs)
            return self._adapter._chat_create_sync(**kwargs)
