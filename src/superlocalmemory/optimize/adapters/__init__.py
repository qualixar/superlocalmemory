"""optimize/adapters — SDK adapters for OpenAI / Anthropic / LangChain.

SECURITY: No pickle anywhere. JSON+pydantic only (SEC-C-04, CWE-502).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("superlocalmemory.optimize.adapters")


def withSLM(client: Any, *, tenant_id: str = "default") -> Any:
    """Wrap any supported SDK client with SLM cache hooks.

    FAIL-OPEN GUARANTEE:
      - If Optimize is disabled (config.enabled == False) → return client unchanged.
      - If cache is unavailable → return client unchanged, log WARNING.
      - If client type unrecognized → return client unchanged, log WARNING.
    """
    try:
        from superlocalmemory.optimize.config import get_optimize_config
        from superlocalmemory.optimize.cache.manager import CacheManager
    except ImportError as exc:
        logger.warning("SLM Optimize not available (%s) — pass-through active", exc)
        return client

    config = get_optimize_config()
    if not config.enabled:
        return client  # DEFAULT STATE — zero behavioral change

    try:
        cache_manager = CacheManager.get_instance()
    except Exception as exc:
        logger.warning("SLM cache unavailable (%s) — pass-through active", exc)
        return client

    # Anthropic (.messages.create)
    if hasattr(client, "messages") and hasattr(client.messages, "create"):
        try:
            from superlocalmemory.optimize.adapters.anthropic_adapter import (
                SLMAnthropicAdapter,
            )
            return SLMAnthropicAdapter(client, cache_manager, config, tenant_id)
        except Exception as exc:
            logger.warning("SLM Anthropic adapter init failed: %s", exc)
            return client

    # OpenAI (.chat.completions)
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        try:
            from superlocalmemory.optimize.adapters.openai_adapter import (
                SLMOpenAIAdapter,
            )
            return SLMOpenAIAdapter(client, cache_manager, config, tenant_id)
        except Exception as exc:
            logger.warning("SLM OpenAI adapter init failed: %s", exc)
            return client

    logger.warning(
        "withSLM: unrecognized client type %s — pass-through active",
        type(client).__name__,
    )
    return client


__all__ = ["withSLM"]
