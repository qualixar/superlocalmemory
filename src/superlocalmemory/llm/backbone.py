# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""LLM backbone — unified interface for LLM providers.

Supports OpenAI, Anthropic, Azure OpenAI, and Ollama via raw HTTP (httpx).
Falls back gracefully when no API key is configured — Mode A still works.

Providers:
- ``"ollama"``: Local Ollama (OpenAI-compatible, no auth needed).
- ``"openai"``: OpenAI API (GPT-4o, etc.).
- ``"anthropic"``: Anthropic API (Claude, etc.).
- ``"azure"``: Azure OpenAI (via AI Foundry deployment).

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: Elastic-2.0
"""

from __future__ import annotations

import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Any

import httpx

from superlocalmemory.core.config import LLMConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_API_VERSION = "2023-06-01"
_AZURE_API_VERSION = "2024-12-01-preview"
_OLLAMA_DEFAULT_BASE = "http://localhost:11434"

_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "ollama": "OLLAMA_HOST",
    "openrouter": "OPENROUTER_API_KEY",
}

_SUPPORTED_PROVIDERS = frozenset({"openai", "anthropic", "azure", "ollama", "openrouter"})

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds, doubles each retry


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMUnavailableError(Exception):
    """Raised when no API key is available for the configured provider."""


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMResponse:
    """Immutable container for a single LLM generation result."""

    text: str
    model: str
    tokens_used: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LLMBackbone:
    """Unified LLM interface driven by LLMConfig.

    All HTTP via httpx — no provider SDKs needed.
    Includes retry logic (3 attempts, exponential backoff) and
    socket-level timeout as a hard backstop for SSL hangs (S15 lesson).
    """

    def __init__(self, config: LLMConfig) -> None:
        if config.provider and config.provider not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider '{config.provider}'. "
                f"Choose from {sorted(_SUPPORTED_PROVIDERS)}."
            )
        self._provider = config.provider
        self._model = config.model
        self._timeout = config.timeout_seconds
        self._default_temperature = config.temperature
        self._default_max_tokens = config.max_tokens

        # Resolve API key: config > environment variable.
        if self._provider == "ollama":
            self._api_key = ""
            host = config.api_base or os.environ.get(
                "OLLAMA_HOST", _OLLAMA_DEFAULT_BASE,
            )
            self._base_url = f"{host.rstrip('/')}/api/chat"
        elif self._provider == "openrouter":
            self._api_key = config.api_key or os.environ.get(
                _ENV_KEYS.get(self._provider, ""), "",
            )
            self._base_url = config.api_base or _OPENROUTER_URL
        elif self._provider:
            self._api_key = config.api_key or os.environ.get(
                _ENV_KEYS.get(self._provider, ""), "",
            )
            self._base_url = config.api_base
        else:
            self._api_key = ""
            self._base_url = ""

    # -- Properties ---------------------------------------------------------

    def is_available(self) -> bool:
        """True when the provider is ready for requests.

        For Ollama: always True (no API key needed). The num_ctx and
        keep_alive guards in _build_ollama() protect against memory spikes.
        The recall-path warm-only guard lives in Summarizer, not here —
        store/fact-extraction should always use the LLM in Mode B.
        """
        if not self._provider:
            return False
        if self._provider == "ollama":
            return True
        # v3.6.12 (modeb-1): a custom local OpenAI-compatible endpoint
        # (llama.cpp, LM Studio, vLLM) needs NO API key — _build_openai already
        # omits the Authorization header when the key is empty. Treat a
        # configured base_url as sufficient, otherwise Mode B silently falls
        # back to Mode A extraction for keyless local endpoints.
        _base = getattr(self, "_base_url", "") or getattr(self, "_api_base", "")
        return bool(self._api_key) or bool(_base)

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    # -- Core generation ----------------------------------------------------

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send prompt to the LLM and return generated text.

        Returns empty string on content-filter errors (Azure 400)
        instead of crashing — lets callers continue gracefully.
        Retries up to 3 times with exponential backoff on transient errors.
        """
        if not self.is_available():
            raise LLMUnavailableError(
                f"No API key for provider '{self._provider}'. "
                f"Set {_ENV_KEYS.get(self._provider, 'API_KEY')} or pass api_key=."
            )

        temp = temperature if temperature is not None else self._default_temperature
        tokens = max_tokens if max_tokens is not None else self._default_max_tokens
        url, headers, payload = self._build_request(prompt, system, tokens, temp)

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._send(url, headers, payload)
                return self._extract_text(response)
            except httpx.HTTPStatusError as exc:
                # Azure content filter returns 400 — not retryable.
                if exc.response.status_code == 400:
                    logger.warning("Content filter or bad request (400). Returning empty.")
                    return ""
                last_error = exc
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_error = exc

            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.info("Retry %d/%d after %.1fs", attempt + 1, _MAX_RETRIES, delay)
                time.sleep(delay)

        logger.error("All %d retries exhausted: %s", _MAX_RETRIES, last_error)
        return ""

    # -- HTTP transport -----------------------------------------------------

    def _send(self, url: str, headers: dict, payload: dict) -> dict:
        """Execute HTTP POST with socket-level SSL backstop."""
        old_default = socket.getdefaulttimeout()
        socket.setdefaulttimeout(self._timeout + 30)
        try:
            timeout = httpx.Timeout(
                connect=10.0,
                read=self._timeout,
                write=10.0,
                pool=10.0,
            )
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                return resp.json()
        finally:
            socket.setdefaulttimeout(old_default)

    # -- Request builders ---------------------------------------------------

    def _build_request(
        self, prompt: str, system: str, max_tokens: int, temperature: float,
    ) -> tuple[str, dict[str, str], dict]:
        """Build provider-specific (url, headers, payload)."""
        builders = {
            "ollama": self._build_ollama,
            "anthropic": self._build_anthropic,
            "azure": self._build_azure,
        }
        builder = builders.get(self._provider, self._build_openai)
        return builder(prompt, system, max_tokens, temperature)

    def _build_openai(
        self, prompt: str, system: str, max_tokens: int, temperature: float,
    ) -> tuple[str, dict[str, str], dict]:
        messages = self._make_messages(system, prompt)
        # V3.5.9: Skip Authorization header when api_key is empty so unauthenticated
        # local endpoints (llama.cpp, LM Studio, etc.) don't reject with HTTP 401.
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # V3.5.9: If base_url is a bare base (e.g. "http://host:port/v1") without
        # the /chat/completions path, append it — matching OpenAI SDK behaviour and
        # fixing the /v1/v1 duplication reported in issue #29.
        url = self._base_url or _OPENAI_URL
        if self._base_url and not self._base_url.rstrip("/").endswith("chat/completions"):
            url = f"{self._base_url.rstrip('/')}/chat/completions"
        return url, headers, payload

    def _build_ollama(
        self, prompt: str, system: str, max_tokens: int, temperature: float,
    ) -> tuple[str, dict[str, str], dict]:
        messages = self._make_messages(system, prompt)
        headers = {"Content-Type": "application/json"}
        # Native /api/chat format — NOT /v1/chat/completions.
        # The OpenAI-compatible endpoint silently ignores options.num_ctx,
        # causing Ollama to use the model's default (131K for llama3.1 = 30 GB).
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "keep_alive": "30s",
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "num_ctx": 4096,
            },
        }
        return self._base_url, headers, payload

    def _build_anthropic(
        self, prompt: str, system: str, max_tokens: int, temperature: float,
    ) -> tuple[str, dict[str, str], dict]:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_API_VERSION,
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        return _ANTHROPIC_URL, headers, payload

    def _build_azure(
        self, prompt: str, system: str, max_tokens: int, temperature: float,
    ) -> tuple[str, dict[str, str], dict]:
        if not self._base_url:
            raise ValueError("Azure provider requires api_base URL.")
        url = (
            f"{self._base_url.rstrip('/')}/openai/deployments/"
            f"{self._model}/chat/completions"
            f"?api-version={_AZURE_API_VERSION}"
        )
        messages = self._make_messages(system, prompt)
        headers = {"api-key": self._api_key, "Content-Type": "application/json"}
        payload: dict[str, Any] = {"messages": messages}
        if "gpt-5" in self._model.lower():
            payload["max_completion_tokens"] = max(max_tokens, 200)
            payload["reasoning_effort"] = "none"
            if temperature > 0:
                payload["temperature"] = temperature
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature
        return url, headers, payload

    # -- Response parsing ---------------------------------------------------

    def _extract_text(self, data: dict) -> str:
        """Extract text from provider-specific JSON response."""
        if self._provider == "anthropic":
            return data.get("content", [{}])[0].get("text", "").strip()
        if self._provider == "ollama":
            # Native /api/chat: {"message": {"content": "..."}}
            return data.get("message", {}).get("content", "").strip()
        # OpenAI / Azure share response format.
        choices = data.get("choices", [{}])
        return choices[0].get("message", {}).get("content", "").strip()

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _make_messages(system: str, prompt: str) -> list[dict[str, str]]:
        """Build messages array with optional system message."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages
