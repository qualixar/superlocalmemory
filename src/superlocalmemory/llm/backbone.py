# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""LLM backbone — unified interface for LLM providers.

Supports OpenAI, Anthropic, Azure OpenAI, and Ollama via raw HTTP (httpx),
sent through the outbound gate (``core.outbound_http``).
Falls back gracefully when no API key is configured — Mode A still works.

Providers:
- ``"ollama"``: Local Ollama (OpenAI-compatible, no auth needed).
- ``"openai"``: OpenAI API (GPT-4o, etc.).
- ``"anthropic"``: Anthropic API (Claude, etc.).
- ``"azure"``: Azure OpenAI (via AI Foundry deployment).

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
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
from superlocalmemory.core import outbound_http
from superlocalmemory.core.outbound_redaction import for_endpoint, is_local_endpoint

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

#: Longest a background generation waits for recalls to finish before it goes
#: ahead anyway, so steady recall load can never park extraction for good.
LLM_YIELD_MAX_S = 20.0


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _yield_to_recall() -> None:
    """Background generation (Mode B extraction, summaries) waits while a recall runs.

    A local model serves one request at a time; a background prompt that takes
    the model first makes every recall behind it wait for a whole generation.
    Foreground calls (a recall's own answer check) never wait here.
    """
    from superlocalmemory.core import recall_gate

    if not recall_gate.is_background_work() or recall_gate.in_flight() == 0:
        return
    import time as _time

    with recall_gate.idle_wait_deadline(_time.monotonic() + LLM_YIELD_MAX_S):
        recall_gate.wait_for_foreground_idle()


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
        # Set once an Ollama server rejects the think field (HTTP 400), so an
        # old server costs one extra request per process, not one per call.
        self._think_unsupported = False

        # Resolve API key: config > environment variable. A custom api_base
        # with an empty api_key is treated as intentionally keyless so local
        # OpenAI-compatible endpoints do not inherit ambient cloud keys.
        if self._provider == "ollama":
            self._api_key = ""
            host = config.api_base or os.environ.get(
                "OLLAMA_HOST", _OLLAMA_DEFAULT_BASE,
            )
            self._ollama_host = host.rstrip("/")
            self._base_url = f"{self._ollama_host}/api/chat"
        elif self._provider == "openrouter":
            self._api_key = self._resolve_api_key(config)
            self._base_url = config.api_base or _OPENROUTER_URL
        elif self._provider:
            self._api_key = self._resolve_api_key(config)
            self._base_url = config.api_base
        else:
            self._api_key = ""
            self._base_url = ""

    def _resolve_api_key(self, config: LLMConfig) -> str:
        if config.api_key:
            return config.api_key
        if config.api_base:
            return ""
        return os.environ.get(_ENV_KEYS.get(self._provider, ""), "")

    # -- Properties ---------------------------------------------------------

    def is_available(self) -> bool:
        """True when the provider is ready for requests.

        For Ollama: no API key is ever needed, but that is not the same
        question as whether anything is actually listening at the
        configured host. 4.1.22: a saved Mode B with no Ollama running (or
        one that has since stopped) reported available forever, so Mode B
        never degraded to Mode A the way every caller of this method
        already expects to for every other unavailable provider. The
        reachability check itself is bounded and cached (see
        ``ollama_reachability``), so this stays cheap on the hot,
        once-per-fact paths that call it. The num_ctx and keep_alive guards
        in _build_ollama() protect against memory spikes once a request is
        actually sent.
        """
        if not self._provider:
            return False
        if self._provider == "ollama":
            from superlocalmemory.llm.ollama_reachability import ollama_reachable

            return ollama_reachable(self._ollama_host)
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
        think: bool | None = None,
        *,
        attempts: int | None = None,
    ) -> str:
        """Send prompt to the LLM and return generated text.

        Returns empty string on content-filter errors (Azure 400)
        instead of crashing — lets callers continue gracefully.
        Retries up to 3 times with exponential backoff on transient errors.

        ``think`` — Ollama-only reasoning control: explicit ``False``
        sends ``think: false`` (content-only answers from thinking
        models); ``None``/``True`` omit the key and keep model defaults.
        ``True`` is never sent: non-thinking models reject it.

        ``attempts`` caps the tries (default three, with backoff). A caller on
        a write path that has its own fallback passes 1, so a provider that is
        down costs one failed request, not seconds of retries.
        """
        if not self.is_available():
            raise LLMUnavailableError(
                f"No API key for provider '{self._provider}'. "
                f"Set {_ENV_KEYS.get(self._provider, 'API_KEY')} or pass api_key=."
            )

        _yield_to_recall()
        temp = temperature if temperature is not None else self._default_temperature
        tokens = max_tokens if max_tokens is not None else self._default_max_tokens
        url, headers, payload = self._build_request(prompt, system, tokens, temp, think)
        if not is_local_endpoint(url):
            # Memories keep their credentials; a model on another machine
            # never receives them. Decided on the final URL, for every provider.
            url, headers, payload = self._build_request(
                for_endpoint(prompt, url), for_endpoint(system, url), tokens, temp, think,
            )

        last_error: Exception | None = None
        think_downgraded = False
        tries = _MAX_RETRIES if attempts is None else max(1, int(attempts))
        for attempt in range(tries):
            try:
                response = self._send(url, headers, payload)
                return self._extract_text(response)
            except httpx.HTTPStatusError as exc:
                # Azure content filter returns 400 — not retryable.
                if exc.response.status_code == 400:
                    # 4.1.14 audit: except ONE case — a 400 on a request
                    # carrying an explicit think:false is plausibly an old
                    # server rejecting the field, not a content filter.
                    # Downgrade once to the model default and retry; a
                    # second 400 is a real refusal and returns empty.
                    # Keyed off the payload (covers both the think=False
                    # argument and the SLM_OLLAMA_DISABLE_THINK env flag).
                    if (
                        not think_downgraded
                        and payload.pop("think", None) is not None
                    ):
                        think_downgraded = True
                        self._think_unsupported = True
                        logger.info(
                            "Ollama rejected think:false (HTTP 400); "
                            "retrying once with model defaults.",
                        )
                        continue
                    logger.warning("Content filter or bad request (400). Returning empty.")
                    return ""
                last_error = exc
            except (httpx.TimeoutException, httpx.ConnectError, ValueError) as exc:
                last_error = exc

            if attempt < tries - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.info("Retry %d/%d after %.1fs", attempt + 1, tries, delay)
                time.sleep(delay)

        logger.error("All %d retries exhausted: %s", tries, last_error)
        return ""

    # -- HTTP transport -----------------------------------------------------

    def _send(self, url: str, headers: dict, payload: dict) -> dict:
        """Execute HTTP POST with socket-level SSL backstop.

        Through ``core.outbound_http``: the body is screened again on the
        final URL, a local model is never reached through a proxy, and a
        redirect is never followed.
        """
        old_default = socket.getdefaulttimeout()
        socket.setdefaulttimeout(self._timeout + 30)
        try:
            timeout = httpx.Timeout(
                connect=10.0,
                read=self._timeout,
                write=10.0,
                pool=10.0,
            )
            resp = outbound_http.post_json(url, payload, headers=headers,
                                           timeout=timeout)
            resp.raise_for_status()
            try:
                return resp.json()
            except Exception as exc:
                raise ValueError(
                    f"LLM endpoint returned HTTP {resp.status_code} with "
                    f"non-JSON body (Content-Length="
                    f"{resp.headers.get('content-length', '?')}): {exc}"
                ) from exc
        finally:
            socket.setdefaulttimeout(old_default)

    # -- Request builders ---------------------------------------------------

    def _build_request(
        self, prompt: str, system: str, max_tokens: int, temperature: float,
        think: bool | None = None,
    ) -> tuple[str, dict[str, str], dict]:
        """Build provider-specific (url, headers, payload)."""
        builders = {
            "ollama": self._build_ollama,
            "anthropic": self._build_anthropic,
            "azure": self._build_azure,
        }
        builder = builders.get(self._provider, self._build_openai)
        if self._provider == "ollama":
            return builder(prompt, system, max_tokens, temperature, think=think)
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
        think: bool | None = None,
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
        if self._ollama_think_off(think):
            payload["think"] = False
        return self._base_url, headers, payload

    def _ollama_think_off(self, think: bool | None) -> bool:
        """Whether this request carries ``think: false``.

        4.2.9: off unless asked for. Every SLM prompt wants a short structured
        answer; a thinking model's default reasoning costs ~6x the tokens
        (gemma4:e2b: ~410 vs ~75 per context call) and, under a token cap,
        can leave content empty. ``True`` is never sent (non-thinking models
        reject it): it, or ``SLM_OLLAMA_ALLOW_THINK=1``, keeps the model's
        default. ``SLM_OLLAMA_DISABLE_THINK=1`` (#128) still forces it off.
        A server that rejected the field once is not asked again.
        """
        if _env_flag("SLM_OLLAMA_DISABLE_THINK"):
            return not self._think_unsupported
        if self._think_unsupported or think is True:
            return False
        if think is False:
            return True
        return not _env_flag("SLM_OLLAMA_ALLOW_THINK")

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
        # Respect custom base_url (e.g. Anthropic-compatible proxy).
        # Append /v1/messages to the root URL, mirroring how _build_openai
        # handles api_base. Falls back to the official Anthropic endpoint.
        url = (
            self._base_url.rstrip("/") + "/v1/messages"
            if self._base_url else _ANTHROPIC_URL
        )
        return url, headers, payload

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
            # Thinking models (qwen3.x, DeepSeek-R1 class) put the whole
            # generation in message.thinking and leave content empty when
            # the task triggers reasoning — without the fallback below,
            # Mode B silently degrades to Mode A (#128). The extractor's
            # array recovery parses the answer out of the trace.
            message = data.get("message", {})
            if not isinstance(message, dict):
                return ""
            content = message.get("content", "")
            content = content.strip() if isinstance(content, str) else ""
            thinking = message.get("thinking", "")
            thinking = thinking.strip() if isinstance(thinking, str) else ""
            if content and thinking:
                # 4.1.14 audit: feed BOTH fields to the parser instead of
                # guessing which holds the answer. Thinking goes FIRST:
                # selection prefers later arrays on schema-fit ties, and
                # the content channel is the model's primary answer —
                # thinking-trace trials must never outrank it. Prose-only
                # content and bracket noise in either field stay
                # recoverable through array selection.
                logger.debug(
                    "Ollama thinking response carries both fields; "
                    "extracting from thinking+content (%d+%d chars).",
                    len(thinking), len(content),
                )
                return thinking + "\n" + content
            if thinking:
                # DEBUG, not INFO: for thinking models empty content is the
                # normal response shape and generate() sits on the store hot
                # path — INFO here would write a line per Mode B store into
                # daemon.log. True emptiness still warns downstream.
                logger.debug(
                    "Ollama thinking model returned empty content; "
                    "extracting from message.thinking (%d chars).",
                    len(thinking),
                )
            return thinking or content
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
