"""capture.py — Lossless shadow-capture of real proxy traffic (v3.6.10, plan §7).

Purpose: build a dogfood corpus of real {request, response, model, tokens,
content_type} pairs so the cache + compression benchmark (benchmarks/optimize/)
can be replayed against authentic traffic instead of only synthetic prompts.

Activation: set ``SLM_OPTIMIZE_CAPTURE=1`` in the daemon's environment. When on:
  * the proxy runs in PURE PASSTHROUGH — cache + compression hooks are disabled
    at load time (see server._load_hooks), so capture never observes a mutated
    request or a cache hit; every line is a genuine upstream exchange.
  * each completed exchange is appended as one JSON line to
    ``~/.superlocalmemory/optimize_capture.jsonl`` (0600, gitignored).

ISOLATION GUARANTEE: this module writes ONLY to optimize_capture.jsonl. It never
opens memory.db, llmcache.db, or any SLM memory store. (Plan §9 hard rule.)

FAIL-OPEN: a capture failure (disk full, permission, encode error) is logged and
swallowed — it MUST NOT break the proxied request the user is waiting on.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from superlocalmemory.infra.data_root import state_path

logger = logging.getLogger("slm.optimize.proxy.capture")

_CAPTURE_FILENAME = "optimize_capture.jsonl"
_CAPTURE_ENV = "SLM_OPTIMIZE_CAPTURE"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Cap a single captured body so a pathological 10 MB request can't bloat the
# corpus line beyond what the replay harness will read back. Bodies above this
# are recorded truncated with a marker (capture is for benchmarking, not audit).
_MAX_CAPTURE_BODY_BYTES = 1 * 1024 * 1024  # 1 MB per side

# Providers whose response bodies follow the OpenAI usage schema
# ({"usage": {"prompt_tokens", "completion_tokens"}, "model"}).
_OPENAI_FORMAT_PROVIDERS = frozenset({"openai", "gemini-openai-compat"})


def capture_enabled() -> bool:
    """True iff ``SLM_OPTIMIZE_CAPTURE`` is set to a truthy value."""
    return os.environ.get(_CAPTURE_ENV, "").strip().lower() in _TRUTHY


def _capture_path() -> Path:
    return state_path(_CAPTURE_FILENAME)


class ShadowCapture:
    """Thread-safe append-only JSONL writer for proxy exchanges (singleton)."""

    _instance: "ShadowCapture | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _capture_path()
        self._write_lock = threading.Lock()
        self._count = 0

    @classmethod
    def get_instance(cls) -> "ShadowCapture":
        # Double-checked locking: cheap fast-path after first construction.
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Test hook — drop the singleton so a fresh path can be injected."""
        with cls._instance_lock:
            cls._instance = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def count(self) -> int:
        return self._count

    def record(self, entry: dict[str, Any]) -> bool:
        """Append one capture entry as a JSON line. Returns True on success.

        Fail-open: any error is logged and False is returned; never raised.

        Security: opens with a single ``os.open`` carrying ``O_CREAT |
        O_APPEND | O_NOFOLLOW`` and mode ``0o600`` on EVERY write. O_NOFOLLOW
        refuses a symlink pre-placed at the path (symlink-append attack), and
        the unconditional 0600-on-create removes the stat/exists TOCTOU that
        could otherwise drop the file to the process umask.
        """
        try:
            line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            logger.warning("capture: entry not JSON-serialisable, dropped: %r", exc)
            return False

        try:
            with self._write_lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                flags = os.O_CREAT | os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(self._path, flags, 0o600)
                with os.fdopen(fd, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                self._count += 1
            return True
        except OSError as exc:
            # PermissionError / symlink-refusal (ELOOP) are security-relevant —
            # surface the errno but still fail open so the request is never blocked.
            logger.warning("capture: write failed (fail-open): %r", exc)
            return False


def _truncate(raw: bytes) -> tuple[str, bool]:
    """Decode bytes for storage; truncate beyond the per-side cap."""
    truncated = len(raw) > _MAX_CAPTURE_BODY_BYTES
    head = raw[:_MAX_CAPTURE_BODY_BYTES] if truncated else raw
    return head.decode("utf-8", errors="replace"), truncated


def build_entry(
    *,
    provider: str,
    model: str,
    request_body: bytes,
    response_body: bytes,
    content_type: str,
    input_tokens: int,
    output_tokens: int,
    status_code: int,
    stream: bool,
) -> dict[str, Any]:
    """Construct a capture entry dict from a completed exchange.

    No timestamp is stamped here (Date.now is intentionally avoided in some
    runtimes); the replay harness keys on content, not time, and the file's own
    line order preserves arrival sequence.
    """
    req_str, req_trunc = _truncate(request_body)
    resp_str, resp_trunc = _truncate(response_body)
    return {
        "provider": provider,
        "model": model,
        "content_type": content_type,
        "stream": stream,
        "status_code": status_code,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "request": req_str,
        "response": resp_str,
        "request_truncated": req_trunc,
        "response_truncated": resp_trunc,
    }


def extract_usage(provider: str, body: bytes | None) -> tuple[int, int, str]:
    """Best-effort (input_tokens, output_tokens, model) from a provider JSON body.

    Works on the normalised JSON the SSE parsers emit AND on non-streaming
    upstream JSON. Returns (0, 0, "") when the body is missing/unparseable —
    capture must never fail because usage couldn't be read.
    """
    if not body:
        return 0, 0, ""
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0, 0, ""
    if not isinstance(data, dict):
        return 0, 0, ""

    if provider == "gemini":
        usage = data.get("usageMetadata") or {}
        return (
            int(usage.get("promptTokenCount", 0) or 0),
            int(usage.get("candidatesTokenCount", 0) or 0),
            str(data.get("modelVersion", "") or ""),
        )

    usage = data.get("usage") or {}
    if provider == "anthropic":
        return (
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
            str(data.get("model", "") or ""),
        )
    # OpenAI-format providers (explicit allowlist so a future provider variant
    # is not silently parsed with the wrong schema — it warns + returns zeros).
    if provider not in _OPENAI_FORMAT_PROVIDERS:
        logger.warning(
            "capture.extract_usage: unknown provider %r — recording zero tokens", provider
        )
        return 0, 0, str(data.get("model", "") or "")
    return (
        int(usage.get("prompt_tokens", 0) or 0),
        int(usage.get("completion_tokens", 0) or 0),
        str(data.get("model", "") or ""),
    )


def record_exchange(
    *,
    provider: str,
    model: str,
    request_body: bytes,
    response_body: bytes,
    content_type: str = "application/json",
    input_tokens: int = 0,
    output_tokens: int = 0,
    status_code: int = 200,
    stream: bool = False,
) -> bool:
    """Build + append a capture entry. Fail-open. Returns True on success."""
    entry = build_entry(
        provider=provider,
        model=model,
        request_body=request_body,
        response_body=response_body,
        content_type=content_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        status_code=status_code,
        stream=stream,
    )
    return ShadowCapture.get_instance().record(entry)


async def record_exchange_async(**kwargs: Any) -> bool:
    """Async wrapper for ``record_exchange`` that offloads the synchronous file
    write to a worker thread so it never blocks the proxy event loop — relevant
    when many streaming responses complete in the same loop iteration.
    """
    return await asyncio.to_thread(lambda: record_exchange(**kwargs))
